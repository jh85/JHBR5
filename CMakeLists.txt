cmake_minimum_required(VERSION 3.18)
project(jhbr5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Temporary one-shot diagnostics for tracing non-finite NN/MCTS values.  The
# validation and containment remain compiled in when this is disabled.
option(ENABLE_NN_DIAGNOSTICS "Log the first invalid NN/MCTS value" ON)
if(ENABLE_NN_DIAGNOSTICS)
  add_compile_definitions(JHBR3_ENABLE_NN_DIAGNOSTICS=1)
endif()

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ---- Instruction set ----
# JHBR5 selects the SIMD level explicitly (no -march=native) so that binaries
# for the AVX2 development machine and the AVX-512 target are reproducible.
# The NNUE kernels (nnue/simd*.h) follow the same option; `scalar` keeps the
# reference kernels only (SSE4.2 is still assumed for the bitboard code).
set(JHBR5_ISA "avx2" CACHE STRING "SIMD level: scalar, avx2 or avx512")
set_property(CACHE JHBR5_ISA PROPERTY STRINGS scalar avx2 avx512)
if(JHBR5_ISA STREQUAL "avx2")
  add_compile_options(-msse4.2 -mpopcnt -mavx2 -mfma -mbmi -mbmi2)
  add_compile_definitions(JHBR5_SIMD_AVX2)
elseif(JHBR5_ISA STREQUAL "avx512")
  add_compile_options(-msse4.2 -mpopcnt -mavx2 -mfma -mbmi -mbmi2
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni)
  add_compile_definitions(JHBR5_SIMD_AVX512)
elseif(JHBR5_ISA STREQUAL "scalar")
  add_compile_options(-msse4.2 -mpopcnt)
  add_compile_definitions(JHBR5_SIMD_SCALAR)
else()
  message(FATAL_ERROR "JHBR5_ISA must be scalar, avx2 or avx512 (got ${JHBR5_ISA})")
endif()
message(STATUS "JHBR5_ISA: ${JHBR5_ISA}")

enable_testing()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

# ---- Sources ----

# Shogi engine (board, bitboard, types, packed sfen)
set(SHOGI_ENGINE_SRCS
  shogi/board.cc
  shogi/bitboard.cc
  shogi/mate1ply.cc
  shogi/packed_sfen.cc
)

set(MCTS_SRCS
  mcts/search_primitives.cc
  mcts/uct_node.cc
  mcts/uct_search.cc
)

# Mate solver
set(MATE_SRCS
  mate/dfpn.cc
  mate/bns.cc
)

# USI
set(USI_SRCS
  usi/search_info.cc
  usi/usi_engine.cc
)

# ---- Include directories ----
# The shogi/ headers use #include "shogi/board.h" — directory and
# include path now match, no shim needed.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: shogi engine core ----
add_library(shogi_core STATIC ${SHOGI_ENGINE_SRCS})
target_include_directories(shogi_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: opening book ----
add_library(opening_book STATIC book/opening_book.cc)
target_include_directories(opening_book PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(opening_book PUBLIC shogi_core)

# ---- Library: pure move-time calculation ----
add_library(time_manager STATIC usi/time_manager.cc)
target_include_directories(time_manager PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# ---- Library: mate solver ----
add_library(mate_solver STATIC ${MATE_SRCS})
target_link_libraries(mate_solver PUBLIC shogi_core)

# =====================================================================
# JHBR5 — CPU NNUE backend
# =====================================================================
set(NNUE_SRCS
  nnue/features.cc
  nnue/move_buckets.cc
  nnue/see.cc
  nnue/net_format.cc
  nnue/value_net.cc
  nnue/policy_net.cc
  nnue/simd.cc
  nnue/random_net.cc
  nnue/record.cc
)
add_library(nnue STATIC ${NNUE_SRCS})
target_include_directories(nnue PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nnue PUBLIC shogi_core)

# ---- Library: dlshogi-style MCTS ----
add_library(mcts STATIC ${MCTS_SRCS})
target_include_directories(mcts PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcts PUBLIC nnue mate_solver time_manager shogi_core)

add_executable(jhbr5 main.cc datagen/datagen.cc ${USI_SRCS})
target_link_libraries(jhbr5 PRIVATE mcts mate_solver
                                    opening_book time_manager shogi_core)

# ---- Test: move generation correctness ----
add_executable(test_movegen test/test_movegen.cc)
target_link_libraries(test_movegen PRIVATE shogi_core)

# ---- Test: maximum legal move bound ----
add_executable(test_max_legal_moves test/test_max_legal_moves.cc)
target_link_libraries(test_max_legal_moves PRIVATE shogi_core)

# ---- Test: MCTS subtree reuse ----
add_executable(test_tree_reuse
  test/test_tree_reuse.cc
  mcts/uct_node.cc
)
target_link_libraries(test_tree_reuse PRIVATE shogi_core)

# ---- Test: multi-threaded MCTS with NNUE evaluation ----
add_executable(test_lockfree_search test/test_lockfree_search.cc)
target_link_libraries(test_lockfree_search PRIVATE mcts)

# ---- Test: root vs descendant repetition search policy ----
add_executable(test_search_repetition test/test_search_repetition.cc)
target_link_libraries(test_search_repetition PRIVATE shogi_core)

# ---- Test: MCTS terminal backup, PUCT/FPU, and proven propagation ----
add_executable(test_search_primitives
  test/test_search_primitives.cc
  mcts/search_primitives.cc
  mcts/uct_node.cc
)
target_link_libraries(test_search_primitives PRIVATE shogi_core)

# ---- Test: game replay (DoMove chain) ----
add_executable(test_replay test/test_replay.cc)
target_link_libraries(test_replay PRIVATE shogi_core)

# ---- Benchmark: move generation ----
add_executable(bench_movegen test/bench_movegen.cc)
target_link_libraries(bench_movegen PRIVATE shogi_core)

# ---- Benchmark: legal move generation from SFEN file ----
add_executable(bench_legal100 test/bench_legal100.cc)
target_link_libraries(bench_legal100 PRIVATE shogi_core)

# ---- Test: mate problem solver ----
add_executable(test_mate test/test_mate.cc)
target_link_libraries(test_mate PRIVATE mate_solver shogi_core)

# ---- Test: df-pn deadline and asynchronous cancellation ----
add_executable(test_dfpn_cancellation test/test_dfpn_cancellation.cc)
target_link_libraries(test_dfpn_cancellation PRIVATE mate_solver shogi_core)

# ---- Test: df-pn repetition handling across the game/search boundary ----
add_executable(test_dfpn_repetition test/test_dfpn_repetition.cc)
target_link_libraries(test_dfpn_repetition PRIVATE mate_solver shogi_core)

# ---- Test: shallow mate (Phase 0+) ----
add_executable(test_shallow_mate test/test_shallow_mate.cc)
target_link_libraries(test_shallow_mate PRIVATE mate_solver shogi_core)

# ---- Bench: shallow mate vs df-pn ----
add_executable(bench_shallow_mate test/bench_shallow_mate.cc)
target_link_libraries(bench_shallow_mate PRIVATE mate_solver shogi_core)

# ---- Bench: mate-in-1 oracle vs specialized implementation ----
add_executable(bench_mate1 test/bench_mate1.cc)
target_link_libraries(bench_mate1 PRIVATE shogi_core)

# ---- Test: specialized mate-in-1 vs move-generation oracle ----
add_executable(test_mate1 test/test_mate1.cc)
target_link_libraries(test_mate1 PRIVATE shogi_core)

# ---- Test: GenerateCheckingMoves property test ----
add_executable(test_check_movegen test/test_check_movegen.cc)
target_link_libraries(test_check_movegen PRIVATE shogi_core)

# ---- Test: BNS (branch number search) mate solver ----
add_executable(test_bns test/test_bns.cc)
target_link_libraries(test_bns PRIVATE mate_solver shogi_core)

# ---- Bench: mate solver comparison (dfpn / bns / bns-pndn) ----
add_executable(bench_mate_solvers test/bench_mate_solvers.cc)
target_link_libraries(bench_mate_solvers PRIVATE mate_solver shogi_core)

# ---- Test: native on-demand YBB lookup ----
add_executable(test_opening_book test/test_opening_book.cc)
target_link_libraries(test_opening_book PRIVATE opening_book)

# ---- Test: YaneuraOu PackedSfen compatibility ----
add_executable(test_packed_sfen test/test_packed_sfen.cc)
target_link_libraries(test_packed_sfen PRIVATE shogi_core)

# ---- Test: asymmetric Gote book-exit retrograde solver ----
add_executable(test_gote_exit_solver test/test_gote_exit_solver.cc)

# ---- Test: side-aware opening-book selection ----
add_executable(test_book_selection test/test_book_selection.cc)

# ---- Tool: generate compact Gote-only early-exit YBB policy ----
add_executable(gote_book_generator tools/gote_book_generator.cc)
target_link_libraries(gote_book_generator PRIVATE shogi_core)

# ---- Test: pure move-time calculation ----
add_executable(test_time_manager test/test_time_manager.cc)
target_link_libraries(test_time_manager PRIVATE time_manager)

# ---- Test: canonical USI search-info formatting ----
add_executable(test_search_info
  test/test_search_info.cc
  usi/search_info.cc
)
target_link_libraries(test_search_info PRIVATE shogi_core)

# ---- Bench: solver-relevant board primitives ----
add_executable(bench_primitives test/bench_primitives.cc)
target_link_libraries(bench_primitives PRIVATE shogi_core)

# ---- Tools ----
add_executable(make_random_net tools/make_random_net.cc)
target_link_libraries(make_random_net PRIVATE nnue)

add_executable(dump_nnue_features tools/dump_nnue_features.cc)
target_link_libraries(dump_nnue_features PRIVATE nnue)

add_executable(eval_positions tools/eval_positions.cc)
target_link_libraries(eval_positions PRIVATE nnue)

add_executable(test_record_io test/test_record_io.cc)
target_link_libraries(test_record_io PRIVATE nnue)
add_test(NAME record_io COMMAND test_record_io ${CMAKE_BINARY_DIR})

# ---- Python bindings (pybind11 from pip: python3 -m pip install pybind11) ----
find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
if(Python3_Interpreter_FOUND AND NOT pybind11_DIR)
  execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_pybind11_dir)
    set(pybind11_DIR ${_pybind11_dir})
  endif()
endif()
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
  message(STATUS "pybind11 ${pybind11_VERSION}: building the jhbr5 Python module")
  pybind11_add_module(jhbr5_py pyext/jhbr5_py.cc)
  target_link_libraries(jhbr5_py PRIVATE nnue)
  set_target_properties(jhbr5_py PROPERTIES OUTPUT_NAME jhbr5)
  add_test(NAME net_roundtrip
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_net_roundtrip.py
      --build $<TARGET_FILE_DIR:jhbr5_py> --eval $<TARGET_FILE:eval_positions>
      --sfens ${CMAKE_CURRENT_SOURCE_DIR}/test/legal100.sfens --tmp ${CMAKE_BINARY_DIR})
  add_test(NAME train_smoke
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_train_smoke.py
      --build $<TARGET_FILE_DIR:jhbr5_py> --eval $<TARGET_FILE:eval_positions>
      --sfens ${CMAKE_CURRENT_SOURCE_DIR}/test/legal100.sfens --tmp ${CMAKE_BINARY_DIR})
  add_test(NAME teacher_smoke
    COMMAND sh -c "$<TARGET_FILE:make_random_net> value ${CMAKE_BINARY_DIR}/dg_value.nn --l1 256 && $<TARGET_FILE:make_random_net> policy ${CMAKE_BINARY_DIR}/dg_policy.nn --l1 512 && ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/teacher.py --engine $<TARGET_FILE:jhbr5> --engine-option ValueNet=${CMAKE_BINARY_DIR}/dg_value.nn --engine-option PolicyNet=${CMAKE_BINARY_DIR}/dg_policy.nn --sfens ${CMAKE_CURRENT_SOURCE_DIR}/test/teacher.sfens --nodes 100 --workers 1 --out ${CMAKE_BINARY_DIR}/teach --build $<TARGET_FILE_DIR:jhbr5_py> | grep -q 'done: 3 records'")
  set_tests_properties(net_roundtrip train_smoke teacher_smoke PROPERTIES SKIP_RETURN_CODE 77)
else()
  message(STATUS "pybind11 not found: Python module and trainer tests disabled")
endif()

# ---- Bench ----
add_executable(bench_nnue test/bench_nnue.cc)
target_link_libraries(bench_nnue PRIVATE nnue)

# ---- Tests ----
add_executable(test_nnue_kernels test/test_nnue_kernels.cc)
target_link_libraries(test_nnue_kernels PRIVATE nnue)
add_test(NAME nnue_kernels COMMAND test_nnue_kernels)

add_executable(test_nnue_features test/test_nnue_features.cc)
target_link_libraries(test_nnue_features PRIVATE nnue)
add_test(NAME nnue_features
  COMMAND test_nnue_features ${CMAKE_CURRENT_SOURCE_DIR}/test/legal100.sfens)
add_test(NAME nnue_features_positions
  COMMAND test_nnue_features ${CMAKE_CURRENT_SOURCE_DIR}/test/positions.txt)

add_executable(test_see test/test_see.cc)
target_link_libraries(test_see PRIVATE nnue)
add_test(NAME see COMMAND test_see)

add_executable(test_nnue_net test/test_nnue_net.cc)
target_link_libraries(test_nnue_net PRIVATE nnue)
add_test(NAME nnue_net
  COMMAND test_nnue_net ${CMAKE_CURRENT_SOURCE_DIR}/test/legal100.sfens ${CMAKE_BINARY_DIR})

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  add_test(NAME nnue_features_python_reference
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_nnue_features_ref.py
      $<TARGET_FILE:dump_nnue_features> ${CMAKE_CURRENT_SOURCE_DIR}/test/legal100.sfens)
endif()

# Inherited JHBR3 tests that run without arguments or with checked-in inputs.
add_test(NAME movegen COMMAND test_movegen ${CMAKE_CURRENT_SOURCE_DIR}/test/positions.txt)
add_test(NAME search_repetition COMMAND test_search_repetition)
add_test(NAME dfpn_repetition COMMAND test_dfpn_repetition)
add_test(NAME shallow_mate COMMAND test_shallow_mate)
add_test(NAME search_primitives COMMAND test_search_primitives)
add_test(NAME packed_sfen COMMAND test_packed_sfen)
add_test(NAME lockfree_search COMMAND test_lockfree_search ${CMAKE_BINARY_DIR})
add_test(NAME tree_reuse COMMAND test_tree_reuse)
add_test(NAME time_manager COMMAND test_time_manager)
add_test(NAME search_info COMMAND test_search_info)
add_test(NAME bns COMMAND test_bns)
add_test(NAME mate1 COMMAND test_mate1 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME datagen_smoke
  COMMAND sh -c "$<TARGET_FILE:make_random_net> value ${CMAKE_BINARY_DIR}/dg_value.nn --l1 256 && $<TARGET_FILE:make_random_net> policy ${CMAKE_BINARY_DIR}/dg_policy.nn --l1 512 && $<TARGET_FILE:jhbr5> datagen --value ${CMAKE_BINARY_DIR}/dg_value.nn --policy ${CMAKE_BINARY_DIR}/dg_policy.nn --out ${CMAKE_BINARY_DIR}/dg_smoke.rec --games 4 --threads 2 --nodes 100 --max-ply 60 --seed 1 && $<TARGET_FILE:test_record_io> ${CMAKE_BINARY_DIR}")
add_test(NAME usi_bench COMMAND sh -c "printf 'bench 2000 2\\nquit\\n' | $<TARGET_FILE:jhbr5>")
