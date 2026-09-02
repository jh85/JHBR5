# JHBR5 changelog

## Unreleased — Phase 2 (search integration)

* `mcts/uct_search.{h,cc}` rewritten for synchronous CPU evaluation: no leaf
  batching, no GPU worker groups. Node state machine `kFresh → kEvaluated
  (value net, first visit) → kExpanded (movegen + policy, second visit)`;
  the iteration that expands a node keeps descending (Monty `perform_one`).
  Per-thread `Evaluator`, per-thread board with make/undo instead of a root
  copy per playout, hashed position mutexes kept for expansion.
* Kept unchanged: PUCT/FPU (`SelectPuctChild`), proven win/loss/draw
  propagation, virtual loss, repetition/declaration/ply-cap handling, shallow
  leaf mate probe, root df-pn/BNS guard, tree reuse, time management, USI
  protocol and info output.
* New: `nnue/eval_cache.h` (8-byte lock-free WDL cache), `mcts/butterfly.h`
  (history bonus), Monty policy temperature and draw-share adjustment, all
  behind USI options and **off by default**; `TreeMemoryMB` limit with a new
  `tree_full` stop reason; live tree memory accounting in `uct_node.cc`.
* USI: `ValueNet`, `PolicyNet`, `Threads` (1..256), `EvalCacheMB`,
  `TreeMemoryMB`, `UseButterfly`, `ButterflyDivisor`, `ButterflyReduction`,
  `UsePolicyTemperature`, `Pst{Root,Depth,WinThreshold,WinMax,Base}`,
  `DrawScale`, `DrawQuadratic`; `bench [nodes] [threads]` command; engine name
  `JHBR5`; binary `jhbr5`. Retired GPU/MLH options are accepted and ignored.
  **`MaxNodes` default raised from 800 to 100,000,000** so timed searches are
  clock limited.
* Removed: `inference/` (TensorRT, ONNX Runtime, NN cache), the 148-plane
  encoder and CUDA unpack kernel, `pyext/`, `batch_eval.cpp`, the JHBR3
  Python model/training/shard scripts, MLH head and moves-left search term,
  ONNX/TensorRT CMake blocks. `-march=native` stays replaced by `JHBR5_ISA`.
* Tests: `test_lockfree_search` rewritten on random networks (threaded search
  and reuse, single-thread determinism, deferred expansion, optional Monty
  features, tree memory limit, leaf mate detection without solvers, adaptive
  deadline, external stop, root mate worker); `test_search_primitives` without
  MLH; `ctest` now also runs tree reuse, time manager, search info, BNS,
  mate-in-1 and a `bench` smoke test through the real binary.
* Measured (dev machine, random M nets, `bench 20000`): 1 thread 50k
  playouts/s (32k net evals/s + cache hits, 35k expansions/s), 4 threads
  175k, 8 threads 281k (5.6×). The depth-5 shallow mate probe costs ~25% of
  single-thread throughput (65k playouts/s with `LeafMateMode off`, 61k at
  depth 3); the default stays 5 pending strength tests.

## Phase 1 (CPU NNUE inference backend)

Decisions taken from the Phase 0 review (`docs/open_questions.txt`): follow
Monty where applicable, otherwise the simpler option. Concretely: one
king-relative table shared by both frames; "M" profile widths (value L1 1024,
policy L1 4096) as defaults with widths carried in the file header; threat
pairs in v1; SEE doubling on; records store move16; MLH dropped; butterfly
and policy temperature default off.

* `nnue/`: `types.h` (all mapping constants), `features.{h,cc}` (group A
  king-relative slots with finny diff, group B absolute slots + threat pairs,
  policy inputs), `move_buckets.{h,cc}` (10,043 buckets, ×2 with SEE),
  `see.cc` (`ShogiBoard::SeeGe`), `net_format.{h,cc}` (versioned header,
  CRC-32C, tensor table), `value_net`, `policy_net`, `evaluator.h`,
  `simd.h` + `simd_scalar.h` + `simd_avx2.h` + `simd_avx512.h` (stub that
  fails to compile on purpose), `random_net` (deterministic nets for tests).
* CMake: `-DJHBR5_ISA=scalar|avx2|avx512` replaces `-march=native`;
  `enable_testing()` with `ctest` registration of the new and the inherited
  argument-free tests.
* Tools: `make_random_net`, `dump_nnue_features`; bench `bench_nnue`.
* Tests: `test_nnue_kernels` (scalar vs AVX2 bit-identical),
  `test_nnue_features` (ranges, uniqueness, 180°-rotation invariance of every
  feature and bucket, diff == scratch), `test_see`,
  `test_nnue_net` (scalar vs AVX2 end-to-end, finny cache == fresh refresh
  along random walks, regression values), and
  `test/test_nnue_features_ref.py` (independent pure-Python reference of the
  feature/bucket mapping and the feature-set id, compared bit-for-bit).
* Deviations from the design doc, recorded there too: PST skip only on
  group B (a shared group-A table cannot carry a signed PST for both
  frames); L1 and readout biases are i16 rather than Monty's i8; positions
  without a king use king square 0; SEE picks the lowest-square attacker
  among equal types, which is not frame invariant in rare double-attacker
  positions (4 of 1.8M moves in `test/positions.txt`), same as Stockfish.
* Bench (i7-10870H, one thread, random "M" nets, 8-ply random walks):
  AVX2 172k value evals/s (5.8 µs; 14.5 group-A rows + 66 group-B rows per
  eval) and 106k policy expansions/s (9.5 µs, 40 legal moves avg);
  scalar 112k / 66k.

## Phase 0 (study and design)

* Forked from JHBR3 `2835d45` ("Update training pipeline docs for v2.1 defaults
  and Attention Residuals"). The JHBR3 remote is kept as `jhbr3`.
* Added `docs/MONTY_NOTES.md`: how Monty's value net, policy net, move buckets,
  SEE doubling, PUCT variant, butterfly history, tree, datagen and trainers work,
  with recomputed constants (80,624 value inputs, 7,840 move buckets).
* Added `docs/YANEURAOU_NNUE_NOTES.md`: YaneuraOu NNUE feature sets,
  accumulator update, quantisation, SIMD kernels, file format, trainer and
  packed-sfen conventions.
* Added `docs/DESIGN.md`: proposed JHBR5 architecture (feature sets, move
  buckets, accumulator strategy, search changes, threading, training, data
  plan, open questions). Awaiting review before Phase 1.
* Added `THIRD_PARTY.md` (licence policy: GPL-3.0, no Monty code copied).
