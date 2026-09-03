# JHBR5 changelog

## Unreleased — Phase 5 (testing) and first training

* Pre-rental trainer work (2026-09-03): king-relative factoriser `KPrel`
  (index computed in C++, `jhbr5.kprel_index`, folded at export),
  validation loss on held-out shards with `ckpt_best.pt`, `--schedule
  flat-cosine|cosine|exp`, `tools/train_both.sh` (two GPUs), importer
  manifest (`imported.json`) with path-based shard names; `docs/TRAINING_PLAN.md`.
* Tree memory: the lazily created child node pointer now lives inside the
  edge (`child_node_t` 24 B, `uct_node_t` 32 B, one allocation per expansion
  instead of two). Floodgate logs showed 56 of ~1000 searches stopping on the
  8 GB `TreeMemoryMB`; the config now allows 24 GB.

* `tools/sprt.py` (pentanomial GSPRT), `tools/sprt_match.py` (rounds of
  paired games until the SPRT decides), `tools/match_vs_jhbr3.sh` (equal
  nodes), `tools/match_vs_yaneuraou.sh` (equal clock and threads),
  `tools/thread_scaling.sh`; `pipeline.py` gains `"gate": "sprt"`;
  `docs/STRENGTH_TESTING.md` gains a JHBR5 section; `tools/cloud_setup.sh`.
* Trainer: policy target falls back to the played move for game records
  without a distribution (`--policy-target auto`); chunked, checkpointed
  policy readout (batch 8192 × 108 legal moves now needs 1.3 GB instead of
  3.4 GB); `BatchReader` interleaves 16 shards for mixing.
* Data: 80 pack files (`../training_data`) imported to `../data/pack`
  (1.02 billion positions from the first 45 files; the rest importing).
* First networks (S profile) trained on the RTX 2060: value L1 512 at 52k
  positions/s, policy L1 2048; installed in `../nets/` for floodgate via
  `../shgterm/my-config.yaml`.

## Phase 4 (data pipeline)

* `tools/import_pack.py`: YaneuraOu `.pack` game records → `.rec` shards via
  cshogi (Apery HCP start positions, move16), multiprocess, ~230–600k
  positions/s per worker. The five available packs (304,841 games, 37.7M
  positions) import in about two minutes.
* `tools/import_psv.py`: PSV/`.bin` → `.rec` by re-framing (numpy, no decode).
* `jhbr5 datagen` (`datagen/datagen.cc`): multi-threaded in-engine self-play
  writing records with root visit distributions, root score and result;
  random opening plies or a book, root Dirichlet noise (new `SearchConfig`
  fields + `Search::ApplyRootNoise`), temperature sampling with decay,
  resignation with a no-resign control fraction, repetition (4th
  occurrence / perpetual check), declaration and ply-cap adjudication.
  `SearchResult` now carries `root_visits` and `root_q`.
* USI option `RootDistOutput` (default off): prints `info string rootdist
  move:visits …` and `rootq` before `bestmove`.
* `tools/jhbr3_rootdist.patch` and branch `jhbr5-teacher` in the JHBR3
  checkout: the same opt-in option for JHBR3 (compile-checked with the
  ONNX Runtime build). Default off, no effect on JHBR3 when unused.
* `tools/teacher.py`: labels positions (sfen file, existing shards keeping
  their results, or teacher self-play via cshogi) with any engine printing
  `rootdist`; one engine process per worker, `<out>.partK.rec` shards.
* `tools/pipeline.py` + `tools/pipeline_example.json`: generate → train →
  export → test (eval_positions, bench, fixed-nodes match with
  `strength_test.py`) → promote (`nets/best`, `nets/manifest.json` with
  SHA-256). Exercised end to end for generations 0 and 1 with tiny settings.
* `docs/DATA_PIPELINE.md`; ctest `datagen_smoke` and `teacher_smoke`.
* Deviations/decisions: datagen uses threads inside one process (a driver
  can launch several processes); records store visits scaled to 65535;
  imported pack positions keep YaneuraOu's ±32000 mate scores (the score→WDL
  sigmoid saturates them); the promotion gate is a plain score threshold
  until Phase 5 adds SPRT; KLD-gain early stopping from Monty's datagen is
  not implemented (fixed nodes per move).

## Phase 3 (training)

Decision (after reviewing BulletOu, MIT, Rust/CUDA): keep a PyTorch trainer
that covers both networks and runs CPU tests; BulletOu stays a reference and
a possible later accelerator for the value net under the same `.nn` header
contract. Rationale in the Phase 3 discussion of the session and
`docs/HOW_TO_TRAIN.md`.

* `nnue/record.{h,cc}` + `docs/DATA_FORMAT.md`: training record format
  (44-byte head whose first 40 bytes are a YaneuraOu `PackedSfenValue`, then
  `(move16, visits)` pairs), reader/writer, `test_record_io`.
* `pyext/jhbr5_py.cc`: pybind11 module `jhbr5` exposing the engine's
  mappings (`value_features`, `policy_features`, `move_bucket`, `see_good`,
  `legal_moves`, `feature_set_id`, `bucket_table_id`), packed-sfen codec,
  CRC-32C, `write_net` (the engine's own file writer), `RecordWriter`, and
  `BatchReader` (C++ shard reader with shuffle buffer that emits EmbeddingBag
  index/offset arrays and per-legal-move bucket/visit arrays; ~100k
  positions/s per thread including legal moves and SEE buckets).
* `train/`: `model.py` (`ValueNet` with three sparse groups, shared group-A
  table, slot-only factoriser, PST skip on group B; `PolicyNet` with
  per-legal-move gathered readout; segment softmax; λ-blended WDL target with
  the nnue-pytorch score→WDL shape; weight clipping to the integer ranges),
  `data.py` (IterableDataset over shards, one C++ reader per worker),
  `train.py` (AdamW, warmup + exponential decay, checkpoints, resume, export),
  `export.py` (factoriser fold, calibrated QB, rounding, `.nn` via
  `jhbr5.write_net`), `quantized.py` (NumPy reference of the engine's integer
  forward pass and the QB calibration).
* Tools/tests: `eval_positions` (prints WDL and per-move logits);
  `test_net_roundtrip.py` (PyTorch → `.nn` → C++ evaluator equals the integer
  reference within 6e-8 WDL / 5e-7 logits; quantisation error vs the float
  model 2e-3 / 2e-2 on random weights); `test_train_smoke.py` (synthetic
  shard → train value and policy nets on CPU → export → engine loads);
  both registered in `ctest` and skipped (code 77) when torch/pybind11 are
  missing. `CMAKE_POSITION_INDEPENDENT_CODE ON` for the module.
* Not done in this phase: the `KPrel` virtual feature (only the slot-only
  factoriser is implemented); a Triton/CUDA kernel for the policy readout
  (the gathered-matmul formulation caps batch size at a few thousand on
  large L1); `bench` of trainer throughput on a GPU (no GPU on the dev
  machine).

## Phase 2 (search integration)

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
