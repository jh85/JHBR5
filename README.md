# JHBR5

JHBR5 is a USI shogi engine that runs AlphaZero/Monty-style MCTS (PUCT with a
learned policy prior and a WDL value) **entirely on CPU**, using NNUE-style
sparse-input networks for both value and policy. It is a fork of JHBR3 (dlshogi-
style MCTS, shallow mate search, parallel root df-pn/BNS solver, USI, time
management) with the GPU evaluation backend replaced by CPU NNUE inference.

Licence: GPL-3.0 (inherited from JHBR3 / Leela Chess Zero); see
`THIRD_PARTY.md` for the origin of every borrowed idea.

## Build

Requires CMake ≥ 3.18 and a C++20 compiler. No CUDA, TensorRT or ONNX. With
`pybind11` installed (`python3 -m pip install pybind11`) the build also
produces the `jhbr5` Python module used by the trainer.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJHBR5_ISA=avx2
cmake --build build -j
ctest --test-dir build
```

`JHBR5_ISA` selects the SIMD level: `avx2` (default), `scalar` (reference
kernels, used to validate the SIMD code), or `avx512` (stub; fails to compile
until implemented on an AVX-512 host).

The executable is `build/jhbr5`.

## Networks

The engine loads two files, set with the `ValueNet` and `PolicyNet` USI
options (defaults `nets/value.nn`, `nets/policy.nn`). The file format is
documented in `docs/NNUE_FORMAT.md`; trained networks come from the Phase 3
trainer. For testing without trained networks:

```bash
build/make_random_net value  nets/value.nn  --l1 1024
build/make_random_net policy nets/policy.nn --l1 4096
```

## Self-play data

```bash
build/jhbr5 datagen --value nets/value.nn --policy nets/policy.nn --out selfplay.rec --games 1000 --threads 8 --nodes 800
```

## Quick check

```bash
printf 'bench 20000 4\nquit\n' | build/jhbr5      # nodes/s, evals/s, expansions/s
build/bench_nnue test/legal100.sfens               # raw network throughput
```

## USI options

| Option | Default | Meaning |
|---|---|---|
| `ValueNet`, `PolicyNet` | `nets/*.nn` | network files |
| `Threads` | 1 | search threads (tree parallelism) |
| `MaxNodes` | 100000000 | playout cap per move |
| `EvalCacheMB` | 64 | value cache (position hash → WDL) |
| `TreeMemoryMB` | 4096 | stop the search when live tree nodes exceed this |
| `CInit`, `CBase`, `FpuReduction` (+`Root` variants) | 1.25, 19652, 0.27 | PUCT |
| `LeafMateMode`, `LeafMateDepth`, `RootMateDepth`, `RootMateSolver` | shallow, 5, 7, bns | mate integration |
| `UseButterfly`, `ButterflyDivisor`, `ButterflyReduction` | false | Monty history bonus on policy logits |
| `UsePolicyTemperature`, `Pst*` | false | Monty depth/Q policy softmax temperature |
| `DrawScale`, `DrawQuadratic` | 0 | Monty draw-share adjustment |
| `DrawValueBlack/White`, `ResignThreshold`, `MaxMovesToDraw` | 0.5, 0.01, 100000 | as JHBR3 |
| time management options (`TimeManagement`, `MoveOverheadMs`, …) | as JHBR3 | see `docs/` |

Retired JHBR3 options (`OnnxModel`, `UseGPU`, `WorkersPerGpu`, `MinibatchSize`,
`NumGPUs`, `NNCacheSize`, `UseMovesLeft`, …) are accepted and ignored.

## Floodgate

`tools/shgterm-config.example.yaml` is a ready-made configuration for the
`shgterm` USI-to-CSA bridge (engine path, network paths, threads, time
management); fill in the server account.

## Documentation

- `docs/DESIGN.md` — architecture and review decisions
- `docs/MONTY_NOTES.md`, `docs/YANEURAOU_NNUE_NOTES.md` — study notes
- `docs/NNUE_FORMAT.md` — weight file format
- `docs/DATA_FORMAT.md` — training record format
- `docs/HOW_TO_TRAIN.md` — PyTorch trainer, export and round-trip verification
- `docs/DATA_PIPELINE.md` — importers, teacher labelling, `datagen`, generation loop
- `docs/CHANGELOG.md` — per-phase changes and decisions
- `docs/STRENGTH_TESTING.md` — A/B testing harness (inherited from JHBR3)
