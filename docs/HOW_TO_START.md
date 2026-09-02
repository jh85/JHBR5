# How to start JHBR5

## Requirements

- CMake ≥ 3.18, a C++20 compiler (GCC 13 or Clang 18 tested), an x86-64 CPU
  with AVX2 (or use `-DJHBR5_ISA=scalar`).
- Two network files (`docs/NNUE_FORMAT.md`). Until trained networks exist,
  `make_random_net` produces deterministic placeholders that exercise every
  code path but play random shogi.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DJHBR5_ISA=avx2
cmake --build build -j
ctest --test-dir build          # all tests, including the USI bench smoke test
```

## Run

```bash
mkdir -p nets
build/make_random_net value  nets/value.nn  --l1 1024     # placeholder
build/make_random_net policy nets/policy.nn --l1 4096     # placeholder
build/jhbr5
```

Then over USI:

```
usi
setoption name Threads value 8
isready
position startpos
go btime 60000 wtime 60000 byoyomi 1000
```

`isready` loads the networks (about 0.5 s for the M profile, 470 MB) and
clears the tree, value cache and history table.

## Benchmark

```
bench [nodes] [threads]
```

runs ten fixed positions and prints nodes/s, evaluations/s, expansions/s and
value-cache hits. With no networks loaded it generates random M-profile
networks in `/tmp`. `build/bench_nnue test/legal100.sfens` benchmarks the
networks alone.

## Strength testing

See `docs/STRENGTH_TESTING.md` (inherited harness). JHBR5 is a CPU engine:
pass `--engine-option Threads=N` and ignore the GPU-topology sections.
