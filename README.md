# JHBR3

JHBR3 is a GPU-accelerated USI shogi engine using a dlshogi-style MCTS search,
neural-network evaluation, shallow mate search, and a parallel root df-pn
solver.

This repository started from JHBR2 commit
`3e444c358118ef6941a02c512dfcf7d8fc293ac8` on
`feat/dlshogi-nyugyoku-148planes`. The corresponding baseline is tagged
`jhbr3-base` here and `jhbr2-final` in the JHBR2 repository. Full development
history is preserved.

## Build

The production backend uses CUDA and TensorRT:

```bash
cmake -S . -B build-trt -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=ON \
  -DCUDAToolkit_ROOT="$CUDA_PATH" \
  -DTENSORRT_ROOT="$TENSORRT_PATH" \
  -DCUDNN_ROOT="$CUDNN_PATH"
cmake --build build-trt -j"$(nproc)"
```

The resulting USI executable is `build-trt/jhbr3`. See
[`docs/HOW_TO_START.md`](docs/HOW_TO_START.md) for model conversion and runtime
configuration.

CPU-only core and search tests can be built without a working CUDA driver:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DUSE_TENSORRT=OFF
cmake --build build-cpu --target \
  test_search_repetition test_dfpn_repetition test_shallow_mate -j"$(nproc)"
```

## Compatibility

JHBR3 retains the internal C++ namespace `jhbr2`, Python encoder module name
`jhbr2_encoder`, and USI model-format value `jhbr2`. These identify the
inherited software/model interface and remain unchanged so existing 148-plane
models, scripts, and serialized TensorRT engines continue to work.

## Documentation

- [`docs/HOW_TO_START.md`](docs/HOW_TO_START.md): build and run the engine
- [`docs/HOW_TO_TRAIN.md`](docs/HOW_TO_TRAIN.md): train and export a network
- [`docs/STRENGTH_TESTING.md`](docs/STRENGTH_TESTING.md): reproducible A/B tests
- [`docs/ENGINE_STRENGTH_ROADMAP.md`](docs/ENGINE_STRENGTH_ROADMAP.md):
  inherited improvement roadmap and technical history
