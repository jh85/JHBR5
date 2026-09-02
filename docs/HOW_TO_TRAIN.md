# How to train JHBR5 networks

The trainer is PyTorch (`train/`). All feature indices, move buckets and SEE
come from the engine's C++ through the `jhbr5` Python module built by CMake;
the trainer contains no second implementation of any mapping
(`docs/DESIGN.md` §9).

## Requirements

- `python3 -m pip install torch numpy pybind11` (CUDA build of torch for real
  training; the CPU build runs the tests).
- Build the engine with the Python module: `cmake -S . -B build && cmake
  --build build -j`. The module is `build/jhbr5.cpython-*.so`; put `build/` on
  `PYTHONPATH` (the scripts also accept it implicitly when run from `build/`).

## Data

Training reads shard files in the format of `docs/DATA_FORMAT.md` (`.rec`).
Phase 4 provides the tools that produce them (PSV/pack import, JHBR3
teacher, self-play). `jhbr5.RecordWriter` writes them from Python:

```python
import jhbr5
w = jhbr5.RecordWriter("shard.rec")
w.write(sfen, score_cp, best_move_usi, ply, result, [(usi, visits), ...], flags)
w.close()
```

## Train

```bash
export PYTHONPATH=build
python3 train/train.py --net value  --shards data/*.rec --l1 1024 --batch-size 16384 \
    --steps 200000 --wdl-lambda 0.7 --out runs/value1 --export nets/value.nn
python3 train/train.py --net policy --shards data/*.rec --l1 4096 --batch-size 16384 \
    --steps 200000 --out runs/policy1 --export nets/policy.nn
```

| Option | Meaning |
|---|---|
| `--net value\|policy` | which network; the policy trainer skips records without a visit distribution |
| `--l1` | accumulator width (value 1024 = "M", 512 = "S"; policy 4096 / 2048); must be a multiple of 256 |
| `--wdl-lambda` | weight of the score-derived WDL vs the game result (0.7 for PSV, 0.5 for self-play) |
| `--score-scale`, `--score-offset` | `w = σ((s−offset)/scale)`, `l = σ((−s−offset)/scale)`, `d = 1−w−l`; defaults 340 / 270 (nnue-pytorch/BulletOu shape) |
| `--no-see` | policy bucket table without SEE doubling (the engine reads which one from the file) |
| `--no-factoriser` | disable the slot-only factoriser on group A |
| `--lr`, `--lr-final`, `--warmup`, `--weight-decay` | AdamW, exponential decay after warmup |
| `--shuffle-buffer`, `--workers` | per-worker C++ shuffle buffer (records) and DataLoader workers |
| `--resume ckpt.pt` | continue from a checkpoint (optimizer state included) |
| `--export path.nn` | quantise and write the engine file at the end |

Weights are clipped after every step to the ranges the integer formats can
hold (`train/model.py`), so export is a rounding, not a projection.

## Export and verify

```bash
python3 train/export.py --checkpoint runs/value1/ckpt_last.pt --out nets/value.nn --calib-sfens positions.txt
python3 train/export.py --checkpoint runs/policy1/ckpt_last.pt --out nets/policy.nn
build/eval_positions nets/value.nn nets/policy.nn test/legal100.sfens | head
```

`export.py` folds the factoriser into the group-A table, calibrates the L2
scale `QB` (largest power of two ≤ 1024 with an 8× overflow margin on the
calibration positions), rounds to the integer types and writes the file
through the engine's own writer (`jhbr5.write_net`), so the header carries
the compiled `feature_set_id` / `bucket_table_id`.

`ctest -R net_roundtrip` checks that the C++ evaluator reproduces the
trainer's integer reference bit-for-bit (differences are float rounding,
about 1e-7) and reports the quantisation error against the float model
(about 2e-3 in WDL, 2e-2 in policy logits for random weights).
`ctest -R train_smoke` writes a synthetic shard, trains both nets for a few
steps on CPU, exports, and loads the result in the engine.

## Speed

Throughput is dominated by the sparse first layer. With the C++ batch reader
(feature extraction, legal moves, buckets with SEE) a DataLoader worker
produces roughly 30–60k positions/s; use `--workers 4..8`. On a GPU the M
profile trains at a few hundred thousand positions/s with `nn.EmbeddingBag`;
if that becomes the bottleneck, the value net can be moved to BulletOu under
the same header contract (see the Phase 3 discussion in `docs/CHANGELOG.md`).
