# Training plan: 1.5B+ positions on a rented 1–2 × RTX 5090 box

Written 2026-09-03 for a one-to-two-week rental. Commands assume the repo at
`~/JHBR5`, shards under `~/data`, and a CUDA torch venv at `~/venv`
(`tools/cloud_setup.sh` creates it). Adjust `--workers` to the core count.

## 0. Machine

| Item | Minimum | Why |
|---|---|---|
| GPU | 1 × RTX 5090 (2 preferred) | one trains value and policy sequentially; two train them concurrently and leave one for the JHBR3 teacher |
| CPU | 32 cores | shard reader ≈ 80k positions/s per worker; self-play is CPU-bound (≈ 35k playouts/s per thread) |
| RAM | 64 GB | 4–6 reader workers with 200k-record shuffle buffers, plus the M-profile optimizer state on the GPU |
| Disk | 300 GB NVMe | 1.5B positions ≈ 66 GB of `.rec` shards, checkpoints 3–10 GB each |

## 1. Setup (day 1, ~2 h)

```bash
git clone <JHBR5 repo> ~/JHBR5 && cd ~/JHBR5
tools/cloud_setup.sh ~/venv                      # apt, venv, torch cu128, cshogi, cmake build, quick tests
export PY=~/venv/bin/python PYTHONPATH=~/JHBR5/build
```

Transfer the shards (rsync from this machine, `~/Downloads/jhbr5/data/{pack,psv}`),
or copy the raw `training_data/` and import on the box:

```bash
$PY tools/import_pack.py --pack-dir ~/training_data --out ~/data/pack --workers 16   # manifest skips duplicates
$PY tools/import_psv.py  --out ~/data/psv ~/training_data/shogi_suisho5_depth9_entering_king/*.bin
```

Hold out validation shards (never trained on):

```bash
mkdir -p ~/data/val && mv ~/data/pack/kif_20251113113227_2000000.rec ~/data/val/ && mv ~/data/psv/*thread_index=126.rec ~/data/val/
```

Sanity: `ctest --test-dir build` and `printf 'bench 20000 8\nquit\n' | build/jhbr5`.

## 2. Supervised stage (days 1–3)

Data: all pack shards (game records with played moves and search scores) plus
the PSV shards (Suisho5 depth-9 positions, value only). The policy trainer uses
the played move of pack records as its target (`--policy-target auto`) and
skips PSV records automatically.

Sizes: value **M** (L1 1024), policy **4096**. Batch 16384 for value, 8192 for
policy. Learning rate 1e-3, flat for 60% of the run then cosine to 1e-5.
Steps = epochs × positions / batch.

Two GPUs, both nets at once:

```bash
VALUE_ARGS="--l1 1024 --batch-size 16384 --steps 275000 --wdl-lambda 0.7 --val-shards ~/data/val/*.rec --val-every 2000" \
POLICY_ARGS="--l1 4096 --batch-size 8192 --steps 250000 --val-shards ~/data/val/kif_*.rec --val-every 2000" \
PY=$PY tools/train_both.sh ~/runs/sup1 ~/data/pack ~/data/psv -- --workers 6 --shuffle-buffer 200000 --schedule flat-cosine --flat-frac 0.6 --save-every 10000
```

(275k × 16384 ≈ 4.5B samples = 3 epochs over 1.5B; 250k × 8192 ≈ 2B policy
samples ≈ 2 epochs over the ~1B pack positions.) Expected wall time at
120–150k and 40–50k positions/s: value ≈ 9–10 h, policy ≈ 12–14 h.

One GPU, sequential:

```bash
$PY train/train.py --net value  --shards ~/data/pack ~/data/psv --l1 1024 --batch-size 16384 --steps 275000 --wdl-lambda 0.7 \
    --schedule flat-cosine --flat-frac 0.6 --workers 6 --val-shards ~/data/val/*.rec --val-every 2000 \
    --out ~/runs/sup1/value --export ~/runs/sup1/value.nn
$PY train/train.py --net policy --shards ~/data/pack --l1 4096 --batch-size 8192 --steps 250000 \
    --schedule flat-cosine --flat-frac 0.6 --workers 6 --val-shards ~/data/val/kif_*.rec --val-every 2000 \
    --out ~/runs/sup1/policy --export ~/runs/sup1/policy.nn
```

Use `ckpt_best.pt` (lowest validation loss) rather than `ckpt_last.pt` if
they differ:

```bash
$PY train/export.py --checkpoint ~/runs/sup1/value/ckpt_best.pt  --out ~/runs/sup1/value.nn --calib-sfens test/positions.txt
$PY train/export.py --checkpoint ~/runs/sup1/policy/ckpt_best.pt --out ~/runs/sup1/policy.nn
```

Gate against the current nets (fixed nodes, SPRT on the candidate):

```bash
tools/generate_strength_openings.py ... build-strength/openings-512.txt     # once; see docs/STRENGTH_TESTING.md
$PY tools/sprt_match.py --engine-a build/jhbr5 --engine-b build/jhbr5 --openings build-strength/openings-512.txt \
    --option-a ValueNet=~/nets/value.nn --option-a PolicyNet=~/nets/policy.nn \
    --option-b ValueNet=~/runs/sup1/value.nn --option-b PolicyNet=~/runs/sup1/policy.nn \
    --option Threads=4 --nodes 5000 --elo0 0 --elo1 10 --round-pairs 25 --max-pairs 400 --output ~/runs/sup1/sprt
```

Then copy the winners to `~/nets/` (and to this machine's `nets/` for floodgate).

Resume after an interruption: add `--resume ~/runs/sup1/value/ckpt_last.pt`.

## 3. Teacher distributions (days 3–5)

Build JHBR3 with TensorRT on the box (its `docs/HOW_TO_START.md`), apply
`git apply tools/jhbr3_rootdist.patch` (or check out its `jhbr5-teacher`
branch), then label positions and self-play games:

```bash
$PY tools/teacher.py --engine ~/JHBR3/build-trt/jhbr3 --engine-option OnnxModel=~/JHBR3/engines/current.engine \
    --engine-option WorkersPerGpu=2 --records ~/data/pack/*.rec --sample 30000000 --nodes 1000 --workers 4 --out ~/data/teacher/pack
$PY tools/teacher.py --engine ~/JHBR3/build-trt/jhbr3 --engine-option OnnxModel=~/JHBR3/engines/current.engine \
    --engine-option WorkersPerGpu=2 --selfplay 100000 --random-plies 8 --nodes 1000 --workers 4 --out ~/data/teacher/selfplay
```

Fine-tune the policy on the distributions (resume from the supervised
checkpoint, lower learning rate), and the value net on the teacher scores:

```bash
$PY train/train.py --net policy --shards ~/data/teacher --resume ~/runs/sup1/policy/ckpt_best.pt --policy-target dist \
    --l1 4096 --batch-size 8192 --steps 40000 --lr 2e-4 --schedule cosine --workers 6 --out ~/runs/dist1/policy --export ~/runs/dist1/policy.nn
$PY train/train.py --net value --shards ~/data/teacher ~/data/pack --resume ~/runs/sup1/value/ckpt_best.pt \
    --l1 1024 --batch-size 16384 --steps 40000 --lr 2e-4 --schedule cosine --wdl-lambda 0.8 --workers 6 --out ~/runs/dist1/value --export ~/runs/dist1/value.nn
```

Gate as in section 2.

## 4. Self-play generations (days 5–14)

`tools/pipeline_example.json` → `~/pipeline.json` with `root`, `openings`,
`generate.threads` = core count, `generate.games` ≈ 20000 (≈ 2.5M positions,
~2–3 h on 32 cores at 800 nodes), `train.data` pointing at
`~/data/pack/*.rec` and `~/data/teacher/*.rec`, `train.value/policy` steps ≈
30000 each (resume from the previous generation), `test.gate = "sprt"`.

```bash
$PY tools/pipeline.py --config ~/pipeline.json --generation 1     # then 2, 3, ...
```

Each generation: datagen with the incumbent → train on the replay window →
export → `eval_positions` sanity, `bench`, SPRT vs the incumbent → promote.
Expect 2–3 generations per day of rental once the first two stages are done.

## 5. Parameter tuning (last day)

`tools/run_spsa_tuning.py --preset puct` with `--engine-option Threads=4`
and fixed nodes (see `docs/STRENGTH_TESTING.md`), then a confirmation match
at the floodgate time control (`--byoyomi-ms`/`--main-time-ms`).

## Bring the nets home

```bash
rsync -av box:~/nets/ ~/Downloads/jhbr5/nets/     # shgterm/my-config.yaml already points there
```
