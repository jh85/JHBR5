# JHBR5 data pipeline

Record format: `docs/DATA_FORMAT.md`. Trainer: `docs/HOW_TO_TRAIN.md`.

## Tools

| Tool | Purpose |
|---|---|
| `tools/import_pack.py` | YaneuraOu gensfen `.pack` game records → `.rec` shards (value-only). Uses cshogi for the Apery HCP start positions and move16. ~230k positions/s per worker. |
| `tools/import_psv.py` | YaneuraOu PSV/`.bin` → `.rec` (pure re-framing, no decoding). |
| `tools/teacher.py` | Labels positions with a USI engine that prints its root visit distribution: JHBR3 (GPU) with the `RootDistOutput` patch, or JHBR5. Sources: sfen file, existing shards (keeps their game results), or teacher self-play. |
| `jhbr5 datagen` | In-engine self-play (docs/DESIGN.md §10.3): random opening plies or a book, root Dirichlet noise, temperature-sampled moves, resignation with a no-resign control fraction, repetition/declaration/ply-cap adjudication; writes records with distributions. |
| `tools/pipeline.py` | generate → train → export → test → promote for one generation, driven by a JSON config (`tools/pipeline_example.json`). |
| `tools/jhbr3_rootdist.patch` | The opt-in `RootDistOutput` option for JHBR3 (also committed on branch `jhbr5-teacher` in the JHBR3 checkout). Off by default; no effect on JHBR3 performance. |

## Stage 0: imported data (value only)

```bash
python3 tools/import_pack.py --pack-dir ../2000000 --out ../data/pack --workers 8
python3 tools/import_psv.py --out ../data/psv teacher1.bin teacher2.bin
python3 train/train.py --net value --shards ../data/pack ../data/psv --l1 1024 --wdl-lambda 0.7 ...
```

The five packs currently available hold 304,841 games and 37.7M positions
(all from arbitrary start positions, average 124 plies; results 42% black,
14% draw, 44% white; scores are YaneuraOu centipawns with ±32000 for mates).
For the first value net that is a small corpus; a few hundred million
positions are the target (docs/DESIGN.md §10.5).

## Stage 1: teacher distillation (policy warm start)

```bash
# JHBR3 (GPU) with the patch applied and built:
python3 tools/teacher.py --engine ../JHBR3/build-trt/jhbr3 \
    --engine-option OnnxModel=/path/model.engine --engine-option WorkersPerGpu=2 \
    --records ../data/pack/*.rec --sample 2000000 --nodes 1000 --workers 4 --out ../data/teacher/pack
python3 tools/teacher.py --engine ../JHBR3/build-trt/jhbr3 --engine-option OnnxModel=... \
    --selfplay 20000 --random-plies 8 --nodes 1000 --workers 4 --out ../data/teacher/selfplay
python3 train/train.py --net policy --shards ../data/teacher --l1 4096 ...
```

Each worker owns one engine process; on a multi-GPU host run one `teacher.py`
per GPU with the JHBR3 GPU options. JHBR5 accepts the same options, so the
tool also serves to relabel positions with a stronger JHBR5 later.

## Stage 2: self-play generations

```bash
python3 tools/pipeline.py --config my_pipeline.json --generation 1
```

Per generation: `jhbr5 datagen` with the incumbent networks → training on
the last `replay_generations` generations plus the imported data (resuming
from the previous checkpoints) → export → `eval_positions` sanity check,
`bench`, and a fixed-nodes match against the incumbent with
`tools/strength_test.py` → promotion when the candidate's score reaches
`promote_score` (Phase 5 replaces this with SPRT). `nets/manifest.json`
records every promotion with SHA-256 hashes.

## Datagen defaults and rationale

| Option | Default | Note |
|---|---|---|
| `--nodes` | 800 | playouts per move |
| `--random-plies` | 8 | uniform 0..8 random legal plies from the start position (or `--book` sfens) |
| `--temperature/--temp-decay/--temp-min` | 1.0 / 0.9 / 0.2 | visits^(1/τ) sampling, greedy after τ < 0.2 (Monty) |
| `--dirichlet-alpha/--dirichlet-epsilon` | 0.15 / 0.25 | α ≈ 10 / average legal moves |
| `--resign-threshold/--resign-plies/--no-resign-fraction` | 0.03 / 8 / 0.1 | a side resigns after 8 of its turns with root Q < 0.03; 10% of games never resign (records carry the flag) |
| `--max-ply` | 320 | floodgate cap → draw |
| `--leaf-mate-depth/--root-mate-depth` | 3 / 0 | cheaper than the play defaults |
| `--eval-cache-mb/--tree-memory-mb` | 16 / 512 | per game thread |

Repetition: draw at the fourth occurrence; perpetual check adjudicated as in
play. Records store the root Q as centipawns (`756·logit(q)`), the best move,
the full root visit distribution and the game result from the mover's side.
