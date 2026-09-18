# NNUE v2 postmortem — why the "better" architecture lost to v1

Date: 2026-09-19 (written after the v2x conclusion)
Status: **v2 line parked.** v1 remains the playing architecture (the 3252-Elo
floodgate pair). All v2 code, nets, checkpoints, and logs are preserved.

Companion docs: `NNUE_V2_DESIGN.md` (the implementation contract),
`/workspace/runs/v2/REPORT.md` (round 1), `/workspace/runs/v2x/STATE.md`
(round 2). This file is the single narrative of the whole attempt.

## 1. Why we tried v2

v1 (the architecture of the 3252 pair) had saturated in training: val loss and
SPRT results were flat across width, init, and data-mix probes
(runs/dist1/FINAL_REPORT.md §4). The diagnosis was that the ceiling was
**feature expressiveness**, not capacity or training length. v2 was designed to
raise expressiveness where it was cheapest in NPS:

1. **King-bucketed king-relative features** — a 3×3 bucket grid (9 buckets)
   replacing v1's exact-king-square (81-row) group A. Strictly more expressive
   per parameter than v1's factorised 81×2344 table, 9× smaller, and the finny
   accumulator cache gets a 9× higher hit rate (king shuffles inside a bucket
   need no refresh).
2. **Full-width SCReLU at L1** — 3·l1 activations instead of v1's pairwise-mul
   halving (3·l1/2). Same int16 range, negligible cost.
3. **Residual, phase-conditioned value head** — L2 widened 16→32, L3 32→32
   with a residual add, L4 conditioned on 8 material-phase buckets.
4. **Policy v2** — king-bucketed piece features added alongside v1's absolute
   attack/defend-flag features; full-width SCReLU; readout unchanged.

Everything was implemented and committed: C++ core (`b1e9865`), pyext bindings
(`4629913`), training/export/quantization (`eb2534b`), `--arch v2` wiring
(`06e4f76`), docs (`738c00b`). Scalar and AVX2 kernels bit-identical; v1 path
untouched (both architectures coexist, dispatched by net-file header version).

Scale: v2 value ≈ 294M params (~300 MB file; group A 21.1M rows·l1 vs v1's
194M — the bucketed table is much smaller, group B shared). v2 policy ≈ 206M
params, **2.6× v1 policy's 79M**.

## 2. Round 1 (v2r1, 2026-09-14): equal budget, isolated variable

Both nets trained from scratch on the exact sup2 shard mix and hyperparameters
(400k steps), `--arch v2` as the only difference vs the run that produced the
3252 nets.

Validation (same holdout, lower better):

| net | v1 sup2 final | v2r1 final |
|---|---|---|
| value | 0.5751 | 0.5764 (wash, ±0.002 all run) |
| policy | 1.8858 | 1.9408 (+0.055 behind all run) |

Gates vs the frozen v1 pair (SPRT elo0=0/elo1=10, ≤400 pairs, Threads=4,
A = v1 always; score shown is the v2 side):

| gate | nodes | result |
|---|---|---|
| value-only | 5k | undecided @400p, v2 ≈ **+8 Elo** (51.2%) |
| policy-only | 5k | **REJECTED @200p, v2 ≈ −75 Elo** (39.4%) |
| pair | 5k | **REJECTED @200p, v2 ≈ −70 Elo** (40.0%) |
| pair | 20k | **REJECTED @175p, v2 ≈ −80 Elo** (39.3%) |

Reading: v2 value = capacity-neutral. v2 policy decisively weaker — but with
2.6× the parameters on the same 400k-step budget, *undertraining* was the
plausible alternative explanation (its val curve was still descending).

## 3. Quantisation ruled out as a factor (2026-09-14)

Before spending 3 days on a longer run, we measured the float-checkpoint vs
engine-integer gap on 512 val records (`/workspace/runs/v2x/quant_gap.py`):

| net | KL(float‖quant) | top1 float-vs-quant | value WDL max\|diff\| mean |
|---|---|---|---|
| v1 policy | 0.00057 | 97.5% | — |
| v2 policy | 0.00064 | 98.0% | — |
| v1 value | — | — | 0.0088 |
| v2 value | — | — | **0.0043** (better) |

Quantisation costs the same for both architectures — the v2r1 policy deficit
lived in the float weights, not the export. No QAT needed; extended training
was the correct fix to test.

## 4. Round 2 (v2x, 2026-09-14 → 09-18): 3× budget, full data bank

v2 policy retrained with **1.2M steps** on everything (pack + psv + teacher
MultiPV-8 + all 70 selfplay gens, 920 shards). 3.2 days at ~34k pos/s.

- Val trajectory: 200k 2.0717 → 400k 2.0631 → 800k 2.0444 → 1M 1.9616 →
  **1.2M 1.9012**. The cosine tail did the work late. Gap vs v1's 1.8858
  narrowed from +0.055 to **+0.015** — undertraining hypothesis confirmed,
  but still behind.
- Quant recheck: KL 0.00079, top1 97.9% — unchanged, fine.
- Gates vs the 3252 pair:

| gate | nodes | result |
|---|---|---|
| policy-only | 5k | undecided @400p, v2x ≈ **−26 Elo** (46.3%) — huge recovery from −75 |
| pair | 5k | undecided @400p, v2 pair ≈ **+1 Elo** (50.2%) — exact parity |
| pair | 20k | **REJECTED @325p, v2 pair ≈ −45 Elo** (43.8%) |

## 5. Verdict

**v2 showed no measured playing-strength advantage at any tested budget or
node count.** The pair went from −70 @5k/−80 @20k (400k steps) to +1 @5k/
−45 @20k (1.2M steps): more budget bought parity at blitz nodes, but the pair
**degrades at longer search** in both rounds — and the degradation grew with
the policy net's quality (−10 Elo from 5k→20k in r1; −46 Elo in v2x).

### Why v2 was not better — our reading

1. **Feature expressiveness was not the binding constraint after all.** The
   exact-king-square table v2 replaced was already sufficient for value; the
   9× parameter saving and higher cache hit rate bought efficiency, not
   accuracy. Value val and gate: both a wash, twice.
2. **Policy targets didn't reward the new features.** v1 policy's absolute
   attack/defend flags already encode what the visit-count targets ask for;
   v2's added king-relative piece features added 127M parameters without
   adding useful signal — at 400k steps that manifested as severe
   undertraining (−75), and even fully trained it never caught v1 on val
   (+0.015) or gates (−26).
3. **Longer search amplifies value fidelity, where v2 has no edge.** At 5k
   nodes a slightly-off policy can be compensated; at 20k the engine leans
   harder on value accuracy, and v2 value's parity means there is nothing to
   absorb the policy deficit — hence parity@5k → rejection@20k in both rounds.
4. **Fixed-node gates, honest caveat:** gates measure strength-per-node, not
   per-second; NPS deltas were not benchmarked. v2 was designed to be
   NPS-neutral-to-better, so time-odds play might shift numbers slightly in
   v2's favor — but it would have to close −45 Elo, and v1x (same arch as the
   3252 nets) carries zero such risk.

### Process lessons (applied to the v1x runs now in flight)

- **Val rank ≠ gate rank** — proven three times in this project
  (sup5p/sup6p, v2r1 value, v2x). SPRT against the frozen playing pair is the
  only promotion criterion.
- **Equal steps ≠ equal optimization** when parameter counts differ 2.6×;
  architecture comparisons need matched *budget*, and the benefit of the doubt
  (a 3× rerun) before a negative verdict.
- **Always run the 20k-node confirmatory.** 5k parity hid a −45 Elo 20k loss;
  a 5k-accept-only policy would have promoted a weaker pair.
- **Measure quantisation first** when a trained net disappoints — it took an
  hour and redirected 3 days of compute at the right hypothesis.

## 6. Assets preserved (for any future round 3)

- `/workspace/runs/v2/nets/{value,policy}.nn` (v2r1), ckpts under
  `/workspace/runs/v2/{value,policy}/` (best/last)
- `/workspace/runs/v2x/nets/policy.nn` (v2x, md5 93d8a3590451f8420a080a43bd63fb09),
  ckpts under `/workspace/runs/v2x/policy/` (best/last)
- `/workspace/runs/v2x/quant_gap.py` — reusable float-vs-engine quantisation
  checker for any future architecture
- v2 engine/training code on branch `nnue-v2` (header-version dispatch; v1 unaffected)
- All gate scripts/logs under `/workspace/runs/v2*/`

## 7. Where the strength effort went instead

v1x: the v1 architecture with a 3× budget (1.2M steps) on the full data bank,
plus the CPU-side selfplay flywheel for fresh data — same architecture means
same NPS, so any measured gate gain is free strength with zero regression
risk. See `/workspace/HANDOFF.md`.
