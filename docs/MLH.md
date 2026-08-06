# Moves-left head (MLH)

## Decision

Keep the head, but keep its search effect opt-in until paired Shogi games show
a gain. The head is a small scalar branch, supplies an auxiliary training
signal to the shared body, and can distinguish short wins from long wins where
WDL cannot. Removing it would also invalidate existing checkpoint and engine
output contracts for little practical saving.

`UseMovesLeft=false` is therefore JHBR3's safe default. When enabled, search
uses the head only if every active inference evaluator reports a real `mlh`
output. A missing head is not represented by the numeric value zero.

## Comparison with lc0

The reference is lc0 master commit
`d8ce48258c39d331c119f8c8729374ceb3df8409` (2026-05-06), especially
`src/search/classic/search.cc`, `node.cc`, `params.cc`, and
`src/trainingdata/trainingdata.cc`.

| Concern | lc0 | JHBR3 after this refactor |
|---|---|---|
| Head | Non-negative scalar, predicted plies left | Same |
| Training target | Plies from the current position; final real row is at least 1 | Same for newly generated shards |
| Tree statistic | Running average M per node | Running average M for the current node and each child edge |
| Search use | Prefer shorter wins and longer losses | Same |
| Formula | Slope, cap, parent-Q threshold, smooth Q scaling, linear/quadratic factors | Same formula, converted from lc0 Q `[-1,1]` to JHBR3 win probability `[0,1]` |
| Capability | Disabled when the backend/network has no MLH | Same |
| Default | Active with tuned chess defaults when MLH exists | Opt-in; chess values are exposed only as starting points for Shogi tuning |
| Other use | UCI display and self-play training-data estimates | Not currently used for USI display or time management |

Lc0 does not use neural MLH as its general time-management estimator; its
smooth time manager has a separate position-based moves-left estimator. JHBR3
likewise should not feed MLH into time allocation without independent testing.

## Compatibility

Older JHBR3 shards used `n_moves - i - 1`, while new generators use
`n_moves - i`. Existing checkpoints and TensorRT engines remain loadable. The
one-ply constant offset mostly cancels in the relative search term, so old
models can be tested without conversion. Do not mix old and new label
conventions within a new training run; regenerate shards when retraining if
consistent absolute MLH calibration matters. New shards store
`mlh_version=1`; the trainer treats an absent version as legacy version 0 and
rejects a dataset prefix that mixes versions.

The old `MovesLeftWeight` / `MovesLeftCap` options implemented a different,
hard-threshold formula and have been replaced by `UseMovesLeft` plus lc0's six
parameters. Strength runs should compare the master switch first and tune only
after that comparison is positive.
