# JHBR3 MLH refactor report

Date: 2026-08-06

## Decision

Keep the moves-left head (MLH), but keep its search effect opt-in until paired
Shogi games demonstrate a strength improvement.

The default JHBR3 model's MLH branch has only 43,657 parameters, about 0.157%
of the 27,810,958-parameter default model. It supplies an auxiliary training
signal to the shared body, distinguishes short wins from long wins where WDL
cannot, and is already part of existing checkpoint and engine output formats.
Removing it would save little while breaking those formats.

Enable its search effect with:

```text
setoption name UseMovesLeft value true
```

It remains disabled by default because lc0's parameter values were tuned for
chess, not Shogi. The next strength-testing step should compare
`UseMovesLeft=true` against the default `false` in paired games.

## lc0 comparison

The comparison used the current official lc0 master commit:

```text
d8ce48258c39d331c119f8c8729374ceb3df8409
```

Relevant lc0 sources:

- `../lc0/src/search/classic/search.cc`: `MEvaluator`, which prefers shorter
  wins and longer losses.
- `../lc0/src/search/classic/node.cc`: running node averages for M.
- `../lc0/src/search/classic/params.cc`: current MLH parameter defaults.
- `../lc0/src/trainingdata/trainingdata.cc`: plies-left target construction.

| Concern | lc0 | JHBR3 before | JHBR3 after |
|---|---|---|---|
| Head output | Non-negative scalar plies | Same architecture | Kept and documented as lc0-style V1 |
| Training target | Plies from current position; final real row is at least 1 | Plies after teacher move; final row was 0 | Plies from current position; final row is at least 1 |
| Tree statistic | Running M average per node | Raw parent NN M versus searched child average | Comparable running averages for parent and children |
| Search formula | Slope, cap, parent-Q threshold, smooth scaling, linear/quadratic factors | Simplified hard-threshold weight and delta cap | Same formula as lc0, converted to JHBR3's `[0,1]` score range |
| Capability handling | Disabled when network/backend lacks MLH | Missing output appeared as numeric zero | Explicit `has_moves_left()` capability |
| Default search use | Active for capable lc0 networks | Disabled through zero weight | Still opt-in for safe Shogi rollout |
| Other uses | UCI display and self-play target estimates | None | Still not used for USI display or time management |

Lc0 does not use the neural MLH as its general time-management estimator; its
smooth time manager uses a separate position-based estimator. JHBR3 therefore
does not feed MLH into time allocation.

## Problems found and fixed

### 1. ONNX Runtime discarded MLH

The ONNX Runtime backend requested every model output but processed only
policy and WDL. `NNOutput.moves_left` therefore stayed at zero for every ORT
evaluation.

`inference/nn_eval.cc` now:

- locates `policy`, `wdl`, and `mlh` by output name;
- reads the MLH tensor when present;
- reports whether the loaded model has MLH;
- retains the historical positional fallback for older policy/WDL outputs.

TensorRT already copied MLH correctly. Its allocation was cleaned up so MLH
device and pinned-host buffers are allocated only when the engine has an MLH
output.

### 2. Missing MLH was indistinguishable from zero

Zero is a legitimate terminal moves-left value, so it cannot also mean that a
model lacks the head.

Both evaluator implementations now expose:

```cpp
bool has_moves_left() const;
```

Search enables the M effect only if every active evaluator has MLH. Model-load
logging now prints `mlh=yes` or `mlh=no`, and warns when `UseMovesLeft` was
requested for a model without the output.

### 3. Parent and child M statistics were inconsistent

The previous search compared:

- the parent's one-time raw NN estimate; and
- the child's average over searched playouts.

Those values drifted into different statistical meanings as the tree grew.

The refactor adds completed MLH sample counts and sums to nodes and child
edges. A node begins with its own NN estimate, then receives one updated M
sample for each completed descendant backup. Virtual visits are excluded from
the MLH Q threshold and scaling calculation.

Terminal backups still start at zero and add one ply per edge toward the root,
matching lc0's convention.

### 4. Search formula was only a rough lc0 approximation

JHBR3 previously used:

```text
-weight * sign(Q - 0.5) * clamp(child_M - parent_M)
```

The new implementation matches lc0's shape:

```text
base = clamp(slope * (child_M - parent_M), -max_effect, max_effect)
base *= sign(-child_Q)
base *= constant + scaled * normalized_abs_Q
                 + quadratic * normalized_abs_Q^2
```

The parent must exceed the configured absolute-Q threshold. The normalized Q
term produces a smooth ramp above the threshold rather than an abrupt full
effect.

Lc0 uses Q in `[-1,1]`; JHBR3 search scores are win probabilities in `[0,1]`.
The final MLH utility is therefore multiplied by 0.5 before being added to the
JHBR3 PUCT score.

The exposed lc0-shaped defaults are:

```text
MovesLeftMaxEffect       0.0345
MovesLeftThreshold       0.8
MovesLeftSlope           0.0027
MovesLeftConstantFactor  0.0
MovesLeftScaledFactor    1.6521
MovesLeftQuadraticFactor -0.6521
```

The retired `MovesLeftWeight` and `MovesLeftCap` options used the old formula.
They are now explicitly reported as retired instead of being silently
accepted.

### 5. Training targets were off by one from lc0

Shard generation encoded the position before teacher move `i`, but labeled it
with `n_moves - i - 1`, meaning plies after that move. This made the final real
training row zero even though one recorded move remained.

New shards use:

```python
n_moves - i
```

The fix applies to:

- `gen_pack_shards.py`
- `pack_to_shards.py`
- `gen_agg_shards.py`
- `psv_to_shards.py`

For PSV `recorded-end` labels, the count remains an approximation of the true
game end because recording can stop early, but its zero point now follows the
same current-position convention.

### 6. Mixed label conventions are now rejected

New shards store:

```text
mlh_version = [1]
```

An absent version is treated as legacy version 0. `ShardedDataset` rejects a
dataset prefix containing both versions, preventing a new training run from
silently mixing different absolute MLH targets.

Existing checkpoints and TensorRT engines remain loadable. Their legacy
one-ply offset mostly cancels in the relative `child_M - parent_M` search term,
so they can be used for MLH strength experiments without conversion. Regenerate
shards when retraining if consistent absolute calibration matters.

## Additional data-path cleanup

While updating the PSV MLH tests, a stale synthetic fixture was found. It wrote
cshogi's native Move16 bits directly into a PSV field, although PSV uses
YaneuraOu's Move16 layout. The fixture now uses `cshogi.move16_to_psv()`, and
its board sequence contains a legal bishop drop.

`pack_to_shards.py` also received the same robust `GameDataDecoder` import
resolution already used by `gen_pack_shards.py`.

## Verification performed

### C++ builds and tests

- Full CPU build: passed.
- TensorRT build: passed.
- `test_search_primitives`: passed in CPU and TensorRT builds.
- `test_lockfree_search`: passed in CPU and TensorRT builds.
- `test_tree_reuse`: passed.
- New tests cover:
  - lc0 parameter behavior at and above the Q threshold;
  - shorter-win preference;
  - longer-loss preference;
  - node and edge M backup distances;
  - use of completed M samples rather than virtual visits.

### Inference backends

- TensorRT live GPU search with an existing epoch-3 engine: passed.
  Model load reported `mlh=yes`, and a 64-node search returned a legal PV and
  best move.
- ONNX Runtime was compiled out-of-tree against the official ORT 1.24.4 C++
  SDK: passed.
- ORT live CPU search using `shogi_bt4_epoch3_dynamic.onnx`: passed.
  Model load reported `mlh=yes`, and an 8-node search returned a legal PV and
  best move.
- Existing ONNX models were inspected and expose outputs named exactly:
  `policy`, `wdl`, and `mlh`.

### Python/data tests

- `test_agg_shards.py`: 15 passed, 0 failed.
- `test_psv_to_shards.py`: 7 passed, 0 failed.
- `test/test_promotion_finetune.py`: 8 passed.
- Python compilation checks passed for the changed model, trainer, generator,
  and test files.
- A direct mixed-version test confirmed that legacy/new MLH shards are
  rejected.

### Functional MLH activation

The current command for an experiment is:

```text
setoption name UseMovesLeft value true
```

Leave all six continuous parameters at their defaults for the first paired
comparison. If that comparison is positive, tune one parameter at a time,
starting with `MovesLeftSlope` or `MovesLeftMaxEffect`.

## Files of interest

- `docs/MLH.md`: concise permanent design and compatibility documentation.
- `mcts/search_primitives.{h,cc}`: lc0-style utility and backup integration.
- `mcts/uct_node.h`: running M statistics.
- `mcts/uct_search.{h,cc}`: evaluator capability gating.
- `inference/nn_eval.{h,cc}`: corrected ONNX Runtime MLH handling.
- `inference/nn_tensorrt.{h,cc}`: capability and conditional buffers.
- `usi/usi_engine.cc`: new option surface and model capability logging.
- `shogi_train.py`: MLH shard-version guard.
- `gen_pack_shards.py`, `pack_to_shards.py`, `gen_agg_shards.py`, and
  `psv_to_shards.py`: corrected target convention.

## Workspace note

`build.sh` already had an unrelated local modification before this work
(epoch-3 model paths changed to step-12000 paths). That user change was left
untouched.
