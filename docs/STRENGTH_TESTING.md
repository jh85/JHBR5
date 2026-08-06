# Reliable strength testing

`tools/run_strength_test.sh` provides a portable A/B match environment for two
USI engines. It installs `cshogi` into `build-strength/venv` when necessary,
detects visible NVIDIA GPUs, runs games in opening pairs with colors reversed,
and continuously saves results and paired confidence intervals.

## 1. Create a fixed opening suite

Create the suite once and reuse the exact file for every comparison:

```bash
./tools/run_strength_test.sh openings \
  /data/new_jhbr2/user_book1.ybb \
  build-strength/openings-512.txt \
  --count 512 --min-ply 8 --max-ply 12
```

The generator is deterministic by default. It follows varied moves within 100
centipawns of the best YBB move and rejects final positions whose best stored
evaluation is more than 250 centipawns from equality. The file records the
source-book SHA-256 and generation parameters.

This opening suite is only a test input. JHBR3's `BookFile` and
`UseGoteExitBook` remain disabled during normal A/B tests.

## 2. Run a fast screening match

The following example compares one option while holding everything else fixed:

```bash
MODEL=/workspace/JHBR3/engines/shogi_bt4_epoch3.engine

./tools/run_strength_test.sh match \
  --engine-a ./build-trt/jhbr3 \
  --engine-b ./build-trt/jhbr3 \
  --openings build-strength/openings-512.txt \
  --pairs 200 --nodes 100000 \
  --option-a OnnxModel="$MODEL" \
  --option-b OnnxModel="$MODEL" \
  --option-a WorkersPerGpu=2 \
  --option-b WorkersPerGpu=1 \
  --gpus-per-worker 1 \
  --output strength-runs/workers-2-vs-1
```

On an eight-GPU host, `--gpus-per-worker 1` automatically launches eight
parallel pair workers. Each worker exposes one GPU to two persistent engine
processes and automatically sets `NumGPUs=1`. The engines move alternately, so
only one searches at a time, although both models occupy GPU memory.

Use fixed nodes for initial algorithm and parameter comparisons. It greatly
reduces noise from machine load and NPS changes. Change only one factor in each
match.

### Automated RTX 5090 topology screening

`tools/run_topology_tuning.py` automates the throughput and paired-game
workflow. It first sweeps workers at the incumbent batch size, then sweeps
batch size at the leading worker count. To avoid assuming those parameters are
independent, it also benchmarks a small interaction grid before launching the
paired match.

For the current eight-RTX-5090 production configuration:

```bash
python3 tools/run_topology_tuning.py \
  --engine ./build-trt/jhbr3 \
  --model /workspace/JHBR3/engines/current.engine \
  --openings build-strength/openings-512.txt \
  --gpus 8 \
  --baseline-workers 16 \
  --baseline-minibatch 256 \
  --nn-cache-size 1000000 \
  --output topology-runs/rtx5090
```

The defaults sweep workers `1,2,4,8,12,16,24,32`, minibatches
`32,64,96,128,192,256`, use 20 two-second benchmark positions, test the top
three worker/batch interaction grid, and run 80 opening pairs at one-second
byoyomi. On an eight-GPU host the paired stage uses eight one-GPU match workers
in parallel. Use `--resume` after an interruption; completed benchmarks and
pairs are reused.

Topology tuning intentionally uses fixed time because its purpose includes
measuring how much useful search each hardware configuration completes.
Fixed-node matches remain the right choice for search-semantic parameters,
where an NPS difference should not change the nominal work budget. Keep
`--minibatch-values` at or below the TensorRT engine's reported `max_batch`;
larger requests are split and do not create a true larger inference batch.
For clock-based matches, the runner automatically raises `MaxNodes` to
10,000,000 unless an explicit `--option-a MaxNodes=...` or
`--option-b MaxNodes=...` is supplied; otherwise JHBR3's default 800-node cap
would end most searches before the clock budget.

Additional engine settings can be held constant with repeatable
`--engine-option NAME=VALUE` arguments. The driver controls `OnnxModel`,
`WorkersPerGpu`, `MinibatchSize`, `MaxNodes`, `NumGPUs`, and `NNCacheSize`
itself.

The script deliberately does not change engine defaults. NPS selects only the
candidate sent to the paired match; inspect `result.json` and the strength
confidence interval before adopting it. A final winner must still be confirmed
with the production eight-GPU topology and production clock.

### Automated SPSA search-parameter tuning

`tools/run_spsa_tuning.py` tunes continuous USI search parameters after the
TensorRT profile, worker count, and minibatch size have been frozen. Each
iteration creates simultaneous plus/minus perturbations around the current
parameter vector and compares them directly in one color-reversed match. The
driver alternates which perturbation is engine A, uses logarithmic coordinates
for `CBase`, bounds noisy updates, tail-averages the later SPSA centers, and can
run a final baseline-versus-recommendation match.

On this two-RTX-3090 host, using the batch-64 TensorRT plan:

```bash
python3 tools/run_spsa_tuning.py \
  --engine ./build-trt/jhbr3 \
  --model ./engines/shogi_bt4_epoch3_trt_o64_m64_ws16384.engine \
  --openings /data/new_jhbr2/JHBR2/build-strength/openings-512.txt \
  --preset nonroot-puct \
  --engine-option WorkersPerGpu=2 \
  --engine-option MinibatchSize=64 \
  --engine-option NNCacheSize=0 \
  --iterations 30 --pairs-per-iteration 8 --nodes 4000 \
  --gpu-devices 0,1 --gpus-per-worker 1 \
  --confirmation-pairs 100 \
  --confirmation-byoyomi-ms 1000 \
  --confirmation-gpus-per-worker 2 \
  --output spsa-runs/rtx3090-nonroot
```

The screening phase above launches one pair worker per GPU and uses fixed
nodes. The confirmation phase exposes both GPUs to each engine and uses the
real one-second clock. A separate opening file can be reserved for confirmation
with `--confirmation-openings`.

For a new eight-RTX-5090 machine, run topology tuning first, then substitute its
selected worker and minibatch values below:

```bash
python3 tools/run_spsa_tuning.py \
  --engine ./build-trt/jhbr3 \
  --model /workspace/JHBR3/engines/current.engine \
  --openings build-strength/openings-512.txt \
  --preset puct \
  --engine-option WorkersPerGpu=BEST_WORKERS \
  --engine-option MinibatchSize=BEST_BATCH \
  --engine-option NNCacheSize=BEST_CACHE \
  --iterations 40 --pairs-per-iteration 16 --nodes TARGET_NODES \
  --gpu-devices 0,1,2,3,4,5,6,7 --gpus-per-worker 1 \
  --confirmation-pairs 100 \
  --confirmation-main-time-ms 300000 \
  --confirmation-byoyomi-ms 10000 \
  --confirmation-gpus-per-worker 8 \
  --output spsa-runs/rtx5090-puct
```

Choose `TARGET_NODES` from the median nodes per move at the intended production
clock. This keeps fixed-node screening close to the search regime that will
actually be deployed. With eight GPUs and `--gpus-per-worker 1`, sixteen
opening pairs take two parallel waves per SPSA iteration. The full-machine
confirmation intentionally uses one pair worker because each engine sees all
eight GPUs.

The built-in presets are:

- `nonroot-puct`: `CInit`, logarithmic `CBase`, and `FpuReduction`;
- `puct`: the three non-root parameters plus their root variants;
- `none`: only explicitly supplied `--parameter` values.

Custom continuous parameters use:

```text
--parameter NAME=INITIAL:MIN:MAX[:linear|log[:float|int]]
```

For example,
Start by testing `UseMovesLeft=true` against the default `false`. If that is
positive, tune one lc0-style continuous parameter at a time, for example
`--parameter MovesLeftSlope=0.0027:0.0:0.01:linear:float`.
Integer/discrete parameters are accepted for controlled experiments, but SPSA
is normally a poor optimizer for mate depths, worker counts, and minibatch
sizes. Use grids for those.

Every run contains `config.json`, append-only `history.jsonl`, `state.json`,
`result.json`, per-iteration strength-test directories, and an optional
`confirmation/` directory. An interrupted run resumes with the identical
command plus `--resume`; both SPSA state and partially completed paired matches
are reused. Use `--dry-run` to validate paths, parameter bounds, GPU grouping,
and the first generated match command without creating a run.

SPSA is a noisy screening optimizer, not its own proof of improvement. Adopt
only the tail-averaged `recommendation_options` from `result.json`, and only
when its fresh confirmation match is convincing. Do not tune topology and
PUCT in the same SPSA run.

## 3. Confirm using the production GPU topology and time control

A change that wins in one-GPU screening should be confirmed in the actual
deployment topology:

```bash
./tools/run_strength_test.sh match \
  --engine-a ./build-trt/jhbr3 \
  --engine-b ./build-trt/jhbr3 \
  --openings build-strength/openings-512.txt \
  --pairs 100 --main-time-ms 300000 --byoyomi-ms 10000 \
  --option-a OnnxModel="$MODEL" \
  --option-b OnnxModel="$MODEL" \
  --option-a WorkersPerGpu=2 \
  --option-b WorkersPerGpu=1 \
  --gpus-per-worker 8 \
  --output strength-runs/workers-production
```

With eight visible GPUs, `--gpus-per-worker 8` runs one pair at a time and
automatically sets `NumGPUs=8`. This is slower but captures multi-GPU scaling,
batching, and real time-management effects.

Use `--gpu-devices 2,3,4,5` to select devices explicitly. The wrapper respects
an existing `CUDA_VISIBLE_DEVICES`. Use `--workers N` only when overriding the
automatic one-worker-per-GPU-group choice.

## Results and resuming

Each run directory contains:

- `config.json`: commands, options, executable/model hashes, opening hash, Git
  revision, host, and GPUs.
- `pairs.jsonl`: append-only game records, including moves and final info lines.
- `summary.json`: W-L-D, paired score, Elo estimate, 95% interval, LOS, and
  pentanomial counts.
- `logs/`: engine stderr, plus complete USI traffic when `--protocol-log` is
  requested.

If a cloud allocation ends, rerun the identical command with `--resume`. The
runner verifies the saved configuration and skips completed pairs.

The approximate confidence interval uses a Jeffreys-smoothed pentanomial model
and treats each color-reversed opening pair as one sample, not two independent
games. This avoids both zero-width intervals in tiny runs and overstating
certainty when an opening strongly favors one side. A result is actionable
when the interval is narrow enough for the decision; nine games are generally
not enough. Keep the raw run directory whenever a result influences an engine
change.

## Practical test order

1. Run a short smoke test (`--pairs 2 --nodes 1000`).
2. Screen candidates on one GPU per worker at fixed nodes.
3. Extend promising tests until the paired interval is useful.
4. Confirm the winner with the production GPU count and production clock.
5. Test the final binary against a frozen previous binary, not merely two
   option sets in a newly rebuilt executable.

For a binary-versus-binary test, set `--engine-a` to the saved baseline and
`--engine-b` to the candidate. Absolute model paths are recommended.
