#!/bin/sh
# Train the value net on GPU 0 and the policy net on GPU 1 at the same time
# (two independent processes; no DDP). Usage:
#   tools/train_both.sh <out dir> <shards...> -- <extra args for both>
# Environment: PY=/path/to/venv/python VALUE_ARGS="..." POLICY_ARGS="..."
set -e
OUT=${1:?out dir}; shift
SHARDS=""
while [ $# -gt 0 ] && [ "$1" != "--" ]; do SHARDS="$SHARDS $1"; shift; done
[ "$1" = "--" ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
export PYTHONPATH="$HERE/../build${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 nohup "$PY" "$HERE/../train/train.py" --net value  --shards $SHARDS --out "$OUT/value"  --export "$OUT/value.nn"  $VALUE_ARGS  "$@" > "$OUT/value.log"  2>&1 &
V=$!
CUDA_VISIBLE_DEVICES=1 nohup "$PY" "$HERE/../train/train.py" --net policy --shards $SHARDS --out "$OUT/policy" --export "$OUT/policy.nn" $POLICY_ARGS "$@" > "$OUT/policy.log" 2>&1 &
P=$!
echo "value pid $V (log $OUT/value.log), policy pid $P (log $OUT/policy.log)"
wait $V; echo "value done"; wait $P; echo "policy done"
