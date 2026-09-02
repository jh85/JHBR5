#!/bin/sh
# JHBR5 vs JHBR3 at equal nodes per move (isolates network quality from speed).
#
#   tools/match_vs_jhbr3.sh <jhbr3 binary> <jhbr3 model (OnnxModel value)> <openings> [pairs] [nodes] [output]
#
# Environment: JHBR5=build/jhbr5 VALUE=nets/value.nn POLICY=nets/policy.nn
# THREADS=4 (JHBR5 threads); JHBR3 options via JHBR3_OPTS="WorkersPerGpu=2 ...".
set -e
JHBR3_BIN=${1:?jhbr3 binary}; MODEL=${2:?jhbr3 model}; OPENINGS=${3:?openings}
PAIRS=${4:-100}; NODES=${5:-5000}; OUT=${6:-strength-runs/jhbr5-vs-jhbr3-n$NODES}
HERE=$(cd "$(dirname "$0")" && pwd)
JHBR5=${JHBR5:-$HERE/../build/jhbr5}
VALUE=${VALUE:-$HERE/../nets/value.nn}; POLICY=${POLICY:-$HERE/../nets/policy.nn}
set -- python3 "$HERE/strength_test.py" --engine-a "$JHBR3_BIN" --engine-b "$JHBR5" \
  --openings "$OPENINGS" --pairs "$PAIRS" --nodes "$NODES" --output "$OUT" \
  --option-a "OnnxModel=$MODEL" --option-a "MaxNodes=$NODES" \
  --option-b "ValueNet=$VALUE" --option-b "PolicyNet=$POLICY" --option-b "Threads=${THREADS:-4}"
for o in ${JHBR3_OPTS:-}; do set -- "$@" --option-a "$o"; done
echo "+ $*"; exec "$@"
