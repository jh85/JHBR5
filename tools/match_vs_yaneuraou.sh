#!/bin/sh
# JHBR5 vs YaneuraOu at equal CPU time and threads (MCTS vs alpha-beta).
#
#   tools/match_vs_yaneuraou.sh <yaneuraou binary> <yaneuraou EvalDir> <openings> [pairs] [byoyomi_ms] [threads] [output]
#
# Environment: JHBR5=build/jhbr5 VALUE=nets/value.nn POLICY=nets/policy.nn
# YANE_OPTS="USI_Hash=1024 ..." for extra YaneuraOu options.
set -e
YANE_BIN=${1:?yaneuraou binary}; EVALDIR=${2:?EvalDir}; OPENINGS=${3:?openings}
PAIRS=${4:-100}; BYOYOMI=${5:-1000}; THREADS=${6:-4}; OUT=${7:-strength-runs/jhbr5-vs-yaneuraou-b$BYOYOMI-t$THREADS}
HERE=$(cd "$(dirname "$0")" && pwd)
JHBR5=${JHBR5:-$HERE/../build/jhbr5}
VALUE=${VALUE:-$HERE/../nets/value.nn}; POLICY=${POLICY:-$HERE/../nets/policy.nn}
set -- python3 "$HERE/strength_test.py" --engine-a "$YANE_BIN" --engine-b "$JHBR5" \
  --openings "$OPENINGS" --pairs "$PAIRS" --byoyomi-ms "$BYOYOMI" --output "$OUT" \
  --option-a "EvalDir=$EVALDIR" --option-a "Threads=$THREADS" --option-a "USI_Ponder=false" \
  --option-b "ValueNet=$VALUE" --option-b "PolicyNet=$POLICY" --option-b "Threads=$THREADS" \
  --option-b "TimeManagement=on"
for o in ${YANE_OPTS:-}; do set -- "$@" --option-a "$o"; done
echo "+ $*"; exec "$@"
