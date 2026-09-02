#!/bin/sh
# Nodes/s per thread count with the USI bench command.
#   tools/thread_scaling.sh build/jhbr5 nets/value.nn nets/policy.nn [nodes] [threads...]
set -e
ENGINE=${1:?engine}; VALUE=${2:?value.nn}; POLICY=${3:?policy.nn}; NODES=${4:-20000}
shift 4 2>/dev/null || shift $#
THREADS=${*:-"1 2 4 8"}
for t in $THREADS; do
  printf 'setoption name ValueNet value %s\nsetoption name PolicyNet value %s\nbench %s %s\nquit\n' "$VALUE" "$POLICY" "$NODES" "$t" \
    | "$ENGINE" | grep "info string bench threads" | sed 's/info string //'
done
