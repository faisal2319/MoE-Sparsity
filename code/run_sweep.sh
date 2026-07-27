#!/usr/bin/env bash
# PHASE 5 - The sweep. 3 expert counts x 2 conditions x 3 seeds = 18 runs.
#
# Three expert counts rather than four, to buy the seeds. If Phase 3 showed
# runs under ~40 min, add E=32 for a fourth point.
#
# Everything else held constant: architecture, optimizer, schedule, token
# budget, non-math data, tokenizer.
set -euo pipefail
TOKENS=${TOKENS:-1000000000}
for E in 4 16 64; do
  for COND in finemath swallowmath; do
    for S in 0 1 2; do
      NAME="E${E}_${COND}_s${S}"
      if [ -f "runs/${NAME}/summary.json" ]; then echo "skip $NAME"; continue; fi
      echo "=== $NAME ==="
      python step07_train.py --experts $E --condition $COND --seed $S \
        --total-tokens $TOKENS --run-name "$NAME"
    done
  done
done