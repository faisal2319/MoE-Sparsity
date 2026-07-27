#!/usr/bin/env bash
# PHASE 4 - Seed variance. RUN THIS BEFORE THE SWEEP.
#
# The effect you are looking for is small. With one seed per config you cannot
# distinguish signal from noise. This establishes the noise floor.
#
# DECISION GATE: if the seed spread is comparable to or larger than the
# between-config differences, the sweep cannot resolve the effect at this
# scale. Report that honestly and stop. It is a legitimate result and far
# better than plotting noise as signal.
set -euo pipefail
TOKENS=${TOKENS:-1000000000}
for E in 4 64; do
  for S in 0 1 2; do
    echo "=== E=$E seed=$S ==="
    python step07_train.py --experts $E --condition finemath --seed $S \
      --total-tokens $TOKENS --run-name "seedvar_E${E}_s${S}"
  done
done
echo "Now: python step08_summarize.py --pattern 'seedvar_*' --report-spread"