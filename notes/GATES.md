# Gate answers

Fill in as you clear each one. These decide what goes in the README and the LOI.

## Gate A - document counts (step01, LOCAL, $0)
- FineMath-4+ documents:
- SwallowMath documents:
- Ratio:
- Filtering confound? (ratio < 0.95 = YES):
- Row-for-row ordering preserved?:
- => LOI wording: "source-matched" / "token-and-distribution-matched" only

## Gate B - harness validation (step02, VAST)
- My GSM8K task loss, E=8:
- Their published value:
- Match?:
- If no, what was wrong:

## Gate C - budget (step06, VAST)
- tokens/sec at E=4 / E=16 / E=64:
- MFU (active FLOPs):
- Peak VRAM:
- Hours per run:
- Token budget chosen (500M or 1B):

## Gate D - noise floor (step11, VAST)
- Max seed range:
- Expected effect size:
- Resolvable? If NO -> STOP, write it up, that is the result:

## Pre-registered prediction (before sweep, 2026-07-27)
TPP at 1B tokens: E=4 -> 68, E=16 -> 19, E=64 -> 4.9
Nakamura et al. report reasoning peaks near TPP ~= 20.
Measured so far (finemath): E=4 = 4.133, E=64 = 3.921
PREDICTION: E=16 beats both, task loss < 3.92.
Minimum detectable effect with 3 seeds: ~0.12 task loss.
