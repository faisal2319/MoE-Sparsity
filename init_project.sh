#!/usr/bin/env bash
# ============================================================================
#  MoE Sparsity x Data Quality Pilot - project scaffold
#
#  Creates every directory AND every file you need, as empty placeholders.
#  Paste the real contents in afterwards.
#
#  SAFE TO RE-RUN: never overwrites a file that already has content.
#  Run from the repo root:   bash code/init_project.sh
# ============================================================================
set -euo pipefail

# --- helper: create file only if missing or empty ---------------------------
stub () {
  local path="$1"; shift
  local note="$*"
  if [ -s "$path" ]; then
    echo "  keep    $path  (already has content)"
    return
  fi
  mkdir -p "$(dirname "$path")"
  case "$path" in
    *.py) printf '# %s\n# --- paste contents here ---\n' "$note" > "$path" ;;
    *.sh) printf '#!/usr/bin/env bash\n# %s\n# --- paste contents here ---\n' "$note" > "$path"; chmod +x "$path" ;;
    *)    printf '<!-- %s -->\n' "$note" > "$path" ;;
  esac
  echo "  created $path"
}

echo "=== Directories ==="
mkdir -p code artifacts/tokenizer data runs out/phaseA out/phaseB out/phaseC figures notes
for d in out/phaseA out/phaseB out/phaseC figures notes; do touch "$d/.gitkeep"; done
echo "  ok"

echo
echo "=== Code files (paste contents into these) ==="
stub code/step00_setup_vast.sh          "step00 (VAST)  - environment bootstrap"
stub code/step01_characterize.py        "step01 (LOCAL) - data characterization + GATE A document counts"
stub code/step02_checkpoint_analysis.py "step02 (VAST)  - released checkpoints, task loss + routing + GATE B"
stub code/step03_train_tokenizer.py     "step03 (VAST)  - 16k BPE on raw FineMath-4+"
stub code/step04_pack.py                "step04 (VAST)  - tokenize + pack both data conditions"
stub code/step05_model.py               "step05 (VAST)  - Mixtral miniature, z-loss, param counts, MFU  [LIBRARY]"
stub code/step06_benchmark.py           "step06 (VAST)  - throughput benchmark + GATE C budget"
stub code/step07_train.py               "step07 (VAST)  - train one (E, condition, seed) cell"
stub code/step08_summarize.py           "step08 (VAST+LOCAL) - aggregate, noise floor, plots"
stub code/run_seeds.sh                  "6 seed-variance runs - GATE D. Run BEFORE the sweep."
stub code/run_sweep.sh                  "18 sweep runs: 3 expert counts x 2 conditions x 3 seeds"

echo
echo "=== Docs ==="
stub IMPLEMENTATION_PLAN.md "the single document to work from"

if [ ! -s requirements.txt ]; then
cat > requirements.txt << 'EOF'
torch>=2.4.0
transformers>=4.44.0
datasets>=2.20.0
tokenizers>=0.19.0
accelerate>=0.33.0
numpy>=1.26.0
pandas>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
tqdm
wandb
pyarrow
sentence-transformers>=3.0.0
scikit-learn
huggingface_hub
EOF
echo "  created requirements.txt"
else echo "  keep    requirements.txt"; fi

if [ ! -s .gitignore ]; then
cat > .gitignore << 'EOF'
# Large artifacts - never commit
data/
runs/
artifacts/
*.npy
*.bin
*.safetensors
wandb/

# Python
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/

# OS
.DS_Store
EOF
echo "  created .gitignore"
else echo "  keep    .gitignore"; fi

# --- README: scope disclaimer must exist from day one -----------------------
if [ ! -s README.md ]; then
cat > README.md << 'EOF'
# MoE Sparsity x Data Quality Pilot

> **Scope.** This is a small-scale methodological pilot at roughly 1/1000 the
> compute of Nakamura et al., *Optimal Sparsity of Mixture-of-Experts Language
> Models for Reasoning Tasks* (ICLR 2026 Oral). It cannot establish a
> compute-optimal sparsity law. Its purpose is to test whether an interaction
> between mathematical pre-training data quality and MoE sparsity is detectable
> in a controlled, resource-constrained setting, and to characterise what the
> SwallowMath rewriting pipeline changed relative to its source corpus
> FineMath-4+.

## Question

Under a fixed pre-training compute budget, does the quality of mathematical
pre-training data change the relationship between MoE sparsity and reasoning
performance? Secondary: are any changes accompanied by differences in expert
utilization and specialization?

## Results

### 1. Data characterization
<!-- what rewriting changed: lengths, diversity, structure, topics, homogeneity -->

### 2. Released checkpoint analysis
<!-- harness validation against published task loss; routing across sparsity -->

### 3. Pilot sweep
<!-- 3 expert counts x 2 conditions x 3 seeds, with error bars -->

## Failures and negative results
<!-- from notes/FAILURES.md - router collapse, diverged runs, discarded configs -->

## Compute cost
| Item | Value |
|---|---|
| Models trained | |
| Total tokens | |
| GPU-hours (incl. failed runs) | |
| Mean MFU (active FLOPs) | |
| Total cost | |

## Limitations
- Possible filtering confound: see notes/GATES.md, Gate A
- Math-enriched mixture (~25%) departs from the 4.79% used in the source paper
- ~1/1000 the compute of the reference study
- SwallowMath's dataset card documents no decontamination procedure
- Task loss at this scale may measure surface form rather than reasoning

## Reproduce
```bash
bash code/init_project.sh
pip install -r requirements.txt
cd code && python step01_characterize.py --help
```
See `IMPLEMENTATION_PLAN.md` for the full step-by-step.
EOF
echo "  created README.md"
else echo "  keep    README.md"; fi

echo
echo "=== Notes ==="
if [ ! -s notes/GATES.md ]; then
cat > notes/GATES.md << 'EOF'
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
EOF
echo "  created notes/GATES.md"
else echo "  keep    notes/GATES.md"; fi

if [ ! -s notes/FAILURES.md ]; then
cat > notes/FAILURES.md << 'EOF'
# Failure log

Every diverged run, router collapse, OOM, and discarded config goes here.
This becomes a README section. Their lab publishes failures and open-sources
logs; matching that norm is itself a signal.

| Date | Run | What happened | Resolution |
|---|---|---|---|
EOF
echo "  created notes/FAILURES.md"
else echo "  keep    notes/FAILURES.md"; fi

echo
echo "============================================================"
echo "Structure:"
find . -not -path './.git/*' -not -path './.venv/*' -not -name '.gitkeep' \
     -not -path './__pycache__*' | sort | sed 's|[^/]*/|  |g'
echo "============================================================"
echo
echo "Next:"
echo "  1. Paste contents into code/step0*.{py,sh} and IMPLEMENTATION_PLAN.md"
echo "  2. python -m venv .venv && source .venv/bin/activate"
echo "  3. pip install -r requirements.txt"
echo "  4. cd code && python step01_characterize.py --n-docs 200000 \\"
echo "       --out ../out/phaseA --embeddings --decontam"
echo
echo "  Gate A runs on your laptop and costs nothing. Clear it before renting a GPU."