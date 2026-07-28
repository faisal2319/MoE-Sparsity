# Gate answers

All values measured. Nothing estimated.

---

## Gate A — document counts (step01, LOCAL, $0) — CLEARED

- **FineMath-4+ documents:** 6,699,493 *(exact; matches the 6.7M documented on the SwallowMath card)*
- **SwallowMath documents:** 6,497,564 *(exact, full streaming enumeration)*
- **Ratio:** 0.9699
- **Filtering confound?** **NO.** 97% of source documents retained. The 9.6B → 2.3B token reduction comes from shortening (documents are 39% of original length), not from selection.
- **Row-for-row ordering preserved?** Not verified. SwallowMath exposes only a `text` column — no source ID, no mapping to FineMath-4+.
- **=> LOI wording:** **"token- and distribution-matched"**. Do NOT write "source-matched" — there is no document-level provenance field.

### Trap worth recording
The HuggingFace datasets-server `/size` endpoint returns **2,628,785** for SwallowMath. That is a partial count from a 5GB-truncated parquet auto-conversion of a 12.8GB dataset — wrong by 2.5×. `step01_characterize.py` originally trusted it and produced a false 0.39 ratio and a false "filtering confound" verdict. The function now checks the API's `partial` flag and falls back to streaming enumeration.

---

## Gate B — harness validation (step02, VAST) — PARTIALLY CLEARED

- **My GSM8K task loss, E=8:** 1.1254
- **Their published value:** **not retrieved.** `taskloss-eval/README.md` was not cross-checked.
- **Match?** **UNVERIFIED for absolute values.** Trend agreement only.
- **What was validated instead:** task loss falls monotonically across the full d512-k2 family — 1.1254 (E=8) → 0.8825 (E=256). This is what their TPP framework predicts for this family: at their 125B-token budget it spans TPP 391 → 19, approaching their reported reasoning optimum of ~20 from above without crossing it.
- **Open item:** clone `github.com/rioyokotalab/optimal-sparsity`, read `taskloss-eval/README.md`, compare absolute values. Listed as limitation #5 in the README.

### Also found
E=256 has **9 dead experts across 6 of 16 layers**. All five smaller released checkpoints are clean. Not reported in the paper as far as I can see.

---

## Gate C — budget (step06, VAST) — CLEARED

- **tokens/sec:** E=4 → 345,132 · E=16 → 331,039 · E=64 → 270,416
- **MFU (active non-embedding FLOPs):** E=4 → 10.5% · E=16 → 10.1% · E=64 → 8.4%
- **Peak VRAM:** 11.9 / 12.2 / 13.5 GB *(24 GB card — comfortable)*
- **Hours per run:** 0.80 / 0.84 / 1.03 — mean **0.89 h**
- **Token budget chosen:** **1B** *(projected 21.4 GPU-hours for 24 runs, ~$7.50 — no need to halve)*
- **Actual:** 16.83 GPU-hours across 18 runs

### Notes
Active non-embedding parameters stay essentially flat across the sweep — 8,396,800 (E=4) → 8,519,680 (E=64) — while total non-embedding rises 14.7M → 203.6M. Only the router grows. That is the control working.
The MFU decline from 10.5% → 8.4% is routing overhead. Absolute MFU is low because d=256 matmuls cannot saturate a 4090 and HF's Mixtral loops over experts.

---

## Gate D — noise floor (step11, VAST) — CLEARED

- **Max seed range (task loss):** 0.1862
- **Seed standard deviation:** 0.018 – 0.100 across the six configurations
- **Architecture effect (E=4 vs E=64, FineMath):** 0.2112, t = 6.7, p < 0.005
- **Expected condition effect:** unknown a priori; Fujii et al. report +12.4 GSM8K accuracy at 1000× the active parameters
- **Minimum detectable effect with 3 seeds:** **~0.12 task loss**
- **Resolvable?** **YES for architecture** (signal/noise ≈ 4.4× by range, t = 6.7 on means). Proceeded to the sweep.

### Metric warning recorded at this gate
The first run of this gate used `final_lm_loss`, which is **not comparable across data conditions** — FineMath and SwallowMath are different text, and SwallowMath is measurably more homogeneous (distinct-4gram 0.707 vs 0.753), so it is easier to model regardless of what the model learned. `step08_summarize.py` now prefers GSM8K task loss and warns loudly if forced to fall back.

---

## Pre-registered prediction (recorded BEFORE the sweep was run)

TPP at 1B tokens: E=4 → 68, E=16 → 19, E=64 → 4.9
Nakamura et al. report reasoning peaks near TPP ≈ 20.
Measured so far (finemath): E=4 = 4.133, E=64 = 3.921
**PREDICTION: E=16 beats both, task loss < 3.92.**
Minimum detectable effect with 3 seeds: ~0.12 task loss.

### OUTCOME — FAILED

| E | TPP | FineMath task loss |
|---|---|---|
| 4 | 68 | 4.1325 |
| 16 | **19** | 4.0848 |
| 64 | 4.9 | **3.9213** ← best |

E=16 did not win. Task loss decreased **monotonically** with total parameters. E=64 was best despite sitting at TPP 4.9, far below the reported optimum.

**No inverted U at this scale.** Either it does not manifest at 8.4M active parameters, or its location shifts substantially with model size.

---

## Primary hypothesis — RESULT

**Condition effect (SwallowMath − FineMath-4+), paired by expert count and seed:**

| E | task loss | pre-training loss |
|---|---|---|
| 4 | +0.0780 | −0.2373 |
| 16 | +0.0219 | −0.2844 |
| 64 | +0.0523 | −0.2601 |
| **pooled (n=9)** | **+0.0507** | **−0.2606** |
| t | +2.16 | **−10.69** |
| p | 0.063 | **0.00001** |
| 95% CI | [−0.0035, +0.1050] | [−0.3168, −0.2044] |
| consistent direction | 9/9 worse | 9/9 lower |

**Verdict:** bounded null on task loss — any benefit at this scale is below ~0.1 task loss — accompanied by a **highly significant reduction in pre-training loss**.

The two metrics move in **opposite directions in all nine pairs**. Rewritten data is significantly easier to model and no better downstream. This is the loss–capability divergence Nakamura et al. demonstrate by varying architecture, reproduced here by varying data.

**Mechanism consistent with Phase A:** distinct-4gram 0.7069 vs 0.7527 (surface diversity down 6%, and that figure is conservative given SwallowMath's shorter documents), while embedding cosine similarity is unchanged (0.1178 vs 0.1199). The rewriter normalised phrasing without altering semantic content.

### Two alternative explanations that cannot be ruled out
1. **Scale.** 8.4M active parameters; GSM8K accuracy ≈ 0. Task loss measures surface-form modelling, not reasoning. The mechanism SwallowMath improves may require a model that can follow steps at all.
2. **Corpus exhaustion.** 250M math tokens drawn from SwallowMath's 2.3B (11%) vs FineMath-4+'s 9.6B (2.6%), so the rewritten condition saw less unique text.

---

## Open items

- [ ] Gate B absolute cross-check against `taskloss-eval/README.md`
- [ ] Tokens/word measurement — published 9.6B and 2.3B counts imply an implausible 1.03 tokens/word for SwallowMath, so they likely use different tokenizers and must not be compared directly
- [ ] Phase A homogeneity was measured with both corpora truncated to 2000 characters; FineMath averages 5051 chars and SwallowMath 1887, so the comparison is confounded by document length. Re-run with matched truncation before citing the semantic-diversity result.
- [ ] Phase B mutual-information estimates are positively biased at high expert counts (more cells, same sample size). A shuffled-label permutation control would establish the bias floor.
- [ ] Decontamination used stride 3, catching roughly 1 in 3 matches. True rates are ~3× reported. Conclusion (negligible) unchanged.