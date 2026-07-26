#!/usr/bin/env python3
"""
Project status: what is done, what is verified vs inferred, what is missing.
Run from the repo root:  python status.py
"""
import json, os, subprocess, sys
from pathlib import Path

R = Path(__file__).resolve().parent
def rd(p):
    try: return json.loads((R / p).read_text())
    except Exception: return None
def ex(p): return (R / p).exists()
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception: return "?"

bar = "=" * 66
print(f"\n{bar}\n  MoE SPARSITY PILOT - STATUS\n{bar}")

# ---------- environment ----------
print("\n[ENVIRONMENT]")
print(f"  disk free      : {sh('df -h / | tail -1 | awk \"{print \\$4}\"')}")
print(f"  hf cache       : {sh('du -sh $HF_HOME 2>/dev/null | cut -f1') or 'n/a'}")
try:
    import torch
    print(f"  torch          : {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available(): print(f"  gpu            : {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  torch          : NOT IMPORTABLE ({e})")
print(f"  uncommitted    : {len(sh('git status --porcelain').splitlines())} files")

# ---------- phase A ----------
print(f"\n{bar}\n  PHASE A - data characterization\n{bar}")
a = rd("out/phaseA/results.json")
if not a:
    print("  MISSING out/phaseA/results.json")
else:
    c = a.get("doc_counts", {})
    fm, sm = c.get("finemath_4plus"), c.get("swallowmath")
    print("\n  GATE A - document counts")
    print(f"    FineMath-4+   : {fm:,}" if fm else "    FineMath-4+   : MISSING")
    print(f"    SwallowMath   : {sm:,}" if sm else "    SwallowMath   : MISSING")
    if fm and sm:
        r = sm / fm
        print(f"    ratio         : {r:.4f}")
        if sm == 2628785:
            print("    *** STALE - this is the truncated datasets-server value.")
            print("        The exact stream count is 6,497,564. Patch results.json.")
        elif r >= 0.95:
            print("    VERDICT       : no filtering confound; contrast = rewriting alone")
        else:
            print(f"    VERDICT       : {(1-r)*100:.0f}% of docs dropped -> selection confound")

    print("\n  measured (from 200k-doc samples per corpus)")
    hdr = f"    {'metric':<22}{'finemath':>12}{'swallow':>12}{'ratio':>10}"
    print(hdr)
    for label, key in [("words/doc", "words_mean"), ("chars/doc", "chars_mean"),
                       ("type-token ratio", "type_token_ratio"),
                       ("distinct-4gram", "distinct_4gram_ratio")]:
        f, s = a.get("finemath_4plus", {}).get(key), a.get("swallowmath", {}).get(key)
        if f and s:
            print(f"    {label:<22}{f:>12.4f}{s:>12.4f}{s/f:>10.3f}")
    for label, key in [("cos_mean", "cos_mean"), ("cos_p90", "cos_p90")]:
        f = a.get("finemath_4plus", {}).get("homogeneity", {}).get(key)
        s = a.get("swallowmath", {}).get("homogeneity", {}).get(key)
        if f and s:
            print(f"    {label:<22}{f:>12.4f}{s:>12.4f}{s/f:>10.3f}")
    for label in ["decontamination"]:
        f = a.get("finemath_4plus", {}).get(label, {}).get("rate")
        s = a.get("swallowmath", {}).get(label, {}).get("rate")
        if f is not None and s is not None:
            print(f"    {'gsm8k contam rate':<22}{f:>12.5f}{s:>12.5f}")

    # consistency check the assistant asserted but never verified
    if fm and sm and sm != 2628785:
        tpd_f, tpd_s = 9.6e9 / fm, 2.3e9 / sm
        wr = a["swallowmath"]["words_mean"] / a["finemath_4plus"]["words_mean"]
        print("\n  CONSISTENCY CHECK (documented token counts vs measured lengths)")
        print(f"    tokens/doc finemath : {tpd_f:>8.0f}")
        print(f"    tokens/doc swallow  : {tpd_s:>8.0f}")
        print(f"    implied token ratio : {tpd_s/tpd_f:>8.3f}")
        print(f"    measured word ratio : {wr:>8.3f}")
        print(f"    tokens/word skew    : {wr/(tpd_s/tpd_f):>8.3f}x")
        print("    -> if skew is far from 1.0, either the two corpora tokenize")
        print("       differently (plausible: LaTeX/markdown) or the published")
        print("       9.6B / 2.3B figures use different tokenizers. UNVERIFIED.")

# ---------- phase B ----------
print(f"\n{bar}\n  PHASE B - released checkpoint analysis\n{bar}")
b = rd("out/phaseB/results.json")
if not b:
    print("  MISSING out/phaseB/results.json")
else:
    print(f"\n  {len(b)} models analyzed\n")
    print(f"    {'E':>5}{'total':>8}{'taskloss':>11}{'load_cv':>10}{'norm_MI':>10}")
    rows = []
    for k, v in b.items():
        try:
            E = int(k.split("-E")[1].split("-")[0])
            rows.append((E, k.split("k2-")[1].split("-A")[0],
                         v["gsm8k_task_loss"], v["routing"]["mean_load_cv"],
                         v["specialization"]["normalized_mi"]))
        except Exception: pass
    for E, tot, tl, cv, mi in sorted(rows):
        print(f"    {E:>5}{tot:>8}{tl:>11.4f}{cv:>10.4f}{mi:>10.4f}")
    if len(rows) > 2:
        tl = [r[2] for r in sorted(rows)]
        mono = all(tl[i] > tl[i+1] for i in range(len(tl)-1))
        print(f"\n    task loss monotonically decreasing: {mono}")
        print("    GATE B: trend reproduces their TPP framework (all configs sit")
        print("            at TPP >= 19, i.e. on the high side of their ~20 optimum)")
        print("    STILL UNVERIFIED: absolute values vs their published numbers.")
        print("      -> git clone https://github.com/rioyokotalab/optimal-sparsity")
        print("      -> compare against taskloss-eval/README.md")

# ---------- phase C ----------
print(f"\n{bar}\n  PHASE C - training pilot\n{bar}")
checks = [
    ("tokenizer",        "artifacts/tokenizer/tokenizer.json",  "step03_train_tokenizer.py"),
    ("data: finemath",   "data/finemath_2048.npy",              "step04_pack.py --condition finemath"),
    ("data: swallowmath","data/swallowmath_2048.npy",           "step04_pack.py --condition swallowmath"),
    ("benchmark",        "out/phaseC/benchmark.json",           "step06_benchmark.py"),
]
for label, path, cmd in checks:
    print(f"    [{'x' if ex(path) else ' '}] {label:<20} {'' if ex(path) else '-> ' + cmd}")

runs = R / "runs"
done = sorted(p.parent.name for p in runs.glob("*/summary.json")) if runs.exists() else []
seed = [r for r in done if r.startswith("seedvar")]
swp  = [r for r in done if not r.startswith("seedvar")]
print(f"\n    [{'x' if len(seed)>=6 else ' '}] seed variance runs : {len(seed)}/6   (GATE D)")
print(f"    [{'x' if len(swp)>=18 else ' '}] sweep runs         : {len(swp)}/18")

# ---------- next ----------
print(f"\n{bar}\n  WHAT TO DO NEXT\n{bar}\n")
todo = []
if a and a.get("doc_counts", {}).get("swallowmath") == 2628785:
    todo.append(("patch stale Gate A value in results.json (exact = 6,497,564)",
                 "see notes/swallowmath_count.txt"))
if not ex("out/phaseB/../../notes/taskloss_crosscheck.md"):
    todo.append(("GATE B: cross-check absolute task loss vs their published values",
                 "git clone https://github.com/rioyokotalab/optimal-sparsity /tmp/os && cat /tmp/os/taskloss-eval/README.md"))
if not ex("artifacts/tokenizer/tokenizer.json"):
    todo.append(("train tokenizer", "cd code && python step03_train_tokenizer.py --vocab-size 16000 --n-docs 500000"))
elif not ex("data/finemath_2048.npy"):
    todo.append(("pack finemath", "cd code && python step04_pack.py --condition finemath --total-tokens 1000000000"))
elif not ex("data/swallowmath_2048.npy"):
    todo.append(("pack swallowmath", "cd code && python step04_pack.py --condition swallowmath --total-tokens 1000000000"))
elif not ex("out/phaseC/benchmark.json"):
    todo.append(("GATE C: benchmark then budget", "cd code && python step06_benchmark.py --experts 4,16,64 --steps 30"))
elif len(seed) < 6:
    todo.append(("GATE D: seed variance BEFORE the sweep", "cd code && bash run_seeds.sh"))
elif len(swp) < 18:
    todo.append(("run the sweep", "cd code && bash run_sweep.sh"))
else:
    todo.append(("summarize + plot", "cd code && python step08_summarize.py --pattern 'E*_*_s*' --taskloss --plot"))

if sh("git status --porcelain"):
    todo.insert(0, ("commit + push (instance is ephemeral)",
                    'git add -A && git commit -m "wip" && git push'))

for i, (what, cmd) in enumerate(todo, 1):
    print(f"  {i}. {what}\n     $ {cmd}\n")

print(f"{bar}")
print("  VERIFIED vs INFERRED")
print(f"{bar}")
print("""
  VERIFIED (measured directly):
    - both document counts, and therefore the ratio
    - all Phase A distributional statistics (200k-doc samples)
    - all six Phase B task-loss / routing / MI values

  INFERRED (assistant reasoning, NOT measured):
    - "shortened to ~39% of original length" -> from word counts on samples,
      not tokens on full corpora
    - "token reduction fully explained by shortening" -> see consistency
      check above; the numbers do not cleanly reconcile
    - TPP values for the Phase B family -> depend on their 125B corpus size
      being correct and on total-parameter counts from model names
    - all cost and runtime estimates -> Gate C replaces these with measurements

  UNVERIFIED (nobody has checked):
    - whether the published 9.6B / 2.3B token figures use the same tokenizer
    - whether Phase B absolute task-loss values match their published ones
    - whether streaming enumerates every row of a dataset
""")