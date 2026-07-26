# step08 (VAST+LOCAL) - aggregate, noise floor, plots
# --- paste contents here ---
"""
Phase 6 - Aggregate runs, compute task loss, produce the plots.

Every number gets error bars from the three seeds. Anything without error bars
is a claim you cannot support.

Usage:
    python step08_summarize.py --pattern 'seedvar_*' --report-spread
    python step08_summarize.py --pattern 'E*_*_s*' --taskloss --plot
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_runs(pattern, runs_dir="runs"):
    out = []
    for p in sorted(glob.glob(f"{runs_dir}/{pattern}/summary.json")):
        d = json.loads(Path(p).read_text())
        d["_dir"] = str(Path(p).parent)
        out.append(d)
    return out


def report_spread(runs):
    """PHASE 4 DECISION GATE."""
    by = defaultdict(list)
    for r in runs:
        by[(r["experts"], r["condition"])].append(r["final_lm_loss"])
    print(f"{'config':<24}{'n':>3}{'mean':>10}{'std':>10}{'range':>10}")
    spreads = []
    for (e, c), v in sorted(by.items()):
        v = np.array(v)
        rng = v.max() - v.min()
        spreads.append(rng)
        print(f"E={e} {c:<16}{len(v):>3}{v.mean():>10.4f}{v.std():>10.4f}{rng:>10.4f}")
    if spreads:
        print(f"\nNOISE FLOOR (max seed range): {max(spreads):.4f}")
        print("Any between-config difference smaller than this is NOT RESOLVABLE.")
        print("If your expected effect is below it, report that and stop.")


def plot(runs, out="out/phase5"):
    import matplotlib.pyplot as plt
    Path(out).mkdir(parents=True, exist_ok=True)
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        key = "task_loss" if "gsm8k_task_loss" in r else "final_lm_loss"
        by[r["condition"]][r["experts"]].append(r.get("gsm8k_task_loss", r["final_lm_loss"]))

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, d in by.items():
        xs = sorted(d)
        ys = [np.mean(d[x]) for x in xs]
        es = [np.std(d[x]) for x in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, label=cond)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of experts (E), k=2")
    ax.set_ylabel("GSM8K task loss")
    ax.set_title("Sparsity vs reasoning task loss, by data condition\n"
                 "(pilot, ~1/1000 the compute of Nakamura et al.)", fontsize=9)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/sparsity_curve.png", dpi=160)
    print(f"wrote {out}/sparsity_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="E*_*_s*")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--report-spread", action="store_true")
    ap.add_argument("--taskloss", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    runs = load_runs(a.pattern, a.runs_dir)
    print(f"{len(runs)} runs matched\n")
    if a.taskloss:
        from step02_checkpoint_analysis import load, task_loss
        for r in runs:
            ck = sorted(glob.glob(f"{r['_dir']}/step*"),
                        key=lambda s: int(re.search(r"step(\d+)", s).group(1)))[-1]
            m, t = load(ck)
            r["gsm8k_task_loss"] = task_loss(m, t, "gsm8k", 500)
            Path(f"{r['_dir']}/summary.json").write_text(json.dumps(r, indent=2))
            print(f"  {r['run']}: {r['gsm8k_task_loss']:.4f}")
    if a.report_spread:
        report_spread(runs)
    if a.plot:
        plot(runs)


if __name__ == "__main__":
    main()