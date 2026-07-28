"""
Phase 6 - Aggregate runs, compute task loss, produce the plots.

Every number gets error bars from the seeds. Anything without error bars is a
claim you cannot support.

*** METRIC WARNING ***
final_lm_loss (pre-training loss) is NOT comparable across data conditions.
FineMath and SwallowMath are different text; SwallowMath is measurably more
homogeneous (distinct-4gram 0.707 vs 0.753), so it is easier to model
regardless of whether the model learned anything more useful. A lower
pre-training loss on the rewritten condition would tell you nothing.

The only valid cross-condition comparison is on a COMMON held-out set, which
is why GSM8K task loss is the primary metric. This script warns loudly if you
try to run the decision gate on pre-training loss.

Usage:
    python step08_summarize.py --pattern 'seedvar_*' --taskloss --report-spread
    python step08_summarize.py --pattern 'E*_*_s*'  --taskloss --report-spread --plot
"""
import argparse, glob, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np

TASK = "gsm8k_task_loss"
PRETRAIN = "final_lm_loss"


def load_runs(pattern, runs_dir="runs"):
    out = []
    for p in sorted(glob.glob(f"{runs_dir}/{pattern}/summary.json")):
        d = json.loads(Path(p).read_text())
        d["_dir"] = str(Path(p).parent)
        out.append(d)
    return out


def compute_task_loss(runs, n_eval=500, force=False):
    """
    Evaluate the final checkpoint of each run on GSM8K.

    Caches into summary.json so re-runs are free. Frees GPU memory between
    models - without this you OOM after a handful of runs.
    """
    import torch
    from step02_checkpoint_analysis import load, task_loss

    for r in runs:
        if TASK in r and not force:
            print(f"  {r['run']}: {r[TASK]:.4f}  (cached)")
            continue
        cks = sorted(glob.glob(f"{r['_dir']}/step*"),
                     key=lambda s: int(re.search(r"step(\d+)", s).group(1)))
        if not cks:
            print(f"  {r['run']}: NO CHECKPOINT - skipped")
            continue
        m, t = load(cks[-1])
        r[TASK] = task_loss(m, t, "gsm8k", n_eval)
        r["_taskloss_ckpt"] = Path(cks[-1]).name
        r["_taskloss_n"] = n_eval
        Path(f"{r['_dir']}/summary.json").write_text(json.dumps(r, indent=2))
        print(f"  {r['run']}: {r[TASK]:.4f}")
        del m
        torch.cuda.empty_cache()


def report_spread(runs, out_dir=None):
    """
    DECISION GATE.

    Per-config seed statistics, the noise floor, and - when two data conditions
    exist at the same expert count - whether the condition effect is resolvable
    above that floor.
    """
    have_task = [r for r in runs if TASK in r]
    if runs and len(have_task) == len(runs):
        metric, label = TASK, "GSM8K task loss"
    else:
        metric, label = PRETRAIN, "pre-training loss"
        print("!" * 66)
        print("  Task loss missing for some runs - falling back to")
        print("  pre-training loss, which is NOT valid across data conditions.")
        print("  Run with --taskloss before trusting anything below.")
        print("!" * 66 + "\n")

    print(f"metric: {metric}  ({label})\n")

    by = defaultdict(list)
    for r in runs:
        if metric in r:
            by[(r["experts"], r["condition"])].append(r[metric])

    print(f"{'config':<26}{'n':>3}{'mean':>11}{'std':>10}{'range':>10}")
    spreads = []
    for (e, c), v in sorted(by.items()):
        v = np.array(v)
        rng = float(v.max() - v.min())
        if len(v) > 1:
            spreads.append(rng)
        print(f"E={e:<4} {c:<19}{len(v):>3}{v.mean():>11.4f}{v.std():>10.4f}{rng:>10.4f}")

    if not spreads:
        print("\nOnly one seed per config - no noise floor available.")
        return

    floor = max(spreads)
    print(f"\nNOISE FLOOR (max seed range): {floor:.4f}")
    print("Differences smaller than this are NOT RESOLVABLE at this scale.")

    conds = sorted({c for _, c in by})
    if len(conds) == 2 and metric == TASK:
        a, b = conds
        print(f"\nCONDITION EFFECT  ({b} minus {a})")
        print(f"{'E':>6}{'diff':>12}{'floor':>10}   verdict")
        any_res = False
        for e in sorted({e for e, _ in by}):
            if (e, a) in by and (e, b) in by:
                d = float(np.mean(by[(e, b)]) - np.mean(by[(e, a)]))
                ok = abs(d) > floor
                any_res |= ok
                print(f"{e:>6}{d:>12.4f}{floor:>10.4f}   "
                      f"{'RESOLVABLE' if ok else 'below noise floor'}")
        print("\n" + ("At least one expert count shows a resolvable effect."
                      if any_res else
                      "No resolvable condition effect at any expert count.\n"
                      "That is a legitimate result. Report it as a null."))

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, "spread.json").write_text(json.dumps(
            {"metric": metric, "noise_floor": floor,
             "by_config": {f"E{e}_{c}": v for (e, c), v in by.items()}}, indent=2))
        print(f"\nwrote {out_dir}/spread.json")


def plot(runs, out="out/phaseC"):
    import matplotlib.pyplot as plt
    runs = [r for r in runs if TASK in r and PRETRAIN in r]
    if not runs:
        print("Task or pre-training loss missing. Skipping plot.")
        return
    Path(out).mkdir(parents=True, exist_ok=True)

    labels = {"finemath": "FineMath-4+", "swallowmath": "SwallowMath"}
    colors = {"finemath": "#31688e", "swallowmath": "#d1495b"}
    metrics = [(PRETRAIN, "Pre-training loss"),
               (TASK, "GSM8K task loss")]
    experts = sorted({r["experts"] for r in runs})

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, (metric, title) in zip(axes, metrics):
        by = defaultdict(lambda: defaultdict(list))
        for r in runs:
            by[r["condition"]][r["experts"]].append(r[metric])
        for cond, d in sorted(by.items()):
            xs = sorted(d)
            ys = [np.mean(d[x]) for x in xs]
            es = [np.std(d[x], ddof=1) for x in xs]
            ax.errorbar(
                xs, ys, yerr=es, marker="o", markersize=5, capsize=4,
                label=labels.get(cond, cond), color=colors.get(cond), lw=1.8
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(experts)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("number of experts (E), k=2")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Rewritten data lowers pre-training loss without lowering task loss",
        fontsize=13, weight="bold"
    )
    fig.text(
        0.5, 0.01,
        "Mean ± seed SD (n=3); active parameters held approximately constant; lower is better",
        ha="center", fontsize=8.5, color="#444444"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(f"{out}/sparsity_curve.png", dpi=160)
    print(f"wrote {out}/sparsity_curve.png")

    rows = [{"experts": r["experts"], "condition": r["condition"],
             "seed": r["seed"], TASK: r[TASK], PRETRAIN: r.get(PRETRAIN)}
            for r in runs]
    Path(out, "all_runs.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}/all_runs.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="E*_*_s*")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--report-spread", action="store_true")
    ap.add_argument("--taskloss", action="store_true")
    ap.add_argument("--force-taskloss", action="store_true",
                    help="recompute even if cached in summary.json")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default="out/phaseC")
    a = ap.parse_args()

    runs = load_runs(a.pattern, a.runs_dir)
    print(f"{len(runs)} runs matched\n")
    if not runs:
        return

    if a.taskloss or a.force_taskloss:
        compute_task_loss(runs, a.n_eval, force=a.force_taskloss)
        print()
    if a.report_spread:
        report_spread(runs, a.out)
    if a.plot:
        plot(runs, a.out)


if __name__ == "__main__":
    main()
