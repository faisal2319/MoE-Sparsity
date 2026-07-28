"""
MI permutation control for Phase B specialization estimates.

WHY THIS EXISTS
---------------
domain_expert_mi() in step02 estimates mutual information between domain label
and expert assignment from a joint table of shape [n_domains x n_experts] per
layer. MI estimators are POSITIVELY BIASED when cells outnumber samples:

    E=8   ->   3 x 8   =  24 cells, filled from ~100 docs/domain
    E=256 ->   3 x 256 = 768 cells, filled from the same ~100 docs/domain

The bias therefore GROWS with expert count - the same direction as the observed
trend (0.066 at E=8 -> 0.203 at E=256). Without a control, an unknown fraction
of that rise is estimator artifact rather than specialization.

METHOD
------
Run inference once, caching per-document expert counts. Then rebuild the joint
table many times with SHUFFLED domain labels. Under shuffling there is no real
domain->expert relationship, so any MI measured is pure bias. Report:

    corrected_MI = observed_MI - mean(shuffled_MI)

Permutations are cheap because they only re-sum cached counts; the model is not
re-run.

Usage:
    python validate_mi_control.py --model <HF_ID> --n-probe 100 --n-perm 20
    python validate_mi_control.py --all --out ../out/phaseB/mi_control.json
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "llm-jp/optimal-sparsity-math-d512-E8-k2-320M-A170M",
    "llm-jp/optimal-sparsity-math-d512-E256-k2-6.6B-A170M",
]


def probe_sets(n=100):
    """Same three domains as step02: math / general web / code."""
    m = load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                     split="train", streaming=True)
    w = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                     split="train", streaming=True)
    c = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    it_m, it_w, it_c = iter(m), iter(w), iter(c)
    return {
        "math": [next(it_m)["text"] for _ in range(n)],
        "web":  [next(it_w)["text"] for _ in range(n)],
        "code": [next(it_c)["content"] for _ in range(n)],
    }


@torch.no_grad()
def per_doc_counts(model, tok, domain_texts, max_len=512, device="cuda"):
    """
    Cache expert token counts per document.

    Returns counts [n_docs, n_layers, n_experts] and integer labels [n_docs].
    Running inference once and permuting the cached counts is what makes the
    control cheap.
    """
    n_layers = model.config.num_hidden_layers
    n_exp = model.config.num_local_experts
    k = model.config.num_experts_per_tok
    domains = list(domain_texts)

    counts, labels = [], []
    for di, d in enumerate(domains):
        for t in tqdm(domain_texts[d], desc=f"  {d}"):
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=max_len).input_ids.to(device)
            out = model(ids, output_router_logits=True)
            c = torch.zeros(n_layers, n_exp, dtype=torch.float64)
            for li, rl in enumerate(out.router_logits):
                topk = F.softmax(rl.float(), -1).topk(k, dim=-1).indices
                c[li] = torch.bincount(topk.flatten().cpu(),
                                       minlength=n_exp).double()
            counts.append(c)
            labels.append(di)
    return torch.stack(counts), np.array(labels), domains


def mi_from_counts(counts, labels, n_domains):
    """Mean MI (nats) across layers, from cached per-document counts."""
    n_docs, n_layers, n_exp = counts.shape
    mis = []
    for li in range(n_layers):
        joint = torch.zeros(n_domains, n_exp, dtype=torch.float64)
        for di in range(n_domains):
            sel = np.where(labels == di)[0]
            if len(sel):
                joint[di] = counts[sel, li].sum(0)
        tot = joint.sum().clamp(min=1)
        p = joint / tot
        pd = p.sum(1, keepdim=True)
        pe = p.sum(0, keepdim=True)
        nz = p > 0
        mis.append((p[nz] * (p[nz] / (pd @ pe)[nz]).log()).sum().item())
    return float(np.mean(mis))


def analyze(model_id, n_probe, n_perm, seed=0):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda",
        output_router_logits=True).eval()

    n_exp = model.config.num_local_experts
    n_layers = model.config.num_hidden_layers
    print(f"\n{model_id}\n  E={n_exp}  layers={n_layers}  "
          f"cells/layer={3*n_exp}  docs/domain={n_probe}")

    probes = probe_sets(n_probe)
    counts, labels, domains = per_doc_counts(model, tok, probes)
    n_dom = len(domains)

    observed = mi_from_counts(counts, labels, n_dom)

    rng = np.random.default_rng(seed)
    shuffled = []
    for _ in tqdm(range(n_perm), desc="  permutations"):
        shuffled.append(mi_from_counts(counts, rng.permutation(labels), n_dom))
    shuffled = np.array(shuffled)

    max_mi = math.log(n_dom)
    corrected = observed - shuffled.mean()
    # z-score of the observed value against the shuffled null
    z = (observed - shuffled.mean()) / max(shuffled.std(ddof=1), 1e-12)

    r = {
        "model": model_id, "n_experts": n_exp, "n_layers": n_layers,
        "n_probe_per_domain": n_probe, "n_permutations": n_perm,
        "cells_per_layer": n_dom * n_exp,
        "observed_mi_nats": observed,
        "shuffled_mi_mean": float(shuffled.mean()),
        "shuffled_mi_std": float(shuffled.std(ddof=1)),
        "corrected_mi_nats": corrected,
        "z_vs_null": float(z),
        "observed_normalized": observed / max_mi,
        "corrected_normalized": corrected / max_mi,
        "bias_fraction_of_observed": float(shuffled.mean() / observed) if observed else None,
    }
    print(f"  observed MI   : {observed:.5f} nats  (normalized {observed/max_mi:.4f})")
    print(f"  shuffled MI   : {shuffled.mean():.5f} +/- {shuffled.std(ddof=1):.5f}")
    print(f"  corrected MI  : {corrected:.5f} nats  (normalized {corrected/max_mi:.4f})")
    print(f"  bias is {100*shuffled.mean()/observed:.1f}% of observed" if observed else "")
    print(f"  z vs null     : {z:.1f}")

    del model
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--all", action="store_true",
                    help=f"run the endpoints: {DEFAULT_MODELS}")
    ap.add_argument("--n-probe", type=int, default=100,
                    help="docs per domain (100 matches the original Phase B run)")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--out", type=str, default="../out/phaseB/mi_control.json")
    a = ap.parse_args()

    models = DEFAULT_MODELS if a.all else [a.model]
    results = []
    for m in models:
        try:
            results.append(analyze(m, a.n_probe, a.n_perm))
        except Exception as e:
            print(f"  FAILED {m}: {e}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")

    if len(results) >= 2:
        lo, hi = results[0], results[-1]
        print(f"\n{'':22}{'E='+str(lo['n_experts']):>12}{'E='+str(hi['n_experts']):>12}{'ratio':>9}")
        for key, label in [("observed_normalized", "observed (norm)"),
                           ("corrected_normalized", "corrected (norm)")]:
            ratio = hi[key] / lo[key] if lo[key] else float("nan")
            print(f"{label:22}{lo[key]:>12.4f}{hi[key]:>12.4f}{ratio:>9.2f}x")
        print("\nIf the corrected ratio is close to the observed ratio, the")
        print("specialization trend is real. If it collapses toward 1.0, the")
        print("original trend was largely estimator bias and must be reported")
        print("as such.")


if __name__ == "__main__":
    main()