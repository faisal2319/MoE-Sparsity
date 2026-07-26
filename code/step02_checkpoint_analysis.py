# step02 (VAST)  - released checkpoints, task loss + routing + GATE B
# --- paste contents here ---
"""
Phase 1 - Released checkpoint analysis.

Two jobs:
  1.2  Validate your eval harness against their PUBLISHED task-loss numbers.
       If these don't match, everything downstream is worthless. Do not skip.
  1.3-1.5  Extract routing behaviour across their sparsity range and test
       whether it explains the reasoning degradation. This answers the LOI's
       secondary question at a scale you could never train yourself.

Inference only - even E=64 (1.7B params) fits a 4090 at bf16 (~3.4GB).
Only TRAINING those doesn't fit.

Usage:
    python step02_checkpoint_analysis.py --list
    python step02_checkpoint_analysis.py --model llm-jp/optimal-sparsity-math-d512-E8-k2-320M-A170M
    python step02_checkpoint_analysis.py --all --out out/phase1
"""
import argparse, json, math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Verify these against https://huggingface.co/collections/llm-jp/optimal-sparsity-math
# before relying on them - naming may differ for some family members.
CANDIDATES = [
    "llm-jp/optimal-sparsity-math-d512-E8-k2-320M-A170M",
    "llm-jp/optimal-sparsity-math-d512-E16-k2",
    "llm-jp/optimal-sparsity-math-d512-E32-k2",
    "llm-jp/optimal-sparsity-math-d512-E64-k2",
]


def load(model_id, device="cuda"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
        output_router_logits=True,
    ).eval()
    return model, tok


@torch.no_grad()
def task_loss(model, tok, dataset="gsm8k", n=500, max_len=1024, device="cuda"):
    """
    Task loss = mean NLL of the answer tokens given the question.

    This is the metric that matters at small scale: it moves long before
    exact-match accuracy does. Their paper separates pre-training loss ->
    task loss -> accuracy for exactly this reason.

    Cross-check the numbers this produces against their taskloss-eval/README.md
    and their published values.
    """
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
        pairs = [(e["question"], e["answer"]) for e in ds]
    else:
        raise ValueError(dataset)

    total_nll, total_tok = 0.0, 0
    for q, ans in tqdm(pairs, desc=f"task_loss[{dataset}]"):
        prompt_ids = tok(q + "\n", return_tensors="pt").input_ids.to(device)
        full_ids = tok(q + "\n" + ans, return_tensors="pt").input_ids.to(device)[:, :max_len]
        n_prompt = prompt_ids.shape[1]
        if full_ids.shape[1] <= n_prompt:
            continue
        logits = model(full_ids).logits[:, :-1]
        targets = full_ids[:, 1:]
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, n_prompt - 1:] = True          # score answer tokens only
        nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1), reduction="none",
        ).reshape(targets.shape)
        total_nll += (nll * mask).sum().item()
        total_tok += mask.sum().item()
    return total_nll / max(total_tok, 1)


@torch.no_grad()
def routing_stats(model, tok, texts, device="cuda", max_len=512):
    """
    1.3 - utilisation, load balance, routing entropy, per layer.

    NOTE: load balance is NOT specialization. A perfectly balanced router may
    have no domain preference at all. See domain_expert_mi() for the real thing.
    """
    n_layers = model.config.num_hidden_layers
    n_exp = model.config.num_local_experts
    k = model.config.num_experts_per_tok

    counts = torch.zeros(n_layers, n_exp, dtype=torch.float64)
    ent_sum = torch.zeros(n_layers, dtype=torch.float64)
    tok_seen = 0

    for t in tqdm(texts, desc="routing"):
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        out = model(ids, output_router_logits=True)
        for li, rl in enumerate(out.router_logits):     # each [n_tokens, n_experts]
            probs = F.softmax(rl.float(), dim=-1)
            topk = probs.topk(k, dim=-1).indices
            counts[li] += torch.bincount(topk.flatten().cpu(), minlength=n_exp).double()
            ent_sum[li] += -(probs * (probs + 1e-9).log()).sum(-1).mean().item() * rl.shape[0]
        tok_seen += ids.shape[1]

    frac = counts / counts.sum(dim=1, keepdim=True).clamp(min=1)
    cv = (frac.std(dim=1) / frac.mean(dim=1)).tolist()          # load imbalance
    max_frac = frac.max(dim=1).values.tolist()
    dead = (frac < 0.01 / n_exp).sum(dim=1).tolist()            # near-zero experts
    return {
        "n_experts": n_exp, "top_k": k, "tokens": tok_seen,
        "expert_fraction_per_layer": frac.tolist(),
        "load_cv_per_layer": cv, "mean_load_cv": sum(cv) / len(cv),
        "max_expert_fraction_per_layer": max_frac,
        "dead_experts_per_layer": dead,
        "routing_entropy_per_layer": (ent_sum / max(tok_seen, 1)).tolist(),
        "max_entropy": math.log(n_exp),
    }


@torch.no_grad()
def domain_expert_mi(model, tok, domain_texts, device="cuda", max_len=512):
    """
    1.4 - SPECIALIZATION, measured properly.

    Mutual information between domain label and expert assignment.
    High MI = experts systematically prefer different domains = specialization.
    Low MI with balanced load = the router is spreading tokens without
    differentiating, which is a different thing entirely.

    domain_texts: {"math": [...], "web": [...], "code": [...]}
    """
    n_layers = model.config.num_hidden_layers
    n_exp = model.config.num_local_experts
    k = model.config.num_experts_per_tok
    domains = list(domain_texts)

    joint = torch.zeros(n_layers, len(domains), n_exp, dtype=torch.float64)
    for di, d in enumerate(domains):
        for t in tqdm(domain_texts[d], desc=f"MI[{d}]"):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            out = model(ids, output_router_logits=True)
            for li, rl in enumerate(out.router_logits):
                topk = F.softmax(rl.float(), -1).topk(k, dim=-1).indices
                joint[li, di] += torch.bincount(topk.flatten().cpu(), minlength=n_exp).double()

    mis = []
    for li in range(n_layers):
        p = joint[li] / joint[li].sum().clamp(min=1)
        pd = p.sum(1, keepdim=True)
        pe = p.sum(0, keepdim=True)
        nz = p > 0
        mi = (p[nz] * (p[nz] / (pd @ pe)[nz]).log()).sum().item()
        mis.append(mi)
    return {
        "domains": domains,
        "mi_per_layer_nats": mis,
        "mean_mi": sum(mis) / len(mis),
        "max_possible_mi": math.log(len(domains)),
        "normalized_mi": (sum(mis) / len(mis)) / math.log(len(domains)),
    }


def probe_sets(n=300):
    math_ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", streaming=True)
    web_ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    code_ds = load_dataset("bigcode/the-stack-smol", "data/python", split="train", streaming=True)
    take = lambda ds, key: [next(it)[key] for it in [iter(ds)] for _ in range(n)]
    it_m, it_w, it_c = iter(math_ds), iter(web_ds), iter(code_ds)
    return {
        "math": [next(it_m)["text"] for _ in range(n)],
        "web": [next(it_w)["text"] for _ in range(n)],
        "code": [next(it_c)["content"] for _ in range(n)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=str, default="out/phase1")
    ap.add_argument("--n-taskloss", type=int, default=500)
    ap.add_argument("--n-probe", type=int, default=300)
    a = ap.parse_args()

    if a.list:
        from huggingface_hub import HfApi
        for m in HfApi().list_models(search="optimal-sparsity"):
            print(m.id)
        return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = CANDIDATES if a.all else [a.model]
    probes = probe_sets(a.n_probe)

    results = {}
    for mid in models:
        print(f"\n{'='*70}\n{mid}\n{'='*70}")
        try:
            model, tok = load(mid)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue
        r = {
            "config": {
                "hidden_size": model.config.hidden_size,
                "num_hidden_layers": model.config.num_hidden_layers,
                "num_local_experts": model.config.num_local_experts,
                "num_experts_per_tok": model.config.num_experts_per_tok,
                "total_params": sum(p.numel() for p in model.parameters()),
            },
            "gsm8k_task_loss": task_loss(model, tok, "gsm8k", a.n_taskloss),
            "routing": routing_stats(model, tok, probes["math"][:100]),
            "specialization": domain_expert_mi(model, tok, {d: v[:100] for d, v in probes.items()}),
        }
        print(json.dumps({k: v for k, v in r.items() if k != "routing"}, indent=2)[:800])
        results[mid] = r
        (out / "results.json").write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()

    print("\n>>> NOW COMPARE gsm8k_task_loss AGAINST THEIR PUBLISHED VALUES.")
    print(">>> If they don't match, fix the harness before doing anything else.")


if __name__ == "__main__":
    main()