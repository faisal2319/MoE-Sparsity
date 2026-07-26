# step06 (VAST)  - throughput benchmark + GATE C budget
# --- paste contents here ---
"""
Phase 3 - Benchmark before budgeting. Do NOT budget from 6ND alone.

Measures real throughput and memory at the extremes of the sweep, then
extrapolates the full experiment cost. The token budget decision comes from
THIS, not from the plan.

Usage:
    python step06_benchmark.py --experts 4,16,64 --steps 30
"""
import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from step05_model import build_model, param_counts, mfu, router_z_loss


def bench(n_experts, vocab, micro_bs, grad_accum, seq_len, steps, peak_flops):
    dev = "cuda"
    model, cfg = build_model(n_experts, vocab, seq_len=seq_len)
    model = model.to(dev, dtype=torch.bfloat16).train()
    counts = param_counts(model, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)

    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, vocab, (micro_bs, seq_len), device=dev)

    for _ in range(3):                                    # warmup
        out = model(input_ids=ids, labels=ids, output_router_logits=True)
        (out.loss + router_z_loss(out.router_logits)).backward()
        opt.step(); opt.zero_grad(set_to_none=True)

    torch.cuda.synchronize(); t0 = time.time(); seen = 0
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            out = model(input_ids=ids, labels=ids, output_router_logits=True)
            loss = out.loss + router_z_loss(out.router_logits).to(out.loss.device)
            (loss / grad_accum).backward()
            seen += ids.numel()
        opt.step()
    torch.cuda.synchronize()
    el = time.time() - t0
    tps = seen / el

    r = {
        "n_experts": n_experts,
        "tokens_per_sec": tps,
        "sec_per_step": el / steps,
        "mfu": mfu(counts["active_nonembed"], tps, peak_flops),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        **counts,
    }
    del model, opt
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=str, default="4,16,64")
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer")
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--peak-flops", type=float, default=165e12)
    ap.add_argument("--total-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--n-runs", type=int, default=24, help="6 seed-variance + 18 sweep")
    ap.add_argument("--price-per-hour", type=float, default=0.35)
    ap.add_argument("--out", type=str, default="out/phase3/benchmark.json")
    a = ap.parse_args()

    vocab = len(AutoTokenizer.from_pretrained(a.tokenizer))
    results = []
    for e in [int(x) for x in a.experts.split(",")]:
        print(f"\n=== E={e} ===")
        try:
            r = bench(e, vocab, a.micro_bs, a.grad_accum, a.seq_len, a.steps, a.peak_flops)
            print(json.dumps(r, indent=2))
            results.append(r)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at E={e} - reduce --micro-bs or drop this config")
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("BUDGET EXTRAPOLATION")
    print("=" * 60)
    tot_h = 0.0
    for r in results:
        h = a.total_tokens / r["tokens_per_sec"] / 3600
        tot_h += h
        print(f"  E={r['n_experts']:3d}  {h:6.2f} h/run   "
              f"MFU {r['mfu']*100:5.1f}%   VRAM {r['peak_vram_gb']:.1f} GB")
    avg = tot_h / max(len(results), 1)
    print(f"\n  mean {avg:.2f} h/run  x {a.n_runs} runs = {avg*a.n_runs:.1f} GPU-hours")
    print(f"  ~${avg*a.n_runs*a.price_per_hour:.2f} at ${a.price_per_hour}/hr (+50% for failures)")
    print(f"\n  If this is too expensive, halve --total-tokens to 500M and re-check.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"per_config": results, "mean_hours_per_run": avg,
         "n_runs": a.n_runs, "total_gpu_hours": avg * a.n_runs}, indent=2))


if __name__ == "__main__":
    main()