# step07 (VAST)  - train one (E, condition, seed) cell
# --- paste contents here ---
"""
Training loop for one (n_experts, data_condition, seed) cell.

Per-expert token counts are logged from step 0 - that is the early-warning
system for router collapse, which is THE standard MoE failure mode. If it
happens, log it and report it. Do not silently tune it away.

Usage:
    python step07_train.py --experts 16 --condition finemath --seed 0 \
        --total-tokens 1_000_000_000 --run-name E16_finemath_s0
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from step05_model import build_model, router_z_loss, param_counts, mfu


class PackedDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path, mmap_mode="r")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.data[i].astype(np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, required=True)
    p.add_argument("--condition", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--tokenizer", type=str, default="artifacts/tokenizer")
    p.add_argument("--out", type=str, default="runs")
    p.add_argument("--run-name", type=str, default=None)
    # schedule
    p.add_argument("--total-tokens", type=int, default=1_000_000_000)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro-bs", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)          # theirs
    p.add_argument("--warmup-frac", type=float, default=0.02) # NOT 2000 steps
    p.add_argument("--wd", type=float, default=0.1)           # theirs
    p.add_argument("--z-loss", type=float, default=1e-3)      # theirs
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ckpt-fracs", type=str, default="0.25,0.5,0.75,1.0")
    p.add_argument("--max-steps", type=int, default=None, help="override, for benchmarking")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = a.run_name or f"E{a.experts}_{a.condition}_s{a.seed}"
    outdir = Path(a.out) / run; outdir.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ds = PackedDataset(f"{a.data_dir}/{a.condition}_{a.seq_len}.npy")
    dl = DataLoader(ds, batch_size=a.micro_bs, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True,
                    generator=torch.Generator().manual_seed(a.seed))

    model, cfg = build_model(a.experts, len(tok), seq_len=a.seq_len)
    model = model.to(dev, dtype=torch.bfloat16)
    model.gradient_checkpointing_disable()

    counts = param_counts(model, cfg)
    tokens_per_step = a.micro_bs * a.grad_accum * a.seq_len
    total_steps = a.max_steps or (a.total_tokens // tokens_per_step)
    warmup = max(1, int(total_steps * a.warmup_frac))
    print(json.dumps({**counts, "total_steps": total_steps, "warmup": warmup,
                      "tokens_per_step": tokens_per_step}, indent=2))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95),
                            weight_decay=a.wd)
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    use_wandb = not a.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project="moe-sparsity-pilot", name=run,
                   config={**vars(a), **counts, "total_steps": total_steps})

    ckpt_steps = {int(total_steps * float(f)) for f in a.ckpt_fracs.split(",")}
    n_exp = cfg.num_local_experts
    expert_counts = torch.zeros(n_exp, dtype=torch.float64)

    step, seen, t0 = 0, 0, time.time()
    it = iter(dl)
    model.train()
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        acc_lm, acc_aux, acc_z = 0.0, 0.0, 0.0
        for _ in range(a.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            ids = batch.to(dev, non_blocking=True)
            out = model(input_ids=ids, labels=ids, output_router_logits=True)
            zl = router_z_loss(out.router_logits, a.z_loss).to(out.loss.device)
            loss = out.loss + zl                       # out.loss already includes aux
            (loss / a.grad_accum).backward()
            acc_lm += out.loss.item() / a.grad_accum
            acc_z += float(zl) / a.grad_accum
            aux = getattr(out, "aux_loss", None)
            acc_aux += (float(aux) if aux is not None else 0.0) / a.grad_accum
            seen += ids.numel()
            # router collapse watch
            with torch.no_grad():
                for rl in out.router_logits:
                    tk = F.softmax(rl.float(), -1).topk(cfg.num_experts_per_tok, -1).indices
                    expert_counts += torch.bincount(tk.flatten().cpu(), minlength=n_exp).double()

        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step(); sched.step(); step += 1

        if step % a.log_every == 0:
            el = time.time() - t0
            tps = seen / el
            frac = (expert_counts / expert_counts.sum().clamp(min=1))
            log = {
                "step": step, "tokens": seen, "lm_loss": acc_lm,
                "aux_loss": acc_aux, "z_loss": acc_z,
                "grad_norm": float(gn), "lr": sched.get_last_lr()[0],
                "tokens_per_sec": tps,
                "mfu": mfu(counts["active_nonembed"], tps),
                "expert_load_cv": float(frac.std() / frac.mean()),
                "min_expert_frac": float(frac.min()),
                "dead_experts": int((frac < 0.01 / n_exp).sum()),
                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in log.items()})
            if use_wandb:
                import wandb; wandb.log(log)
            if log["dead_experts"] > 0:
                print(f"  !! ROUTER COLLAPSE WARNING: {log['dead_experts']} dead experts")
            expert_counts.zero_()

        if step in ckpt_steps:
            d = outdir / f"step{step}"; d.mkdir(exist_ok=True)
            model.save_pretrained(d); tok.save_pretrained(d)
            print(f"  saved {d}")

    (outdir / "summary.json").write_text(json.dumps({
        "run": run, "experts": a.experts, "condition": a.condition, "seed": a.seed,
        "final_lm_loss": acc_lm, "total_steps": total_steps, "tokens": seen,
        "wall_clock_sec": time.time() - t0, **counts,
    }, indent=2))
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()