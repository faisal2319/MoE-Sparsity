# step05 (VAST)  - Mixtral miniature, z-loss, param counts, MFU  [LIBRARY]
# --- paste contents here ---
"""
Architecture-faithful miniature of Nakamura et al. (ICLR 2026 Oral).

NOT a replication. ~1/1000 the compute. Matches:
  - model class (Mixtral, so the eval path is shared with their checkpoints)
  - FFN width = 2 * d
  - top-k = 2
  - aux load-balancing loss coefficient 1e-2
  - router z-loss 1e-3
  - AdamW / LR 4e-4 / cosine / wd 0.1

Departs deliberately:
  - d=256, 8 layers instead of d=512, 16 layers.
    Aspect ratio is preserved at 32. Do NOT use 16 layers at d=256 - that is
    an aspect ratio of 16, which is narrow-deep, less stable, and gets poor MFU
    because the matmuls are too small to saturate the GPU.
"""
import torch
import torch.nn.functional as F
from transformers import MixtralConfig, MixtralForCausalLM


def build_config(n_experts, vocab_size, hidden=256, layers=8, heads=8,
                 top_k=2, seq_len=2048):
    return MixtralConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        intermediate_size=hidden * 2,        # FFN width = 2d, matching theirs
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,           # no GQA at this scale
        num_local_experts=n_experts,
        num_experts_per_tok=top_k,
        max_position_embeddings=seq_len,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        sliding_window=None,                 # full attention
        attention_dropout=0.0,
        router_aux_loss_coef=1e-2,           # theirs
        output_router_logits=True,           # required for aux loss + z-loss
        tie_word_embeddings=True,            # keeps embedding cost down at 16k vocab
    )


def build_model(n_experts, vocab_size, **kw):
    cfg = build_config(n_experts, vocab_size, **kw)
    model = MixtralForCausalLM(cfg)
    return model, cfg


def router_z_loss(router_logits, coef=1e-3):
    """
    Router z-loss from ST-MoE. Penalises large router logits, which stabilises
    training at high expert counts.

    HuggingFace's Mixtral implements the aux load-balancing loss but NOT the
    z-loss. VERIFY THIS on your transformers version - if it is present, remove
    this to avoid double-counting.

    router_logits: tuple of [n_tokens, n_experts], one per MoE layer.
    """
    if router_logits is None:
        return torch.tensor(0.0)
    total = 0.0
    for rl in router_logits:
        total = total + torch.logsumexp(rl.float(), dim=-1).pow(2).mean()
    return coef * total / len(router_logits)


def param_counts(model, cfg):
    """
    Active non-embedding parameters - the correct denominator for MFU.

    Using TOTAL parameters would count dormant experts and inflate MFU.
    Getting this right is one of the details that signals competence to an
    HPC audience.
    """
    d, L, E, k = (cfg.hidden_size, cfg.num_hidden_layers,
                  cfg.num_local_experts, cfg.num_experts_per_tok)
    inter = cfg.intermediate_size

    attn_per_layer = 4 * d * d                    # q, k, v, o
    expert_params = 3 * d * inter                 # w1, w2, w3 (SwiGLU)
    router_per_layer = d * E

    active_nonembed = L * (attn_per_layer + k * expert_params + router_per_layer)
    total_nonembed = L * (attn_per_layer + E * expert_params + router_per_layer)
    embed = cfg.vocab_size * d

    return {
        "active_nonembed": active_nonembed,
        "total_nonembed": total_nonembed,
        "embedding": embed,
        "total_all": sum(p.numel() for p in model.parameters()),
        "sparsity_ratio": total_nonembed / active_nonembed,
    }


def mfu(active_nonembed_params, tokens_per_sec, peak_flops=165e12):
    """
    MFU = 6 * N_active * tokens/sec / peak_FLOPS

    6N = forward (2N) + backward (4N).
    peak_flops default: RTX 4090 bf16 dense ~165 TFLOPS. VERIFY against your
    card and driver - the number varies with clocks and whether the spec you
    read includes 2:4 sparsity (it usually does; halve it if so).
    """
    return (6 * active_nonembed_params * tokens_per_sec) / peak_flops