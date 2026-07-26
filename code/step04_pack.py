# step04 (VAST)  - tokenize + pack both data conditions
# --- paste contents here ---
"""
Tokenize and pack a data condition into fixed-length sequences.

Mixture: ~25% math / 75% general web (FineWeb-Edu).

Why 25% and not their 4.79%: at 1B total tokens a 5% math slice is only 50M
math tokens - too dilute for the variable to register at this scale. This is a
deliberate amplification of the experimental variable and MUST be stated in the
README as a departure from their distribution.

The non-math half is byte-identical across conditions. Only the math slice
changes. Token allocation is matched exactly.

Usage:
    python step04_pack.py --condition finemath   --total-tokens 1_000_000_000
    python step04_pack.py --condition swallowmath --total-tokens 1_000_000_000
"""
import argparse, json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

MATH_SOURCES = {
    "finemath":    ("HuggingFaceTB/finemath", "finemath-4plus", "text"),
    "swallowmath": ("tokyotech-llm/swallow-math", None, "text"),
}
WEB_SOURCE = ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text")


def pack_stream(repo, config, field, tok, target_tokens, seq_len, seed):
    """Tokenize a stream, concatenate with EOS separators, chunk to seq_len."""
    ds = load_dataset(repo, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    eos = tok.eos_token_id
    buf, chunks, n_tok = [], [], 0
    pbar = tqdm(total=target_tokens, desc=f"{repo}", unit="tok")
    for ex in ds:
        ids = tok(ex[field], truncation=False)["input_ids"] + [eos]
        buf.extend(ids)
        n_tok += len(ids)
        pbar.update(len(ids))
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len])
            buf = buf[seq_len:]
        if n_tok >= target_tokens:
            break
    pbar.close()
    return np.array(chunks, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(MATH_SOURCES), required=True)
    ap.add_argument("--total-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--math-frac", type=float, default=0.25)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer")
    ap.add_argument("--out", type=str, default="data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    math_tokens = int(a.total_tokens * a.math_frac)
    web_tokens = a.total_tokens - math_tokens
    print(f"math={math_tokens/1e6:.0f}M  web={web_tokens/1e6:.0f}M")

    repo, cfg, field = MATH_SOURCES[a.condition]
    math_chunks = pack_stream(repo, cfg, field, tok, math_tokens, a.seq_len, a.seed)

    # Web half is shared - build once, reuse for both conditions.
    web_path = out / f"web_{web_tokens}_{a.seq_len}.npy"
    if web_path.exists():
        print(f"reusing {web_path}")
        web_chunks = np.load(web_path)
    else:
        web_chunks = pack_stream(*WEB_SOURCE, tok, web_tokens, a.seq_len, a.seed)
        np.save(web_path, web_chunks)

    allc = np.concatenate([math_chunks, web_chunks])
    rng = np.random.default_rng(a.seed)
    rng.shuffle(allc)

    dest = out / f"{a.condition}_{a.seq_len}.npy"
    np.save(dest, allc)
    meta = {
        "condition": a.condition, "seq_len": a.seq_len,
        "n_sequences": int(allc.shape[0]),
        "total_tokens": int(allc.size),
        "math_sequences": int(math_chunks.shape[0]),
        "web_sequences": int(web_chunks.shape[0]),
        "math_frac_actual": float(math_chunks.shape[0] / allc.shape[0]),
        "vocab_size": len(tok),
    }
    (out / f"{a.condition}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\nsaved -> {dest}")
    print("VERIFY: math_frac_actual and total_tokens must match across conditions.")


if __name__ == "__main__":
    main()