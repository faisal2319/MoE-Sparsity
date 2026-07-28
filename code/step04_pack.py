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


def pack_stream(repo, config, field, tok, target_tokens, seq_len, seed, out_path):
    """
    Tokenize a stream, concatenate with EOS separators, chunk to seq_len,
    writing INCREMENTALLY to a memmap on disk.

    The obvious implementation - accumulate every chunk in a Python list, then
    np.array(...) at the end - does not work at this scale. 1B tokens is
    ~488k chunks of 2048 Python ints; the list-of-lists alone is several GB of
    pointers before the int objects, and it OOMs or thrashes. Writing rows into
    a preallocated memmap keeps memory flat regardless of corpus size.
    """
    ds = load_dataset(repo, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    eos = tok.eos_token_id

    max_chunks = target_tokens // seq_len + 16          # small slack
    mm = np.lib.format.open_memmap(out_path, mode="w+",
                                   dtype=np.uint16, shape=(max_chunks, seq_len))

    buf = []
    written = 0
    n_tok = 0
    pbar = tqdm(total=target_tokens, desc=f"{repo}", unit="tok", unit_scale=True)
    for ex in ds:
        ids = tok(ex[field], truncation=False)["input_ids"] + [eos]
        buf.extend(ids)
        n_tok += len(ids)
        pbar.update(len(ids))

        # drain buf into the memmap without repeated list slicing
        n_full = len(buf) // seq_len
        if n_full:
            take = n_full * seq_len
            block = np.asarray(buf[:take], dtype=np.uint16).reshape(n_full, seq_len)
            end = min(written + n_full, max_chunks)
            mm[written:end] = block[:end - written]
            written = end
            del buf[:take]
            if written >= max_chunks:
                break
        if n_tok >= target_tokens:
            break
    pbar.close()

    mm.flush()
    del mm

    # truncate to what was actually written
    full = np.load(out_path, mmap_mode="r")
    real = np.array(full[:written])
    del full
    np.save(out_path, real)
    print(f"  {out_path}: {written:,} sequences  ({written*seq_len/1e6:.1f}M tokens)")
    return real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(MATH_SOURCES), required=True)
    ap.add_argument("--total-tokens", type=int, default=1_000_000_000)
    ap.add_argument("--math-frac", type=float, default=0.25)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--tokenizer", type=str, default="artifacts/tokenizer")
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    math_tokens = int(a.total_tokens * a.math_frac)
    web_tokens = a.total_tokens - math_tokens
    print(f"math={math_tokens/1e6:.0f}M  web={web_tokens/1e6:.0f}M")

    repo, cfg, field = MATH_SOURCES[a.condition]
    math_tmp = out / f"_tmp_math_{a.condition}.npy"
    math_chunks = pack_stream(repo, cfg, field, tok, math_tokens, a.seq_len, a.seed, math_tmp)

    # Web half is shared - build once, reuse for both conditions.
    web_path = out / f"web_{web_tokens}_{a.seq_len}.npy"
    if web_path.exists():
        print(f"reusing {web_path}")
        web_chunks = np.load(web_path)
    else:
        web_chunks = pack_stream(*WEB_SOURCE, tok, web_tokens, a.seq_len, a.seed, web_path)

    n_math, n_web = math_chunks.shape[0], web_chunks.shape[0]
    allc = np.concatenate([math_chunks, web_chunks])
    del math_chunks, web_chunks
    rng = np.random.default_rng(a.seed)
    rng.shuffle(allc)

    dest = out / f"{a.condition}_{a.seq_len}.npy"
    np.save(dest, allc)
    meta = {
        "condition": a.condition, "seq_len": a.seq_len,
        "n_sequences": int(allc.shape[0]),
        "total_tokens": int(allc.size),
        "math_sequences": int(n_math),
        "web_sequences": int(n_web),
        "math_frac_actual": float(n_math / allc.shape[0]),
        "vocab_size": len(tok),
    }
    (out / f"{a.condition}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    math_tmp.unlink(missing_ok=True)
    print(f"\nsaved -> {dest}")
    print("VERIFY: math_frac_actual and total_tokens must match across conditions.")


if __name__ == "__main__":
    main()