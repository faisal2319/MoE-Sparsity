# step03 (VAST)  - 16k BPE on raw FineMath-4+
# --- paste contents here ---
"""
Train a 16k BPE tokenizer on raw FineMath-4+.

Why 16k and not an off-the-shelf tokenizer: at d=256, a Llama-3 128k vocab
would give ~33M embedding parameters against ~2M non-embedding active
parameters. The embedding table would BE the model. 16k keeps the parameter
budget where the science is.

Why FineMath-4+ as the source: it is the origin of BOTH conditions, so the
tokenizer favours neither. The same tokenizer is used for both conditions -
non-negotiable, or the comparison is meaningless.

Usage:
    python step03_train_tokenizer.py --vocab-size 16000 --n-docs 500000
"""
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--n-docs", type=int, default=500_000)
    ap.add_argument("--out", type=str, default="artifacts/tokenizer")
    a = ap.parse_args()

    ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                      split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)

    def it(batch=1000):
        buf = []
        for i, ex in enumerate(ds):
            if i >= a.n_docs:
                break
            buf.append(ex["text"])
            if len(buf) == batch:
                yield buf
                buf = []
        if buf:
            yield buf

    base = AutoTokenizer.from_pretrained("gpt2")
    tok = base.train_new_from_iterator(it(), vocab_size=a.vocab_size)
    tok.pad_token = tok.eos_token
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}  (vocab={len(tok)})")


if __name__ == "__main__":
    main()