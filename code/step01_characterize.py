"""
step01 (LOCAL) - Data characterization.

LAST EDITED: 2026-07-26 18:42 UTC
CHANGELOG:
  2026-07-26 18:42 - fixed OOM hang in char_stats (4-gram set was ~16GB at
                     200k docs); now subsamples and hashes. Rewrote
                     exact_doc_count to use the HF datasets-server API, since
                     load_dataset_builder returns UNKNOWN for SwallowMath
                     (JSON-backed, no precomputed split metadata). Default
                     --n-docs lowered 200000 -> 50000.

Quantifies what the SwallowMath rewriting pipeline actually changed relative to
its source corpus FineMath-4+. Produces the guaranteed deliverable of this
project: numbers the dataset authors flagged as concerns but never measured.

Runs on a MacBook. Uses streaming - never downloads the full corpora.

Usage:
    python step01_characterize.py --n-docs 50000 --out out/phaseA
    python step01_characterize.py --n-docs 50000 --out out/phaseA --embeddings --decontam

MEMORY NOTE: 50k docs is plenty for distributional statistics. 200k pushed a
16GB MacBook into swap because of the n-gram counting, which is now bounded by
--ngram-docs regardless of --n-docs.
"""
import argparse, json, os, re, random, sys
from collections import Counter
from pathlib import Path

# Python randomises str hashing per process, so distinct_4gram_ratio would not
# be reproducible across runs. Re-exec once with a fixed seed.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

FINEMATH = ("HuggingFaceTB/finemath", "finemath-4plus")
SWALLOWMATH = ("tokyotech-llm/swallow-math", None)

# The rewriter's stylistic fingerprint. SwallowMath's card names
# "influence of Llama-3.3-70B-Instruct's preferences in solution style and
# formatting" as a suspected bias. These regexes measure it.
STRUCTURE_PATTERNS = {
    "step_header":      re.compile(r"^#{1,4}\s*Step\s*\d+", re.M | re.I),
    "md_header":        re.compile(r"^#{1,4}\s+", re.M),
    "bold_marker":      re.compile(r"\*\*[^*]+\*\*"),
    "numbered_list":    re.compile(r"^\s*\d+\.\s+", re.M),
    "latex_inline":     re.compile(r"\$[^$\n]+\$"),
    "latex_block":      re.compile(r"\$\$|\\\[|\\begin\{"),
    "boxed_answer":     re.compile(r"\\boxed\{"),
    "final_answer":     re.compile(r"(final answer|the answer is)", re.I),
}

TOPIC_KEYWORDS = {
    "algebra":     ["equation", "solve for", "polynomial", "quadratic", "factor", "variable"],
    "geometry":    ["triangle", "circle", "angle", "polygon", "perimeter", "vertex", "parallel"],
    "calculus":    ["derivative", "integral", "limit", "differentiat", "antiderivative"],
    "probability": ["probability", "random variable", "distribution", "expected value", "variance"],
    "arithmetic":  ["multiply", "divide", "fraction", "decimal", "percent", "sum of"],
    "linalg":      ["matrix", "eigenvalue", "vector", "determinant", "linear system"],
}


def stream_docs(repo, config, n, seed=0):
    """Stream n documents. Streaming avoids downloading the full corpus."""
    ds = load_dataset(repo, config, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for ex in tqdm(ds, total=n, desc=f"sampling {repo}"):
        out.append(ex["text"])
        if len(out) >= n:
            break
    return out


def exact_doc_count(repo, config, allow_stream=True):
    """
    GATE A.

    If SwallowMath has materially fewer documents than FineMath-4+, then the
    contrast confounds *rewriting* with *filtering*, and the one-variable claim
    in the LOI does not hold. Resolve this before writing any other code.

    Four fallbacks, cheapest first.

      1. HF datasets-server /size - reads parquet footers server-side.
         *** MUST check the `partial` flag. *** The server caps auto-conversion
         at 5GB, so for larger datasets it reports the row count of a TRUNCATED
         conversion. SwallowMath is 12.8GB, and trusting this returned
         2,628,785 instead of the true count - the bug that broke Gate A on the
         first run.
      2. load_dataset_builder split metadata - works for FineMath (parquet),
         returns UNKNOWN for SwallowMath (JSON-backed, no precomputed splits).
      3. Direct parquet footer read - fails for JSON-backed datasets.
      4. Stream and count. Slow (~25 min for SwallowMath) but always correct.
    """
    # 1. datasets-server
    try:
        import requests
        url = f"https://datasets-server.huggingface.co/size?dataset={repo}"
        if config:
            url += f"&config={config}"
        r = requests.get(url, timeout=30)
        if r.ok:
            j = r.json()
            partial = j.get("partial", False)
            size = j["size"]
            key = "config" if config and "config" in size else "dataset"
            n = size.get(key, {}).get("num_rows")
            if n and not partial:
                return int(n)
            if n and partial:
                print(f"    [1/4] API says {int(n):,} but response is PARTIAL "
                      f"(dataset >5GB) - ignoring, falling through")
    except Exception as e:
        print(f"    [1/4] datasets-server failed: {e}")

    # 2. builder metadata
    try:
        from datasets import load_dataset_builder
        b = load_dataset_builder(repo, config)
        splits = b.info.splits
        if splits and "train" in splits and splits["train"].num_examples:
            return int(splits["train"].num_examples)
        print("    [2/4] builder has no split metadata")
    except Exception as e:
        print(f"    [2/4] load_dataset_builder failed: {e}")

    # 3. parquet footers directly
    try:
        import fsspec, pyarrow.parquet as pq
        from huggingface_hub import HfApi
        files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                 if f.endswith(".parquet")]
        if files:
            fs = fsspec.filesystem("hf")
            total = 0
            for f in tqdm(files, desc="    parquet footers"):
                with fs.open(f"datasets/{repo}/{f}") as fh:
                    total += pq.ParquetFile(fh).metadata.num_rows
            return int(total)
        print("    [3/4] no parquet files (dataset is JSON-backed)")
    except Exception as e:
        print(f"    [3/4] parquet footer read failed: {e}")

    # 4. stream and count - slow but always correct
    if not allow_stream:
        return None
    try:
        print(f"    [4/4] streaming count for {repo} (~25 min, constant memory)")
        ds = load_dataset(repo, config, split="train", streaming=True)
        return sum(1 for _ in tqdm(ds, unit="doc", desc="    counting"))
    except Exception as e:
        print(f"    [4/4] streaming count failed: {e}")

    return None


def char_stats(texts, ngram_docs=20_000):
    """
    ngram_docs bounds the n-gram work independently of len(texts).

    Storing 4-grams as tuples across 200k documents is ~120M tuples and roughly
    16GB of RAM - that is what hung the first run. Two fixes: subsample the
    documents used for n-grams, and store 8-byte hashes instead of tuples.
    distinct_4gram_ratio is a ratio, so a subsample estimates it fine.
    """
    lens = np.array([len(t) for t in texts])
    words = [t.split() for t in texts]
    wlens = np.array([len(w) for w in words])

    # vocabulary over everything - bounded by vocab size, cheap
    vocab, total_tokens = Counter(), 0
    for w in tqdm(words, desc="  vocab", leave=False):
        vocab.update(w)
        total_tokens += len(w)

    # n-grams on a subsample, hashed
    sub = random.sample(words, min(ngram_docs, len(words)))
    four_grams, four_gram_total = set(), 0
    for w in tqdm(sub, desc="  4-grams", leave=False):
        for i in range(len(w) - 3):
            four_grams.add(hash((w[i], w[i + 1], w[i + 2], w[i + 3])))
            four_gram_total += 1

    struct = {}
    for name, pat in STRUCTURE_PATTERNS.items():
        hits = sum(1 for t in texts if pat.search(t))
        struct[f"{name}_doc_frac"] = hits / len(texts)

    topics = {}
    lowered = [t.lower() for t in texts]
    for topic, kws in TOPIC_KEYWORDS.items():
        hits = sum(1 for t in lowered if any(k in t for k in kws))
        topics[topic] = hits / len(texts)

    return {
        "n_docs": len(texts),
        "chars_mean": float(lens.mean()), "chars_median": float(np.median(lens)),
        "chars_p10": float(np.percentile(lens, 10)), "chars_p90": float(np.percentile(lens, 90)),
        "words_mean": float(wlens.mean()), "words_median": float(np.median(wlens)),
        "type_token_ratio": len(vocab) / max(total_tokens, 1),
        "distinct_4gram_ratio": len(four_grams) / max(four_gram_total, 1),
        "ngram_docs_sampled": len(sub),
        "structure": struct,
        "topics": topics,
        "_len_hist": np.histogram(wlens, bins=50, range=(0, 3000))[0].tolist(),
    }


def homogeneity(texts, n=4000, batch=64):
    """
    THE HOMOGENIZATION TEST.

    Embed a sample, compute the pairwise cosine similarity distribution.
    If rewriting collapses stylistic variety, this distribution shifts right.
    This is the empirical basis for the LOI's counter-hypothesis: better data
    might reduce expert differentiation and push the optimum denser.
    """
    import torch
    from sentence_transformers import SentenceTransformer
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  embedding device: {dev}")
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    sample = random.sample(texts, min(n, len(texts)))
    sample = [t[:2000] for t in sample]
    emb = m.encode(sample, batch_size=batch, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True)
    idx = np.random.choice(len(emb), size=(200_000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    sims = np.einsum("ij,ij->i", emb[idx[:, 0]], emb[idx[:, 1]])
    return {
        "cos_mean": float(sims.mean()), "cos_std": float(sims.std()),
        "cos_p50": float(np.percentile(sims, 50)),
        "cos_p90": float(np.percentile(sims, 90)),
        "cos_p99": float(np.percentile(sims, 99)),
        "_hist": np.histogram(sims, bins=60, range=(-0.2, 1.0))[0].tolist(),
    }


def decontamination(texts, n_gram=13):
    """
    SwallowMath's dataset card has an empty Decontamination section.
    Primary metric here is GSM8K task loss, so this is not optional.
    """
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    test_ngrams = set()
    for ex in gsm:
        w = (ex["question"] + " " + ex["answer"]).split()
        for i in range(len(w) - n_gram):
            test_ngrams.add(" ".join(w[i:i + n_gram]))

    hits = 0
    for t in tqdm(texts, desc="decontamination"):
        w = t.split()
        for i in range(0, max(len(w) - n_gram, 0), 3):
            if " ".join(w[i:i + n_gram]) in test_ngrams:
                hits += 1
                break
    return {"gsm8k_contaminated_docs": hits, "rate": hits / len(texts), "n_gram": n_gram}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=50_000,
                    help="docs sampled for distributional stats (50k is plenty)")
    ap.add_argument("--ngram-docs", type=int, default=20_000,
                    help="docs used for 4-gram stats; bounds memory independently of --n-docs")
    ap.add_argument("--out", type=str, default="out/phaseA")
    ap.add_argument("--embeddings", action="store_true",
                    help="run the homogenization test (uses cuda/mps/cpu automatically)")
    ap.add_argument("--decontam", action="store_true")
    ap.add_argument("--skip-counts", action="store_true",
                    help="skip GATE A (only if already answered - see notes/GATES.md)")
    ap.add_argument("--gate-only", action="store_true",
                    help="run GATE A document counts and nothing else")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = {}

    # ---- GATE A: exact document counts -----------------------------------
    if not a.skip_counts:
        print("\n" + "=" * 62)
        print("GATE A: document counts")
        print("=" * 62)
        counts = {}
        for name, (repo, cfg) in [("finemath_4plus", FINEMATH), ("swallowmath", SWALLOWMATH)]:
            print(f"  {name} ...")
            c = exact_doc_count(repo, cfg)
            counts[name] = c
            print(f"  {name:20s} {c if c else 'UNKNOWN'}")
        results["doc_counts"] = counts

        fm, sm = counts.get("finemath_4plus"), counts.get("swallowmath")
        if fm and sm:
            ratio = sm / fm
            results["doc_count_ratio"] = ratio
            print(f"\n  ratio = {sm:,} / {fm:,} = {ratio:.4f}")
            if ratio < 0.95:
                print("\n  *** VERDICT: FILTERING CONFOUND ***")
                print("  SwallowMath drops documents. The contrast is rewriting")
                print("  PLUS selection, not rewriting alone.")
                print("  -> README limitations must say so.")
                print("  -> LOI must say 'token- and distribution-matched',")
                print("     NOT 'source-matched'.")
            else:
                print("\n  *** VERDICT: ~1:1, contrast is rewriting alone ***")
            print("\n  Record this in notes/GATES.md before continuing.")
        else:
            print("\n  *** GATE A UNRESOLVED. ***")

        (out / "gate_a.json").write_text(json.dumps(
            {"counts": counts, "ratio": results.get("doc_count_ratio")}, indent=2))
        print(f"  wrote {out/'gate_a.json'}")

    if a.gate_only:
        return

    # ---- distributional statistics ---------------------------------------
    for name, (repo, cfg) in [("finemath_4plus", FINEMATH), ("swallowmath", SWALLOWMATH)]:
        print(f"\n=== {name} ===")
        texts = stream_docs(repo, cfg, a.n_docs)
        results[name] = char_stats(texts, ngram_docs=a.ngram_docs)
        if a.embeddings:
            results[name]["homogeneity"] = homogeneity(texts)
        if a.decontam:
            results[name]["decontamination"] = decontamination(texts[:50_000])
        # keep 200 docs for manual side-by-side inspection
        (out / f"{name}_sample.json").write_text(json.dumps(texts[:200], indent=2))
        del texts
        import gc; gc.collect()
        (out / "results.json").write_text(json.dumps(results, indent=2))  # checkpoint

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out/'results.json'}")

    # ---- ordering spot-check --------------------------------------------
    print("\n=== ORDERING SPOT-CHECK ===")
    print("Manually compare the first 50 docs of each _sample.json.")
    print("If row i of SwallowMath is the rewrite of row i of FineMath-4+,")
    print("you have de facto provenance even without an explicit ID field,")
    print("and can promise source-matching in the LOI. Otherwise you cannot.")


if __name__ == "__main__":
    main()