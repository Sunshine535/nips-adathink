#!/usr/bin/env python3
"""Create canonical sample manifest for reproducible A/B/C comparisons.

Manifest stores: benchmark, split, HF dataset id, sample index, question hash,
gold hash, seed. Ensures all variants run on IDENTICAL samples.

Usage:
    python scripts/make_sample_manifest.py \
        --benchmark math500 --n_samples 200 --seed 42 \
        --output results/manifests/math500_n200_seed42.json
"""
import argparse, hashlib, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_manifest(benchmark: str, n: int, seed: int) -> dict:
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        items = []
        for k, i in enumerate(idxs[:n]):
            raw = ds[i]
            q = raw["question"]
            gold = raw["answer"].split("####")[-1].strip().replace(",", "")
            items.append({
                "order": k,
                "hf_index": i,
                "question_hash": hash_str(q),
                "gold_hash": hash_str(gold),
                "question_preview": q[:80],
                "gold": gold,
            })
        ds_id = "openai/gsm8k"
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        all_items = list(ds)
        random.seed(seed); random.shuffle(all_items)
        items = []
        for k, s in enumerate(all_items[:n]):
            q = s["problem"]
            gold = s["answer"]
            items.append({
                "order": k,
                "hf_index": k,
                "question_hash": hash_str(q),
                "gold_hash": hash_str(str(gold)),
                "question_preview": q[:80],
                "gold": str(gold)[:40],
            })
        ds_id = "HuggingFaceH4/MATH-500"

    return {
        "meta": {"benchmark": benchmark, "dataset_id": ds_id,
                 "split": "test", "n_samples": n, "seed": seed},
        "items": items,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    m = build_manifest(args.benchmark, args.n_samples, args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Wrote manifest: {args.output}")
    print(f"n={len(m['items'])}, benchmark={args.benchmark}, seed={args.seed}")


if __name__ == "__main__":
    main()
