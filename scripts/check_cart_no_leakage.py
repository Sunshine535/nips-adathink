#!/usr/bin/env python3
"""Verify CART training data has NO eval/test leakage.

Checks that no question_hash in training data matches any eval benchmark sample.

Usage:
    python3 scripts/check_cart_no_leakage.py \
        --train_data results/cart/train_prefixes.jsonl \
        --eval_benchmark math500
"""
import argparse, hashlib, json, sys


def hash_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_eval_hashes(benchmark):
    from datasets import load_dataset
    if benchmark == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return {hash_str(s["problem"]) for s in ds}
    elif benchmark == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return {hash_str(s["question"]) for s in ds}
    else:
        raise ValueError(f"Unknown eval benchmark: {benchmark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True)
    p.add_argument("--eval_benchmark", default="math500")
    args = p.parse_args()

    eval_hashes = load_eval_hashes(args.eval_benchmark)
    print(f"Eval benchmark: {args.eval_benchmark}, {len(eval_hashes)} unique hashes")

    train_hashes = set()
    n_records = 0
    with open(args.train_data) as f:
        for line in f:
            r = json.loads(line)
            train_hashes.add(r["question_hash"])
            n_records += 1

    overlap = train_hashes & eval_hashes
    print(f"Train records: {n_records}, unique questions: {len(train_hashes)}")
    print(f"Overlap with eval: {len(overlap)}")

    if overlap:
        print(f"\n**LEAKAGE DETECTED**: {len(overlap)} train questions match eval set!")
        for h in list(overlap)[:5]:
            print(f"  hash: {h}")
        sys.exit(1)
    else:
        print("\n**NO LEAKAGE DETECTED** — PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
