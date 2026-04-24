#!/usr/bin/env python3
"""Create canonical sample manifest for reproducible A/B/C comparisons.

V2 (GPT-5.5 review fixes):
- Preserve ORIGINAL HF index (not post-shuffle order)
- Store full question hash AND gold hash (not 40-char truncation)
- Runners must support --sample_manifest to enforce same-sample evaluation

Usage:
    python scripts/make_sample_manifest.py --benchmark math500 \\
        --n_samples 200 --seed 42 --output results/manifests/math200_seed42.json
"""
import argparse, hashlib, json, os, random


def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_manifest(benchmark: str, n: int, seed: int) -> dict:
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        items = []
        for k, orig_i in enumerate(idxs[:n]):
            raw = ds[orig_i]
            q = raw["question"]
            gold = raw["answer"].split("####")[-1].strip().replace(",", "")
            items.append({
                "order": k,
                "hf_original_index": orig_i,  # V2: ACTUAL HF index
                "question_hash": hash_str(q),
                "gold_hash": hash_str(gold),
                "question_preview": q[:80],
                "gold": gold,
            })
        ds_id = "openai/gsm8k"
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        n_total = len(ds)
        idxs = list(range(n_total))
        random.seed(seed); random.shuffle(idxs)
        items = []
        for k, orig_i in enumerate(idxs[:n]):
            s = ds[orig_i]
            q = s["problem"]
            gold = str(s["answer"])
            items.append({
                "order": k,
                "hf_original_index": orig_i,  # V2: ACTUAL HF index
                "question_hash": hash_str(q),
                "gold_hash": hash_str(gold),
                "question_preview": q[:80],
                "gold": gold,  # V2: full gold, not 40-char truncation
            })
        ds_id = "HuggingFaceH4/MATH-500"

    return {
        "meta": {"benchmark": benchmark, "dataset_id": ds_id, "split": "test",
                 "n_samples": n, "seed": seed, "schema_version": 2},
        "items": items,
    }


def load_manifest(path: str) -> dict:
    """Load a manifest. Runners use this for same-sample enforcement."""
    with open(path) as f:
        m = json.load(f)
    assert m.get("meta", {}).get("schema_version", 1) >= 1
    return m


def load_items_from_manifest(manifest: dict) -> list:
    """Return list of {q, gold, idx} dicts using ACTUAL HF indices."""
    from datasets import load_dataset
    bm = manifest["meta"]["benchmark"]
    if bm == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for item in manifest["items"]:
            i = item["hf_original_index"]
            raw = ds[i]
            q = raw["question"]
            gold = raw["answer"].split("####")[-1].strip().replace(",", "")
            # Verify hash match
            if hash_str(q) != item["question_hash"]:
                raise ValueError(f"Manifest hash mismatch at order={item['order']}")
            out.append({"q": q, "gold": gold, "idx": i, "order": item["order"]})
        return out
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for item in manifest["items"]:
            i = item["hf_original_index"]
            s = ds[i]
            q = s["problem"]
            gold = str(s["answer"])
            if hash_str(q) != item["question_hash"]:
                raise ValueError(f"Manifest hash mismatch at order={item['order']}")
            out.append({"q": q, "gold": gold, "idx": i, "order": item["order"]})
        return out


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
    print(f"First 3 HF indices: {[x['hf_original_index'] for x in m['items'][:3]]}")


if __name__ == "__main__":
    main()
