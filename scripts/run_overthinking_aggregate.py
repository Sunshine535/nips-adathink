#!/usr/bin/env python3
import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def bootstrap_ci(values: List[float], n_bootstrap: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rnd = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        s = 0.0
        for _ in range(n):
            s += values[rnd.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = int((alpha / 2.0) * n_bootstrap)
    hi = int((1.0 - alpha / 2.0) * n_bootstrap)
    hi = min(max(0, hi), n_bootstrap - 1)
    return means[lo], means[hi]


def load_rows(input_csvs: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in input_csvs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"input csv not found: {path}")
        with p.open("r", encoding="utf-8", newline="") as f:
            table = list(csv.DictReader(f))
        if not table:
            raise RuntimeError(f"empty csv: {path}")
        rows.extend(table)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Aggregate pooled overthinking metrics from per-sample CSVs")
    ap.add_argument("--input_csvs", nargs="+", required=True, help="per_sample_*.csv list")
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260228)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    rows = load_rows(args.input_csvs)

    f128_acc, f128_tok = [], []
    f256_acc, f256_tok = [], []
    f512_acc, f512_tok = [], []
    ada_acc, ada_tok = [], []

    d_acc_256_512, d_tok_256_512 = [], []
    d_acc_ada_256, d_tok_ada_256 = [], []

    for r in rows:
        c128 = to_int(r.get("fixed_128_correct", 0))
        t128 = to_float(r.get("fixed_128_tokens", 0.0))
        c256 = to_int(r.get("fixed_256_correct", 0))
        t256 = to_float(r.get("fixed_256_tokens", 0.0))
        c512 = to_int(r.get("fixed_512_correct", 0))
        t512 = to_float(r.get("fixed_512_tokens", 0.0))
        ca = to_int(r.get("adaptive_correct", 0))
        ta = to_float(r.get("adaptive_tokens", 0.0))

        f128_acc.append(c128)
        f128_tok.append(t128)
        f256_acc.append(c256)
        f256_tok.append(t256)
        f512_acc.append(c512)
        f512_tok.append(t512)
        ada_acc.append(ca)
        ada_tok.append(ta)

        d_acc_256_512.append(c256 - c512)
        d_tok_256_512.append(t256 - t512)
        d_acc_ada_256.append(ca - c256)
        d_tok_ada_256.append(ta - t256)

    acc_ci_256_512 = bootstrap_ci(d_acc_256_512, args.n_bootstrap, args.bootstrap_seed + 1, args.alpha)
    tok_ci_256_512 = bootstrap_ci(d_tok_256_512, args.n_bootstrap, args.bootstrap_seed + 2, args.alpha)
    acc_ci_ada_256 = bootstrap_ci(d_acc_ada_256, args.n_bootstrap, args.bootstrap_seed + 3, args.alpha)
    tok_ci_ada_256 = bootstrap_ci(d_tok_ada_256, args.n_bootstrap, args.bootstrap_seed + 4, args.alpha)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = (
        Path(args.output_json)
        if args.output_json
        else Path(f"methods/01_adathink/results/qwen35_27b_overthinking_{len(rows)}rows_{ts}.json")
    )

    summary = {
        "meta": {
            "timestamp_utc": ts,
            "input_csvs": args.input_csvs,
            "n_samples": len(rows),
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "alpha": args.alpha,
        },
        "means": {
            "fixed_128": {"accuracy": mean(f128_acc), "avg_tokens": mean(f128_tok)},
            "fixed_256": {"accuracy": mean(f256_acc), "avg_tokens": mean(f256_tok)},
            "fixed_512": {"accuracy": mean(f512_acc), "avg_tokens": mean(f512_tok)},
            "adaptive": {"accuracy": mean(ada_acc), "avg_tokens": mean(ada_tok)},
        },
        "paired_delta": {
            "fixed256_minus_fixed512": {
                "accuracy_mean": mean(d_acc_256_512),
                "accuracy_ci95": [acc_ci_256_512[0], acc_ci_256_512[1]],
                "avg_tokens_mean": mean(d_tok_256_512),
                "avg_tokens_ci95": [tok_ci_256_512[0], tok_ci_256_512[1]],
            },
            "adaptive_minus_fixed256": {
                "accuracy_mean": mean(d_acc_ada_256),
                "accuracy_ci95": [acc_ci_ada_256[0], acc_ci_ada_256[1]],
                "avg_tokens_mean": mean(d_tok_ada_256),
                "avg_tokens_ci95": [tok_ci_ada_256[0], tok_ci_ada_256[1]],
            },
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
