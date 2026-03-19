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


def bootstrap_ci(values: List[float], n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rnd = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rnd.randrange(n)]
        means.append(s / n)
    means.sort()
    lo_idx = int((alpha / 2.0) * n_boot)
    hi_idx = int((1.0 - alpha / 2.0) * n_boot)
    hi_idx = min(max(0, hi_idx), n_boot - 1)
    return means[lo_idx], means[hi_idx]


def main():
    ap = argparse.ArgumentParser(description="Paired bootstrap significance for template budget controller rows")
    ap.add_argument("--rows_csv", required=True, help="template_controller_rows_*.csv path")
    ap.add_argument("--compare_budget", type=int, default=256, help="fixed budget baseline for paired deltas")
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument("--n_bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=20260228)
    ap.add_argument("--output_json", type=str, default="")
    args = ap.parse_args()

    rows_path = Path(args.rows_csv)
    if not rows_path.exists():
        raise FileNotFoundError(f"rows_csv not found: {rows_path}")

    with rows_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {rows_path}")

    test_csv_paths = sorted({r["test_csv"] for r in rows})
    cache: Dict[str, Dict[int, Dict[str, str]]] = {}
    for p in test_csv_paths:
        pp = Path(p)
        if not pp.exists():
            raise FileNotFoundError(f"Referenced test_csv not found: {p}")
        with pp.open("r", encoding="utf-8", newline="") as f:
            table = list(csv.DictReader(f))
        cache[p] = {to_int(rr.get("idx", 0)): rr for rr in table}

    b = args.compare_budget
    acc_delta = []
    tok_delta = []
    util_delta = []
    learned_acc = []
    learned_tok = []
    learned_util = []
    fixed_acc = []
    fixed_tok = []
    fixed_util = []

    for r in rows:
        test_csv = r["test_csv"]
        idx = to_int(r.get("idx", 0))
        base = cache[test_csv].get(idx)
        if base is None:
            raise RuntimeError(f"Missing idx={idx} in {test_csv}")

        lc = to_int(r.get("correct", 0))
        lt = to_float(r.get("tokens", 0.0))
        lu = lc - args.lambda_cost * (lt / args.norm_tokens)

        fc = to_int(base.get(f"fixed_{b}_correct", 0))
        ft = to_float(base.get(f"fixed_{b}_tokens", 0.0))
        fu = fc - args.lambda_cost * (ft / args.norm_tokens)

        learned_acc.append(lc)
        learned_tok.append(lt)
        learned_util.append(lu)
        fixed_acc.append(fc)
        fixed_tok.append(ft)
        fixed_util.append(fu)

        acc_delta.append(lc - fc)
        tok_delta.append(lt - ft)
        util_delta.append(lu - fu)

    def mean(vs: List[float]) -> float:
        return sum(vs) / max(1, len(vs))

    acc_ci = bootstrap_ci(acc_delta, args.n_bootstrap, args.bootstrap_seed + 1)
    tok_ci = bootstrap_ci(tok_delta, args.n_bootstrap, args.bootstrap_seed + 2)
    util_ci = bootstrap_ci(util_delta, args.n_bootstrap, args.bootstrap_seed + 3)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = (
        Path(args.output_json)
        if args.output_json
        else rows_path.with_name(
            rows_path.stem + f"_significance_vs_fixed{args.compare_budget}_{ts}.json"
        )
    )

    summary = {
        "meta": {
            "timestamp_utc": ts,
            "rows_csv": str(rows_path),
            "compare_budget": args.compare_budget,
            "lambda_cost": args.lambda_cost,
            "norm_tokens": args.norm_tokens,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "n_samples": len(rows),
        },
        "learned_mean": {
            "accuracy": mean(learned_acc),
            "avg_tokens": mean(learned_tok),
            "avg_utility": mean(learned_util),
        },
        "fixed_mean": {
            "accuracy": mean(fixed_acc),
            "avg_tokens": mean(fixed_tok),
            "avg_utility": mean(fixed_util),
        },
        "paired_delta_learned_minus_fixed": {
            "accuracy": {
                "mean": mean(acc_delta),
                "ci95": [acc_ci[0], acc_ci[1]],
            },
            "avg_tokens": {
                "mean": mean(tok_delta),
                "ci95": [tok_ci[0], tok_ci[1]],
            },
            "avg_utility": {
                "mean": mean(util_delta),
                "ci95": [util_ci[0], util_ci[1]],
            },
        },
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
