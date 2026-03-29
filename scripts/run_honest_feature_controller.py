#!/usr/bin/env python3
"""
Honest 3-bit feature controller as described in paper:
- answer_presence: whether "Final answer" appears in probe output
- token_utilization: whether probe used full budget (>95%)
- answer_consistency: whether answer is consistent across features
"""
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def to_float(v, default=0.0):
    try:
        return float(v)
    except:
        return default


def to_int(v, default=0):
    try:
        return int(float(v))
    except:
        return default


def detect_budgets(fieldnames: List[str]) -> List[int]:
    budgets = []
    for k in fieldnames:
        m = re.match(r"fixed_(\d+)_correct", k)
        if m:
            budgets.append(int(m.group(1)))
    return sorted(set(budgets))


def utility(row: Dict[str, str], budget: int, lambda_cost: float, norm_tokens: float) -> float:
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / norm_tokens)


def extract_features(row: Dict[str, str], probe_budget: int) -> Tuple[int, int, int]:
    """Extract 3-bit features from probe pass."""
    raw = (row.get(f"fixed_{probe_budget}_raw") or "").lower()
    tokens = to_float(row.get(f"fixed_{probe_budget}_tokens", 0.0))

    # Feature 1: answer_presence
    answer_presence = 1 if "final answer" in raw else 0

    # Feature 2: token_utilization (>95% of budget)
    token_utilization = 1 if tokens >= 0.95 * probe_budget else 0

    # Feature 3: answer_consistency (has answer AND didn't use full budget)
    answer_consistency = 1 if (answer_presence == 1 and token_utilization == 0) else 0

    return (answer_presence, token_utilization, answer_consistency)


def make_key(features: Tuple[int, int, int]) -> str:
    """Convert 3-bit features to string key."""
    return f"{features[0]}{features[1]}{features[2]}"


def build_policy(
    rows: List[Dict[str, str]],
    budgets: List[int],
    probe_budget: int,
    lambda_cost: float,
    norm_tokens: float
) -> Tuple[Dict[str, int], int]:
    """Build feature-to-budget mapping."""
    stats: Dict[str, Dict[int, List[float]]] = {}

    for r in rows:
        features = extract_features(r, probe_budget)
        k = make_key(features)

        if k not in stats:
            stats[k] = {b: [0.0, 0] for b in budgets}

        for b in budgets:
            u = utility(r, b, lambda_cost, norm_tokens)
            stats[k][b][0] += u
            stats[k][b][1] += 1

    # Select best budget for each feature pattern
    mapping: Dict[str, int] = {}
    for k, st in stats.items():
        best_b = budgets[0]
        best_u = st[best_b][0] / max(1, st[best_b][1])
        for b in budgets[1:]:
            u = st[b][0] / max(1, st[b][1])
            if u > best_u:
                best_u = u
                best_b = b
        mapping[k] = best_b

    # Default budget (global best)
    default_budget = budgets[0]
    best_global = -1e18
    for b in budgets:
        avg_u = sum(utility(r, b, lambda_cost, norm_tokens) for r in rows) / max(1, len(rows))
        if avg_u > best_global:
            best_global = avg_u
            default_budget = b

    return mapping, default_budget


def evaluate(
    rows: List[Dict[str, str]],
    budgets: List[int],
    probe_budget: int,
    mapping: Dict[str, int],
    default_budget: int,
    lambda_cost: float,
    norm_tokens: float
) -> Dict:
    """Evaluate policy on test set."""
    n = len(rows)
    acc = tok = util = 0.0
    oracle_match = 0
    out_rows = []
    budget_dist = {b: 0 for b in budgets}

    for r in rows:
        features = extract_features(r, probe_budget)
        k = make_key(features)
        chosen_b = mapping.get(k, default_budget)
        budget_dist[chosen_b] += 1

        c = to_int(r.get(f"fixed_{chosen_b}_correct", 0))
        t = to_float(r.get(f"fixed_{chosen_b}_tokens", 0.0))
        u = c - lambda_cost * (t / norm_tokens)

        acc += c
        tok += t
        util += u

        # Oracle budget
        best_b = budgets[0]
        best_u = utility(r, budgets[0], lambda_cost, norm_tokens)
        for b in budgets[1:]:
            u_b = utility(r, b, lambda_cost, norm_tokens)
            if u_b > best_u:
                best_u = u_b
                best_b = b

        if chosen_b == best_b:
            oracle_match += 1

        out_rows.append({
            "idx": r.get("idx", ""),
            "features": k,
            "chosen_budget": chosen_b,
            "oracle_budget": best_b,
            "correct": c,
            "tokens": t,
            "utility": u
        })

    return {
        "accuracy": acc / n,
        "avg_tokens": tok / n,
        "avg_utility": util / n,
        "oracle_match_rate": oracle_match / n,
        "budget_distribution": {b: budget_dist[b] / n for b in budgets},
        "rows": out_rows
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csvs", nargs="+", required=True)
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=1000.0)
    ap.add_argument("--output_json", type=str, required=True)
    ap.add_argument("--output_csv", type=str, default="")
    args = ap.parse_args()

    # Load datasets
    datasets = []
    budgets = None
    for p in args.input_csvs:
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        b = detect_budgets(list(rows[0].keys()))
        budgets = b if budgets is None else budgets
        datasets.append((p, rows))

    probe_budget = budgets[0]  # Use minimum budget as probe

    # Leave-one-out cross-validation
    results = []
    all_rows = []

    for i, (test_path, test_rows) in enumerate(datasets):
        train_rows = []
        for j, (_, rows) in enumerate(datasets):
            if i != j:
                train_rows.extend(rows)

        mapping, default_budget = build_policy(
            train_rows, budgets, probe_budget, args.lambda_cost, args.norm_tokens
        )

        result = evaluate(
            test_rows, budgets, probe_budget, mapping, default_budget,
            args.lambda_cost, args.norm_tokens
        )
        result["test_csv"] = test_path
        result["mapping"] = {k: int(v) for k, v in mapping.items()}
        result["default_budget"] = int(default_budget)
        results.append(result)
        all_rows.extend(result["rows"])

    # Aggregate
    n_total = sum(len(ds[1]) for ds in datasets)
    agg = {
        "accuracy": sum(r["accuracy"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
        "avg_tokens": sum(r["avg_tokens"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
        "avg_utility": sum(r["avg_utility"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
        "oracle_match_rate": sum(r["oracle_match_rate"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
    }

    # Budget distribution
    budget_counts = {b: 0 for b in budgets}
    for r in all_rows:
        budget_counts[r["chosen_budget"]] += 1
    agg["budget_distribution"] = {b: budget_counts[b] / n_total for b in budgets}

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "method": "honest_3bit_feature_controller",
            "probe_budget": probe_budget,
            "budgets": budgets,
            "lambda_cost": args.lambda_cost,
            "norm_tokens": args.norm_tokens,
            "n_datasets": len(datasets),
            "n_total": n_total
        },
        "aggregate": agg,
        "per_fold": results
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "features", "chosen_budget", "oracle_budget", "correct", "tokens", "utility"])
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"Honest feature controller results:")
    print(f"  Accuracy: {agg['accuracy']:.4f}")
    print(f"  Avg tokens: {agg['avg_tokens']:.2f}")
    print(f"  Avg utility: {agg['avg_utility']:.4f}")
    print(f"  Oracle match: {agg['oracle_match_rate']:.4f}")
    print(f"  Budget dist: {agg['budget_distribution']}")


if __name__ == "__main__":
    main()
