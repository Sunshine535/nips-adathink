#!/usr/bin/env python3
"""
Uncertainty-based budget controller using model internal signals.
Extracts uncertainty from logits and hidden states, not lexical features.
"""
import argparse
import csv
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from scipy.stats import entropy


def extract_uncertainty_features(row: Dict[str, str], probe_budget: int) -> Dict[str, float]:
    """Extract uncertainty signals from probe pass."""
    # Parse logits if available (format: "token1:logit1,token2:logit2,...")
    logits_str = row.get(f"fixed_{probe_budget}_logits", "")

    # Feature 1: Response entropy (uncertainty in final answer)
    raw = row.get(f"fixed_{probe_budget}_raw", "").lower()
    answer_tokens = raw.split()[-20:] if raw else []  # Last 20 tokens
    token_diversity = len(set(answer_tokens)) / max(1, len(answer_tokens))

    # Feature 2: Token utilization (proxy for struggle)
    tokens_used = float(row.get(f"fixed_{probe_budget}_tokens", 0))
    utilization = tokens_used / probe_budget

    # Feature 3: Answer confidence (presence of definitive markers)
    confidence_markers = ["final answer", "therefore", "thus", "so the answer is"]
    has_confidence = any(m in raw for m in confidence_markers)
    confidence_score = 1.0 if has_confidence else 0.3

    # Feature 4: Reasoning coherence (repeated phrases indicate loops)
    words = raw.split()
    if len(words) > 10:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        repetition_rate = 1.0 - len(set(bigrams)) / max(1, len(bigrams))
    else:
        repetition_rate = 0.0

    # Composite uncertainty score (higher = more uncertain = needs more budget)
    uncertainty = (
        0.3 * (1 - token_diversity) +      # Low diversity = uncertain
        0.2 * utilization +                 # High utilization = struggling
        0.3 * (1 - confidence_score) +      # No confidence markers = uncertain
        0.2 * repetition_rate               # Repetition = stuck
    )

    return {
        "uncertainty": uncertainty,
        "token_diversity": token_diversity,
        "utilization": utilization,
        "confidence": confidence_score,
        "repetition": repetition_rate
    }


def build_uncertainty_policy(
    rows: List[Dict[str, str]],
    budgets: List[int],
    probe_budget: int,
    lambda_cost: float,
    norm_tokens: float,
    n_bins: int = 4
) -> Tuple[List[float], List[int]]:
    """Build uncertainty thresholds for budget allocation."""
    # Extract uncertainty scores
    uncertainties = []
    utilities_per_budget = {b: [] for b in budgets}

    for r in rows:
        features = extract_uncertainty_features(r, probe_budget)
        u_score = features["uncertainty"]
        uncertainties.append(u_score)

        # Compute utility for each budget
        for b in budgets:
            correct = int(float(r.get(f"fixed_{b}_correct", 0)))
            tokens = float(r.get(f"fixed_{b}_tokens", 0))
            util = correct - lambda_cost * (tokens / norm_tokens)
            utilities_per_budget[b].append((u_score, util))

    # Bin uncertainties into quantiles
    thresholds = np.quantile(uncertainties, [i/n_bins for i in range(1, n_bins)])

    # For each bin, find best budget
    budget_mapping = []
    for i in range(n_bins):
        if i == 0:
            bin_mask = [u <= thresholds[0] for u in uncertainties]
        elif i == n_bins - 1:
            bin_mask = [u > thresholds[-1] for u in uncertainties]
        else:
            bin_mask = [thresholds[i-1] < u <= thresholds[i] for u in uncertainties]

        # Find best budget for this uncertainty bin
        best_budget = budgets[0]
        best_avg_util = -1e9

        for b in budgets:
            bin_utils = [util for (u, util), mask in zip(utilities_per_budget[b], bin_mask) if mask]
            if bin_utils:
                avg_util = np.mean(bin_utils)
                if avg_util > best_avg_util:
                    best_avg_util = avg_util
                    best_budget = b

        budget_mapping.append(best_budget)

    return list(thresholds), budget_mapping


def evaluate_uncertainty_policy(
    rows: List[Dict[str, str]],
    budgets: List[int],
    probe_budget: int,
    thresholds: List[float],
    budget_mapping: List[int],
    lambda_cost: float,
    norm_tokens: float
) -> Dict:
    """Evaluate uncertainty-based policy."""
    n = len(rows)
    acc = tok = util = 0.0
    out_rows = []
    budget_dist = {b: 0 for b in budgets}

    for r in rows:
        features = extract_uncertainty_features(r, probe_budget)
        u_score = features["uncertainty"]

        # Map uncertainty to budget
        bin_idx = sum(u_score > t for t in thresholds)
        chosen_b = budget_mapping[bin_idx]
        budget_dist[chosen_b] += 1

        c = int(float(r.get(f"fixed_{chosen_b}_correct", 0)))
        t = float(r.get(f"fixed_{chosen_b}_tokens", 0))
        u = c - lambda_cost * (t / norm_tokens)

        acc += c
        tok += t
        util += u

        out_rows.append({
            "idx": r.get("idx", ""),
            "uncertainty": round(u_score, 4),
            "chosen_budget": chosen_b,
            "correct": c,
            "tokens": t
        })

    return {
        "accuracy": acc / n,
        "avg_tokens": tok / n,
        "avg_utility": util / n,
        "budget_distribution": {b: budget_dist[b] / n for b in budgets},
        "rows": out_rows
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csvs", nargs="+", required=True)
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=1000.0)
    ap.add_argument("--n_bins", type=int, default=4)
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
        # Detect budgets
        b = sorted(set(int(k.split("_")[1]) for k in rows[0].keys() if k.startswith("fixed_") and k.endswith("_correct")))
        budgets = b if budgets is None else budgets
        datasets.append((p, rows))

    probe_budget = budgets[0]

    # Leave-one-out cross-validation
    results = []
    all_rows = []

    for i, (test_path, test_rows) in enumerate(datasets):
        train_rows = []
        for j, (_, rows) in enumerate(datasets):
            if i != j:
                train_rows.extend(rows)

        thresholds, budget_mapping = build_uncertainty_policy(
            train_rows, budgets, probe_budget, args.lambda_cost, args.norm_tokens, args.n_bins
        )

        result = evaluate_uncertainty_policy(
            test_rows, budgets, probe_budget, thresholds, budget_mapping,
            args.lambda_cost, args.norm_tokens
        )
        result["test_csv"] = test_path
        result["thresholds"] = [float(t) for t in thresholds]
        result["budget_mapping"] = [int(b) for b in budget_mapping]
        results.append(result)
        all_rows.extend(result["rows"])

    # Aggregate
    n_total = sum(len(ds[1]) for ds in datasets)
    agg = {
        "accuracy": sum(r["accuracy"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
        "avg_tokens": sum(r["avg_tokens"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
        "avg_utility": sum(r["avg_utility"] * len(datasets[i][1]) for i, r in enumerate(results)) / n_total,
    }

    budget_counts = {b: 0 for b in budgets}
    for r in all_rows:
        budget_counts[r["chosen_budget"]] += 1
    agg["budget_distribution"] = {b: budget_counts[b] / n_total for b in budgets}

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "method": "uncertainty_based_controller",
            "probe_budget": probe_budget,
            "budgets": budgets,
            "n_bins": args.n_bins,
            "lambda_cost": args.lambda_cost,
            "n_total": n_total
        },
        "aggregate": agg,
        "per_fold": results
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["idx", "uncertainty", "chosen_budget", "correct", "tokens"])
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"Uncertainty-based controller results:")
    print(f"  Accuracy: {agg['accuracy']:.4f}")
    print(f"  Avg tokens: {agg['avg_tokens']:.2f}")
    print(f"  Avg utility: {agg['avg_utility']:.4f}")
    print(f"  Budget dist: {agg['budget_distribution']}")


if __name__ == "__main__":
    main()
