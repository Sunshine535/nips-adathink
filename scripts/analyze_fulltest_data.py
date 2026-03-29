#!/usr/bin/env python3
"""
Comprehensive analysis of Qwen3-8B GSM8K per-sample results across fixed budgets.

Produces:
  1. Per-budget accuracy, avg tokens, early-stop rate, accuracy breakdown
  2. Cross-budget analysis (oracle, difficulty correlation)
  3. Cascade strategy simulation (128→512, 256→512)
  4. Token distribution buckets for Fixed-512

Usage:
    python scripts/analyze_fulltest_data.py [--csv PATH]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def pct(x):
    return f"{x * 100:.1f}%"


def print_header(title):
    w = 72
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def print_subheader(title):
    print(f"\n--- {title} ---")


# ──────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze per-sample GSM8K results")
    parser.add_argument(
        "--csv",
        type=str,
        default="results_kun/fulltest/per_sample_gsm8k_Qwen3_8B_20260324_120316.csv",
        help="Path to per-sample CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"Loaded {n} samples from {csv_path.name}")

    budgets = [128, 256, 512]

    # Detect hit-budget token counts (max tokens per budget = budget + answer overhead)
    max_tokens = {}
    for b in budgets:
        col = f"fixed_{b}_tokens"
        max_tokens[b] = int(df[col].max())
    print(f"Max token values per budget: {max_tokens}")

    # ================================================================
    # Part 1: Per-Budget Statistics
    # ================================================================
    print_header("Part 1: Per-Budget Statistics")

    rows = []
    for b in budgets:
        c_col = f"fixed_{b}_correct"
        t_col = f"fixed_{b}_tokens"

        correct = df[c_col].values
        tokens = df[t_col].values

        acc = correct.mean()
        avg_tok = tokens.mean()

        # Early-stop: tokens < max_tokens (i.e. did NOT hit the budget ceiling)
        hit_budget_tok = max_tokens[b]
        early_stop = tokens < hit_budget_tok
        early_stop_rate = early_stop.mean()

        # Accuracy split
        acc_early = correct[early_stop].mean() if early_stop.sum() > 0 else float("nan")
        acc_hit = correct[~early_stop].mean() if (~early_stop).sum() > 0 else float("nan")
        n_early = early_stop.sum()
        n_hit = (~early_stop).sum()

        rows.append({
            "Budget": b,
            "Accuracy": pct(acc),
            "Avg Tokens": f"{avg_tok:.1f}",
            "Early-Stop Rate": pct(early_stop_rate),
            f"N (early/hit)": f"{n_early}/{n_hit}",
            "Acc (early-stop)": pct(acc_early) if not np.isnan(acc_early) else "N/A",
            "Acc (hit-budget)": pct(acc_hit) if not np.isnan(acc_hit) else "N/A",
        })

    tbl1 = pd.DataFrame(rows)
    print(tbl1.to_string(index=False))

    # ================================================================
    # Part 2: Cross-Budget Analysis
    # ================================================================
    print_header("Part 2: Cross-Budget Analysis")

    # 2a. Samples correct at 128 but wrong at 512 (and vice versa)
    print_subheader("2a. Correct-at-X but Wrong-at-Y")
    for a, b_val in [(128, 512), (256, 512), (128, 256)]:
        ca = df[f"fixed_{a}_correct"].values.astype(bool)
        cb = df[f"fixed_{b_val}_correct"].values.astype(bool)
        right_a_wrong_b = (ca & ~cb).sum()
        right_b_wrong_a = (~ca & cb).sum()
        both_right = (ca & cb).sum()
        both_wrong = (~ca & ~cb).sum()
        print(f"  Fixed-{a} vs Fixed-{b_val}:")
        print(f"    Both correct:  {both_right:>5}  ({pct(both_right / n):>6})")
        print(f"    {a} right, {b_val} wrong: {right_a_wrong_b:>5}  ({pct(right_a_wrong_b / n):>6})")
        print(f"    {b_val} right, {a} wrong: {right_b_wrong_a:>5}  ({pct(right_b_wrong_a / n):>6})")
        print(f"    Both wrong:    {both_wrong:>5}  ({pct(both_wrong / n):>6})")

    # 2b. Difficulty vs token usage
    print_subheader("2b. Difficulty vs Token Usage Correlation")
    # Difficulty proxy: number of budgets that got it wrong (0=easy, 3=hard)
    difficulty = 3 - (
        df["fixed_128_correct"].values
        + df["fixed_256_correct"].values
        + df["fixed_512_correct"].values
    )
    for b in budgets:
        t = df[f"fixed_{b}_tokens"].values
        r, p = sp_stats.spearmanr(difficulty, t)
        print(f"  Spearman(difficulty, fixed_{b}_tokens) = {r:.3f}  (p = {p:.2e})")

    # 2c. Perfect oracle routing
    print_subheader("2c. Perfect Oracle Routing")
    print("  (Routes each sample to cheapest budget that gives correct answer)")

    oracle_tokens = []
    oracle_correct = 0
    for i in range(n):
        assigned = None
        for b in budgets:  # 128 → 256 → 512 (cheapest first)
            if df[f"fixed_{b}_correct"].iloc[i] == 1:
                assigned = df[f"fixed_{b}_tokens"].iloc[i]
                oracle_correct += 1
                break
        if assigned is not None:
            oracle_tokens.append(assigned)
        else:
            # No budget got it right; assign max-budget cost
            oracle_tokens.append(df["fixed_512_tokens"].iloc[i])

    oracle_tokens = np.array(oracle_tokens)
    print(f"  Oracle accuracy:   {pct(oracle_correct / n)}  (= Fixed-512 upper bound since oracle can't do better)")
    print(f"  Oracle avg tokens: {oracle_tokens.mean():.1f}")
    print(f"  Fixed-512 avg tok: {df['fixed_512_tokens'].mean():.1f}")
    print(f"  Token savings:     {pct(1 - oracle_tokens.mean() / df['fixed_512_tokens'].mean())}")

    # Breakdown: where does oracle route?
    route_counts = {b: 0 for b in budgets}
    route_counts["none"] = 0
    for i in range(n):
        routed = False
        for b in budgets:
            if df[f"fixed_{b}_correct"].iloc[i] == 1:
                route_counts[b] += 1
                routed = True
                break
        if not routed:
            route_counts["none"] += 1
    print(f"  Oracle routing distribution:")
    for k, v in route_counts.items():
        print(f"    → Budget {k}: {v:>5} ({pct(v / n):>6})")

    # ================================================================
    # Part 3: Cascade Strategy Analysis
    # ================================================================
    print_header("Part 3: Cascade Strategy Analysis")

    def cascade_analysis(lo_budget, hi_budget):
        """
        Cascade strategy: run at lo_budget first.
        - If model naturally stops (early-stop), accept that answer.
        - If it hits the budget ceiling, route to hi_budget.
        Returns (accuracy, avg_tokens, details).
        """
        lo_c = f"fixed_{lo_budget}_correct"
        lo_t = f"fixed_{lo_budget}_tokens"
        hi_c = f"fixed_{hi_budget}_correct"
        hi_t = f"fixed_{hi_budget}_tokens"
        lo_max = max_tokens[lo_budget]

        cascade_correct = 0
        cascade_tokens = []
        n_accepted = 0
        n_routed = 0
        acc_accepted = []
        acc_routed = []

        for i in range(n):
            tok_lo = df[lo_t].iloc[i]
            if tok_lo < lo_max:
                # Early-stop → accept lo_budget answer
                n_accepted += 1
                c = df[lo_c].iloc[i]
                cascade_correct += c
                cascade_tokens.append(tok_lo)
                acc_accepted.append(c)
            else:
                # Hit ceiling → route to hi_budget
                n_routed += 1
                c = df[hi_c].iloc[i]
                cascade_correct += c
                # Cost = lo_budget attempt + hi_budget attempt
                cascade_tokens.append(tok_lo + df[hi_t].iloc[i])
                acc_routed.append(c)

        cascade_tokens = np.array(cascade_tokens)
        acc = cascade_correct / n
        avg_tok = cascade_tokens.mean()
        acc_accept = np.mean(acc_accepted) if acc_accepted else float("nan")
        acc_route = np.mean(acc_routed) if acc_routed else float("nan")

        return {
            "Strategy": f"{lo_budget}→{hi_budget}",
            "Accuracy": pct(acc),
            "Avg Tokens": f"{avg_tok:.1f}",
            "Accepted (early)": f"{n_accepted} ({pct(n_accepted / n)})",
            "Routed (hit)": f"{n_routed} ({pct(n_routed / n)})",
            "Acc (accepted)": pct(acc_accept) if not np.isnan(acc_accept) else "N/A",
            "Acc (routed)": pct(acc_route) if not np.isnan(acc_route) else "N/A",
        }

    cascade_rows = []
    cascade_rows.append(cascade_analysis(128, 512))
    cascade_rows.append(cascade_analysis(256, 512))

    # Add baselines for comparison
    for b in budgets:
        cascade_rows.append({
            "Strategy": f"Fixed-{b} (baseline)",
            "Accuracy": pct(df[f"fixed_{b}_correct"].mean()),
            "Avg Tokens": f"{df[f'fixed_{b}_tokens'].mean():.1f}",
            "Accepted (early)": "-",
            "Routed (hit)": "-",
            "Acc (accepted)": "-",
            "Acc (routed)": "-",
        })

    tbl3 = pd.DataFrame(cascade_rows)
    print(tbl3.to_string(index=False))

    # ================================================================
    # Part 4: Token Distribution Analysis (Fixed-512)
    # ================================================================
    print_header("Part 4: Token Distribution Analysis (Fixed-512)")

    tok512 = df["fixed_512_tokens"].values
    cor512 = df["fixed_512_correct"].values

    # Note: tokens include answer tokens beyond the thinking budget.
    # We define buckets based on total token count.
    bucket_edges = [0, 64, 128, 256, 384, 512, 9999]
    bucket_labels = ["<64", "64-128", "128-256", "256-384", "384-512", ">512"]

    bucket_rows = []
    for i in range(len(bucket_edges) - 1):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (tok512 >= lo) & (tok512 < hi)
        cnt = mask.sum()
        if cnt > 0:
            acc = cor512[mask].mean()
            avg_t = tok512[mask].mean()
        else:
            acc = float("nan")
            avg_t = float("nan")
        bucket_rows.append({
            "Token Bucket": bucket_labels[i],
            "N Samples": cnt,
            "% of Total": pct(cnt / n),
            "Accuracy": pct(acc) if not np.isnan(acc) else "N/A",
            "Avg Tokens": f"{avg_t:.1f}" if not np.isnan(avg_t) else "N/A",
        })

    tbl4 = pd.DataFrame(bucket_rows)
    print(tbl4.to_string(index=False))

    # Cumulative: % of samples that naturally stop below each threshold
    print_subheader("Cumulative Early-Stop (Fixed-512)")
    thresholds = [64, 128, 256, 384, 512]
    for thr in thresholds:
        below = (tok512 < thr).sum()
        print(f"  Tokens < {thr:>4}: {below:>5} ({pct(below / n):>6})")

    # ================================================================
    # Summary Table
    # ================================================================
    print_header("Compact Summary")

    summary_rows = []
    for b in budgets:
        c_col = f"fixed_{b}_correct"
        t_col = f"fixed_{b}_tokens"
        acc = df[c_col].mean()
        avg_tok = df[t_col].mean()
        summary_rows.append({
            "Method": f"Fixed-{b}",
            "Accuracy": pct(acc),
            "Avg Tokens": f"{avg_tok:.1f}",
        })

    # Add cascade
    for lo, hi in [(128, 512), (256, 512)]:
        info = cascade_analysis(lo, hi)
        summary_rows.append({
            "Method": f"Cascade {lo}→{hi}",
            "Accuracy": info["Accuracy"],
            "Avg Tokens": info["Avg Tokens"],
        })

    # Add oracle
    summary_rows.append({
        "Method": "Perfect Oracle",
        "Accuracy": pct(oracle_correct / n),
        "Avg Tokens": f"{oracle_tokens.mean():.1f}",
    })

    tbl_summary = pd.DataFrame(summary_rows)
    print(tbl_summary.to_string(index=False))

    print("\n" + "=" * 72)
    print("  Analysis complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
