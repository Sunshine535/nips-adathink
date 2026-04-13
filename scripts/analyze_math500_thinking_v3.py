#!/usr/bin/env python3
"""Analyze MATH-500 thinking experiment results (v3) for Qwen3-8B.

Loads JSON result files from math500_thinking_v3/, computes per-budget statistics,
verifies the F_L(b) theory prediction, and compares against paper claims.

Usage:
    python scripts/analyze_math500_thinking_v3.py
    python scripts/analyze_math500_thinking_v3.py --results_dir /workspace/nips-adathink/results/math500_thinking_v3
"""

import argparse
import glob
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Paper-claimed accuracy numbers (think@budget, %)
PAPER_CLAIMS = {
    512: 6.2,
    1024: 18.0,
    2048: 44.0,
}

BUDGETS_OF_INTEREST = [256, 512, 1024, 2048, 4096]


def load_json_results(results_dir):
    """Load all JSON result files and group samples by budget.

    Returns:
        dict[int, list[dict]]: budget -> list of sample result dicts
    """
    pattern = os.path.join(results_dir, "math500_Qwen3-8B_thinking_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        # Fallback: try any .json in the directory
        pattern = os.path.join(results_dir, "*.json")
        files = sorted(glob.glob(pattern))

    if not files:
        log.error("No JSON files found in %s", results_dir)
        sys.exit(1)

    log.info("Found %d JSON file(s) in %s", len(files), results_dir)
    for f in files:
        log.info("  %s", os.path.basename(f))

    by_budget = defaultdict(list)
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Might be wrapped: {"results": [...], "config": {...}}
            if "results" in data:
                samples = data["results"]
            else:
                log.warning("Unexpected dict format in %s, skipping", fpath)
                continue
        elif isinstance(data, list):
            samples = data
        else:
            log.warning("Unexpected format in %s, skipping", fpath)
            continue

        for s in samples:
            budget = int(s.get("budget", 0))
            if budget > 0:
                by_budget[budget].append(s)

    log.info("Loaded budgets: %s", sorted(by_budget.keys()))
    return dict(by_budget)


def compute_budget_stats(samples):
    """Compute statistics for a list of samples at a single budget.

    Returns:
        dict with keys: accuracy, n_samples, avg_tokens, natural_stop_rate,
        has_final_answer_rate, acc_natural, acc_truncated, n_natural, n_truncated
    """
    n = len(samples)
    if n == 0:
        return None

    corrects = [int(bool(s.get("correct", False))) for s in samples]
    tokens = [s.get("num_tokens", 0) for s in samples]

    # natural_stop might be stored as "natural_stop" or inferred from "early_stop"
    natural_flags = []
    for s in samples:
        if "natural_stop" in s:
            natural_flags.append(bool(s["natural_stop"]))
        elif "early_stop" in s:
            # early_stop=True means it stopped early (natural), NOT truncated
            # Careful: naming can be confusing. Assume early_stop=True means
            # the model stopped on its own before hitting the budget.
            natural_flags.append(bool(s["early_stop"]))
        else:
            # Heuristic: if num_tokens < budget, likely natural stop
            budget = int(s.get("budget", 0))
            natural_flags.append(s.get("num_tokens", budget) < budget)

    has_final = [int(bool(s.get("has_final_answer", False))) for s in samples]

    # Split by natural stop vs truncated
    nat_correct = [c for c, ns in zip(corrects, natural_flags) if ns]
    trunc_correct = [c for c, ns in zip(corrects, natural_flags) if not ns]

    acc_natural = np.mean(nat_correct) * 100 if nat_correct else float("nan")
    acc_truncated = np.mean(trunc_correct) * 100 if trunc_correct else float("nan")

    return {
        "accuracy": np.mean(corrects) * 100,
        "n_samples": n,
        "avg_tokens": np.mean(tokens),
        "natural_stop_rate": np.mean(natural_flags) * 100,
        "has_final_answer_rate": np.mean(has_final) * 100,
        "acc_natural": acc_natural,
        "acc_truncated": acc_truncated,
        "n_natural": len(nat_correct),
        "n_truncated": len(trunc_correct),
    }


def compute_theory_prediction(stats):
    """Compute Acc_pred = F_L * alpha_c + (1 - F_L) * alpha_t.

    F_L(b) = natural_stop_rate / 100  (fraction that finish within budget)
    alpha_c = accuracy among natural stops
    alpha_t = accuracy among truncated
    """
    f_l = stats["natural_stop_rate"] / 100.0
    alpha_c = stats["acc_natural"] / 100.0 if not np.isnan(stats["acc_natural"]) else 0.0
    alpha_t = stats["acc_truncated"] / 100.0 if not np.isnan(stats["acc_truncated"]) else 0.0
    acc_pred = (f_l * alpha_c + (1 - f_l) * alpha_t) * 100
    return {
        "F_L": f_l,
        "alpha_c": alpha_c,
        "alpha_t": alpha_t,
        "acc_pred": acc_pred,
    }


def print_summary_table(all_stats, all_theory):
    """Print a clear summary table suitable for copy-pasting into the paper."""
    sep = "=" * 100
    thin_sep = "-" * 100

    print(f"\n{sep}")
    print("MATH-500 Thinking Experiment Results — Qwen3-8B (v3)")
    print(sep)

    # ── Per-budget overview ──
    print(f"\n{'Budget':>7} | {'Acc(%)':>7} | {'N':>5} | {'AvgTok':>7} | {'NatStop%':>9} | "
          f"{'FinalAns%':>10} | {'AccNat%':>8} | {'AccTrunc%':>10} | {'Nnat':>5} | {'Ntrunc':>6}")
    print(thin_sep)
    for b in sorted(all_stats.keys()):
        s = all_stats[b]
        print(f"{b:>7} | {s['accuracy']:>7.2f} | {s['n_samples']:>5} | {s['avg_tokens']:>7.1f} | "
              f"{s['natural_stop_rate']:>9.2f} | {s['has_final_answer_rate']:>10.2f} | "
              f"{s['acc_natural']:>8.2f} | {s['acc_truncated']:>10.2f} | "
              f"{s['n_natural']:>5} | {s['n_truncated']:>6}")

    # ── Theory verification ──
    print(f"\n{sep}")
    print("Theory Verification: Acc_pred = F_L * alpha_c + (1 - F_L) * alpha_t")
    print(sep)
    print(f"{'Budget':>7} | {'F_L(b)':>7} | {'α_c':>7} | {'α_t':>7} | {'Acc_pred%':>10} | "
          f"{'Acc_obs%':>9} | {'Δ(pp)':>7}")
    print(thin_sep)
    for b in sorted(all_theory.keys()):
        t = all_theory[b]
        s = all_stats[b]
        delta = t["acc_pred"] - s["accuracy"]
        print(f"{b:>7} | {t['F_L']:>7.4f} | {t['alpha_c']:>7.4f} | {t['alpha_t']:>7.4f} | "
              f"{t['acc_pred']:>10.2f} | {s['accuracy']:>9.2f} | {delta:>+7.2f}")

    # ── Comparison with paper claims ──
    print(f"\n{sep}")
    print("Comparison with Paper Claims (think@budget)")
    print(sep)
    print(f"{'Budget':>7} | {'Claimed%':>9} | {'Observed%':>10} | {'Δ(pp)':>7} | {'Match?':>7}")
    print(thin_sep)
    for b in sorted(PAPER_CLAIMS.keys()):
        claimed = PAPER_CLAIMS[b]
        if b in all_stats:
            observed = all_stats[b]["accuracy"]
            delta = observed - claimed
            match = "✓" if abs(delta) < 3.0 else "✗"
            print(f"{b:>7} | {claimed:>9.1f} | {observed:>10.2f} | {delta:>+7.2f} | {match:>7}")
        else:
            print(f"{b:>7} | {claimed:>9.1f} | {'N/A':>10} | {'—':>7} | {'—':>7}")

    print(f"\n{sep}")
    print("NOTE: Match criterion = |Δ| < 3.0 pp")
    print(sep)


def save_results(all_stats, all_theory, output_path):
    """Save analysis results as JSON."""
    result = {
        "model": "Qwen3-8B",
        "benchmark": "MATH-500",
        "version": "v3",
        "per_budget": {},
        "paper_claims": PAPER_CLAIMS,
    }
    for b in sorted(all_stats.keys()):
        s = all_stats[b]
        t = all_theory.get(b, {})
        entry = {**s, **{f"theory_{k}": v for k, v in t.items()}}
        # Convert NaN to None for JSON serialization
        for k, v in entry.items():
            if isinstance(v, float) and np.isnan(v):
                entry[k] = None
        result["per_budget"][str(b)] = entry

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved analysis JSON to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MATH-500 thinking v3 results for Qwen3-8B"
    )
    parser.add_argument(
        "--results_dir",
        default="results/math500_thinking_v3/",
        help="Directory containing JSON result files (default: results/math500_thinking_v3/)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <results_dir>/analysis_summary.json)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results_dir, "analysis_summary.json")

    log.info("Results dir: %s", args.results_dir)

    # Load data
    by_budget = load_json_results(args.results_dir)

    # Compute per-budget statistics
    all_stats = {}
    all_theory = {}
    for budget in sorted(by_budget.keys()):
        samples = by_budget[budget]
        stats = compute_budget_stats(samples)
        if stats is None:
            log.warning("No samples for budget %d, skipping", budget)
            continue
        all_stats[budget] = stats
        all_theory[budget] = compute_theory_prediction(stats)

    if not all_stats:
        log.error("No valid budget data found. Exiting.")
        sys.exit(1)

    # Print summary
    print_summary_table(all_stats, all_theory)

    # Save JSON
    save_results(all_stats, all_theory, args.output)

    log.info("Done.")


if __name__ == "__main__":
    main()
