#!/usr/bin/env python3
"""Analyze DeepSeek-R1-Distill-Llama-8B results and compute template controller performance.

Run after deploy_deepseek_full.sh completes:
    python scripts/analyze_deepseek_results.py --results_dir results/deepseek
"""

import argparse
import csv
import json
import os
import glob
import numpy as np
from collections import defaultdict


def load_per_sample_csv(path):
    """Load per-sample CSV and return list of dicts."""
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_fixed_budget_accuracy(rows, budget_col_prefix="correct_at_"):
    """Compute accuracy at each fixed budget from per-sample data."""
    budgets = []
    for key in rows[0].keys():
        if key.startswith(budget_col_prefix):
            budgets.append(int(key.replace(budget_col_prefix, "")))
    budgets.sort()
    
    results = {}
    for b in budgets:
        col = f"{budget_col_prefix}{b}"
        corrects = [int(float(r[col])) for r in rows if col in r]
        acc = np.mean(corrects) if corrects else 0.0
        results[b] = {"accuracy": acc, "n": len(corrects)}
    return results


def bootstrap_ci(data, n_boot=10000, ci=0.95, seed=20260228):
    """Compute bootstrap CI for mean."""
    rng = np.random.RandomState(seed)
    data = np.array(data)
    n = len(data)
    boot_means = np.array([np.mean(rng.choice(data, n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [100*alpha, 100*(1-alpha)])
    return np.mean(data), lo, hi


def paired_bootstrap_ci(deltas, n_boot=10000, ci=0.95, seed=20260228):
    """Compute paired bootstrap CI for mean of deltas."""
    return bootstrap_ci(deltas, n_boot, ci, seed)


def analyze_results(results_dir, benchmark="gsm8k"):
    """Analyze full-dataset results."""
    pattern = os.path.join(results_dir, f"per_sample_{benchmark}_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        pattern2 = os.path.join(results_dir, f"per_sample_*{benchmark}*.csv")
        files = glob.glob(pattern2)
    
    if not files:
        print(f"No per_sample CSV files found for {benchmark} in {results_dir}")
        return None
    
    print(f"\nFound {len(files)} per-sample files for {benchmark}:")
    for f in sorted(files):
        print(f"  {os.path.basename(f)}")
    
    all_results = {}
    for fpath in sorted(files):
        rows = load_per_sample_csv(fpath)
        budgets_acc = compute_fixed_budget_accuracy(rows)
        fname = os.path.basename(fpath)
        all_results[fname] = {
            "n_samples": len(rows),
            "budgets": budgets_acc
        }
        
        print(f"\n{fname} (n={len(rows)}):")
        for b, info in sorted(budgets_acc.items()):
            print(f"  Budget {b}: acc={info['accuracy']:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/deepseek")
    parser.add_argument("--benchmark", default="gsm8k")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"DeepSeek-R1-Distill-Llama-8B Results Analysis")
    print(f"Results dir: {args.results_dir}")
    print("=" * 60)
    
    for bench in ["gsm8k", "math500"]:
        results = analyze_results(args.results_dir, bench)
        if results:
            summary = {
                "benchmark": bench,
                "model": "DeepSeek-R1-Distill-Llama-8B",
                "results": {}
            }
            for fname, data in results.items():
                summary["results"][fname] = data
            
            out_path = os.path.join(args.results_dir, f"deepseek_{bench}_analysis.json")
            with open(out_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nSaved analysis to {out_path}")
    
    print("\n" + "=" * 60)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
