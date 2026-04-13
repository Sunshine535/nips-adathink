#!/usr/bin/env python3
"""
Analyze full-scale (n=500) IRIS + TOWN results for both B2048 and B4096.
Outputs paper-ready statistics: Wilson CIs, McNemar tests, stage breakdowns.

Usage:
    python scripts/analyze_fullscale_iris.py \
        --b2048 results/iris_math500_fullscale/checkpoint_iris_500.json \
        --b4096 results/iris_math500_fullscale_b4096/checkpoint_iris_500.json
"""
import json
import argparse
import numpy as np
from collections import Counter
from scipy import stats


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI, returns (lower%, upper%)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(max(0, center - spread) * 100, 1), round(min(1, center + spread) * 100, 1)


def mcnemar_test(correct_a, correct_b):
    """McNemar test on paired binary outcomes. Returns (chi2, p-value)."""
    assert len(correct_a) == len(correct_b)
    # b: A correct, B wrong; c: A wrong, B correct
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if not a and bb)
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return round(chi2, 2), round(p, 6)


def analyze_iris(data, label):
    """Analyze IRIS results from checkpoint JSON."""
    results = data.get("iris_results", data.get("results", []))
    n = len(results)
    correct = sum(1 for r in results if r.get("correct", 0))
    lo, hi = wilson_ci(correct, n)

    print(f"\n{'='*60}")
    print(f"IRIS {label} (n={n})")
    print(f"{'='*60}")
    print(f"  Accuracy: {correct}/{n} = {correct/n*100:.1f}%  CI [{lo}, {hi}]")

    # Stage breakdown
    for stage in [1, 2, 3]:
        sr = [r for r in results if r.get("final_stage", 0) == stage]
        if not sr:
            continue
        sc = sum(1 for r in sr if r.get("correct", 0))
        slo, shi = wilson_ci(sc, len(sr))
        pct = len(sr) / n * 100
        print(f"  Stage {stage}: {len(sr)} ({pct:.1f}%), acc={sc}/{len(sr)} = {sc/len(sr)*100:.1f}% [{slo}, {shi}]")

    # S2 natural stop rate (among escalated)
    s2 = [r for r in results if r.get("final_stage", 0) == 2]
    s3 = [r for r in results if r.get("final_stage", 0) == 3]
    escalated = len(s2) + len(s3)
    if escalated > 0:
        print(f"  S2 natural stop: {len(s2)}/{escalated} = {len(s2)/escalated*100:.1f}%")

    # Avg tokens
    avg_tok = np.mean([r.get("tokens_total", 0) for r in results])
    print(f"  Avg tokens: {avg_tok:.0f}")

    # First 200 samples (pilot reproduction)
    first200 = results[:200]
    c200 = sum(1 for r in first200 if r.get("correct", 0))
    lo200, hi200 = wilson_ci(c200, 200)
    print(f"  First 200 (pilot check): {c200}/200 = {c200/200*100:.1f}% [{lo200}, {hi200}]")

    return results


def analyze_town(data, label):
    """Analyze TOWN results from the same JSON."""
    town = data.get("per_sample_town", [])
    if not town:
        # Try alternative key
        town = data.get("town_results_list", data.get("town_per_sample", []))
    if not town:
        print(f"\n  TOWN {label}: No per-sample results found")
        # Check summary
        ts = data.get("town_results", {})
        if isinstance(ts, dict) and "accuracy" in ts:
            print(f"  TOWN summary: acc={ts['accuracy']*100:.1f}%, avg_tok={ts.get('avg_tokens', '?')}")
        return None

    n = len(town)
    correct = sum(1 for r in town if r.get("correct", 0))
    lo, hi = wilson_ci(correct, n)
    print(f"\n  TOWN {label} (n={n})")
    print(f"  Accuracy: {correct}/{n} = {correct/n*100:.1f}%  CI [{lo}, {hi}]")

    avg_tok = np.mean([r.get("tokens_total", 0) for r in town])
    print(f"  Avg tokens: {avg_tok:.0f}")

    # First 200
    first200 = town[:200]
    c200 = sum(1 for r in first200 if r.get("correct", 0))
    lo200, hi200 = wilson_ci(c200, 200)
    print(f"  First 200 (pilot check): {c200}/200 = {c200/200*100:.1f}% [{lo200}, {hi200}]")

    return town


def compare_paired(iris_results, town_results, label):
    """Paired comparison: IRIS vs TOWN on same samples."""
    if not town_results:
        return
    n = min(len(iris_results), len(town_results))
    iris_c = [r.get("correct", 0) for r in iris_results[:n]]
    town_c = [r.get("correct", 0) for r in town_results[:n]]

    iris_acc = sum(iris_c) / n
    town_acc = sum(town_c) / n
    gap = (iris_acc - town_acc) * 100

    chi2, p = mcnemar_test(iris_c, town_c)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"\n  IRIS vs TOWN {label} (n={n}):")
    print(f"    IRIS: {iris_acc*100:.1f}%  vs  TOWN: {town_acc*100:.1f}%")
    print(f"    Gap: {'+' if gap>0 else ''}{gap:.1f}pp  McNemar chi2={chi2}, p={p} {sig}")

    # Disagreement analysis
    iris_only = sum(1 for a, b in zip(iris_c, town_c) if a and not b)
    town_only = sum(1 for a, b in zip(iris_c, town_c) if not a and b)
    both_correct = sum(1 for a, b in zip(iris_c, town_c) if a and b)
    both_wrong = sum(1 for a, b in zip(iris_c, town_c) if not a and not b)
    print(f"    Both correct: {both_correct}, Both wrong: {both_wrong}")
    print(f"    IRIS-only: {iris_only}, TOWN-only: {town_only}")


def latex_table_row(label, correct, n, avg_tok):
    """Generate LaTeX table row."""
    lo, hi = wilson_ci(correct, n)
    acc = correct / n * 100
    return f"  {label} & {acc:.1f}\\% \\ci{{{lo}}}{{{hi}}} & {avg_tok:.0f} \\\\"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2048", type=str, help="Path to B2048 final JSON")
    parser.add_argument("--b4096", type=str, help="Path to B4096 final JSON")
    args = parser.parse_args()

    configs = []
    if args.b2048:
        with open(args.b2048) as f:
            configs.append(("B_think=2048", json.load(f)))
    if args.b4096:
        with open(args.b4096) as f:
            configs.append(("B_think=4096", json.load(f)))

    if not configs:
        print("No input files provided. Use --b2048 and/or --b4096.")
        return

    all_iris = {}
    all_town = {}

    for label, data in configs:
        iris_r = analyze_iris(data, label)
        town_r = analyze_town(data, label)
        all_iris[label] = iris_r
        all_town[label] = town_r
        if iris_r and town_r:
            compare_paired(iris_r, town_r, label)

    # Cross-budget comparison
    if len(all_iris) == 2:
        labels = list(all_iris.keys())
        print(f"\n{'='*60}")
        print(f"Cross-budget: {labels[0]} vs {labels[1]}")
        print(f"{'='*60}")
        for label in labels:
            r = all_iris[label]
            c = sum(1 for x in r if x.get("correct", 0))
            n = len(r)
            lo, hi = wilson_ci(c, n)
            print(f"  {label}: {c}/{n} = {c/n*100:.1f}% [{lo}, {hi}]")

    # LaTeX output
    print(f"\n{'='*60}")
    print("LaTeX table rows (for paper)")
    print(f"{'='*60}")
    for label, data in configs:
        r = all_iris[label]
        c = sum(1 for x in r if x.get("correct", 0))
        avg = np.mean([x.get("tokens_total", 0) for x in r])
        print(latex_table_row(f"IRIS ({label}, $n{{=}}500$)", c, len(r), avg))

        if all_town.get(label):
            tr = all_town[label]
            tc = sum(1 for x in tr if x.get("correct", 0))
            tavg = np.mean([x.get("tokens_total", 0) for x in tr])
            print(latex_table_row(f"TOWN ({label}, $n{{=}}500$)", tc, len(tr), tavg))

    # Reference baselines
    print("\n  % Reference baselines")
    print("  nothink@1024 ($n{=}500$) & 59.8\\% & 1024 \\\\")
    print("  think@1024 ($n{=}500$) & 18.0\\% & 1024 \\\\")


if __name__ == "__main__":
    main()
