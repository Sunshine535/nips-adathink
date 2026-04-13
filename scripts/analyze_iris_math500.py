#!/usr/bin/env python3
"""Analyze IRIS MATH-500 results from checkpoint/final JSON files."""
import json
import sys
import numpy as np
from collections import Counter

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI."""
    if n == 0:
        return 0, 0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0, center - spread)*100, min(1, center + spread)*100

def analyze(path):
    with open(path) as f:
        d = json.load(f)
    
    results = d.get("iris_results", d.get("results", []))
    n = len(results)
    if n == 0:
        print("No results found!")
        return
    
    # Overall accuracy
    correct = sum(1 for r in results if r.get("correct", False))
    lo, hi = wilson_ci(correct, n)
    print(f"\n=== IRIS MATH-500 ({n} samples) ===")
    print(f"Overall: {correct}/{n} = {correct/n*100:.1f}% [95% CI: {lo:.1f}, {hi:.1f}]")
    
    # Stage distribution
    stages = Counter(r.get("final_stage", "?") for r in results)
    print(f"\nStage distribution: {dict(sorted(stages.items()))}")
    
    # Per-stage accuracy
    for stage in sorted(stages.keys()):
        sr = [r for r in results if r.get("final_stage", "?") == stage]
        sc = sum(1 for r in sr if r.get("correct", False))
        slo, shi = wilson_ci(sc, len(sr))
        print(f"  Stage {stage}: {sc}/{len(sr)} = {sc/len(sr)*100:.1f}% [{slo:.1f}, {shi:.1f}]")
    
    # Escalated-only
    esc = [r for r in results if r.get("final_stage", 0) > 1]
    if esc:
        esc_c = sum(1 for r in esc if r.get("correct", False))
        elo, ehi = wilson_ci(esc_c, len(esc))
        print(f"\nEscalated only: {esc_c}/{len(esc)} = {esc_c/len(esc)*100:.1f}% [{elo:.1f}, {ehi:.1f}]")
    
    # Stop reasons
    reasons = Counter(r.get("stop_reason", "unknown") for r in results)
    print(f"\nStop reasons: {dict(sorted(reasons.items()))}")
    
    # Avg tokens
    avg_tok = np.mean([r.get("tokens_total", 0) for r in results])
    print(f"\nAvg tokens: {avg_tok:.0f}")
    
    # Comparison points
    print(f"\n=== Key comparisons ===")
    print(f"IRIS ({n}): {correct/n*100:.1f}% [{lo:.1f}, {hi:.1f}]")
    print(f"MRSD pilot (200): 61.0% [54.1, 67.5]")
    print(f"Nothink@1024 pilot (200): 69.5% [62.8, 75.5]")
    print(f"Nothink@1024 full (500): 59.8%")
    print(f"TOWN@1024 pilot (200): 69.5%")
    
    if correct/n > 0.695:
        print(f"\n*** IRIS BEATS pilot nothink@1024 (69.5%) ***")
    if correct/n > 0.598:
        print(f"*** IRIS BEATS full-scale nothink@1024 (59.8%) ***")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results/iris_math500/checkpoint_iris_50.json"
    analyze(path)
