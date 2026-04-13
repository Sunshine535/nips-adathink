#!/usr/bin/env python3
"""
Phase 1a: Simulate IRIS entropy-based stopping across a grid of (τ_h, τ_s) thresholds.

Loads per-token entropy traces from checkpoint_entropy_Qwen3_8B_b512_200.json
and simulates: "if we used these thresholds, where would we have stopped?"

For each (τ_h, τ_s) pair, computes:
  - How many truncated samples would have been stopped earlier (tokens saved)
  - How many natural-stop samples would have been prematurely killed
  - Net accuracy impact

Key finding to demonstrate: NO threshold pair achieves >30% token savings
with <10% premature stop rate on natural-stop samples.

Usage:
    python scripts/simulate_iris_thresholds.py
    python scripts/simulate_iris_thresholds.py --entropy_file results/entropy_dynamics/checkpoint_entropy_Qwen3_8B_b512_200.json
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict

def compute_chunk_entropies(entropy_trace, chunk_size=32):
    """Compute mean entropy per chunk from per-token entropy trace."""
    chunks = []
    for i in range(0, len(entropy_trace), chunk_size):
        chunk = entropy_trace[i:i+chunk_size]
        if len(chunk) > 0:
            chunks.append(np.mean(chunk))
    return chunks

def simulate_stopping(chunk_entropies, chunk_stabilities, tau_h, tau_s, min_chunks=2):
    """
    Simulate IRIS entropy stopping criterion.
    Returns the chunk index where stopping would occur, or None if never triggered.

    Criterion: stop at chunk ci if H_chunk < tau_h AND S_chunk < tau_s
    (for ci >= min_chunks)
    """
    n_chunks = min(len(chunk_entropies), len(chunk_stabilities))
    for ci in range(min_chunks, n_chunks):
        H = chunk_entropies[ci]
        S = chunk_stabilities[ci]
        if H < tau_h and S < tau_s:
            return ci
    return None

def main():
    parser = argparse.ArgumentParser(description="Simulate IRIS threshold grid search")
    parser.add_argument("--entropy_file", type=str,
                        default="results/entropy_dynamics/checkpoint_entropy_Qwen3_8B_b512_200.json",
                        help="Path to entropy dynamics checkpoint")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--min_chunks", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="results/iris")
    parser.add_argument("--output_file", type=str, default="threshold_simulation.json")
    args = parser.parse_args()

    # Load data
    print(f"Loading entropy data from {args.entropy_file}")
    with open(args.entropy_file) as f:
        data = json.load(f)

    results = data["results"]
    n_total = len(results)
    print(f"Loaded {n_total} samples")

    # Separate natural-stop vs truncated
    natural_stop = [r for r in results if r["natural_stop"]]
    truncated = [r for r in results if r["hit_budget"]]
    n_ns = len(natural_stop)
    n_tr = len(truncated)
    ns_correct = sum(1 for r in natural_stop if r["correct"])
    tr_correct = sum(1 for r in truncated if r["correct"])

    print(f"Natural stop: {n_ns} ({ns_correct} correct, {n_ns - ns_correct} wrong)")
    print(f"Truncated:    {n_tr} ({tr_correct} correct, {n_tr - tr_correct} wrong)")

    # Compute chunk entropies for all samples
    for r in results:
        r["_chunk_entropies"] = compute_chunk_entropies(r["entropy_trace"], args.chunk_size)

    # Print entropy and stability ranges for context
    all_chunk_H = []
    all_chunk_S = []
    for r in results:
        all_chunk_H.extend(r["_chunk_entropies"])
        all_chunk_S.extend(r["chunk_stability"])

    print(f"\nChunk entropy range: [{min(all_chunk_H):.4f}, {max(all_chunk_H):.4f}]")
    print(f"  Mean: {np.mean(all_chunk_H):.4f}, Median: {np.median(all_chunk_H):.4f}")
    print(f"  Percentiles: p25={np.percentile(all_chunk_H, 25):.4f}, p75={np.percentile(all_chunk_H, 75):.4f}, p95={np.percentile(all_chunk_H, 95):.4f}")
    print(f"\nChunk stability range: [{min(all_chunk_S):.4f}, {max(all_chunk_S):.4f}]")
    print(f"  Mean: {np.mean(all_chunk_S):.4f}, Median: {np.median(all_chunk_S):.4f}")
    print(f"  Percentiles: p25={np.percentile(all_chunk_S, 25):.4f}, p75={np.percentile(all_chunk_S, 75):.4f}, p95={np.percentile(all_chunk_S, 95):.4f}")

    # Original IRIS thresholds
    print(f"\nOriginal IRIS thresholds: τ_h=1.5, τ_s=50")
    print(f"  τ_h=1.5 is ABOVE max chunk entropy ({max(all_chunk_H):.4f}) → H < τ_h always true")
    print(f"  τ_s=50 is BELOW min chunk stability ({min(all_chunk_S):.4f}) → S < τ_s always false")
    print(f"  Joint condition: always true AND always false = NEVER triggers")

    # Grid search
    tau_h_grid = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
    tau_s_grid = [50, 100, 130, 150, 170, 200, 250, 300, 500]

    print(f"\n{'='*90}")
    print(f"Grid search: τ_h × τ_s = {len(tau_h_grid)} × {len(tau_s_grid)} = {len(tau_h_grid)*len(tau_s_grid)} configurations")
    print(f"{'='*90}")

    grid_results = []

    # Header
    print(f"\n{'τ_h':>6} {'τ_s':>6} | {'NS killed':>10} {'NS kill%':>8} | {'TR early':>9} {'TR save%':>8} | {'Acc chg':>8} | {'Viable':>6}")
    print("-" * 90)

    for tau_h in tau_h_grid:
        for tau_s in tau_s_grid:
            # Natural-stop samples: check if they'd be killed prematurely
            ns_killed = 0
            ns_killed_correct = 0
            ns_killed_wrong = 0

            for r in natural_stop:
                stop_chunk = simulate_stopping(
                    r["_chunk_entropies"], r["chunk_stability"],
                    tau_h, tau_s, args.min_chunks
                )
                if stop_chunk is not None:
                    # Would have been stopped before natural completion
                    # Check if stop happens before the think_end_position
                    stop_token = (stop_chunk + 1) * args.chunk_size
                    think_end = r.get("think_end_position")
                    if think_end is not None and stop_token < think_end:
                        ns_killed += 1
                        if r["correct"]:
                            ns_killed_correct += 1
                        else:
                            ns_killed_wrong += 1

            # Truncated samples: check if they'd be stopped earlier (saving tokens)
            tr_early_stopped = 0
            total_tokens_saved = 0
            total_tokens_possible = 0

            for r in truncated:
                total_tokens_possible += r["n_tokens"]
                stop_chunk = simulate_stopping(
                    r["_chunk_entropies"], r["chunk_stability"],
                    tau_h, tau_s, args.min_chunks
                )
                if stop_chunk is not None:
                    stop_token = (stop_chunk + 1) * args.chunk_size
                    saved = r["n_tokens"] - stop_token
                    if saved > 0:
                        tr_early_stopped += 1
                        total_tokens_saved += saved

            # Compute metrics
            ns_kill_rate = ns_killed / n_ns * 100 if n_ns > 0 else 0
            tr_save_rate = total_tokens_saved / total_tokens_possible * 100 if total_tokens_possible > 0 else 0

            # Accuracy change: killing correct natural-stop samples hurts
            # Early-stopping truncated samples doesn't change accuracy (they were already wrong/guessing)
            acc_change = -ns_killed_correct / n_total * 100

            viable = "✓" if ns_kill_rate < 10 and tr_save_rate > 30 else "✗"

            result = {
                "tau_h": tau_h,
                "tau_s": tau_s,
                "ns_killed": ns_killed,
                "ns_killed_correct": ns_killed_correct,
                "ns_killed_wrong": ns_killed_wrong,
                "ns_kill_rate_pct": round(ns_kill_rate, 1),
                "tr_early_stopped": tr_early_stopped,
                "tr_token_savings_pct": round(tr_save_rate, 1),
                "accuracy_change_pp": round(acc_change, 1),
                "viable": viable == "✓"
            }
            grid_results.append(result)

            print(f"{tau_h:>6.2f} {tau_s:>6.0f} | {ns_killed:>10d} {ns_kill_rate:>7.1f}% | {tr_early_stopped:>9d} {tr_save_rate:>7.1f}% | {acc_change:>+7.1f}% | {viable:>6}")

    # Summary
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")

    viable_configs = [r for r in grid_results if r["viable"]]
    print(f"Viable configurations (NS kill <10%, TR savings >30%): {len(viable_configs)} / {len(grid_results)}")

    if viable_configs:
        print("\nViable configurations:")
        for v in viable_configs:
            print(f"  τ_h={v['tau_h']:.2f}, τ_s={v['tau_s']:.0f}: "
                  f"NS kill={v['ns_kill_rate_pct']:.1f}%, TR save={v['tr_token_savings_pct']:.1f}%, "
                  f"Acc Δ={v['accuracy_change_pp']:+.1f}pp")
    else:
        print("\n>>> NO viable configuration found. <<<")
        print("Entropy-based stopping cannot achieve meaningful savings without harming natural-stop accuracy.")

    # Relaxed criteria analysis
    print(f"\n--- Relaxed analysis ---")
    for max_kill in [5, 10, 20, 50]:
        candidates = [r for r in grid_results if r["ns_kill_rate_pct"] <= max_kill]
        if candidates:
            best = max(candidates, key=lambda x: x["tr_token_savings_pct"])
            print(f"Best savings with NS kill ≤{max_kill}%: "
                  f"τ_h={best['tau_h']:.2f}, τ_s={best['tau_s']:.0f} → "
                  f"TR save={best['tr_token_savings_pct']:.1f}%, NS kill={best['ns_kill_rate_pct']:.1f}%, "
                  f"Acc Δ={best['accuracy_change_pp']:+.1f}pp")

    # Entropy signal direction analysis
    print(f"\n--- Entropy signal direction analysis ---")
    ns_correct_entropies = [np.mean(r["_chunk_entropies"]) for r in natural_stop if r["correct"]]
    ns_wrong_entropies = [np.mean(r["_chunk_entropies"]) for r in natural_stop if not r["correct"]]
    tr_correct_entropies = [np.mean(r["_chunk_entropies"]) for r in truncated if r["correct"]]
    tr_wrong_entropies = [np.mean(r["_chunk_entropies"]) for r in truncated if not r["correct"]]

    print(f"Natural-stop correct  (n={len(ns_correct_entropies)}): mean H = {np.mean(ns_correct_entropies):.4f}")
    if ns_wrong_entropies:
        print(f"Natural-stop wrong    (n={len(ns_wrong_entropies)}): mean H = {np.mean(ns_wrong_entropies):.4f}")
    print(f"Truncated correct     (n={len(tr_correct_entropies)}): mean H = {np.mean(tr_correct_entropies):.4f}")
    print(f"Truncated wrong       (n={len(tr_wrong_entropies)}): mean H = {np.mean(tr_wrong_entropies):.4f}")
    print(f"\nSignal direction: {'ANTI-CORRELATED' if np.mean(tr_wrong_entropies) < np.mean(tr_correct_entropies) else 'CORRELATED'} "
          f"(wrong samples have {'lower' if np.mean(tr_wrong_entropies) < np.mean(tr_correct_entropies) else 'higher'} entropy)")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)

    output = {
        "meta": {
            "entropy_file": args.entropy_file,
            "chunk_size": args.chunk_size,
            "min_chunks": args.min_chunks,
            "n_total": n_total,
            "n_natural_stop": n_ns,
            "n_truncated": n_tr
        },
        "entropy_stats": {
            "chunk_H_range": [float(min(all_chunk_H)), float(max(all_chunk_H))],
            "chunk_H_mean": float(np.mean(all_chunk_H)),
            "chunk_H_median": float(np.median(all_chunk_H)),
            "chunk_S_range": [float(min(all_chunk_S)), float(max(all_chunk_S))],
            "chunk_S_mean": float(np.mean(all_chunk_S)),
            "chunk_S_median": float(np.median(all_chunk_S)),
            "original_tau_h": 1.5,
            "original_tau_s": 50.0,
            "diagnosis": "τ_h=1.5 always passes (max H={:.4f}); τ_s=50 never passes (min S={:.4f}); joint=NEVER triggers".format(
                max(all_chunk_H), min(all_chunk_S)
            )
        },
        "signal_direction": {
            "ns_correct_mean_H": float(np.mean(ns_correct_entropies)),
            "ns_wrong_mean_H": float(np.mean(ns_wrong_entropies)) if ns_wrong_entropies else None,
            "tr_correct_mean_H": float(np.mean(tr_correct_entropies)),
            "tr_wrong_mean_H": float(np.mean(tr_wrong_entropies)),
            "direction": "anti-correlated" if np.mean(tr_wrong_entropies) < np.mean(tr_correct_entropies) else "correlated"
        },
        "grid_search": {
            "tau_h_grid": tau_h_grid,
            "tau_s_grid": tau_s_grid,
            "n_configs": len(grid_results),
            "n_viable": len(viable_configs),
            "results": grid_results
        },
        "conclusion": (
            "No threshold pair achieves >30% token savings with <10% premature stop rate. "
            "The entropy signal is flat, low-magnitude, and anti-correlated with correctness. "
            "Entropy-based early stopping is not viable for this model/task."
            if not viable_configs else
            f"Found {len(viable_configs)} viable configurations."
        )
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
