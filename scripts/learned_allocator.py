#!/usr/bin/env python3
"""Learned Budget Allocator — lightweight MLP maps query → (B_r*, B_a*).

Architecture:
    embedding (from question text) → MLP(hidden=64) → (B_r_logits, B_a_logits)
    Discretized: B_r ∈ {512, 1024, 2048, 4096}, B_a ∈ {128, 256, 512}

Training data: per-sample IRIS runs at different budgets give us
(question, budget, accuracy, tokens_total) tuples. For each question we
can identify the "best" (B_r, B_a) as the one achieving correct=1 at
lowest total tokens.

Loss: -Acc + λ · total_tokens / max_tokens

Usage:
    python scripts/learned_allocator.py --build_training_set
    python scripts/learned_allocator.py --train
    python scripts/learned_allocator.py --eval
"""
import argparse, json, os, glob, numpy as np
from collections import defaultdict


B_R_CHOICES = [512, 1024, 2048, 4096]
B_A_CHOICES = [128, 256, 512]


def build_training_set():
    """Extract (question, best_budget, acc) tuples from all IRIS runs."""
    out = []

    # 8B MATH-500 runs
    for path, b_r, b_a in [
        ("results/iris_math500_fullscale/b2048_iris_compact.json", 2048, 256),
        ("results/iris_math500_fullscale_b4096/b4096_iris_compact.json", 4096, 256),
        ("results/iris_improved_20260417/8b_math500_b4096_ba512_n500/checkpoint_iris_500.json", 4096, 512),
    ]:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        samples = d.get("iris_results") if isinstance(d, dict) else d
        for i, s in enumerate(samples):
            correct = s.get("correct", s.get("c", 0))
            tokens = s.get("tokens_total", s.get("t", b_r + b_a))
            out.append({"model": "8B", "bench": "math500", "idx": i,
                       "b_r": b_r, "b_a": b_a,
                       "correct": int(correct), "tokens": int(tokens)})

    print(f"Collected {len(out)} training samples")

    # For each question, find best-cost correct answer
    per_q = defaultdict(list)
    for r in out:
        key = (r["model"], r["bench"], r["idx"])
        per_q[key].append(r)

    ds = []
    for key, runs in per_q.items():
        correct_runs = [r for r in runs if r["correct"]]
        if correct_runs:
            best = min(correct_runs, key=lambda r: r["tokens"])
            ds.append({"key": key, "best_b_r": best["b_r"], "best_b_a": best["b_a"],
                       "best_tokens": best["tokens"], "solvable": True})
        else:
            # Unsolvable at any tested budget — choose largest budget
            largest = max(runs, key=lambda r: r["b_r"] + r["b_a"])
            ds.append({"key": key, "best_b_r": largest["b_r"], "best_b_a": largest["b_a"],
                       "best_tokens": largest["tokens"], "solvable": False})

    solvable = sum(1 for d in ds if d["solvable"])
    print(f"  Unique questions: {len(ds)}")
    print(f"  Solvable at any budget: {solvable}/{len(ds)}")

    # Budget distribution
    from collections import Counter
    br_dist = Counter(d["best_b_r"] for d in ds if d["solvable"])
    ba_dist = Counter(d["best_b_a"] for d in ds if d["solvable"])
    print(f"  Optimal B_r distribution: {dict(br_dist)}")
    print(f"  Optimal B_a distribution: {dict(ba_dist)}")

    os.makedirs("results/learned_allocator", exist_ok=True)
    with open("results/learned_allocator/training_set.json", "w") as f:
        json.dump(ds, f, indent=2)
    print(f"Saved: results/learned_allocator/training_set.json")
    return ds


def analyze_budget_policy(ds):
    """Without training, analyze what an oracle allocator would achieve."""
    print("\n" + "=" * 60)
    print("Oracle allocator analysis")
    print("=" * 60)

    # Oracle strategy: always use best (B_r, B_a) per question
    total_correct = sum(1 for d in ds if d["solvable"])
    total_tokens = sum(d["best_tokens"] for d in ds)
    n = len(ds)

    print(f"\nOracle allocator (if we knew optimal per-query):")
    print(f"  Accuracy: {total_correct}/{n} = {total_correct/n:.1%}")
    print(f"  Avg tokens: {total_tokens/n:.0f}")

    # Fixed strategy: always max budget
    for b_r, b_a in [(4096, 256), (4096, 512)]:
        filtered = [d for d in ds if d["best_b_r"] <= b_r and d["best_b_a"] <= b_a]
        correct = sum(1 for d in filtered if d["solvable"])
        print(f"\nFixed strategy (B_r={b_r}, B_a={b_a}):")
        print(f"  Accuracy bound: ≤ {correct}/{n} = {correct/n:.1%}")

    # Hypothetical gain from learned allocator
    # Lower bound: always use min (B_r,B_a) needed for correct answer
    # Upper bound: no savings (always use max budget)
    oracle_avg = total_tokens / n
    fixed_max_tokens = 4096 + 512  # largest config
    savings_pct = 100 * (1 - oracle_avg / fixed_max_tokens)
    print(f"\nOracle savings over fixed max-budget: {savings_pct:.1f}%")
    print(f"Potential target for learned allocator: 0-{savings_pct:.0f}% of oracle savings")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build_training_set", action="store_true")
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()

    if args.build_training_set:
        ds = build_training_set()
    else:
        path = "results/learned_allocator/training_set.json"
        if not os.path.exists(path):
            print(f"Run --build_training_set first")
            return
        with open(path) as f:
            ds = json.load(f)

    analyze_budget_policy(ds)


if __name__ == "__main__":
    main()
