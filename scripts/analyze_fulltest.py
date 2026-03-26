#!/usr/bin/env python3
"""Analyze full-test results: compute template controller performance + bootstrap CIs.

For full-test evaluation, we use k-fold cross-validation (k=10) on the full dataset
to train and evaluate the template controller, eliminating subset overlap entirely.
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np


def to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default

def to_int(v, default=0):
    try:
        return int(float(v))
    except Exception:
        return default

def detect_budgets(fieldnames):
    import re
    budgets = sorted(set(int(m.group(1)) for k in fieldnames for m in [re.match(r"fixed_(\d+)_correct", k)] if m))
    if not budgets:
        raise RuntimeError("No budget columns found.")
    return budgets

def utility(row, budget, lam, norm):
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lam * (t / norm)

def extract_features(row, budgets):
    b1 = budgets[0]
    has_answer = to_int(row.get(f"fixed_{b1}_has_final", 0))
    tokens_used = to_float(row.get(f"fixed_{b1}_tokens", 0.0))
    utilization = tokens_used / b1 if b1 > 0 else 0.0
    util_bin = 0 if utilization < 0.5 else (1 if utilization < 0.9 else 2)
    return (has_answer, util_bin)

def build_template(rows, budgets, lam, norm):
    cats = {}
    for r in rows:
        feat = extract_features(r, budgets)
        if feat not in cats:
            cats[feat] = {b: [] for b in budgets}
        for b in budgets:
            cats[feat][b].append(utility(r, b, lam, norm))

    mapping = {}
    for feat, bdata in cats.items():
        best_b = max(budgets, key=lambda b: np.mean(bdata[b]) if bdata[b] else -1e18)
        mapping[feat] = best_b
    
    all_utils = {b: [] for b in budgets}
    for r in rows:
        for b in budgets:
            all_utils[b].append(utility(r, b, lam, norm))
    default = max(budgets, key=lambda b: np.mean(all_utils[b]))
    return mapping, default

def evaluate_template(rows, budgets, mapping, default, lam, norm):
    b1 = budgets[0]
    results = []
    for r in rows:
        feat = extract_features(r, budgets)
        chosen = mapping.get(feat, default)
        c = to_int(r.get(f"fixed_{chosen}_correct", 0))
        t = to_float(r.get(f"fixed_{chosen}_tokens", 0.0))
        e2e_t = t + (to_float(r.get(f"fixed_{b1}_tokens", 0.0)) if chosen > b1 else 0)
        u = c - lam * (e2e_t / norm)
        results.append({
            "correct": c,
            "tokens_second_pass": t,
            "tokens_e2e": e2e_t,
            "utility": u,
            "chosen_budget": chosen,
            "question": r.get("question", "")[:50],
        })
    return results

def bootstrap_ci(deltas, n_boot=10000, alpha=0.05, seed=20260228):
    rng = np.random.RandomState(seed)
    deltas = np.array(deltas)
    n = len(deltas)
    boot_means = np.array([rng.choice(deltas, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(deltas.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Full-test per-sample CSV")
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--k_folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.csv}")

    budgets = detect_budgets(list(rows[0].keys()))
    norm = float(budgets[-1])
    b_mid = budgets[1] if len(budgets) >= 2 else budgets[0]
    print(f"Budgets: {budgets}, norm={norm}, mid-tier baseline: Fixed-{b_mid}")

    # Fixed baseline stats
    for b in budgets:
        acc = np.mean([to_int(r.get(f"fixed_{b}_correct", 0)) for r in rows])
        tok = np.mean([to_float(r.get(f"fixed_{b}_tokens", 0.0)) for r in rows])
        util = np.mean([utility(r, b, args.lam, norm) for r in rows])
        print(f"  Fixed-{b}: acc={acc:.4f}, tok={tok:.1f}, util={util:.4f}")

    # Oracle
    oracle_results = []
    for r in rows:
        best_b = max(budgets, key=lambda b: utility(r, b, args.lam, norm))
        c = to_int(r.get(f"fixed_{best_b}_correct", 0))
        t = to_float(r.get(f"fixed_{best_b}_tokens", 0.0))
        oracle_results.append({"correct": c, "tokens": t, "utility": utility(r, best_b, args.lam, norm)})
    oracle_acc = np.mean([x["correct"] for x in oracle_results])
    oracle_tok = np.mean([x["tokens"] for x in oracle_results])
    oracle_util = np.mean([x["utility"] for x in oracle_results])
    print(f"  Oracle: acc={oracle_acc:.4f}, tok={oracle_tok:.1f}, util={oracle_util:.4f}")

    # K-fold cross-validation template controller
    rng = random.Random(args.seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    fold_size = len(rows) // args.k_folds
    folds = []
    for i in range(args.k_folds):
        start = i * fold_size
        end = start + fold_size if i < args.k_folds - 1 else len(rows)
        folds.append(indices[start:end])

    all_template_results = []
    fold_summaries = []
    for i in range(args.k_folds):
        test_idx = set(folds[i])
        train_rows = [rows[j] for j in range(len(rows)) if j not in test_idx]
        test_rows = [rows[j] for j in folds[i]]

        mapping, default = build_template(train_rows, budgets, args.lam, norm)
        results = evaluate_template(test_rows, budgets, mapping, default, args.lam, norm)
        all_template_results.extend(results)

        acc = np.mean([x["correct"] for x in results])
        tok = np.mean([x["tokens_e2e"] for x in results])
        util = np.mean([x["utility"] for x in results])
        fold_summaries.append({"fold": i, "n": len(test_rows), "acc": acc, "tok": tok, "util": util})
        print(f"  Fold {i}: n={len(test_rows)}, acc={acc:.4f}, tok={tok:.1f}, util={util:.4f}")

    template_acc = np.mean([x["correct"] for x in all_template_results])
    template_tok = np.mean([x["tokens_e2e"] for x in all_template_results])
    template_util = np.mean([x["utility"] for x in all_template_results])
    print(f"\n=== Template Controller ({args.k_folds}-fold CV) ===")
    print(f"  Accuracy: {template_acc:.4f}")
    print(f"  End-to-end tokens: {template_tok:.1f}")
    print(f"  Utility (λ={args.lam}): {template_util:.4f}")

    # Budget allocation distribution
    alloc = {}
    for r in all_template_results:
        b = r["chosen_budget"]
        alloc[b] = alloc.get(b, 0) + 1
    total = sum(alloc.values())
    print(f"  Allocation: {', '.join(f'{b}={c/total:.1%}' for b, c in sorted(alloc.items()))}")

    # Bootstrap CIs vs mid-tier
    fixed_mid_acc = [to_int(r.get(f"fixed_{b_mid}_correct", 0)) for r in rows]
    fixed_mid_tok = [to_float(r.get(f"fixed_{b_mid}_tokens", 0.0)) for r in rows]

    reorder = []
    idx = 0
    for fold_idx in folds:
        for _ in fold_idx:
            reorder.append(all_template_results[idx])
            idx += 1

    ctrl_acc = [x["correct"] for x in all_template_results]
    ctrl_tok = [x["tokens_e2e"] for x in all_template_results]

    n = min(len(ctrl_acc), len(fixed_mid_acc))
    delta_acc = [ctrl_acc[i] - fixed_mid_acc[i] for i in range(n)]
    delta_tok = [ctrl_tok[i] - fixed_mid_tok[i] for i in range(n)]

    mean_da, lo_da, hi_da = bootstrap_ci(delta_acc)
    mean_dt, lo_dt, hi_dt = bootstrap_ci(delta_tok)

    print(f"\n=== Paired Deltas vs Fixed-{b_mid} ===")
    print(f"  ΔAcc:  {mean_da:+.4f} [{lo_da:+.4f}, {hi_da:+.4f}]")
    print(f"  ΔTok:  {mean_dt:+.1f} [{lo_dt:+.1f}, {hi_dt:+.1f}]")

    fixed_mid_util = [utility(rows[i], b_mid, args.lam, norm) for i in range(n)]
    ctrl_util = [all_template_results[i]["utility"] for i in range(n)]
    delta_util = [ctrl_util[i] - fixed_mid_util[i] for i in range(n)]
    mean_du, lo_du, hi_du = bootstrap_ci(delta_util)
    print(f"  ΔUtil: {mean_du:+.4f} [{lo_du:+.4f}, {hi_du:+.4f}]")

    # Save results
    outdir = os.path.dirname(args.csv)
    bench = os.path.basename(args.csv).split("_")[0] if "_" in os.path.basename(args.csv) else "unknown"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(outdir, f"fulltest_analysis_{bench}_{ts}.json")

    summary = {
        "csv": args.csv,
        "n_questions": len(rows),
        "budgets": budgets,
        "lambda": args.lam,
        "k_folds": args.k_folds,
        "fixed_baselines": {str(b): {
            "accuracy": float(np.mean([to_int(r.get(f"fixed_{b}_correct", 0)) for r in rows])),
            "avg_tokens": float(np.mean([to_float(r.get(f"fixed_{b}_tokens", 0.0)) for r in rows])),
        } for b in budgets},
        "oracle": {"accuracy": float(oracle_acc), "avg_tokens": float(oracle_tok), "utility": float(oracle_util)},
        "template_controller": {
            "accuracy": float(template_acc),
            "avg_tokens_e2e": float(template_tok),
            "utility": float(template_util),
            "allocation": {str(b): c/total for b, c in sorted(alloc.items())},
        },
        "paired_deltas_vs_mid": {
            "delta_acc": {"mean": float(mean_da), "ci_lo": float(lo_da), "ci_hi": float(hi_da)},
            "delta_tok": {"mean": float(mean_dt), "ci_lo": float(lo_dt), "ci_hi": float(hi_dt)},
            "delta_util": {"mean": float(mean_du), "ci_lo": float(lo_du), "ci_hi": float(hi_du)},
        },
        "fold_summaries": fold_summaries,
    }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
