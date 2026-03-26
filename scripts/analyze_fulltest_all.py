#!/usr/bin/env python3
"""Analyze full-test results across all benchmarks and models.

Reads per-sample CSVs from results/fulltest/, computes:
1. Fixed-budget accuracy at each tier
2. Template controller accuracy via K-fold cross-validation
3. Paired bootstrap CIs for accuracy and utility deltas
4. Comparison with subset-based estimates from the paper

Usage:
    python scripts/analyze_fulltest_all.py
"""

import csv
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

LAMBDA = 0.15
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260228
K_FOLDS = 10

SETTINGS = {
    "gsm8k_8b": {
        "budgets": [128, 256, 512],
        "model_tag": "Qwen3_8B",
    },
    "math500_8b": {
        "budgets": [512, 1024, 2048],
        "model_tag": "Qwen3_8B",
    },
    "bbh_8b": {
        "budgets": [256, 512, 1024],
        "model_tag": "Qwen3_8B",
    },
    "gsm8k_27b": {
        "budgets": [128, 256, 512],
        "model_tag": "Qwen3.5_27B",
    },
    "math500_27b": {
        "budgets": [2048, 4096, 8192],
        "model_tag": "Qwen3.5_27B",
    },
    "bbh_27b": {
        "budgets": [1024, 2048, 4096],
        "model_tag": "Qwen3.5_27B",
    },
}


def find_csv(results_dir: str, benchmark: str, model_tag: str) -> Optional[str]:
    """Find the most recent per-sample CSV for a benchmark/model combo."""
    candidates = []
    for f in os.listdir(results_dir):
        if f.startswith(f"per_sample_{benchmark}_{model_tag}") and f.endswith(".csv"):
            candidates.append(os.path.join(results_dir, f))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_per_sample(csv_path: str, budgets: List[int]) -> List[Dict]:
    """Load per-sample CSV and extract fixed-budget metrics."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {"idx": row.get("idx", ""), "question": row.get("question", "")}
            for b in budgets:
                entry[f"correct_{b}"] = int(float(row.get(f"fixed_{b}_correct", 0)))
                entry[f"tokens_{b}"] = float(row.get(f"fixed_{b}_tokens", b))
                entry[f"has_final_{b}"] = int(float(row.get(f"fixed_{b}_has_final", 0)))
            rows.append(entry)
    return rows


def extract_difficulty_features(row: Dict, b1: int) -> Tuple[int, int, int]:
    """Extract template controller difficulty features from a sample row."""
    has_final = row.get(f"has_final_{b1}", 0)
    tokens = row.get(f"tokens_{b1}", b1)
    utilization = tokens / b1 if b1 > 0 else 1.0
    util_bin = 1 if utilization >= 0.95 else 0
    consistency = 1 if has_final and util_bin == 0 else 0
    return has_final, util_bin, consistency


def template_search(train_rows: List[Dict], budgets: List[int], lam: float) -> Dict:
    """Exhaustive search over template assignments."""
    b1, b_mid, b_max = budgets[0], budgets[1], budgets[2]
    norm = b_max

    categories = defaultdict(list)
    for row in train_rows:
        feat = extract_difficulty_features(row, b1)
        categories[feat].append(row)

    cat_keys = sorted(categories.keys())
    if not cat_keys:
        return {k: b_max for k in cat_keys}

    best_util = -float("inf")
    best_template = {}

    budget_options = budgets
    from itertools import product
    for assignment in product(budget_options, repeat=len(cat_keys)):
        template = dict(zip(cat_keys, assignment))
        total_util = 0.0
        n = 0
        for cat, assigned_b in template.items():
            for row in categories[cat]:
                correct = row[f"correct_{assigned_b}"]
                tokens = row[f"tokens_{assigned_b}"]
                probe_cost = row[f"tokens_{b1}"] if assigned_b > b1 else 0
                e2e_tokens = tokens + probe_cost
                util = correct - lam * (e2e_tokens / norm)
                total_util += util
                n += 1
        avg_util = total_util / max(n, 1)
        if avg_util > best_util:
            best_util = avg_util
            best_template = template.copy()

    return best_template


def evaluate_template(rows: List[Dict], template: Dict, budgets: List[int], lam: float) -> Dict:
    """Evaluate a template on held-out rows."""
    b1, b_mid, b_max = budgets[0], budgets[1], budgets[2]
    norm = b_max

    results = []
    for row in rows:
        feat = extract_difficulty_features(row, b1)
        assigned_b = template.get(feat, b_max)
        correct = row[f"correct_{assigned_b}"]
        tokens = row[f"tokens_{assigned_b}"]
        probe_cost = row[f"tokens_{b1}"] if assigned_b > b1 else 0
        e2e_tokens = tokens + probe_cost

        fixed_mid_correct = row[f"correct_{b_mid}"]
        fixed_mid_tokens = row[f"tokens_{b_mid}"]
        fixed_max_correct = row[f"correct_{b_max}"]
        fixed_max_tokens = row[f"tokens_{b_max}"]

        results.append({
            "ctrl_correct": correct,
            "ctrl_tokens": e2e_tokens,
            "ctrl_budget": assigned_b,
            "fixed_mid_correct": fixed_mid_correct,
            "fixed_mid_tokens": fixed_mid_tokens,
            "fixed_max_correct": fixed_max_correct,
            "fixed_max_tokens": fixed_max_tokens,
        })
    return results


def bootstrap_ci(data: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float, float]:
    """Compute mean and 95% CI via percentile bootstrap."""
    rng = np.random.RandomState(seed)
    n = len(data)
    means = np.array([data[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_setting(name: str, cfg: Dict, results_dir: str) -> Optional[Dict]:
    """Full analysis for one benchmark-model setting."""
    benchmark = name.split("_")[0]
    budgets = cfg["budgets"]
    model_tag = cfg["model_tag"]

    csv_path = find_csv(results_dir, benchmark, model_tag)
    if csv_path is None:
        print(f"  [SKIP] No CSV found for {name}")
        return None

    rows = load_per_sample(csv_path, budgets)
    n = len(rows)
    print(f"  [{name}] Loaded {n} samples from {os.path.basename(csv_path)}")

    b1, b_mid, b_max = budgets[0], budgets[1], budgets[2]
    norm = b_max

    fixed_results = {}
    for b in budgets:
        accs = [r[f"correct_{b}"] for r in rows]
        toks = [r[f"tokens_{b}"] for r in rows]
        avg_acc = np.mean(accs)
        avg_tok = np.mean(toks)
        avg_util = avg_acc - LAMBDA * (avg_tok / norm)
        fixed_results[b] = {"acc": avg_acc, "tokens": avg_tok, "utility": avg_util}
        print(f"    Fixed-{b}: Acc={avg_acc:.4f}, Tok={avg_tok:.1f}, Util={avg_util:.4f}")

    # K-fold cross-validation for template controller
    rng = np.random.RandomState(42)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, K_FOLDS)

    all_eval_results = []
    for fold_idx in range(K_FOLDS):
        test_idx = set(folds[fold_idx])
        train_rows = [rows[i] for i in range(n) if i not in test_idx]
        test_rows = [rows[i] for i in folds[fold_idx]]

        template = template_search(train_rows, budgets, LAMBDA)
        fold_results = evaluate_template(test_rows, template, budgets, LAMBDA)
        all_eval_results.extend(fold_results)

    ctrl_accs = np.array([r["ctrl_correct"] for r in all_eval_results])
    ctrl_toks = np.array([r["ctrl_tokens"] for r in all_eval_results])
    ctrl_utils = ctrl_accs - LAMBDA * (ctrl_toks / norm)

    fixed_mid_accs = np.array([r["fixed_mid_correct"] for r in all_eval_results])
    fixed_mid_toks = np.array([r["fixed_mid_tokens"] for r in all_eval_results])
    fixed_mid_utils = fixed_mid_accs - LAMBDA * (fixed_mid_toks / norm)

    fixed_max_accs = np.array([r["fixed_max_correct"] for r in all_eval_results])
    fixed_max_toks = np.array([r["fixed_max_tokens"] for r in all_eval_results])
    fixed_max_utils = fixed_max_accs - LAMBDA * (fixed_max_toks / norm)

    acc_delta = ctrl_accs - fixed_mid_accs
    util_delta = ctrl_utils - fixed_mid_utils

    acc_mean, acc_lo, acc_hi = bootstrap_ci(acc_delta)
    util_mean, util_lo, util_hi = bootstrap_ci(util_delta)

    budget_dist = defaultdict(int)
    for r in all_eval_results:
        budget_dist[r["ctrl_budget"]] += 1

    result = {
        "n": n,
        "k_folds": K_FOLDS,
        "fixed": {str(b): fixed_results[b] for b in budgets},
        "template_ctrl": {
            "acc": float(ctrl_accs.mean()),
            "tokens": float(ctrl_toks.mean()),
            "utility": float(ctrl_utils.mean()),
        },
        "delta_vs_mid": {
            "acc": {"mean": acc_mean, "ci_lo": acc_lo, "ci_hi": acc_hi},
            "utility": {"mean": util_mean, "ci_lo": util_lo, "ci_hi": util_hi},
        },
        "delta_vs_max": {
            "acc_mean": float((ctrl_accs - fixed_max_accs).mean()),
            "util_mean": float((ctrl_utils - fixed_max_utils).mean()),
        },
        "budget_distribution": {str(b): budget_dist[b] for b in budgets},
        "csv_path": csv_path,
    }

    print(f"    Template Ctrl ({K_FOLDS}-fold): Acc={ctrl_accs.mean():.4f}, "
          f"Tok={ctrl_toks.mean():.1f}, Util={ctrl_utils.mean():.4f}")
    print(f"    ΔAcc vs mid: {acc_mean*100:+.1f}pp [{acc_lo*100:+.1f}, {acc_hi*100:+.1f}]")
    print(f"    ΔUtil vs mid: {util_mean*100:+.1f}pp [{util_lo*100:+.1f}, {util_hi*100:+.1f}]")
    print(f"    Budget dist: {dict(budget_dist)}")

    return result


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "fulltest")
    if not os.path.isdir(results_dir):
        print(f"Results dir not found: {results_dir}")
        print("Trying alternative: results/")
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

    print(f"Analyzing full-test results from: {results_dir}")
    print(f"Lambda={LAMBDA}, K-folds={K_FOLDS}, Bootstrap={N_BOOTSTRAP}")
    print("=" * 70)

    all_results = {}
    for name, cfg in SETTINGS.items():
        print(f"\n--- {name} ---")
        result = analyze_setting(name, cfg, results_dir)
        if result:
            all_results[name] = result

    output_path = os.path.join(results_dir, "fulltest_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE: Full-dataset Template Controller Results")
    print("=" * 70)
    print(f"{'Setting':<20} {'n':>6} {'Ctrl Acc':>9} {'Fixed Mid':>9} {'ΔAcc (pp)':>10} {'95% CI':>18} {'ΔUtil (pp)':>10} {'95% CI':>18}")
    print("-" * 100)
    for name, r in all_results.items():
        d = r["delta_vs_mid"]
        budgets = SETTINGS[name]["budgets"]
        mid_acc = r["fixed"][str(budgets[1])]["acc"]
        print(f"{name:<20} {r['n']:>6} {r['template_ctrl']['acc']:>9.4f} {mid_acc:>9.4f} "
              f"{d['acc']['mean']*100:>+10.1f} [{d['acc']['ci_lo']*100:>+6.1f}, {d['acc']['ci_hi']*100:>+6.1f}] "
              f"{d['utility']['mean']*100:>+10.1f} [{d['utility']['ci_lo']*100:>+6.1f}, {d['utility']['ci_hi']*100:>+6.1f}]")


if __name__ == "__main__":
    main()
