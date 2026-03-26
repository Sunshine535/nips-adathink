#!/usr/bin/env python3
"""Cross-benchmark template transfer test.

Demonstrates that the template PATTERN (mapping from difficulty features
to budget tier indices) transfers across benchmarks, even when the absolute
budget values differ.

Key idea: template maps feature patterns → tier index {0, 1, 2} = {low, mid, high}.
A template learned on GSM8K-27B can be applied to MATH500-27B by mapping
tier indices to MATH500's budget values.
"""

import csv
import json
import os
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

LAMBDA = 0.15
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260228
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def load_all_csvs(pattern_prefix: str) -> List[Dict]:
    """Load all per-sample CSVs matching prefix, keep ALL rows (including duplicates from seeds)."""
    all_rows = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith(pattern_prefix) and f.endswith(".csv") and "template" not in f and "controller" not in f:
            path = os.path.join(RESULTS_DIR, f)
            with open(path) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    row["_source_csv"] = f
                    all_rows.append(row)
    return all_rows


def extract_features(row: Dict, b1: int) -> Tuple[int, int, int]:
    has_final = int(float(row.get(f"fixed_{b1}_has_final", 0)))
    tokens = float(row.get(f"fixed_{b1}_tokens", b1))
    util = tokens / b1 if b1 > 0 else 1.0
    util_bin = 1 if util >= 0.95 else 0
    consistency = 1 if has_final and util_bin == 0 else 0
    return has_final, util_bin, consistency


def search_template_indexed(rows: List[Dict], budgets: List[int], lam: float) -> Dict:
    """Search for best template mapping features → tier INDEX (0=low, 1=mid, 2=high)."""
    b1 = budgets[0]
    norm = budgets[-1]
    categories = defaultdict(list)
    for row in rows:
        feat = extract_features(row, b1)
        categories[feat].append(row)

    cat_keys = sorted(categories.keys())
    tier_indices = list(range(len(budgets)))
    best_util, best_template = -float("inf"), {}

    for assignment in product(tier_indices, repeat=len(cat_keys)):
        template = dict(zip(cat_keys, assignment))
        total_u, n = 0.0, 0
        for cat, tier_idx in template.items():
            b = budgets[tier_idx]
            for row in categories[cat]:
                correct = int(float(row.get(f"fixed_{b}_correct", 0)))
                tok = float(row.get(f"fixed_{b}_tokens", b))
                probe = float(row.get(f"fixed_{b1}_tokens", b1)) if b > b1 else 0
                total_u += correct - lam * ((tok + probe) / norm)
                n += 1
        avg = total_u / max(n, 1)
        if avg > best_util:
            best_util, best_template = avg, template.copy()

    return best_template


def evaluate_indexed(rows: List[Dict], template: Dict, budgets: List[int], lam: float) -> Dict:
    """Evaluate a tier-indexed template on target benchmark data."""
    b1, b_mid = budgets[0], budgets[1]
    norm = budgets[-1]
    ctrl_accs, ctrl_toks, mid_accs, mid_toks = [], [], [], []

    for row in rows:
        feat = extract_features(row, b1)
        tier_idx = template.get(feat, len(budgets) - 1)
        b = budgets[min(tier_idx, len(budgets) - 1)]
        c = int(float(row.get(f"fixed_{b}_correct", 0)))
        t = float(row.get(f"fixed_{b}_tokens", b))
        probe = float(row.get(f"fixed_{b1}_tokens", b1)) if b > b1 else 0
        ctrl_accs.append(c)
        ctrl_toks.append(t + probe)
        mid_accs.append(int(float(row.get(f"fixed_{b_mid}_correct", 0))))
        mid_toks.append(float(row.get(f"fixed_{b_mid}_tokens", b_mid)))

    ca, ma = np.array(ctrl_accs), np.array(mid_accs)
    ct, mt = np.array(ctrl_toks), np.array(mid_toks)
    cu, mu = ca - lam * ct / norm, ma - lam * mt / norm
    da, du = ca - ma, cu - mu

    rng = np.random.RandomState(BOOTSTRAP_SEED)
    n = len(da)
    ab = [da[rng.randint(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    ub = [du[rng.randint(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]

    return {
        "n": n,
        "ctrl_acc": float(ca.mean()),
        "ctrl_tok": float(ct.mean()),
        "mid_acc": float(ma.mean()),
        "delta_acc": float(da.mean()),
        "delta_acc_ci": [float(np.percentile(ab, 2.5)), float(np.percentile(ab, 97.5))],
        "delta_util": float(du.mean()),
        "delta_util_ci": [float(np.percentile(ub, 2.5)), float(np.percentile(ub, 97.5))],
    }


CONFIGS = {
    "gsm8k_27b": {"prefix": "per_sample_gsm8k_Qwen3.5_27B", "budgets": [128, 256, 512]},
    "math500_27b": {"prefix": "per_sample_math500_Qwen3.5_27B", "budgets": [2048, 4096, 8192]},
    "bbh_27b": {"prefix": "per_sample_bbh_Qwen3.5_27B", "budgets": [1024, 2048, 4096]},
    "gsm8k_8b": {"prefix": "per_sample_gsm8k_Qwen3_8B", "budgets": [128, 256, 512]},
    "math500_8b": {"prefix": "per_sample_math500_Qwen3_8B", "budgets": [512, 1024, 2048]},
    "bbh_8b": {"prefix": "per_sample_bbh_Qwen3_8B", "budgets": [256, 512, 1024]},
}


def main():
    print("=" * 80)
    print("CROSS-BENCHMARK TEMPLATE TRANSFER TEST")
    print("Template pattern: features → tier index {low, mid, high}")
    print("Transfer: learn pattern on source → apply to target's budget tiers")
    print("=" * 80)

    data = {}
    for name, cfg in CONFIGS.items():
        rows = load_all_csvs(cfg["prefix"])
        if rows:
            data[name] = {"rows": rows, "budgets": cfg["budgets"]}
            print(f"  {name}: {len(rows)} samples (all seeds)")

    results = {}

    print(f"\n{'Transfer':<35} {'n':>5} {'ΔAcc':>8} {'95% CI':>18} {'ΔUtil':>8} {'95% CI':>18}")
    print("-" * 98)

    for model_tag in ["27b", "8b"]:
        model_benches = {k: v for k, v in data.items() if k.endswith(model_tag)}

        for src_name, src_data in model_benches.items():
            template = search_template_indexed(src_data["rows"], src_data["budgets"], LAMBDA)
            src_bench = src_name.split("_")[0]
            pattern_str = {str(k): f"tier_{v}" for k, v in template.items()}

            for tgt_name, tgt_data in model_benches.items():
                label = f"{src_name} → {tgt_name}"
                is_transfer = src_name != tgt_name
                result = evaluate_indexed(tgt_data["rows"], template, tgt_data["budgets"], LAMBDA)
                result["is_transfer"] = is_transfer
                result["template_pattern"] = pattern_str
                results[label] = result

                marker = "TRANSFER" if is_transfer else "in-domain"
                ci_a = result["delta_acc_ci"]
                ci_u = result["delta_util_ci"]
                print(f"  [{marker:>8}] {label:<25} {result['n']:>5} "
                      f"{result['delta_acc']*100:>+7.1f}pp [{ci_a[0]*100:>+5.1f},{ci_a[1]*100:>+5.1f}] "
                      f"{result['delta_util']*100:>+7.1f}pp [{ci_u[0]*100:>+5.1f},{ci_u[1]*100:>+5.1f}]")

        print()

    output = os.path.join(RESULTS_DIR, "cross_benchmark_transfer.json")
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
