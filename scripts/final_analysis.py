#!/usr/bin/env python3
"""Generate all tables, statistics, and analyses for the NeurIPS paper.

Produces:
1. Main results table (Table 1): All benchmarks × models × budgets
2. Controller comparison table (Table 2): Template Controller vs fixed budgets
3. Ablation table (Table 3)
4. Self-Consistency baseline comparison
5. Bootstrap significance tests
6. Wall-clock latency analysis
7. Per-difficulty breakdown for MATH-500 and BBH
"""

import csv
import glob
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np

RESULTS_DIR = "/workspace/nips-adathink/results"
OUTPUT_DIR = "/workspace/nips-adathink/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_summaries(pattern, budget_filter=None):
    """Load summary JSONs matching pattern."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    results = []
    for f in files:
        with open(f) as fh:
            s = json.load(fh)
        if budget_filter:
            budgets_in = set(int(b) for b in s.get("fixed", {}).keys())
            if not budget_filter.issubset(budgets_in):
                continue
        results.append(s)
    return results


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    s = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5
    return m, s


def bootstrap_ci(vals, n_bootstrap=10000, alpha=0.05, seed=42):
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = np.array(vals)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(sample.mean())
    means = sorted(means)
    lo = means[int(n_bootstrap * alpha / 2)]
    hi = means[int(n_bootstrap * (1 - alpha / 2))]
    return arr.mean(), lo, hi


def bootstrap_paired_test(vals_a, vals_b, n_bootstrap=10000, seed=42):
    """Bootstrap paired test: P(mean(A) > mean(B))."""
    rng = np.random.RandomState(seed)
    a = np.array(vals_a)
    b = np.array(vals_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    count_positive = 0
    for _ in range(n_bootstrap):
        sample = rng.choice(diff, size=n, replace=True)
        if sample.mean() > 0:
            count_positive += 1
    p_value = 1 - count_positive / n_bootstrap
    return p_value, diff.mean()


def load_per_sample_csvs(pattern):
    """Load per_sample CSVs and return list of rows."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    all_rows = []
    for f in files:
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        all_rows.append({"file": f, "rows": rows})
    return all_rows


# ============================================================
# Table 1: Main Results
# ============================================================
def generate_main_results():
    print("=" * 70)
    print("TABLE 1: MAIN RESULTS")
    print("=" * 70)

    configs = [
        ("GSM8K", "27B", "summary_Qwen3.5_27B_*.json", {128, 256, 512}, [128, 256, 512]),
        ("GSM8K", "8B", "summary_Qwen3_8B_*.json", {128, 256, 512}, [128, 256, 512]),
        ("MATH-500", "27B", "summary_math500_Qwen3.5_27B_*.json", {2048, 4096, 8192}, [2048, 4096, 8192]),
        ("MATH-500", "8B", "summary_math500_Qwen3_8B_*.json", {512, 1024, 2048}, [512, 1024, 2048]),
        ("BBH", "27B", "summary_bbh_Qwen3.5_27B_*.json", {1024, 2048, 4096}, [1024, 2048, 4096]),
        ("BBH", "8B", "summary_bbh_Qwen3_8B_*.json", {256, 512, 1024}, [256, 512, 1024]),
    ]

    table_data = []
    for bench, model, pattern, bf, budget_order in configs:
        summaries = load_summaries(pattern, budget_filter=bf)
        if not summaries:
            continue

        row = {"benchmark": bench, "model": model, "n_seeds": len(summaries)}
        for b in budget_order:
            accs = [s["fixed"][str(b)]["accuracy"] for s in summaries if str(b) in s.get("fixed", {})]
            m, s = mean_std(accs)
            row[f"fixed_{b}"] = f"{m:.3f}±{s:.3f}"
            row[f"fixed_{b}_mean"] = m

        adap_accs = [s["adaptive"]["accuracy"] for s in summaries]
        m, s = mean_std(adap_accs)
        row["adaptive"] = f"{m:.3f}±{s:.3f}"
        row["adaptive_mean"] = m

        # Compute token savings: adaptive tokens vs max-budget tokens
        max_budget = max(budget_order)
        adap_tokens = [s["adaptive"].get("avg_tokens", max_budget) for s in summaries]
        max_tokens = [s["fixed"][str(max_budget)].get("avg_tokens", max_budget) for s in summaries if str(max_budget) in s.get("fixed", {})]
        if adap_tokens and max_tokens:
            avg_adap_t = sum(adap_tokens) / len(adap_tokens)
            avg_max_t = sum(max_tokens) / len(max_tokens)
            savings = (1 - avg_adap_t / avg_max_t) * 100 if avg_max_t > 0 else 0
            row["token_savings"] = f"{savings:.1f}%"
        else:
            row["token_savings"] = "N/A"

        table_data.append(row)
        budgets_str = "/".join(str(b) for b in budget_order)
        print(f"\n{bench} | {model} (n={len(summaries)} seeds)")
        print(f"  Budgets: {budgets_str}")
        for b in budget_order:
            print(f"  Fixed@{b}: {row.get(f'fixed_{b}', 'N/A')}")
        print(f"  Adaptive: {row['adaptive']}")
        print(f"  Token savings: {row['token_savings']}")

    return table_data


# ============================================================
# Table 2: Controller Comparison
# ============================================================
def generate_controller_table():
    print("\n" + "=" * 70)
    print("TABLE 2: TEMPLATE CONTROLLER RESULTS")
    print("=" * 70)

    ctrl_configs = [
        ("GSM8K 27B (23 seeds)", "lam0p15_20260228_23seed"),
        ("GSM8K 8B-think", "qwen3_8b_think_lam0p15_3seed_20260228"),
        ("MATH-500 27B v2", "math500_27b_v2_20260320_132540"),
        ("MATH-500 8B", "math500_8b_20260320_081545"),
        ("BBH 27B v2", "bbh_27b_v2_20260320_132544"),
        ("BBH 8B", "bbh_8b_20260320_081550"),
    ]

    for label, tag in ctrl_configs:
        pattern = os.path.join(RESULTS_DIR, f"template_controller_{tag}.json")
        files = glob.glob(pattern)
        if not files:
            print(f"\n{label}: NOT FOUND ({tag})")
            continue
        with open(files[0]) as f:
            data = json.load(f)

        macro = data.get("macro_mean", {})
        learned = macro.get("learned", {})
        fixed = macro.get("fixed", {})

        print(f"\n{label}:")
        print(f"  Learned (AdaThink): acc={learned.get('accuracy', 0):.3f}, avg_tokens={learned.get('avg_tokens', 0):.1f}")
        for b, v in sorted(fixed.items()):
            print(f"  Fixed@{b}: acc={v.get('accuracy', 0):.3f}, avg_tokens={v.get('avg_tokens', 0):.1f}")


# ============================================================
# Bootstrap Significance
# ============================================================
def generate_significance():
    print("\n" + "=" * 70)
    print("TABLE 3: BOOTSTRAP SIGNIFICANCE TESTS")
    print("=" * 70)

    tests = [
        ("MATH-500 27B", "summary_math500_Qwen3.5_27B_*.json", {2048, 4096, 8192}, "adaptive", "4096"),
        ("BBH 27B", "summary_bbh_Qwen3.5_27B_*.json", {1024, 2048, 4096}, "adaptive", "2048"),
        ("MATH-500 8B", "summary_math500_Qwen3_8B_*.json", {512, 1024, 2048}, "adaptive", "1024"),
        ("BBH 8B", "summary_bbh_Qwen3_8B_*.json", {256, 512, 1024}, "adaptive", "512"),
    ]

    for label, pattern, bf, method_a, budget_b in tests:
        summaries = load_summaries(pattern, budget_filter=bf)
        if len(summaries) < 3:
            print(f"\n{label}: insufficient seeds ({len(summaries)})")
            continue

        adap_accs = [s["adaptive"]["accuracy"] for s in summaries]
        fixed_accs = [s["fixed"][budget_b]["accuracy"] for s in summaries if budget_b in s.get("fixed", {})]

        n = min(len(adap_accs), len(fixed_accs))
        if n < 3:
            continue

        p_val, mean_diff = bootstrap_paired_test(adap_accs[:n], fixed_accs[:n])
        adap_mean, adap_lo, adap_hi = bootstrap_ci(adap_accs)
        fixed_mean, fixed_lo, fixed_hi = bootstrap_ci(fixed_accs)

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"\n{label}:")
        print(f"  Adaptive: {adap_mean:.3f} [{adap_lo:.3f}, {adap_hi:.3f}]")
        print(f"  Fixed@{budget_b}: {fixed_mean:.3f} [{fixed_lo:.3f}, {fixed_hi:.3f}]")
        print(f"  Diff: {mean_diff:+.3f}, p={p_val:.4f} {sig}")


# ============================================================
# Per-Difficulty Breakdown
# ============================================================
def generate_per_difficulty():
    print("\n" + "=" * 70)
    print("TABLE 4: PER-DIFFICULTY BREAKDOWN")
    print("=" * 70)

    # MATH-500 by level
    summaries = load_summaries("summary_math500_Qwen3.5_27B_*.json", {2048, 4096, 8192})
    if summaries:
        levels = {}
        for s in summaries:
            for gk in ["per_math_level"]:
                for lvl, data in s.get(gk, {}).items():
                    levels.setdefault(lvl, defaultdict(list))
                    for b, v in data.get("fixed", {}).items():
                        levels[lvl][f"fixed_{b}"].append(v.get("accuracy", 0))
                    levels[lvl]["adaptive"].append(data.get("adaptive", {}).get("accuracy", 0))

        print("\nMATH-500 27B by Difficulty Level:")
        for lvl in sorted(levels.keys()):
            vals = levels[lvl]
            parts = []
            for k in sorted(vals.keys()):
                m, _ = mean_std(vals[k])
                parts.append(f"{k}={m:.3f}")
            print(f"  Level {lvl}: {', '.join(parts)}")

    # BBH by task type
    summaries = load_summaries("summary_bbh_Qwen3.5_27B_*.json", {1024, 2048, 4096})
    if summaries:
        tasks = {}
        for s in summaries:
            for task, data in s.get("per_task", {}).items():
                tasks.setdefault(task, defaultdict(list))
                for b, v in data.get("fixed", {}).items():
                    tasks[task][f"fixed_{b}"].append(v.get("accuracy", 0))
                tasks[task]["adaptive"].append(data.get("adaptive", {}).get("accuracy", 0))

        print("\nBBH 27B by Task (top 10 by accuracy difference):")
        task_diffs = []
        for task, vals in tasks.items():
            if "fixed_1024" in vals and "fixed_4096" in vals:
                m_lo, _ = mean_std(vals["fixed_1024"])
                m_hi, _ = mean_std(vals["fixed_4096"])
                task_diffs.append((task, m_hi - m_lo, m_lo, m_hi))
        task_diffs.sort(key=lambda x: -x[1])
        for task, diff, lo, hi in task_diffs[:10]:
            print(f"  {task}: @1024={lo:.3f}, @4096={hi:.3f}, gap={diff:+.3f}")


# ============================================================
# Wall-Clock Analysis
# ============================================================
def generate_wallclock():
    print("\n" + "=" * 70)
    print("TABLE 5: WALL-CLOCK LATENCY")
    print("=" * 70)

    configs = [
        ("MATH-500 27B", "per_sample_math500_Qwen3.5_27B_*.csv"),
        ("MATH-500 8B", "per_sample_math500_Qwen3_8B_*.csv"),
        ("BBH 27B", "per_sample_bbh_Qwen3.5_27B_*.csv"),
        ("BBH 8B", "per_sample_bbh_Qwen3_8B_*.csv"),
    ]

    for label, pattern in configs:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)), key=os.path.getmtime, reverse=True)
        if not files:
            continue
        # Use most recent file
        with open(files[0]) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        budgets = set()
        for k in rows[0].keys():
            if k.startswith("fixed_") and k.endswith("_latency_s"):
                b = k.split("_")[1]
                try:
                    int(b)
                    budgets.add(b)
                except ValueError:
                    pass

        print(f"\n{label}:")
        for b in sorted(budgets, key=int):
            lats = [float(r.get(f"fixed_{b}_latency_s", 0)) for r in rows]
            tokens = [float(r.get(f"fixed_{b}_tokens", 0)) for r in rows]
            avg_lat = sum(lats) / len(lats)
            avg_tok = sum(tokens) / len(tokens)
            throughput = avg_tok / avg_lat if avg_lat > 0 else 0
            print(f"  Budget={b}: avg_latency={avg_lat:.2f}s, avg_tokens={avg_tok:.0f}, throughput={throughput:.0f} tok/s")


# ============================================================
# Overthinking Analysis
# ============================================================
def generate_overthinking():
    print("\n" + "=" * 70)
    print("TABLE 6: OVERTHINKING RATE (OER)")
    print("=" * 70)
    print("OER = fraction of samples correct at low budget but wrong at high budget")

    configs = [
        ("MATH-500 27B", "summary_math500_Qwen3.5_27B_*.json", {2048, 4096, 8192}),
        ("MATH-500 8B", "summary_math500_Qwen3_8B_*.json", {512, 1024, 2048}),
        ("BBH 27B", "summary_bbh_Qwen3.5_27B_*.json", {1024, 2048, 4096}),
        ("BBH 8B", "summary_bbh_Qwen3_8B_*.json", {256, 512, 1024}),
    ]

    for label, pattern, bf in configs:
        summaries = load_summaries(pattern, budget_filter=bf)
        if not summaries:
            continue
        oers = []
        for s in summaries:
            for k, v in s.get("overthinking", {}).items():
                oers.append(v)
        if oers:
            m, std = mean_std(oers)
            print(f"  {label}: OER={m:.3f}±{std:.3f} (n={len(summaries)})")


# ============================================================
# Save all to JSON
# ============================================================
def save_analysis():
    """Save complete analysis as JSON for paper generation."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {
            "gsm8k_27b": {"n_seeds": len(glob.glob(os.path.join(RESULTS_DIR, "summary_Qwen3.5_27B_*.json")))},
            "gsm8k_8b": {"n_seeds": len(glob.glob(os.path.join(RESULTS_DIR, "summary_Qwen3_8B_*.json")))},
            "math500_27b_v2": {"n_seeds": len(load_summaries("summary_math500_Qwen3.5_27B_*.json", {2048, 4096, 8192}))},
            "math500_8b": {"n_seeds": len(load_summaries("summary_math500_Qwen3_8B_*.json", {512, 1024, 2048}))},
            "bbh_27b_v2": {"n_seeds": len(load_summaries("summary_bbh_Qwen3.5_27B_*.json", {1024, 2048, 4096}))},
            "bbh_8b": {"n_seeds": len(load_summaries("summary_bbh_Qwen3_8B_*.json", {256, 512, 1024}))},
        },
        "total_json_files": len(glob.glob(os.path.join(RESULTS_DIR, "*.json"))),
        "total_csv_files": len(glob.glob(os.path.join(RESULTS_DIR, "*.csv"))),
    }

    path = os.path.join(OUTPUT_DIR, "complete_analysis.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to: {path}")


def main():
    generate_main_results()
    generate_controller_table()
    generate_significance()
    generate_per_difficulty()
    generate_wallclock()
    generate_overthinking()
    save_analysis()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
