#!/usr/bin/env python3
"""Controller-level ablation study.

Ablation 1: halting-only — controller can only choose lowest or highest budget
Ablation 2: no-branch — controller can only choose from first two budgets
Ablation 3: mid-only — force always pick the middle budget (naive baseline)

All done purely computationally from existing per_sample CSVs.
"""

import argparse
import csv
import glob
import json
import math
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Tuple

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


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
    budgets = []
    for k in fieldnames:
        m = re.match(r"fixed_(\d+)_correct", k)
        if m:
            budgets.append(int(m.group(1)))
    return sorted(set(budgets))


def utility(row, budget, lam, norm):
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lam * (t / norm)


def make_key(row, mode):
    toks = WORD_RE.findall((row.get("question") or "").lower())
    if mode == "first2":
        return "|".join(toks[:2]) if toks else "_"
    if mode == "first3":
        return "|".join(toks[:3]) if toks else "_"
    if mode == "first2_lenbin":
        base = "|".join(toks[:2]) if toks else "_"
        return f"{len(toks)//6}|{base}"
    return "|".join(toks[:2]) if toks else "_"


def split_inner(rows):
    train, val = [], []
    for r in rows:
        idx = to_int(r.get("idx", 0))
        if idx % 5 == 0:
            val.append(r)
        else:
            train.append(r)
    if not val:
        k = max(1, len(rows) // 5)
        val, train = rows[-k:], rows[:-k]
    return train, val


def build_and_eval_template(train_rows, test_rows, allowed_budgets, mode, lam, norm):
    """Build template controller restricted to allowed_budgets and evaluate on test."""
    stats = {}
    for r in train_rows:
        k = make_key(r, mode)
        if k not in stats:
            stats[k] = {b: [0.0, 0] for b in allowed_budgets}
        for b in allowed_budgets:
            stats[k][b][0] += utility(r, b, lam, norm)
            stats[k][b][1] += 1

    mapping = {}
    for k, st in stats.items():
        best_b = allowed_budgets[0]
        best_u = st[best_b][0] / max(1, st[best_b][1])
        for b in allowed_budgets[1:]:
            u = st[b][0] / max(1, st[b][1])
            if u > best_u:
                best_u = u
                best_b = b
        mapping[k] = best_b

    default_budget = allowed_budgets[0]
    best_global = -1e18
    for b in allowed_budgets:
        avg_u = sum(utility(r, b, lam, norm) for r in train_rows) / max(1, len(train_rows))
        if avg_u > best_global:
            best_global = avg_u
            default_budget = b

    n = max(1, len(test_rows))
    acc = tok = util_sum = 0.0
    out_rows = []
    for r in test_rows:
        k = make_key(r, mode)
        b = mapping.get(k, default_budget)
        c = to_int(r.get(f"fixed_{b}_correct", 0))
        t = to_float(r.get(f"fixed_{b}_tokens", 0.0))
        u = utility(r, b, lam, norm)
        acc += c
        tok += t
        util_sum += u
        out_rows.append({"idx": r.get("idx", ""), "chosen_budget": b, "correct": c, "tokens": t, "utility": u})
    return {"accuracy": acc / n, "avg_tokens": tok / n, "avg_utility": util_sum / n, "rows": out_rows}


def bootstrap_ci(values, n_boot=10000, seed=42, alpha=0.05):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    rnd = random.Random(seed)
    means = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(alpha / 2 * n_boot)], means[int((1 - alpha / 2) * n_boot)]


def run_ablation_on_benchmark(tag, csvs, all_budgets, lam, norm):
    """Run all ablation variants on a benchmark."""
    datasets = []
    for p in csvs:
        with open(p) as f:
            rows = list(csv.DictReader(f))
        datasets.append((p, rows))

    modes = ["first2", "first3", "first2_lenbin"]
    b_lo, b_mid, b_hi = all_budgets[0], all_budgets[len(all_budgets) // 2], all_budgets[-1]

    ablation_configs = {
        "full": all_budgets,
        "halting_only": [b_lo, b_hi],
        "no_branch": all_budgets[:2],
        "max_only": [b_hi],
        "mid_only": [b_mid],
    }

    results = {}
    for abl_name, allowed in ablation_configs.items():
        fold_results = []
        all_rows = []

        for i in range(len(datasets)):
            test_name, test_rows = datasets[i]
            train_rows = []
            for j, (_, rows) in enumerate(datasets):
                if j != i:
                    train_rows.extend(rows)

            inner_train, inner_val = split_inner(train_rows)
            best_mode = modes[0]
            best_u = -1e18

            if len(allowed) > 1:
                for mode in modes:
                    ev = build_and_eval_template(inner_train, inner_val, allowed, mode, lam, norm)
                    if ev["avg_utility"] > best_u:
                        best_u = ev["avg_utility"]
                        best_mode = mode
                learned = build_and_eval_template(train_rows, test_rows, allowed, best_mode, lam, norm)
            else:
                b = allowed[0]
                n = max(1, len(test_rows))
                acc = sum(to_int(r.get(f"fixed_{b}_correct", 0)) for r in test_rows) / n
                tok_avg = sum(to_float(r.get(f"fixed_{b}_tokens", 0.0)) for r in test_rows) / n
                util_avg = acc - lam * (tok_avg / norm)
                learned = {"accuracy": acc, "avg_tokens": tok_avg, "avg_utility": util_avg, "rows": []}

            fold_results.append(learned)
            all_rows.extend(learned.get("rows", []))

        macro_acc = sum(f["accuracy"] for f in fold_results) / len(fold_results)
        macro_tok = sum(f["avg_tokens"] for f in fold_results) / len(fold_results)
        macro_util = sum(f["avg_utility"] for f in fold_results) / len(fold_results)

        results[abl_name] = {
            "allowed_budgets": allowed,
            "accuracy": macro_acc,
            "avg_tokens": macro_tok,
            "avg_utility": macro_util,
            "n_folds": len(fold_results),
        }

    return results


def find_csvs(results_dir, pattern, budget_filter):
    csvs = []
    for f in sorted(glob.glob(os.path.join(results_dir, pattern))):
        sf = f.replace("per_sample_", "summary_").replace(".csv", ".json")
        if os.path.exists(sf):
            with open(sf) as fh:
                s = json.load(fh)
            if all(str(b) in s.get("fixed", {}) for b in budget_filter):
                csvs.append(f)
    return csvs


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    lam = 0.15
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    benchmarks = {
        "gsm8k_27b": {
            "pattern": "per_sample_Qwen3.5_27B_*.csv",
            "budget_filter": [128, 256, 512],
            "budgets": [128, 256, 512],
            "norm": 512.0,
        },
        "math500_27b": {
            "pattern": "per_sample_math500_Qwen3.5_27B_*.csv",
            "budget_filter": [2048, 4096, 8192],
            "budgets": [2048, 4096, 8192],
            "norm": 8192.0,
        },
        "bbh_27b": {
            "pattern": "per_sample_bbh_Qwen3.5_27B_*.csv",
            "budget_filter": [1024, 2048, 4096],
            "budgets": [1024, 2048, 4096],
            "norm": 4096.0,
        },
        "math500_8b": {
            "pattern": "per_sample_math500_Qwen3_8B_*.csv",
            "budget_filter": [512, 1024, 2048],
            "budgets": [512, 1024, 2048],
            "norm": 2048.0,
        },
        "bbh_8b": {
            "pattern": "per_sample_bbh_Qwen3_8B_*.csv",
            "budget_filter": [256, 512, 1024],
            "budgets": [256, 512, 1024],
            "norm": 1024.0,
        },
    }

    all_results = {}
    for tag, cfg in benchmarks.items():
        csvs = find_csvs(results_dir, cfg["pattern"], cfg["budget_filter"])
        if len(csvs) < 3:
            print(f"  Skipping {tag}: only {len(csvs)} CSVs")
            continue
        print(f"\nRunning ablation on {tag} ({len(csvs)} seeds)...")
        res = run_ablation_on_benchmark(tag, csvs, cfg["budgets"], lam, cfg["norm"])
        all_results[tag] = res

        for abl_name, r in res.items():
            print(f"  {abl_name:15s}: acc={r['accuracy']:.4f}  tok={r['avg_tokens']:.0f}  util={r['avg_utility']:.4f}  budgets={r['allowed_budgets']}")

    out_path = os.path.join(results_dir, f"ablation_analysis_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": ts, "lambda_cost": lam, "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print LaTeX-ready table
    print("\n=== ABLATION TABLE (LaTeX-ready) ===")
    print("Benchmark & Full & Halting-Only & No-Branch & Max-Only & Mid-Only \\\\")
    print("\\hline")
    for tag, res in all_results.items():
        vals = []
        for abl in ["full", "halting_only", "no_branch", "max_only", "mid_only"]:
            r = res.get(abl, {})
            vals.append(f"{r.get('accuracy', 0):.3f}/{r.get('avg_tokens', 0):.0f}")
        print(f"{tag} & {' & '.join(vals)} \\\\")


if __name__ == "__main__":
    main()
