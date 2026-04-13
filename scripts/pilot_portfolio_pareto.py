#!/usr/bin/env python3
"""
Portfolio Theory / Pareto Frontier Pilot Analysis
==================================================
Goal: Given per-sample accuracy data for think/nothink at multiple budgets,
compute the exact Pareto frontier and check if mixing strategies beats any
pure strategy.

Key question: does the optimal mixture beat nothink@512 (93.1%)?

Data sources (all seed=42, n=1319 GSM8K):
  - nothink_fullset/  : think & nothink @ {128, 256}
  - gap_fill/8b_highbudget/ : think & nothink @ {512, 1024}

Output: results/pilot_portfolio/
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "pilot_portfolio"

DATA_SOURCES = {
    # (mode, budget): (file_path_relative_to_project, per_sample_key)
    ("think", 128): ("results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_063213.json", "thinking_128"),
    ("think", 256): ("results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_000345.json", "thinking_256"),
    ("think", 512): ("results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_121035.json", "thinking_512"),
    ("think", 1024): ("results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_220034.json", "thinking_1024"),
    ("nothink", 128): ("results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_063213.json", "nothink_128"),
    ("nothink", 256): ("results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_000345.json", "nothink_256"),
    ("nothink", 512): ("results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_121035.json", "nothink_512"),
    ("nothink", 1024): ("results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_220034.json", "nothink_1024"),
}

# Budget levels for Pareto analysis
PARETO_BUDGETS = [128, 192, 256, 384, 512, 640, 768, 1024]

N_SAMPLES = 1319
SEED = 42

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "pilot_portfolio.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_per_sample_data() -> dict:
    """Load per-sample correctness from all sources. Returns {strategy_name: np.array[bool]}."""
    data = {}
    file_cache = {}  # avoid re-reading same file

    for (mode, budget), (rel_path, key) in DATA_SOURCES.items():
        full_path = str(PROJECT_ROOT / rel_path)
        if full_path not in file_cache:
            with open(full_path) as f:
                file_cache[full_path] = json.load(f)
            log.info(f"Loaded {rel_path}")

        d = file_cache[full_path]
        samples = d["per_sample"][key]
        assert len(samples) == N_SAMPLES, f"Expected {N_SAMPLES}, got {len(samples)} for {key}"

        # Sort by idx to ensure alignment
        samples_sorted = sorted(samples, key=lambda s: s["idx"])
        correct = np.array([s["correct"] for s in samples_sorted], dtype=bool)

        strategy_name = f"{mode}@{budget}"
        data[strategy_name] = correct
        acc = correct.mean()
        log.info(f"  {strategy_name}: acc={acc:.4f} ({correct.sum()}/{N_SAMPLES})")

    return data


# ---------------------------------------------------------------------------
# Analysis 1: Correlation matrix
# ---------------------------------------------------------------------------
def compute_correlation_matrix(data: dict) -> tuple:
    """Compute phi-coefficient (binary correlation) between all strategy pairs."""
    names = sorted(data.keys(), key=lambda s: (s.split("@")[0], int(s.split("@")[1])))
    n = len(names)
    corr = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            a = data[ni].astype(float)
            b = data[nj].astype(float)
            corr[i, j] = np.corrcoef(a, b)[0, 1]

    return names, corr


# ---------------------------------------------------------------------------
# Analysis 2: Oracle Pareto frontier (per-sample best)
# ---------------------------------------------------------------------------
def oracle_pareto_frontier(data: dict) -> dict:
    """For each budget B, find the best per-sample strategy selection (oracle).

    Strategy: for each sample, pick the cheapest strategy that gets it right.
    If multiple strategies at the same cost get it right, pick any.
    """
    # Parse strategies into (mode, budget, correct_array)
    strategies = []
    for name, correct in data.items():
        mode, budget = name.split("@")
        strategies.append((mode, int(budget), correct))

    results = {}
    for target_budget in PARETO_BUDGETS:
        # Available strategies: those with cost <= target_budget
        available = [(m, b, c) for m, b, c in strategies if b <= target_budget]
        if not available:
            results[target_budget] = {
                "oracle_acc": 0.0,
                "available_strategies": [],
                "per_sample_best": None,
            }
            continue

        # Oracle: for each sample, can it be solved by ANY available strategy?
        any_correct = np.zeros(N_SAMPLES, dtype=bool)
        for m, b, c in available:
            any_correct |= c

        oracle_acc = any_correct.mean()
        avail_names = [f"{m}@{b}" for m, b, c in available]
        results[target_budget] = {
            "oracle_acc": float(oracle_acc),
            "oracle_correct": int(any_correct.sum()),
            "available_strategies": avail_names,
        }
        log.info(f"Oracle @ B={target_budget}: acc={oracle_acc:.4f} using {avail_names}")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Static mixture Pareto frontier
# ---------------------------------------------------------------------------
def static_mixture_frontier(data: dict) -> dict:
    """For each budget B, find the best STATIC mixture of two strategies.

    A static mixture: allocate fraction p of samples to strategy A,
    (1-p) to strategy B. Without per-sample oracle, we randomly assign.
    Expected accuracy = p * acc_A + (1-p) * acc_B, subject to:
      p * cost_A + (1-p) * cost_B <= B

    Also consider: cascaded strategies where you run cheap first,
    then expensive on failures (but we don't know failures without oracle).
    """
    strategies = {}
    for name, correct in data.items():
        mode, budget = name.split("@")
        strategies[name] = {
            "mode": mode,
            "budget": int(budget),
            "correct": correct,
            "acc": float(correct.mean()),
        }

    results = {}
    for target_budget in PARETO_BUDGETS:
        best_acc = 0.0
        best_config = None

        # 1. Pure strategies
        for name, info in strategies.items():
            if info["budget"] <= target_budget:
                if info["acc"] > best_acc:
                    best_acc = info["acc"]
                    best_config = {"type": "pure", "strategy": name, "acc": info["acc"]}

        # 2. Two-strategy mixtures (random assignment)
        strat_names = list(strategies.keys())
        for i in range(len(strat_names)):
            for j in range(i + 1, len(strat_names)):
                s_a = strategies[strat_names[i]]
                s_b = strategies[strat_names[j]]
                c_a, c_b = s_a["budget"], s_b["budget"]
                a_a, a_b = s_a["acc"], s_b["acc"]

                if c_a == c_b:
                    # Same cost: just pick the better one (already handled as pure)
                    continue

                # p * c_a + (1-p) * c_b <= target_budget
                # p * (c_a - c_b) <= target_budget - c_b
                # If c_a > c_b: p <= (B - c_b) / (c_a - c_b)
                # If c_a < c_b: p >= (B - c_b) / (c_a - c_b) = (c_b - B) / (c_b - c_a)
                if c_a > c_b:
                    p_max = min(1.0, (target_budget - c_b) / (c_a - c_b))
                    p_min = 0.0
                else:
                    p_min = max(0.0, (c_b - target_budget) / (c_b - c_a))
                    p_max = 1.0

                if p_max < p_min:
                    continue

                # Optimal p: maximize p * a_a + (1-p) * a_b
                # This is linear in p, so optimal is at boundary
                if a_a > a_b:
                    p_opt = p_max
                elif a_a < a_b:
                    p_opt = p_min
                else:
                    p_opt = p_min  # doesn't matter

                mix_acc = p_opt * a_a + (1 - p_opt) * a_b
                mix_cost = p_opt * c_a + (1 - p_opt) * c_b

                if mix_acc > best_acc and mix_cost <= target_budget + 1e-6:
                    best_acc = mix_acc
                    best_config = {
                        "type": "mixture",
                        "strategy_a": strat_names[i],
                        "strategy_b": strat_names[j],
                        "p": float(p_opt),
                        "acc": float(mix_acc),
                        "avg_cost": float(mix_cost),
                    }

        results[target_budget] = best_config if best_config else {"type": "none", "acc": 0.0}
        if best_config:
            log.info(f"Static best @ B={target_budget}: acc={best_acc:.4f} config={best_config['type']}"
                     f" {'→ ' + best_config.get('strategy', best_config.get('strategy_a','')) }")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Cascade (cheap → expensive on failure)
# ---------------------------------------------------------------------------
def cascade_analysis(data: dict) -> dict:
    """Cascade: run cheap strategy first, then expensive on (predicted) failures.

    Without a router, the best cascade is:
      - Run cheap strategy on all samples (cost = cheap_budget)
      - For wrong answers: run expensive strategy (additional cost = expensive_budget)
      - Total cost = cheap_budget + (1 - cheap_acc) * expensive_budget

    But we need to know which are wrong → need a confidence signal.
    Here we compute the ORACLE cascade (knowing which are wrong).
    """
    strategies = {}
    for name, correct in data.items():
        mode, budget = name.split("@")
        strategies[name] = {
            "mode": mode, "budget": int(budget), "correct": correct,
            "acc": float(correct.mean()),
        }

    results = {}
    strat_names = list(strategies.keys())

    for i in range(len(strat_names)):
        for j in range(len(strat_names)):
            if i == j:
                continue
            first = strategies[strat_names[i]]
            second = strategies[strat_names[j]]

            # Run first on all, then second on first's failures
            first_correct = first["correct"]
            second_correct = second["correct"]

            # Cascade correctness: correct if first is correct, OR second is correct
            cascade_correct = first_correct | second_correct
            cascade_acc = cascade_correct.mean()

            # Oracle cost: first on all + second only on first's failures
            n_failures = (~first_correct).sum()
            oracle_avg_cost = first["budget"] + (n_failures / N_SAMPLES) * second["budget"]

            cascade_name = f"{strat_names[i]} → {strat_names[j]}"
            results[cascade_name] = {
                "first": strat_names[i],
                "second": strat_names[j],
                "cascade_acc": float(cascade_acc),
                "oracle_avg_cost": float(oracle_avg_cost),
                "first_acc": float(first["acc"]),
                "fallback_rate": float(n_failures / N_SAMPLES),
                "second_saves": int((~first_correct & second_correct).sum()),
            }

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Think/nothink complementarity
# ---------------------------------------------------------------------------
def complementarity_analysis(data: dict) -> dict:
    """Analyze how think and nothink complement each other at same budget."""
    results = {}
    budgets = [128, 256, 512, 1024]

    for b in budgets:
        t_key = f"think@{b}"
        nt_key = f"nothink@{b}"
        if t_key not in data or nt_key not in data:
            continue

        t = data[t_key]
        nt = data[nt_key]

        both_correct = (t & nt).sum()
        only_think = (t & ~nt).sum()
        only_nothink = (~t & nt).sum()
        neither = (~t & ~nt).sum()
        union = (t | nt).sum()

        results[b] = {
            "think_acc": float(t.mean()),
            "nothink_acc": float(nt.mean()),
            "both_correct": int(both_correct),
            "only_think": int(only_think),
            "only_nothink": int(only_nothink),
            "neither": int(neither),
            "union_acc": float(union / N_SAMPLES),
            "jaccard": float(both_correct / union) if union > 0 else 0.0,
            "conditional_p_think_given_nothink_wrong": float(
                (t & ~nt).sum() / (~nt).sum()
            ) if (~nt).sum() > 0 else 0.0,
            "conditional_p_nothink_given_think_wrong": float(
                (nt & ~t).sum() / (~t).sum()
            ) if (~t).sum() > 0 else 0.0,
        }

    return results


# ---------------------------------------------------------------------------
# Analysis 6: Pareto valley detection
# ---------------------------------------------------------------------------
def pareto_valley_analysis(data: dict) -> dict:
    """Is there a budget range where neither think nor nothink dominates?

    Pareto valley: budget B where best pure think < best pure nothink,
    but both are suboptimal compared to a mixture.
    """
    # Collect pure strategy performance at each budget
    think_curve = {}
    nothink_curve = {}
    for name, correct in data.items():
        mode, budget_str = name.split("@")
        budget = int(budget_str)
        acc = float(correct.mean())
        if mode == "think":
            think_curve[budget] = acc
        else:
            nothink_curve[budget] = acc

    # For each budget, find the dominant mode
    valley_info = {}
    for b in sorted(set(list(think_curve.keys()) + list(nothink_curve.keys()))):
        t_acc = think_curve.get(b)
        nt_acc = nothink_curve.get(b)
        valley_info[b] = {
            "think_acc": t_acc,
            "nothink_acc": nt_acc,
            "dominant": "think" if (t_acc or 0) > (nt_acc or 0) else "nothink",
            "gap": abs((t_acc or 0) - (nt_acc or 0)),
        }

    # Find crossover point
    budgets_sorted = sorted(valley_info.keys())
    crossover = None
    for i in range(len(budgets_sorted) - 1):
        b1, b2 = budgets_sorted[i], budgets_sorted[i + 1]
        d1 = valley_info[b1]["dominant"]
        d2 = valley_info[b2]["dominant"]
        if d1 != d2:
            crossover = (b1, b2)
            break

    return {
        "per_budget": valley_info,
        "crossover_between": crossover,
        "think_budgets": think_curve,
        "nothink_budgets": nothink_curve,
    }


# ---------------------------------------------------------------------------
# Analysis 7: Best achievable mixtures (knapsack-style)
# ---------------------------------------------------------------------------
def best_achievable_mixture(data: dict) -> dict:
    """For mixed-budget portfolios: what if we allocate different budgets
    to different fractions of samples?

    E.g., total budget B=512 per sample on average:
    - Could run nothink@256 on 50% + nothink@768 on 50% (but no 768 data)
    - Could run nothink@128 on 25% + nothink@1024 on 25% + think@256 on 50%

    We search over all two-strategy mixtures with proportions.
    The key insight: with random assignment (no oracle), the expected accuracy
    of a mixture is just the weighted average of individual accuracies.
    So a mixture can only beat a pure strategy if the Pareto frontier is concave.
    """
    strategies = []
    for name, correct in data.items():
        mode, budget_str = name.split("@")
        strategies.append({
            "name": name,
            "budget": int(budget_str),
            "acc": float(correct.mean()),
            "correct": correct,
        })

    # Sort by budget
    strategies.sort(key=lambda s: s["budget"])

    results = {}
    for target_budget in PARETO_BUDGETS:
        best = {"acc": 0.0, "config": "none"}

        # Pure strategies
        for s in strategies:
            if s["budget"] <= target_budget and s["acc"] > best["acc"]:
                best = {"acc": s["acc"], "config": f"pure {s['name']}",
                        "avg_cost": s["budget"]}

        # Two-strategy mixtures
        for i, sa in enumerate(strategies):
            for j, sb in enumerate(strategies):
                if i >= j:
                    continue
                ca, cb = sa["budget"], sb["budget"]
                aa, ab = sa["acc"], sb["acc"]

                # Sweep p in [0, 1]: average cost = p*ca + (1-p)*cb <= target
                # and acc = p*aa + (1-p)*ab
                if ca == cb:
                    continue

                if ca < cb:
                    # p is fraction of sa (cheaper)
                    p_min = max(0.0, (cb - target_budget) / (cb - ca))
                    p_max = 1.0
                else:
                    p_min = 0.0
                    p_max = min(1.0, (target_budget - cb) / (ca - cb))

                if p_max < p_min - 1e-9:
                    continue

                # Linear in p → optimal at boundary
                for p in [max(0, p_min), min(1, p_max)]:
                    mix_acc = p * aa + (1 - p) * ab
                    mix_cost = p * ca + (1 - p) * cb
                    if mix_cost <= target_budget + 1e-6 and mix_acc > best["acc"]:
                        best = {
                            "acc": float(mix_acc),
                            "config": f"mix {sa['name']}({p:.3f}) + {sb['name']}({1-p:.3f})",
                            "avg_cost": float(mix_cost),
                            "p": float(p),
                            "strategy_a": sa["name"],
                            "strategy_b": sb["name"],
                        }

        results[target_budget] = best

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(
    data: dict,
    corr_names: list,
    corr_matrix: np.ndarray,
    oracle: dict,
    static: dict,
    cascades: dict,
    complementarity: dict,
    valley: dict,
    achievable: dict,
):
    """Print comprehensive analysis report."""
    sep = "=" * 78

    print(f"\n{sep}")
    print("PORTFOLIO THEORY / PARETO FRONTIER — PILOT ANALYSIS")
    print(f"Model: Qwen3-8B | Benchmark: GSM8K (n={N_SAMPLES}) | Seed: {SEED}")
    print(sep)

    # --- Strategy accuracies ---
    print("\n1. STRATEGY ACCURACIES")
    print("-" * 50)
    for name in sorted(data.keys(), key=lambda s: (s.split("@")[0], int(s.split("@")[1]))):
        acc = data[name].mean()
        n_correct = data[name].sum()
        print(f"  {name:>15s}: {acc:.4f}  ({n_correct}/{N_SAMPLES})")

    # --- Correlation matrix ---
    print(f"\n2. CORRELATION MATRIX (phi coefficient)")
    print("-" * 50)
    # Header
    header = "              " + " ".join(f"{n:>12s}" for n in corr_names)
    print(header)
    for i, ni in enumerate(corr_names):
        row = f"  {ni:>12s}" + " ".join(f"{corr_matrix[i,j]:12.3f}" for j in range(len(corr_names)))
        print(row)

    # --- Complementarity ---
    print(f"\n3. THINK / NOTHINK COMPLEMENTARITY (same budget)")
    print("-" * 60)
    print(f"  {'Budget':>6s} {'Think':>8s} {'Nothink':>8s} {'Both✓':>7s} {'Only-T':>7s} {'Only-NT':>8s} {'Neither':>8s} {'Union':>8s}")
    for b, info in sorted(complementarity.items()):
        print(f"  {b:>6d} {info['think_acc']:>8.4f} {info['nothink_acc']:>8.4f} "
              f"{info['both_correct']:>7d} {info['only_think']:>7d} {info['only_nothink']:>8d} "
              f"{info['neither']:>8d} {info['union_acc']:>8.4f}")

    print(f"\n  Conditional probabilities:")
    for b, info in sorted(complementarity.items()):
        print(f"    @{b}: P(think✓|nothink✗) = {info['conditional_p_think_given_nothink_wrong']:.4f}"
              f"    P(nothink✓|think✗) = {info['conditional_p_nothink_given_think_wrong']:.4f}")

    # --- Pareto valley ---
    print(f"\n4. PARETO VALLEY ANALYSIS")
    print("-" * 50)
    print(f"  {'Budget':>6s} {'Think':>8s} {'Nothink':>8s} {'Dominant':>10s} {'Gap':>8s}")
    for b, info in sorted(valley["per_budget"].items()):
        t = f"{info['think_acc']:.4f}" if info['think_acc'] is not None else "   N/A"
        nt = f"{info['nothink_acc']:.4f}" if info['nothink_acc'] is not None else "   N/A"
        print(f"  {b:>6d} {t:>8s} {nt:>8s} {info['dominant']:>10s} {info['gap']:>8.4f}")

    if valley["crossover_between"]:
        b1, b2 = valley["crossover_between"]
        print(f"\n  → Crossover point: between budget {b1} and {b2}")
    else:
        print(f"\n  → No crossover detected — one mode dominates at all budgets")

    # --- Oracle Pareto ---
    print(f"\n5. ORACLE PARETO FRONTIER (per-sample best)")
    print("-" * 60)
    print(f"  {'Budget':>6s} {'Oracle Acc':>11s} {'# Strategies':>13s} {'Strategies'}")
    for b in PARETO_BUDGETS:
        info = oracle.get(b, {})
        acc = info.get("oracle_acc", 0)
        n_strats = len(info.get("available_strategies", []))
        strats = ", ".join(info.get("available_strategies", []))
        print(f"  {b:>6d} {acc:>11.4f} {n_strats:>13d}   {strats}")

    # --- Achievable Pareto ---
    print(f"\n6. ACHIEVABLE PARETO FRONTIER (static mixtures)")
    print("-" * 78)
    print(f"  {'Budget':>6s} {'Best Acc':>9s} {'Avg Cost':>9s} {'Configuration'}")
    for b in PARETO_BUDGETS:
        info = achievable.get(b, {})
        acc = info.get("acc", 0)
        cost = info.get("avg_cost", 0)
        config = info.get("config", "none")
        print(f"  {b:>6d} {acc:>9.4f} {cost:>9.1f}   {config}")

    # --- Top cascades ---
    print(f"\n7. TOP CASCADE STRATEGIES (oracle fallback)")
    print("-" * 78)
    # Sort cascades by accuracy, then cost
    cascade_list = sorted(cascades.items(), key=lambda x: (-x[1]["cascade_acc"], x[1]["oracle_avg_cost"]))
    print(f"  {'Cascade':>35s} {'Acc':>7s} {'Avg Cost':>9s} {'Fallback%':>10s} {'#Saved':>7s}")
    for name, info in cascade_list[:15]:
        print(f"  {name:>35s} {info['cascade_acc']:>7.4f} {info['oracle_avg_cost']:>9.1f} "
              f"{info['fallback_rate']:>10.4f} {info['second_saves']:>7d}")

    # --- Key findings ---
    print(f"\n{sep}")
    print("KEY FINDINGS")
    print(sep)

    # Best pure nothink
    best_pure_nt = max(
        [(n, data[n].mean()) for n in data if n.startswith("nothink")],
        key=lambda x: x[1]
    )
    print(f"\n  Best pure nothink: {best_pure_nt[0]} = {best_pure_nt[1]:.4f}")

    # Best pure think
    best_pure_t = max(
        [(n, data[n].mean()) for n in data if n.startswith("think")],
        key=lambda x: x[1]
    )
    print(f"  Best pure think:   {best_pure_t[0]} = {best_pure_t[1]:.4f}")

    # Oracle at same budget as best pure
    nothink_512_acc = data.get("nothink@512", np.zeros(1)).mean()
    oracle_512 = oracle.get(512, {}).get("oracle_acc", 0)
    achievable_512 = achievable.get(512, {}).get("acc", 0)
    print(f"\n  At budget=512:")
    print(f"    nothink@512 (pure):     {nothink_512_acc:.4f}")
    print(f"    Best static mixture:    {achievable_512:.4f}")
    print(f"    Oracle (per-sample):    {oracle_512:.4f}")
    print(f"    Mixture gain over pure: {achievable_512 - nothink_512_acc:+.4f}")
    print(f"    Oracle gap:             {oracle_512 - nothink_512_acc:+.4f}")

    # Does mixing beat nothink@512 (93.1%)?
    print(f"\n  ★ Does mixing beat nothink@512 ({nothink_512_acc:.4f})?")
    for b in PARETO_BUDGETS:
        a = achievable.get(b, {})
        if a.get("acc", 0) > nothink_512_acc and a.get("avg_cost", 9999) <= 512:
            print(f"    YES at avg_cost={a['avg_cost']:.0f}: {a['acc']:.4f} via {a['config']}")
            break
    else:
        print(f"    Checking oracle...")
        if oracle_512 > nothink_512_acc:
            print(f"    Oracle YES ({oracle_512:.4f}), but static mixture NO ({achievable_512:.4f})")
            print(f"    → A router is needed to unlock the oracle gain")
        else:
            print(f"    NO — nothink@512 is already Pareto optimal")

    # Complementarity summary
    print(f"\n  Think/nothink complementarity summary:")
    for b in [256, 512, 1024]:
        if b in complementarity:
            c = complementarity[b]
            print(f"    @{b}: {c['only_think']} only-think + {c['only_nothink']} only-nothink"
                  f" = {c['only_think']+c['only_nothink']} unique solvers"
                  f" → union {c['union_acc']:.4f} (vs {max(c['think_acc'], c['nothink_acc']):.4f} best pure)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Portfolio/Pareto frontier pilot analysis")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("Loading per-sample data...")
    data = load_per_sample_data()

    log.info("\nComputing correlation matrix...")
    corr_names, corr_matrix = compute_correlation_matrix(data)

    log.info("\nComputing oracle Pareto frontier...")
    oracle = oracle_pareto_frontier(data)

    log.info("\nComputing static mixture frontier...")
    static = static_mixture_frontier(data)

    log.info("\nComputing cascade analysis...")
    cascades = cascade_analysis(data)

    log.info("\nComputing complementarity analysis...")
    complementarity = complementarity_analysis(data)

    log.info("\nComputing Pareto valley analysis...")
    valley = pareto_valley_analysis(data)

    log.info("\nComputing achievable mixture frontier...")
    achievable = best_achievable_mixture(data)

    # --- Print report ---
    print_report(data, corr_names, corr_matrix, oracle, static, cascades,
                 complementarity, valley, achievable)

    # --- Save results ---
    output = {
        "meta": {
            "model": "Qwen/Qwen3-8B",
            "benchmark": "gsm8k",
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "strategies": list(data.keys()),
            "pareto_budgets": PARETO_BUDGETS,
        },
        "strategy_accuracies": {name: float(data[name].mean()) for name in data},
        "correlation_matrix": {
            "names": corr_names,
            "matrix": corr_matrix.tolist(),
        },
        "oracle_pareto": {str(k): v for k, v in oracle.items()},
        "achievable_pareto": {str(k): v for k, v in achievable.items()},
        "cascades_top10": dict(sorted(
            cascades.items(),
            key=lambda x: (-x[1]["cascade_acc"], x[1]["oracle_avg_cost"])
        )[:10]),
        "complementarity": {str(k): v for k, v in complementarity.items()},
        "valley_analysis": {
            "per_budget": {str(k): v for k, v in valley["per_budget"].items()},
            "crossover_between": valley["crossover_between"],
        },
    }

    out_path = Path(args.output_dir) / "portfolio_pareto_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Also save a compact summary
    summary_path = Path(args.output_dir) / "portfolio_pareto_summary.txt"
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        print_report(data, corr_names, corr_matrix, oracle, static, cascades,
                     complementarity, valley, achievable)
    with open(summary_path, "w") as f:
        f.write(buf.getvalue())
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
