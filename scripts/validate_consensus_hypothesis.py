#!/usr/bin/env python3
"""Phase 0: Validate the Consensus-Difficulty Hypothesis.

Uses existing fulltest (8B, 1319 GSM8K samples) and 23-seed (27B, 920 samples)
data to answer:

  1. Cross-budget agreement: When predictions at different budgets agree,
     is the question easier (higher oracle accuracy)?
  2. Overthinking vs. underthinking: For questions where budget consensus
     is low, is the oracle accuracy also low?
  3. Simulated K-path consensus: If we had K independent reasoning paths,
     what accuracy improvement would consensus-based allocation give?

Note: Because fixed_128/256/512 come from the SAME trace (truncated at
different lengths), cross-budget "consensus" is a LOWER BOUND on what
truly independent paths would provide.  The real multi-path experiment
(temperature>0, different seeds) will be stronger.

Output: results/consensus_validation.json
"""

import argparse
import csv
import glob
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------- helpers ----------------------------------------------------------
NUM_RE_PAT = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"

def to_float(s):
    if s is None or str(s).strip() in ("", "nan", "None"):
        return None
    s = str(s).replace(",", "").strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                d = float(parts[1])
                return float(parts[0]) / d if d != 0 else None
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None

def is_correct(pred, gold, tol=1e-6):
    p, g = to_float(pred), to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))

def preds_agree(preds):
    """Check how many of the non-None preds agree (majority)."""
    vals = []
    for p in preds:
        v = to_float(p)
        if v is not None:
            vals.append(round(v, 6))
    if not vals:
        return 0, 0  # agreement_ratio, n_valid
    ctr = Counter(vals)
    majority_count = ctr.most_common(1)[0][1]
    return majority_count / len(vals), len(vals)


# ---------- Load fulltest data -----------------------------------------------
def load_fulltest_csv(path):
    """Load the 8B fulltest CSV and return list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def detect_budget_columns(fieldnames):
    """Detect available budget columns from CSV headers."""
    budgets = set()
    for col in fieldnames:
        if col.startswith("fixed_") and col.endswith("_correct"):
            budget = col.replace("fixed_", "").replace("_correct", "")
            try:
                int(budget)
                budgets.add(int(budget))
            except ValueError:
                pass
    return sorted(budgets)

# ---------- Analysis 1: Cross-budget consensus vs difficulty -----------------
def analyze_cross_budget_consensus(rows, budgets):
    """For each question, compute agreement among budget predictions
    and correlate with oracle correctness."""
    results = []
    for row in rows:
        gold = row.get("gold", "")
        preds = {}
        corrects = {}
        for b in budgets:
            pred_col = f"fixed_{b}_pred"
            correct_col = f"fixed_{b}_correct"
            preds[b] = row.get(pred_col, None)
            c = row.get(correct_col, None)
            corrects[b] = bool(int(float(c))) if c and c not in ("", "nan", "None") else False

        # Consensus: do all budgets agree on the same answer?
        pred_vals = [to_float(preds[b]) for b in budgets]
        valid_preds = [v for v in pred_vals if v is not None]

        if not valid_preds:
            consensus_ratio = 0.0
            n_valid = 0
        else:
            rounded = [round(v, 4) for v in valid_preds]
            ctr = Counter(rounded)
            majority = ctr.most_common(1)[0][1]
            consensus_ratio = majority / len(rounded)
            n_valid = len(rounded)

        # Oracle: correct at ANY budget
        oracle_correct = any(corrects[b] for b in budgets)
        # Best budget: highest accuracy
        best_budget_correct = corrects[budgets[-1]]  # highest budget
        # Lowest budget correct
        lowest_correct = corrects[budgets[0]]

        # Difficulty proxy: how many budgets get it right
        n_correct = sum(1 for b in budgets if corrects[b])
        difficulty = 1.0 - n_correct / len(budgets)  # 0=easy, 1=hard

        results.append({
            "consensus_ratio": consensus_ratio,
            "n_valid_preds": n_valid,
            "oracle_correct": oracle_correct,
            "best_budget_correct": best_budget_correct,
            "lowest_correct": lowest_correct,
            "n_budgets_correct": n_correct,
            "difficulty": difficulty,
            "corrects": {str(b): corrects[b] for b in budgets},
        })
    return results

# ---------- Analysis 2: Consensus-based allocation simulation ----------------
def simulate_consensus_allocation(rows, budgets, strategy="threshold"):
    """Simulate what would happen if we used consensus to allocate budgets.

    Strategy:
    - If all budgets agree → use lowest budget (save tokens)
    - If partial agreement → use middle budget
    - If no agreement → use highest budget
    """
    total = len(rows)
    allocations = {str(b): 0 for b in budgets}
    correct_count = 0
    total_tokens = 0

    # Also compute fixed-budget baselines
    fixed_results = {}
    for b in budgets:
        fixed_correct = sum(1 for r in rows if r.get(f"fixed_{b}_correct") and
                          bool(int(float(r.get(f"fixed_{b}_correct", "0")))))
        fixed_tokens = sum(float(r.get(f"fixed_{b}_tokens", "0")) for r in rows)
        fixed_results[str(b)] = {
            "accuracy": fixed_correct / total,
            "avg_tokens": fixed_tokens / total,
        }

    # Oracle
    oracle_correct = 0
    oracle_tokens = 0

    for row in rows:
        gold = row.get("gold", "")
        preds = {}
        corrects = {}
        tokens_map = {}
        for b in budgets:
            preds[b] = row.get(f"fixed_{b}_pred", None)
            c = row.get(f"fixed_{b}_correct", "0")
            corrects[b] = bool(int(float(c))) if c and c not in ("", "nan", "None") else False
            tokens_map[b] = float(row.get(f"fixed_{b}_tokens", "0"))

        # Compute consensus
        pred_vals = []
        for b in budgets:
            v = to_float(preds[b])
            if v is not None:
                pred_vals.append((b, round(v, 4)))

        if len(pred_vals) <= 1:
            # Not enough predictions → use highest budget
            chosen = budgets[-1]
        else:
            vals_only = [v for _, v in pred_vals]
            ctr = Counter(vals_only)
            majority_val, majority_count = ctr.most_common(1)[0]
            agreement = majority_count / len(vals_only)

            if agreement >= 1.0:
                # Full consensus → use lowest budget that got the majority answer
                chosen = min(b for b, v in pred_vals if round(v, 4) == majority_val)
            elif agreement >= 0.5:
                # Partial consensus → use middle budget
                chosen = budgets[len(budgets) // 2]
            else:
                # No consensus → use highest budget
                chosen = budgets[-1]

        allocations[str(chosen)] += 1
        is_right = corrects[chosen]
        correct_count += int(is_right)
        total_tokens += tokens_map[chosen]

        # Oracle: pick budget with correct answer and lowest tokens
        oracle_budget = None
        for b in budgets:
            if corrects[b]:
                oracle_budget = b
                break
        if oracle_budget is None:
            oracle_budget = budgets[-1]
        oracle_correct += int(corrects.get(oracle_budget, False))
        oracle_tokens += tokens_map[oracle_budget]

    return {
        "consensus_allocation": {
            "accuracy": correct_count / total,
            "avg_tokens": total_tokens / total,
            "budget_distribution": {k: v / total for k, v in allocations.items()},
        },
        "fixed_baselines": fixed_results,
        "oracle": {
            "accuracy": oracle_correct / total,
            "avg_tokens": oracle_tokens / total,
        },
        "n_samples": total,
    }


# ---------- Analysis 3: Difficulty bucket analysis ----------------------------
def difficulty_bucket_analysis(analysis_results):
    """Group questions by difficulty and show consensus patterns."""
    buckets = defaultdict(list)
    for r in analysis_results:
        d = r["difficulty"]
        if d == 0:
            label = "easy (all budgets correct)"
        elif d < 0.5:
            label = "medium (most budgets correct)"
        elif d < 1.0:
            label = "hard (some budgets correct)"
        else:
            label = "impossible (no budget correct)"
        buckets[label].append(r)

    summary = {}
    for label, items in sorted(buckets.items()):
        consensus_vals = [r["consensus_ratio"] for r in items]
        summary[label] = {
            "count": len(items),
            "fraction": len(items) / len(analysis_results),
            "avg_consensus": np.mean(consensus_vals),
            "std_consensus": np.std(consensus_vals),
            "avg_difficulty": np.mean([r["difficulty"] for r in items]),
        }
    return summary

# ---------- Analysis 4: Spearman correlation ---------------------------------
def compute_spearman(xs, ys):
    """Compute Spearman rank correlation."""
    from scipy import stats
    rho, pval = stats.spearmanr(xs, ys)
    return float(rho), float(pval)

def consensus_difficulty_correlation(analysis_results):
    """Spearman correlation between consensus ratio and difficulty."""
    consensus = [r["consensus_ratio"] for r in analysis_results]
    difficulty = [r["difficulty"] for r in analysis_results]
    n_correct = [r["n_budgets_correct"] for r in analysis_results]

    try:
        rho_diff, p_diff = compute_spearman(consensus, difficulty)
        rho_ncorr, p_ncorr = compute_spearman(consensus, n_correct)
    except Exception as e:
        log.warning(f"Spearman computation failed: {e}")
        rho_diff, p_diff = 0.0, 1.0
        rho_ncorr, p_ncorr = 0.0, 1.0

    return {
        "consensus_vs_difficulty": {
            "spearman_rho": rho_diff,
            "p_value": p_diff,
            "interpretation": (
                "STRONG negative" if rho_diff < -0.5 else
                "moderate negative" if rho_diff < -0.3 else
                "weak" if abs(rho_diff) < 0.3 else
                "moderate positive" if rho_diff < 0.5 else
                "STRONG positive"
            ),
        },
        "consensus_vs_n_correct": {
            "spearman_rho": rho_ncorr,
            "p_value": p_ncorr,
        },
        "note": (
            "Negative rho means higher consensus → lower difficulty (easier questions). "
            "This is what we WANT for Reasoning Speculation to work."
        ),
    }


# ---------- Analysis 5: Information gain of multi-path over single-path ------
def information_gain_analysis(analysis_results):
    """Estimate information gain from multi-path consensus over single path."""
    # For questions where all budgets agree AND are correct
    full_agree_correct = sum(1 for r in analysis_results
                            if r["consensus_ratio"] >= 1.0 and r["oracle_correct"])
    full_agree_wrong = sum(1 for r in analysis_results
                          if r["consensus_ratio"] >= 1.0 and not r["oracle_correct"])
    partial_agree = sum(1 for r in analysis_results if 0 < r["consensus_ratio"] < 1.0)
    no_agree = sum(1 for r in analysis_results if r["consensus_ratio"] == 0)

    total = len(analysis_results)

    # Precision of "full consensus → easy" heuristic
    full_agree_total = full_agree_correct + full_agree_wrong
    precision = full_agree_correct / full_agree_total if full_agree_total > 0 else 0

    return {
        "full_consensus_correct": full_agree_correct,
        "full_consensus_wrong": full_agree_wrong,
        "partial_consensus": partial_agree,
        "no_consensus": no_agree,
        "total": total,
        "consensus_precision": precision,
        "note": (
            f"When all budgets fully agree, {precision:.1%} of the time the question "
            f"is actually correct (oracle). This is the precision of consensus as a "
            f"difficulty signal. Higher = better for Reasoning Speculation."
        ),
    }


# ---------- main -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate Consensus-Difficulty Hypothesis")
    parser.add_argument("--result_dir", default="results_kun", help="Results directory")
    parser.add_argument("--output", default="results/consensus_validation.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    results = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ---- Load 8B fulltest data (best available: 1319 GSM8K questions) -------
    fulltest_dir = os.path.join(args.result_dir, "fulltest")
    gsm8k_csvs = sorted(glob.glob(os.path.join(fulltest_dir, "per_sample_gsm8k_Qwen3_8B*.csv")))

    if gsm8k_csvs:
        csv_path = gsm8k_csvs[0]  # Should be the main one
        log.info(f"Loading fulltest CSV: {csv_path}")
        rows = load_fulltest_csv(csv_path)
        log.info(f"Loaded {len(rows)} rows")

        # Detect budgets
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        budgets = detect_budget_columns(fieldnames)
        log.info(f"Detected budgets: {budgets}")

        if budgets:
            # Analysis 1: Cross-budget consensus
            log.info("Running cross-budget consensus analysis...")
            analysis = analyze_cross_budget_consensus(rows, budgets)

            # Analysis 2: Consensus-based allocation simulation
            log.info("Running consensus allocation simulation...")
            allocation_sim = simulate_consensus_allocation(rows, budgets)

            # Analysis 3: Difficulty buckets
            log.info("Running difficulty bucket analysis...")
            bucket_analysis = difficulty_bucket_analysis(analysis)

            # Analysis 4: Spearman correlation
            log.info("Computing Spearman correlation...")
            correlation = consensus_difficulty_correlation(analysis)

            # Analysis 5: Information gain
            log.info("Computing information gain...")
            info_gain = information_gain_analysis(analysis)

            results["fulltest_8b_gsm8k"] = {
                "n_samples": len(rows),
                "budgets": budgets,
                "csv_path": csv_path,
                "allocation_simulation": allocation_sim,
                "difficulty_buckets": bucket_analysis,
                "correlation": correlation,
                "information_gain": info_gain,
            }
    else:
        log.warning("No fulltest CSV found")

    # ---- Load 27B 23-seed subset data (find overlapping questions) ----------
    log.info("Looking for 27B 23-seed data for cross-seed analysis...")
    seed_csvs = sorted(glob.glob(os.path.join(args.result_dir, "per_sample_Qwen3.5_27B_202602*.csv")))

    if len(seed_csvs) >= 3:
        log.info(f"Found {len(seed_csvs)} 27B seed CSVs")

        # Group by question text (find overlapping questions)
        question_data = defaultdict(list)  # question_text -> list of (csv_idx, row)

        for csv_idx, csv_path in enumerate(seed_csvs):
            rows_27b = load_fulltest_csv(csv_path)
            for row in rows_27b:
                q = row.get("question", "").strip()
                if q:
                    question_data[q].append((csv_idx, row))

        # Find questions that appear in >= 2 seeds
        overlap_questions = {q: data for q, data in question_data.items() if len(data) >= 2}
        log.info(f"Found {len(overlap_questions)} questions with >=2 seed overlap")

        if overlap_questions:
            # For overlapping questions, compute cross-seed consensus
            cross_seed_results = []
            for q, seed_rows in overlap_questions.items():
                gold = seed_rows[0][1].get("gold", "")

                # Detect budgets from first row
                if not budgets:
                    budgets = [128, 256, 512]

                for b in budgets:
                    preds = []
                    corrects = []
                    for csv_idx, row in seed_rows:
                        pred = row.get(f"fixed_{b}_pred", None)
                        c = row.get(f"fixed_{b}_correct", "0")
                        is_c = bool(int(float(c))) if c and c not in ("", "nan", "None") else False
                        preds.append(pred)
                        corrects.append(is_c)

                    agreement, n_valid = preds_agree(preds)
                    any_correct = any(corrects)
                    all_correct = all(corrects) if corrects else False

                    cross_seed_results.append({
                        "budget": b,
                        "n_seeds": len(seed_rows),
                        "agreement": agreement,
                        "any_correct": any_correct,
                        "all_correct": all_correct,
                        "n_correct": sum(corrects),
                    })

            # Compute cross-seed consensus-accuracy correlation
            agreements = [r["agreement"] for r in cross_seed_results]
            n_corrects = [r["n_correct"] / r["n_seeds"] for r in cross_seed_results]

            try:
                rho, pval = compute_spearman(agreements, n_corrects)
            except Exception:
                rho, pval = 0.0, 1.0

            results["cross_seed_27b"] = {
                "n_overlap_questions": len(overlap_questions),
                "n_total_observations": len(cross_seed_results),
                "avg_seeds_per_question": np.mean([len(v) for v in overlap_questions.values()]),
                "consensus_accuracy_correlation": {
                    "spearman_rho": rho,
                    "p_value": pval,
                },
                "summary_by_budget": {},
            }

            for b in budgets:
                b_results = [r for r in cross_seed_results if r["budget"] == b]
                if b_results:
                    high_agree = [r for r in b_results if r["agreement"] >= 0.8]
                    low_agree = [r for r in b_results if r["agreement"] < 0.5]
                    results["cross_seed_27b"]["summary_by_budget"][str(b)] = {
                        "n": len(b_results),
                        "high_agreement_accuracy": (
                            np.mean([r["n_correct"]/r["n_seeds"] for r in high_agree])
                            if high_agree else None
                        ),
                        "low_agreement_accuracy": (
                            np.mean([r["n_correct"]/r["n_seeds"] for r in low_agree])
                            if low_agree else None
                        ),
                    }

    # ---- Summary & verdict --------------------------------------------------
    verdict = {
        "hypothesis": "Multi-path consensus correlates with question difficulty",
        "timestamp": timestamp,
    }

    if "fulltest_8b_gsm8k" in results:
        corr = results["fulltest_8b_gsm8k"]["correlation"]
        rho = corr["consensus_vs_difficulty"]["spearman_rho"]
        alloc = results["fulltest_8b_gsm8k"]["allocation_simulation"]

        consensus_acc = alloc["consensus_allocation"]["accuracy"]
        best_fixed_acc = max(v["accuracy"] for v in alloc["fixed_baselines"].values())
        oracle_acc = alloc["oracle"]["accuracy"]

        consensus_tok = alloc["consensus_allocation"]["avg_tokens"]
        best_fixed_tok = min(v["avg_tokens"] for k, v in alloc["fixed_baselines"].items()
                           if v["accuracy"] >= consensus_acc * 0.95) if consensus_acc > 0 else 9999

        verdict["cross_budget_spearman_rho"] = rho
        verdict["consensus_accuracy"] = consensus_acc
        verdict["best_fixed_accuracy"] = best_fixed_acc
        verdict["oracle_accuracy"] = oracle_acc
        verdict["consensus_avg_tokens"] = consensus_tok

        # Decision
        if abs(rho) > 0.5:
            verdict["signal_strength"] = "STRONG"
        elif abs(rho) > 0.3:
            verdict["signal_strength"] = "MODERATE"
        else:
            verdict["signal_strength"] = "WEAK"

        if consensus_acc > best_fixed_acc:
            verdict["allocation_beats_fixed"] = True
            verdict["delta_acc"] = consensus_acc - best_fixed_acc
        else:
            verdict["allocation_beats_fixed"] = False
            verdict["delta_acc"] = consensus_acc - best_fixed_acc

        verdict["proceed_to_phase1"] = (
            abs(rho) > 0.2 or consensus_acc > best_fixed_acc * 0.95
        )
        verdict["recommendation"] = (
            "✅ PROCEED: Consensus signal is informative. Implement multi-path generation."
            if verdict["proceed_to_phase1"]
            else "⚠️ WEAK SIGNAL: Consider alternative approaches."
        )

    results["verdict"] = verdict

    # ---- Save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"Results saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("CONSENSUS-DIFFICULTY HYPOTHESIS VALIDATION")
    print("=" * 70)
    if "fulltest_8b_gsm8k" in results:
        r = results["fulltest_8b_gsm8k"]
        print(f"\n📊 8B Fulltest GSM8K (n={r['n_samples']}):")
        print(f"   Budgets: {r['budgets']}")

        corr = r["correlation"]
        print(f"\n   Spearman ρ (consensus vs difficulty): {corr['consensus_vs_difficulty']['spearman_rho']:.4f}")
        print(f"   p-value: {corr['consensus_vs_difficulty']['p_value']:.2e}")
        print(f"   Interpretation: {corr['consensus_vs_difficulty']['interpretation']}")

        alloc = r["allocation_simulation"]
        print(f"\n   📈 Allocation Simulation:")
        print(f"   {'Method':<25} {'Accuracy':>10} {'Avg Tokens':>12}")
        print(f"   {'-'*25} {'-'*10} {'-'*12}")
        for b, v in alloc["fixed_baselines"].items():
            print(f"   Fixed {b:<20} {v['accuracy']:>10.4f} {v['avg_tokens']:>12.1f}")
        ca = alloc["consensus_allocation"]
        print(f"   {'Consensus Alloc':<25} {ca['accuracy']:>10.4f} {ca['avg_tokens']:>12.1f}")
        o = alloc["oracle"]
        print(f"   {'Oracle':<25} {o['accuracy']:>10.4f} {o['avg_tokens']:>12.1f}")

        print(f"\n   Budget Distribution: {ca['budget_distribution']}")

        ig = r["information_gain"]
        print(f"\n   🔍 Consensus Precision: {ig['consensus_precision']:.1%}")
        print(f"   Full consensus correct: {ig['full_consensus_correct']}")
        print(f"   Full consensus wrong: {ig['full_consensus_wrong']}")

        print(f"\n   📊 Difficulty Buckets:")
        for label, info in r["difficulty_buckets"].items():
            print(f"   {label}: {info['count']} ({info['fraction']:.1%}), "
                  f"avg consensus={info['avg_consensus']:.3f}")

    if "cross_seed_27b" in results:
        r = results["cross_seed_27b"]
        print(f"\n📊 27B Cross-Seed Analysis:")
        print(f"   Overlap questions: {r['n_overlap_questions']}")
        print(f"   Consensus-Accuracy ρ: {r['consensus_accuracy_correlation']['spearman_rho']:.4f}")

    v = results["verdict"]
    print(f"\n{'='*70}")
    print(f"🎯 VERDICT: {v.get('signal_strength', 'N/A')} signal")
    print(f"   {v.get('recommendation', 'N/A')}")
    if "delta_acc" in v:
        print(f"   Consensus vs best fixed: {v['delta_acc']:+.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
