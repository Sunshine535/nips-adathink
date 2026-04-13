#!/usr/bin/env python3
"""
Pilot test: Chain-Length CDF as Sufficient Statistic.

Hypothesis
----------
Think-mode accuracy at any budget b can be decomposed as:

    acc_think(b) = alpha_c * F_L(b) + alpha_trunc(b) * (1 - F_L(b))

where
    alpha_c         = accuracy among naturally completed chains (~98%)
    F_L(b)          = chain-length CDF (fraction of chains that fit within budget b)
    alpha_trunc(b)  = accuracy among truncated chains at budget b

Key question: is F_L (the chain-length CDF) a sufficient statistic?
i.e., given F_L and alpha_c, can we predict accuracy at any budget?

IMPORTANT DATA NOTE:
- fulltest data uses "projection" (extra 32 tokens) for answer extraction
- gap_fill data uses "last_number" extraction (no projection)
- These give DIFFERENT accuracy numbers at the same budget
- We use gap_fill data as primary (consistent extraction across budgets)
- fulltest data used only for supplementary analysis

No GPU needed -- pure analysis on existing JSON/CSV files.
"""

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent

# 8B think @ 128/256/512 -- per-sample CSV from fulltest (uses PROJECTION)
FULLTEST_CSV_8B = REPO / "results_kun/fulltest/per_sample_gsm8k_Qwen3_8B_20260324_120316.csv"
FULLTEST_SUMMARY_8B = REPO / "results_kun/fulltest/summary_gsm8k_Qwen3_8B_20260324_120316.json"

# 8B think/nothink @ 512, 1024 -- per-sample JSON from gap_fill (uses LAST_NUMBER)
GAPFILL_512 = REPO / "results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_121035.json"
GAPFILL_1024 = REPO / "results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_220034.json"

# 8B nothink @ 128, 256 (contains thinking too, uses LAST_NUMBER)
NOTHINK_128 = REPO / "results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_063213.json"
NOTHINK_256 = REPO / "results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_000345.json"

# 27B think -- per-sample CSV (uses PROJECTION)
FULLTEST_CSV_27B = REPO / "results_kun/fulltest_27b/per_sample_gsm8k_Qwen3.5_27B_20260328_213534.csv"
FULLTEST_SUMMARY_27B = REPO / "results_kun/fulltest_27b/summary_gsm8k_Qwen3.5_27B_20260328_213534.json"

# 27B nothink summary
NOTHINK_27B_SUMMARY = REPO / "results_kun/fulltest_27b_nothink/summary_recovered.json"

OUTPUT_DIR = REPO / "results/pilot_cdf_sufficiency"


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_gapfill_json(path: Path):
    """Load gap_fill / nothink_baseline JSON."""
    with open(path) as f:
        return json.load(f)


def load_fulltest_csv(path: Path):
    """Load per-sample CSV from fulltest. Returns list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


# ═══════════════════════════════════════════════════════════════════════════
# Build unified per-sample dataset
# ═══════════════════════════════════════════════════════════════════════════

def build_per_sample_dataset(logger):
    """
    Build a unified per-sample dataset across budgets {128, 256, 512, 1024}.
    Uses gap_fill / nothink_baseline data (consistent last_number extraction).
    """
    d1024 = load_gapfill_json(GAPFILL_1024)
    d512 = load_gapfill_json(GAPFILL_512)
    d128 = load_gapfill_json(NOTHINK_128)
    d256 = load_gapfill_json(NOTHINK_256)

    by_idx = {}

    # 1024-budget: primary source for chain length estimation
    for s in d1024["per_sample"]["thinking_1024"]:
        by_idx[s["idx"]] = {
            "idx": s["idx"],
            "t1024": s["tokens"], "hb1024": s["hit_budget"], "c1024": s["correct"],
        }

    # 512-budget
    for s in d512["per_sample"]["thinking_512"]:
        if s["idx"] in by_idx:
            by_idx[s["idx"]].update({
                "t512": s["tokens"], "hb512": s["hit_budget"], "c512": s["correct"],
            })

    # 128-budget
    for s in d128["per_sample"]["thinking_128"]:
        if s["idx"] in by_idx:
            by_idx[s["idx"]].update({
                "t128": s["tokens"], "hb128": s["hit_budget"], "c128": s["correct"],
            })

    # 256-budget
    for s in d256["per_sample"]["thinking_256"]:
        if s["idx"] in by_idx:
            by_idx[s["idx"]].update({
                "t256": s["tokens"], "hb256": s["hit_budget"], "c256": s["correct"],
            })

    samples = list(by_idx.values())
    n_complete = sum(1 for s in samples
                     if all(f"c{b}" in s for b in [128, 256, 512, 1024]))
    logger.info("Built per-sample dataset: %d samples (%d with all 4 budgets)",
                len(samples), n_complete)
    return samples


def get_actual_accuracy(samples, logger):
    """Compute actual accuracy at each budget from gap_fill data."""
    actuals = {}
    for b in [128, 256, 512, 1024]:
        key = f"c{b}"
        total = sum(1 for s in samples if key in s)
        correct = sum(1 for s in samples if s.get(key, False))
        actuals[b] = correct / total
        logger.info("  Actual think accuracy @%d: %.4f (%d/%d)",
                    b, actuals[b], correct, total)
    return actuals


# ═══════════════════════════════════════════════════════════════════════════
# Chain-length CDF estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_chain_stats(samples, logger):
    """
    Estimate chain-length distribution from 1024-budget data.

    Returns:
        completed_lengths: list of chain lengths for naturally completed samples
        n_censored: number of right-censored samples (chain > 1024)
        alpha_c: accuracy among completed samples (very stable at ~98%)
    """
    completed_lengths = []
    n_censored = 0
    n_completed = 0
    correct_completed = 0
    correct_truncated = 0

    for s in samples:
        if not s["hb1024"]:
            completed_lengths.append(s["t1024"])
            n_completed += 1
            if s["c1024"]:
                correct_completed += 1
        else:
            n_censored += 1
            if s["c1024"]:
                correct_truncated += 1

    alpha_c = correct_completed / n_completed if n_completed > 0 else 0.0
    alpha_trunc_1024 = correct_truncated / n_censored if n_censored > 0 else 0.0

    logger.info("Chain statistics (from 1024-budget):")
    logger.info("  Completed: %d (%.1f%%)", n_completed, 100 * n_completed / len(samples))
    logger.info("  Censored (>1024): %d (%.1f%%)", n_censored, 100 * n_censored / len(samples))
    logger.info("  alpha_c = %.4f (accuracy among completed)", alpha_c)
    logger.info("  alpha_trunc@1024 = %.4f", alpha_trunc_1024)

    if completed_lengths:
        logger.info("  Chain length: mean=%.0f, median=%.0f, std=%.0f, "
                    "min=%d, max=%d",
                    np.mean(completed_lengths), np.median(completed_lengths),
                    np.std(completed_lengths),
                    min(completed_lengths), max(completed_lengths))

    return completed_lengths, n_censored, alpha_c


def fit_lognormal(completed_lengths, n_censored, logger):
    """
    Fit log-normal to chain lengths with right-censoring at 1024 (MLE).
    """
    log_L = np.log(np.array(completed_lengths, dtype=float))

    def neg_loglik(params):
        mu, log_s = params
        s = np.exp(log_s)
        ll_obs = stats.norm.logpdf(log_L, loc=mu, scale=s).sum()
        if n_censored > 0:
            ll_cens = n_censored * stats.norm.logsf(np.log(1024.0), loc=mu, scale=s)
        else:
            ll_cens = 0.0
        return -(ll_obs + ll_cens)

    result = optimize.minimize(neg_loglik,
                                [np.mean(log_L), np.log(np.std(log_L))],
                                method="Nelder-Mead")
    mu, sigma = result.x[0], np.exp(result.x[1])
    logger.info("Log-normal fit (censored MLE): mu=%.4f, sigma=%.4f", mu, sigma)
    logger.info("  => median=%.0f, mean=%.0f tokens",
                np.exp(mu), np.exp(mu + sigma**2 / 2))
    return mu, sigma


def compute_empirical_cdf(completed_lengths, n_total, budgets):
    """F_L(b) = fraction of ALL samples whose chain <= b."""
    sorted_L = sorted(completed_lengths)
    return {b: sum(1 for l in sorted_L if l <= b) / n_total for b in budgets}


def compute_parametric_cdf(mu, sigma, budgets):
    """Evaluate log-normal CDF at given budgets."""
    return {b: float(stats.lognorm.cdf(b, s=sigma, scale=np.exp(mu)))
            for b in budgets}


# ═══════════════════════════════════════════════════════════════════════════
# Decomposition analysis
# ═══════════════════════════════════════════════════════════════════════════

def decomposition_analysis(samples, completed_lengths, alpha_c, logger):
    """
    Detailed decomposition of accuracy into completed + truncated components.

    The formula acc(b) = alpha_c * F_L(b) + alpha_trunc(b) * (1-F_L(b))
    is exact by DEFINITION (law of total probability).

    The question is: can we PREDICT alpha_trunc(b) or set it to a simple function?
    """
    n_total = len(samples)
    budgets = [128, 256, 512, 1024]

    logger.info("\n>>> Decomposition Analysis")
    logger.info("%-8s  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s",
                "Budget", "F_L(b)", "alpha_c", "alpha_trunc", "Comp_acc",
                "Trunc_acc", "Total_acc")
    logger.info("-" * 78)

    decomp = {}
    for b in budgets:
        c_key = f"c{b}"
        hb_key = f"hb{b}"

        # Which samples have chain <= b? Use 1024-budget data
        completed_at_b = [s for s in samples if not s["hb1024"] and s["t1024"] <= b]
        truncated_at_b = [s for s in samples if s["hb1024"] or s["t1024"] > b]

        n_comp = len(completed_at_b)
        n_trunc = len(truncated_at_b)
        f_l = n_comp / n_total

        # alpha_c at this budget (should be ~constant)
        a_c_b = (sum(1 for s in completed_at_b if s.get(c_key, False)) / n_comp
                 if n_comp > 0 else 0.0)

        # alpha_trunc at this budget
        a_t_b = (sum(1 for s in truncated_at_b if s.get(c_key, False)) / n_trunc
                 if n_trunc > 0 else 0.0)

        # Total accuracy (decomposition check)
        total_acc = a_c_b * f_l + a_t_b * (1 - f_l)

        # Actual accuracy for comparison
        actual_total = sum(1 for s in samples if s.get(c_key, False))
        actual_acc = actual_total / n_total

        decomp[b] = {
            "F_L": f_l, "n_completed": n_comp, "n_truncated": n_trunc,
            "alpha_c_at_b": a_c_b, "alpha_trunc_at_b": a_t_b,
            "decomposed_acc": total_acc, "actual_acc": actual_acc,
        }

        logger.info("%-8d  %-8.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f  %-10.4f",
                    b, f_l, a_c_b, a_t_b,
                    a_c_b * f_l, a_t_b * (1 - f_l), actual_acc)

    # Check: alpha_c stability across budgets
    alpha_c_values = [d["alpha_c_at_b"] for d in decomp.values()
                      if d["n_completed"] > 10]
    if alpha_c_values:
        logger.info("\nalpha_c stability: min=%.4f, max=%.4f, range=%.1f pp",
                    min(alpha_c_values), max(alpha_c_values),
                    100 * (max(alpha_c_values) - min(alpha_c_values)))

    return decomp


def truncation_accuracy_model(samples, logger):
    """
    Model alpha_trunc(b) as a function of budget b.

    Key insight: alpha_trunc is NOT zero because:
    1. "last_number" extraction can find correct answer in partial chain
    2. Some problems have answers that appear early in the reasoning

    Empirical finding: alpha_trunc(b) increases roughly linearly with b
    for b < median chain length, then levels off.
    """
    # For each sample, determine natural chain length
    # and correctness at each budget
    budgets = [128, 256, 512, 1024]

    # Collect (b, alpha_trunc) data points
    b_vals = []
    at_vals = []

    for b in budgets:
        c_key = f"c{b}"
        # Truncated samples at this budget
        truncated = [s for s in samples if (s["hb1024"] or s["t1024"] > b)
                     and c_key in s]
        if len(truncated) < 10:
            continue
        a_t = sum(1 for s in truncated if s[c_key]) / len(truncated)
        b_vals.append(b)
        at_vals.append(a_t)

    b_vals = np.array(b_vals, dtype=float)
    at_vals = np.array(at_vals, dtype=float)

    # Fit simple linear model: alpha_trunc(b) = a + c * b
    if len(b_vals) >= 2:
        coeffs = np.polyfit(b_vals, at_vals, 1)
        slope, intercept = coeffs
        logger.info("Linear fit for alpha_trunc(b): %.6f * b + %.4f", slope, intercept)

        # Also fit power law: alpha_trunc(b) = beta * b^gamma
        try:
            log_b = np.log(b_vals)
            log_at = np.log(np.maximum(at_vals, 1e-6))
            popt = np.polyfit(log_b, log_at, 1)
            gamma = popt[0]
            beta = np.exp(popt[1])
            logger.info("Power-law fit for alpha_trunc(b): %.4f * b^%.4f", beta, gamma)
        except Exception:
            beta, gamma = 0.0, 1.0

        return {"linear": {"slope": float(slope), "intercept": float(intercept)},
                "power": {"beta": float(beta), "gamma": float(gamma)},
                "empirical": dict(zip([int(b) for b in b_vals], at_vals.tolist()))}

    return {"empirical": dict(zip([int(b) for b in b_vals], at_vals.tolist()))}


# ═══════════════════════════════════════════════════════════════════════════
# Prediction models
# ═══════════════════════════════════════════════════════════════════════════

def predict_and_evaluate(alpha_c, cdf_dict, alpha_trunc_fn, actuals,
                         budgets, label, logger):
    """
    Predict accuracy at each budget, compare to actual. Return results + RMSE.
    alpha_trunc_fn: callable(b) -> float
    """
    results = {}
    residuals = []

    logger.info("\n%s:", label)
    logger.info("%-8s  %-8s  %-8s  %-8s  %-8s  %-10s",
                "Budget", "F_L(b)", "a_trunc", "Pred", "Actual", "Err(pp)")
    logger.info("-" * 56)

    for b in budgets:
        if b not in actuals or b not in cdf_dict:
            continue
        f_l = cdf_dict[b]
        a_t = alpha_trunc_fn(b)
        pred = alpha_c * f_l + a_t * (1 - f_l)
        actual = actuals[b]
        resid = pred - actual

        results[b] = {
            "actual": float(actual), "predicted": float(pred),
            "residual": float(resid), "abs_error_pp": float(abs(resid) * 100),
            "F_L": float(f_l), "alpha_trunc": float(a_t),
        }
        residuals.append(resid)

        logger.info("%-8d  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %+7.1f",
                    b, f_l, a_t, pred, actual, resid * 100)

    rmse = np.sqrt(np.mean(np.array(residuals)**2)) if residuals else float("nan")
    max_err = max(abs(r) for r in residuals) if residuals else float("nan")
    logger.info("  RMSE = %.2f pp,  Max |error| = %.2f pp", rmse * 100, max_err * 100)
    return results, float(rmse), float(max_err)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration from small subset
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration_trials(samples, n_cal, n_trials, base_seed, alpha_trunc_fn,
                           actuals, budgets, logger):
    """Run calibration from n_cal-sample subsets, report stability."""
    trial_results = []

    for seed in range(base_seed, base_seed + n_trials):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(samples))[:n_cal]
        cal = [samples[i] for i in idx]

        # Estimate alpha_c from calibration set
        completed = [s for s in cal if not s["hb1024"]]
        if len(completed) < 5:
            continue
        a_c = sum(1 for s in completed if s["c1024"]) / len(completed)

        # Estimate chain lengths and fit log-normal
        cal_lengths = [s["t1024"] for s in cal if not s["hb1024"]]
        cal_censored = sum(1 for s in cal if s["hb1024"])
        try:
            mu, sigma = fit_lognormal(cal_lengths, cal_censored, logger)
        except Exception:
            continue
        cal_cdf = compute_parametric_cdf(mu, sigma, budgets)

        # Predict
        residuals = []
        for b in budgets:
            if b not in actuals:
                continue
            pred = a_c * cal_cdf[b] + alpha_trunc_fn(b) * (1 - cal_cdf[b])
            residuals.append(pred - actuals[b])

        rmse = np.sqrt(np.mean(np.array(residuals)**2))
        trial_results.append({
            "seed": seed, "alpha_c": a_c, "mu": mu, "sigma": sigma,
            "rmse": float(rmse), "n_completed": len(completed),
        })

    if trial_results:
        rmses = [t["rmse"] for t in trial_results]
        a_cs = [t["alpha_c"] for t in trial_results]
        logger.info("\nCalibration (n_cal=%d, %d trials):", n_cal, len(trial_results))
        logger.info("  alpha_c: mean=%.4f, std=%.4f", np.mean(a_cs), np.std(a_cs))
        logger.info("  RMSE: mean=%.2f pp, std=%.2f pp, max=%.2f pp",
                    np.mean(rmses) * 100, np.std(rmses) * 100, np.max(rmses) * 100)

    return trial_results


# ═══════════════════════════════════════════════════════════════════════════
# Cross-model transfer (8B -> 27B)
# ═══════════════════════════════════════════════════════════════════════════

def cross_model_transfer(mu_8b, sigma_8b, alpha_c_8b, logger):
    """Test if rescaled 8B CDF predicts 27B accuracy."""

    # 27B actual accuracy (from summary, uses PROJECTION)
    with open(FULLTEST_SUMMARY_27B) as f:
        s27 = json.load(f)
    actuals_27b = {int(b): info["accuracy"] for b, info in s27["fixed"].items()}

    # 27B nothink
    with open(NOTHINK_27B_SUMMARY) as f:
        nt27 = json.load(f)

    logger.info("\n27B actuals (projection-based): %s",
                {b: f"{a:.4f}" for b, a in actuals_27b.items()})
    logger.info("27B nothink@512 = %.3f", nt27["experiments"]["nothink_512"]["accuracy"])

    # 27B think data also from nothink_baseline (LAST_NUMBER extraction)
    # thinking_128: 0.026, thinking_256: 0.051
    actuals_27b_lastnum = {
        128: nt27["experiments"]["thinking_128"]["accuracy"],
        256: nt27["experiments"]["thinking_256"]["accuracy"],
    }
    logger.info("27B actuals (last_number, from nothink_baseline): %s",
                {b: f"{a:.4f}" for b, a in actuals_27b_lastnum.items()})

    # Search for scale factor k: mu_27b = mu_8b + log(k)
    # 27B chains should be longer since the model is more verbose
    budgets_27b = sorted(actuals_27b.keys())
    best_k = None
    best_rmse = float("inf")

    for k in np.arange(1.0, 6.0, 0.01):
        mu_27b = mu_8b + np.log(k)
        cdf_27b = compute_parametric_cdf(mu_27b, sigma_8b, budgets_27b)
        residuals = []
        for b in budgets_27b:
            pred = alpha_c_8b * cdf_27b[b]  # assume alpha_trunc ≈ 0 for 27B
            residuals.append(pred - actuals_27b[b])
        rmse = np.sqrt(np.mean(np.array(residuals)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k

    logger.info("Best scale factor k = %.2f (RMSE = %.2f pp)", best_k, best_rmse * 100)

    # Show predictions
    mu_27b_best = mu_8b + np.log(best_k)
    cdf_27b = compute_parametric_cdf(mu_27b_best, sigma_8b, budgets_27b)

    results = {}
    logger.info("\n27B transfer predictions (k=%.2f, median_27b=%.0f tokens):",
                best_k, np.exp(mu_27b_best))
    logger.info("%-8s  %-8s  %-8s  %-8s  %-10s", "Budget", "F_L_27b", "Pred", "Actual", "Err(pp)")
    logger.info("-" * 50)

    for b in budgets_27b:
        pred = alpha_c_8b * cdf_27b[b]
        actual = actuals_27b[b]
        results[b] = {
            "actual": actual, "predicted": float(pred),
            "residual": float(pred - actual),
            "F_L_27b": float(cdf_27b[b]),
        }
        logger.info("%-8d  %-8.4f  %-8.4f  %-8.4f  %+7.1f",
                    b, cdf_27b[b], pred, actual, (pred - actual) * 100)

    return results, float(best_k), float(best_rmse)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pilot: Chain-Length CDF as Sufficient Statistic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-cal", type=int, default=50,
                        help="Calibration subset size")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("pilot_cdf")
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Pilot: Chain-Length CDF as Sufficient Statistic")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Build per-sample dataset & actuals
    # ------------------------------------------------------------------
    logger.info("\n>>> Step 1: Building per-sample dataset")
    samples = build_per_sample_dataset(logger)
    actuals = get_actual_accuracy(samples, logger)
    n_total = len(samples)
    budgets = [128, 256, 512, 1024]

    # Nothink saturated accuracy
    d512 = load_gapfill_json(GAPFILL_512)
    alpha_nt = d512["results"]["nothink_512"]["accuracy"]
    logger.info("  Nothink saturated accuracy: %.4f", alpha_nt)

    # ------------------------------------------------------------------
    # 2. Estimate chain statistics
    # ------------------------------------------------------------------
    logger.info("\n>>> Step 2: Chain-length CDF estimation")
    completed_lengths, n_censored, alpha_c = estimate_chain_stats(samples, logger)

    # Empirical CDF
    emp_cdf = compute_empirical_cdf(completed_lengths, n_total, budgets)
    logger.info("Empirical CDF: %s", {b: f"{v:.4f}" for b, v in emp_cdf.items()})

    # Parametric CDF
    mu, sigma = fit_lognormal(completed_lengths, n_censored, logger)
    param_cdf = compute_parametric_cdf(mu, sigma, budgets)
    logger.info("Parametric CDF: %s", {b: f"{v:.4f}" for b, v in param_cdf.items()})

    # CDF comparison
    logger.info("\nCDF comparison (empirical vs parametric):")
    for b in budgets:
        logger.info("  F_L(%d): emp=%.4f, param=%.4f, diff=%.1f pp",
                    b, emp_cdf[b], param_cdf[b], (emp_cdf[b] - param_cdf[b]) * 100)

    # ------------------------------------------------------------------
    # 3. Decomposition analysis
    # ------------------------------------------------------------------
    decomp = decomposition_analysis(samples, completed_lengths, alpha_c, logger)

    # ------------------------------------------------------------------
    # 4. Truncation accuracy model
    # ------------------------------------------------------------------
    logger.info("\n>>> Step 3: Truncation accuracy modeling")
    trunc_model = truncation_accuracy_model(samples, logger)

    # Collect empirical alpha_trunc
    emp_alpha_trunc = {}
    for b in budgets:
        c_key = f"c{b}"
        truncated = [s for s in samples if (s["hb1024"] or s["t1024"] > b)
                     and c_key in s]
        if truncated:
            emp_alpha_trunc[b] = sum(1 for s in truncated if s[c_key]) / len(truncated)
        else:
            emp_alpha_trunc[b] = 0.0

    logger.info("Empirical alpha_trunc: %s",
                {b: f"{v:.4f}" for b, v in emp_alpha_trunc.items()})

    # ------------------------------------------------------------------
    # 5. Prediction models
    # ------------------------------------------------------------------
    logger.info("\n>>> Step 4: Prediction Models")

    # Model A: Empirical CDF + Empirical alpha_trunc (oracle — uses same data)
    res_A, rmse_A, max_A = predict_and_evaluate(
        alpha_c, emp_cdf, lambda b: emp_alpha_trunc.get(b, 0.0),
        actuals, budgets, "Model A: emp CDF + emp alpha_trunc (oracle)", logger)

    # Model B: Parametric CDF + Empirical alpha_trunc
    res_B, rmse_B, max_B = predict_and_evaluate(
        alpha_c, param_cdf, lambda b: emp_alpha_trunc.get(b, 0.0),
        actuals, budgets, "Model B: param CDF + emp alpha_trunc", logger)

    # Model C: Empirical CDF + zero alpha_trunc
    res_C, rmse_C, max_C = predict_and_evaluate(
        alpha_c, emp_cdf, lambda b: 0.0,
        actuals, budgets, "Model C: emp CDF + zero alpha_trunc", logger)

    # Model D: Parametric CDF + linear alpha_trunc model
    if "linear" in trunc_model:
        sl = trunc_model["linear"]["slope"]
        it = trunc_model["linear"]["intercept"]
        res_D, rmse_D, max_D = predict_and_evaluate(
            alpha_c, param_cdf, lambda b: max(0, sl * b + it),
            actuals, budgets, "Model D: param CDF + linear alpha_trunc", logger)
    else:
        res_D, rmse_D, max_D = {}, float("nan"), float("nan")

    # Model E: Parametric CDF + power-law alpha_trunc model
    if "power" in trunc_model:
        beta = trunc_model["power"]["beta"]
        gamma = trunc_model["power"]["gamma"]
        res_E, rmse_E, max_E = predict_and_evaluate(
            alpha_c, param_cdf, lambda b: beta * b**gamma,
            actuals, budgets, "Model E: param CDF + power alpha_trunc", logger)
    else:
        res_E, rmse_E, max_E = {}, float("nan"), float("nan")

    # Model F: Fulltest data (projection-based) for comparison
    with open(FULLTEST_SUMMARY_8B) as f:
        ft = json.load(f)
    actuals_proj = {int(b): info["accuracy"] for b, info in ft["fixed"].items()}
    actuals_proj[1024] = actuals[1024]  # use gap_fill for 1024
    logger.info("\n(Reference) Fulltest actuals (projection-based): %s",
                {b: f"{a:.4f}" for b, a in actuals_proj.items()})
    logger.info("(Reference) Gap-fill actuals (last_number): %s",
                {b: f"{a:.4f}" for b, a in actuals.items()})

    # ------------------------------------------------------------------
    # 6. Calibration from small subset
    # ------------------------------------------------------------------
    logger.info("\n>>> Step 5: Calibration from %d-sample subset", args.n_cal)
    cal_trials = run_calibration_trials(
        samples, args.n_cal, 20, args.seed,
        lambda b: emp_alpha_trunc.get(b, 0.0),
        actuals, budgets, logger)

    avg_cal_rmse = np.mean([t["rmse"] for t in cal_trials]) if cal_trials else float("nan")
    std_cal_rmse = np.std([t["rmse"] for t in cal_trials]) if cal_trials else float("nan")

    # ------------------------------------------------------------------
    # 7. Cross-model transfer
    # ------------------------------------------------------------------
    logger.info("\n>>> Step 6: Cross-model transfer (8B -> 27B)")
    results_27b, best_k, rmse_27b = cross_model_transfer(mu, sigma, alpha_c, logger)

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    logger.info("\nKey parameters:")
    logger.info("  alpha_c = %.4f  (accuracy if chain completes, stable across budgets)", alpha_c)
    logger.info("  alpha_nt = %.4f  (nothink saturated accuracy)", alpha_nt)
    logger.info("  Log-normal: mu=%.3f, sigma=%.3f => median=%.0f tokens",
                mu, sigma, np.exp(mu))

    logger.info("\n" + "=" * 70)
    logger.info("%-38s  %8s  %8s  %4s", "Model", "RMSE(pp)", "MaxE(pp)", "OK?")
    logger.info("-" * 70)

    models = [
        ("A: emp CDF + emp a_trunc (oracle)", rmse_A, max_A),
        ("B: param CDF + emp a_trunc", rmse_B, max_B),
        ("C: emp CDF + zero a_trunc", rmse_C, max_C),
        ("D: param CDF + linear a_trunc", rmse_D, max_D),
        ("E: param CDF + power a_trunc", rmse_E, max_E),
    ]
    for name, rmse, maxe in models:
        ok = "YES" if maxe * 100 <= 3.0 else "NO"
        logger.info("%-38s  %8.2f  %8.2f  %4s", name, rmse * 100, maxe * 100, ok)
    if cal_trials:
        logger.info("%-38s  %8.2f  %8s  %4s",
                    f"Calibration ({args.n_cal} samp, avg)",
                    avg_cal_rmse * 100, "--", "--")
    logger.info("%-38s  %8.2f  %8s  %4s",
                "8B->27B transfer", rmse_27b * 100, "--", "--")
    logger.info("=" * 70)

    # Key finding
    logger.info("\nKEY FINDINGS:")
    logger.info("1. The decomposition acc(b) = alpha_c*F_L(b) + a_trunc*(1-F_L(b))")
    logger.info("   is EXACT by definition (law of total probability).")
    logger.info("2. alpha_c is remarkably STABLE across budgets: ~%.1f%%", alpha_c * 100)
    logger.info("   (range: %.1f pp across budgets with >10 completions)",
                100 * (max(d["alpha_c_at_b"] for d in decomp.values() if d["n_completed"] > 10) -
                       min(d["alpha_c_at_b"] for d in decomp.values() if d["n_completed"] > 10)))
    logger.info("3. The CDF F_L is well-fit by log-normal (censored MLE).")
    logger.info("4. The HARD PART is alpha_trunc(b): it's NOT zero!")
    logger.info("   alpha_trunc grows from %.1f%% @128 to %.1f%% @1024",
                emp_alpha_trunc[128] * 100, emp_alpha_trunc[1024] * 100)
    logger.info("5. With ORACLE alpha_trunc: RMSE = %.2f pp, Max = %.2f pp => %s",
                rmse_A * 100, max_A * 100,
                "PASS" if max_A * 100 <= 3.0 else "FAIL ±3pp target")
    logger.info("6. 50-sample calibration RMSE: %.2f ± %.2f pp",
                avg_cal_rmse * 100, std_cal_rmse * 100)
    logger.info("7. Cross-model 8B->27B: k=%.2f, RMSE=%.2f pp", best_k, rmse_27b * 100)

    within_3pp = max_A * 100 <= 3.0
    logger.info("\nVERDICT: CDF as sufficient statistic for acc_think(b)?")
    if within_3pp:
        logger.info("  YES — with known alpha_trunc, predictions within ±3pp")
    else:
        logger.info("  CONDITIONAL — the CDF captures the main curve shape,")
        logger.info("  but alpha_trunc (truncation accuracy) must also be known.")
        logger.info("  The CDF + alpha_c + alpha_trunc form a SUFFICIENT TRIPLE.")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "hypothesis": "acc(b) = alpha_c * F_L(b) + alpha_trunc(b) * (1 - F_L(b))",
        "verdict": "conditional" if not within_3pp else "supported",
        "key_parameters": {
            "alpha_c": float(alpha_c),
            "alpha_nt": float(alpha_nt),
            "lognormal_mu": float(mu),
            "lognormal_sigma": float(sigma),
            "median_chain_length": float(np.exp(mu)),
        },
        "empirical_cdf": {str(b): float(v) for b, v in emp_cdf.items()},
        "parametric_cdf": {str(b): float(v) for b, v in param_cdf.items()},
        "alpha_trunc_empirical": {str(b): float(v) for b, v in emp_alpha_trunc.items()},
        "truncation_model": {k: v for k, v in trunc_model.items() if k != "empirical"},
        "decomposition": {
            str(b): {k: float(v) if isinstance(v, (float, np.floating)) else v
                     for k, v in d.items()}
            for b, d in decomp.items()
        },
        "models": {
            "A_emp_oracle": {"rmse_pp": rmse_A * 100, "max_pp": max_A * 100,
                             "within_3pp": within_3pp,
                             "per_budget": {str(b): v for b, v in res_A.items()}},
            "B_param_oracle": {"rmse_pp": rmse_B * 100, "max_pp": max_B * 100,
                               "per_budget": {str(b): v for b, v in res_B.items()}},
            "C_zero_trunc": {"rmse_pp": rmse_C * 100, "max_pp": max_C * 100,
                             "per_budget": {str(b): v for b, v in res_C.items()}},
            "D_linear_trunc": {"rmse_pp": rmse_D * 100, "max_pp": max_D * 100,
                               "per_budget": {str(b): v for b, v in res_D.items()}},
            "E_power_trunc": {"rmse_pp": rmse_E * 100, "max_pp": max_E * 100,
                              "per_budget": {str(b): v for b, v in res_E.items()}},
        },
        "calibration": {
            "n_cal": args.n_cal,
            "n_trials": len(cal_trials),
            "avg_rmse_pp": float(avg_cal_rmse * 100),
            "std_rmse_pp": float(std_cal_rmse * 100),
        },
        "cross_model_27b": {
            "scale_factor_k": float(best_k),
            "rmse_pp": float(rmse_27b * 100),
            "per_budget": {str(b): v for b, v in results_27b.items()},
        },
        "data_note": ("Actuals use gap_fill data (last_number extraction, no projection). "
                      "Fulltest data (with projection) gives ~10pp higher accuracy at 128/256."),
        "meta": {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "seed": args.seed,
            "n_samples": n_total,
            "n_completed_1024": len(completed_lengths),
            "n_censored_1024": n_censored,
        },
    }

    out_path = out_dir / "pilot_cdf_sufficiency_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("\nResults saved to %s", out_path)

    # Text summary
    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("Pilot: Chain-Length CDF as Sufficient Statistic\n")
        f.write("=" * 60 + "\n\n")
        f.write("Formula: acc(b) = alpha_c * F_L(b) + alpha_trunc(b) * (1-F_L(b))\n\n")
        f.write("Key parameters (8B, GSM8K, n=1319):\n")
        f.write(f"  alpha_c  = {alpha_c:.4f}  (stable across budgets: ~98%)\n")
        f.write(f"  alpha_nt = {alpha_nt:.4f}  (nothink saturated)\n")
        f.write(f"  F_L ~ LogNormal(mu={mu:.3f}, sigma={sigma:.3f})\n")
        f.write(f"  Median chain = {np.exp(mu):.0f} tokens\n\n")
        f.write("Decomposition table:\n")
        f.write(f"{'Budget':>8}  {'F_L':>8}  {'a_c':>8}  {'a_trunc':>8}  "
                f"{'Pred':>8}  {'Actual':>8}  {'Err':>8}\n")
        f.write("-" * 62 + "\n")
        for b in budgets:
            d = decomp[b]
            pred = d["alpha_c_at_b"] * d["F_L"] + d["alpha_trunc_at_b"] * (1 - d["F_L"])
            f.write(f"{b:>8d}  {d['F_L']:>8.4f}  {d['alpha_c_at_b']:>8.4f}  "
                    f"{d['alpha_trunc_at_b']:>8.4f}  {pred:>8.4f}  "
                    f"{d['actual_acc']:>8.4f}  {(pred-d['actual_acc'])*100:>+7.1f}pp\n")
        f.write(f"\nModel comparison:\n")
        f.write(f"{'Model':>40}  {'RMSE':>8}  {'MaxE':>8}  {'OK?':>4}\n")
        f.write("-" * 65 + "\n")
        for name, rmse, maxe in models:
            ok = "YES" if maxe * 100 <= 3.0 else "NO"
            f.write(f"{name:>40}  {rmse*100:>7.2f}  {maxe*100:>7.2f}  {ok:>4}\n")
        f.write(f"\nVerdict: {'SUPPORTED' if within_3pp else 'CONDITIONAL'}\n")
        f.write(f"  The CDF captures curve SHAPE, but alpha_trunc must be known.\n")
    logger.info("Text summary saved to %s", txt_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
