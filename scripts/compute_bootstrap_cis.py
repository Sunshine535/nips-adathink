#!/usr/bin/env python3
"""Compute 95% bootstrap confidence intervals for pilot experiment results.

Deterministic decoding (τ=0) → only uncertainty is sample selection.
We construct binary outcome vectors and resample 10,000 times.

Output: CIs for each method and paired differences, plus LaTeX strings.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
B = 10_000  # bootstrap resamples
ALPHA = 0.05  # 95% CI

rng = np.random.default_rng(SEED)


def make_binary(n: int, k: int) -> np.ndarray:
    """Return a length-n binary vector with exactly k ones."""
    v = np.zeros(n, dtype=np.int32)
    v[:k] = 1
    return v


def bootstrap_ci(x: np.ndarray, B: int = B, alpha: float = ALPHA):
    """Percentile bootstrap CI for the mean of binary vector x."""
    n = len(x)
    # shape (B, n) → means shape (B,)
    samples = rng.choice(x, size=(B, n), replace=True)
    means = samples.mean(axis=1) * 100  # percentage
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    point = x.mean() * 100
    return point, lo, hi


def paired_bootstrap_ci(x: np.ndarray, y: np.ndarray, B: int = B, alpha: float = ALPHA):
    """Percentile bootstrap CI for mean(x) - mean(y), paired by sample index."""
    assert len(x) == len(y)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    diff_means = (x[idx].mean(axis=1) - y[idx].mean(axis=1)) * 100
    lo = np.percentile(diff_means, 100 * alpha / 2)
    hi = np.percentile(diff_means, 100 * (1 - alpha / 2))
    point = (x.mean() - y.mean()) * 100
    return point, lo, hi


def fmt_ci(point, lo, hi):
    return f"{point:.1f}% [{lo:.1f}, {hi:.1f}]"


def latex_ci(point, lo, hi):
    return rf"{point:.1f}\% \ci{{{lo:.1f}}}{{{hi:.1f}}}"


# ===================================================================
# GSM8K pilot (n=200, Qwen3-8B)
# ===================================================================
n_gsm = 200
gsm_mrsd = make_binary(n_gsm, 188)        # 94.0%
gsm_nothink256 = make_binary(n_gsm, 178)  # 89.0%

# ===================================================================
# MATH-500 pilot (n=200, Qwen3-8B)
# ===================================================================
n_math = 200
math_mrsd = make_binary(n_math, 122)        # 61.0%
math_nothink512 = make_binary(n_math, 84)   # 42.0%
math_town = make_binary(n_math, 84)         # 42.0%
math_iris = make_binary(n_math, 111)        # 55.5%

# ===================================================================
# Compute CIs
# ===================================================================
print("=" * 70)
print("95% Bootstrap Confidence Intervals (B=10,000)")
print("=" * 70)

results = {}

# --- GSM8K ---
print("\n### GSM8K (n=200, Qwen3-8B)")
for name, vec in [("MRSD", gsm_mrsd), ("Nothink@256", gsm_nothink256)]:
    pt, lo, hi = bootstrap_ci(vec)
    results[f"gsm_{name}"] = (pt, lo, hi)
    print(f"  {name:20s}: {fmt_ci(pt, lo, hi)}")

pt, lo, hi = paired_bootstrap_ci(gsm_mrsd, gsm_nothink256)
results["gsm_diff"] = (pt, lo, hi)
print(f"  {'MRSD − Nothink@256':20s}: {fmt_ci(pt, lo, hi)}  (paired)")

# --- MATH-500 ---
print("\n### MATH-500 (n=200, Qwen3-8B)")
for name, vec in [
    ("MRSD", math_mrsd),
    ("Nothink@512", math_nothink512),
    ("TOWN", math_town),
    ("IRIS (1-round)", math_iris),
]:
    pt, lo, hi = bootstrap_ci(vec)
    results[f"math_{name}"] = (pt, lo, hi)
    print(f"  {name:20s}: {fmt_ci(pt, lo, hi)}")

pt, lo, hi = paired_bootstrap_ci(math_mrsd, math_nothink512)
results["math_diff"] = (pt, lo, hi)
print(f"  {'MRSD − Nothink@512':20s}: {fmt_ci(pt, lo, hi)}  (paired)")

pt, lo, hi = paired_bootstrap_ci(math_mrsd, math_iris)
results["math_diff_iris"] = (pt, lo, hi)
print(f"  {'MRSD − IRIS':20s}: {fmt_ci(pt, lo, hi)}  (paired)")

# ===================================================================
# LaTeX-ready strings
# ===================================================================
print("\n" + "=" * 70)
print("LaTeX-ready strings  (define \\newcommand{\\ci}[2]{[#1, #2]})")
print("=" * 70)

print("\n% GSM8K")
for name, vec in [("MRSD", gsm_mrsd), ("Nothink@256", gsm_nothink256)]:
    pt, lo, hi = results[f"gsm_{name}"]
    print(f"% {name}: {latex_ci(pt, lo, hi)}")

pt, lo, hi = results["gsm_diff"]
print(f"% MRSD − Nothink@256 (Δ): {latex_ci(pt, lo, hi)}")

print("\n% MATH-500")
for name, vec in [
    ("MRSD", math_mrsd),
    ("Nothink@512", math_nothink512),
    ("TOWN", math_town),
    ("IRIS (1-round)", math_iris),
]:
    pt, lo, hi = results[f"math_{name}"]
    print(f"% {name}: {latex_ci(pt, lo, hi)}")

pt, lo, hi = results["math_diff"]
print(f"% MRSD − Nothink@512 (Δ): {latex_ci(pt, lo, hi)}")

pt, lo, hi = results["math_diff_iris"]
print(f"% MRSD − IRIS (Δ): {latex_ci(pt, lo, hi)}")

print("\n% \\newcommand{\\ci}[2]{\\,[#1,\\,#2]}")
print("% Usage: $94.0\\% \\ci{90.5}{97.0}$ → 94.0% [90.5, 97.0]")
