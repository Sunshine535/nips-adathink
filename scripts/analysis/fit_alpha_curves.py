#!/usr/bin/env python3
"""Fit α_c(b) and α_t(b) curves across budgets + out-of-sample prediction.

Tests three parametric forms for α_t(b):
  - logistic: α_t(b) = α_inf / (1 + exp(-k(b - b0)))
  - power:    α_t(b) = min(1, c · b^p) + α_0
  - linear:   α_t(b) = min(1, a + c·b)

Then uses {b=256, 512} data to predict {b=1024, 2048, 4096} accuracy
via the decomposition:
    Acc_think(b) = F_L(b) · α_c + (1-F_L(b)) · α_t(b)

Usage:
    python scripts/analysis/fit_alpha_curves.py
"""
import json, numpy as np
from scipy.optimize import curve_fit

# Ground-truth data from paper (Qwen3-8B on GSM8K, n=1319, seed=42)
# Source: results/gap_fill/8b_highbudget/nothink_baseline_*.json
GSM8K_8B = {
    # budget: (acc_think, early_stop_rate, avg_tokens)
    128:  (0.030, 0.000, 128),   # early_stop ~= F_L(b) proxy
    256:  (0.180, 0.014, 255),
    512:  (0.569, 0.374, 460),
    1024: (0.861, 0.789, 590),   # from 8b_highbudget 220034 file
    2048: None,  # TBD
}

# For each completed budget, we have:
#   α_c(b) = acc_among_natural_stops
#   α_t(b) = acc_among_truncated
# These require per-sample data. For now, use approximations.

ALPHA_C_GSM8K_8B = {
    256: 1.000,   # paper table: natural-stop acc at b=256
    512: 0.990,   # paper value
}

ALPHA_T_GSM8K_8B = {
    128: 0.030,   # all truncated, acc ≈ 3% (≈ α_t since F_L ≈ 0)
    256: 0.168,   # from paper Table decomposition
    512: 0.318,   # from paper Table (99% natural_stop → 31.8% truncated_acc)
    1024: 0.417,  # from BBH section
}

F_L_GSM8K_8B = {
    128: 0.000,   # 0% natural stop
    256: 0.014,
    512: 0.374,
    1024: 0.789,
}


def logistic(b, alpha_inf, k, b0):
    return alpha_inf / (1.0 + np.exp(-k * (b - b0)))


def power(b, c, p, alpha_0):
    return np.minimum(1.0, c * np.power(b, p)) + alpha_0


def linear(b, a, c):
    return np.minimum(1.0, a + c * b)


def fit_and_predict(budgets_train, alpha_train, budgets_test):
    """Fit three forms, return best + predictions."""
    b_train = np.array(budgets_train, dtype=float)
    a_train = np.array(alpha_train, dtype=float)

    results = {}
    for name, f, p0 in [
        ("logistic", logistic, [0.5, 0.005, 500]),
        ("power", power, [0.001, 0.7, 0.0]),
        ("linear", linear, [0.0, 0.0005]),
    ]:
        try:
            popt, _ = curve_fit(f, b_train, a_train, p0=p0, maxfev=5000)
            train_pred = f(b_train, *popt)
            train_rmse = np.sqrt(np.mean((train_pred - a_train) ** 2))
            test_pred = f(np.array(budgets_test), *popt)
            results[name] = {"params": popt.tolist(), "train_rmse": float(train_rmse),
                            "predictions": {int(b): float(p) for b, p in
                                           zip(budgets_test, test_pred)}}
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def main():
    print("=" * 70)
    print("α_t(b) curve fitting — GSM8K Qwen3-8B")
    print("=" * 70)

    # Fit on {128, 256, 512}, predict {1024}
    train_budgets = [128, 256, 512]
    train_alpha = [ALPHA_T_GSM8K_8B[b] for b in train_budgets]
    test_budgets = [1024, 2048]

    print(f"\nTraining set: b in {train_budgets}, α_t = {train_alpha}")
    print(f"Test set: b in {test_budgets}")
    print(f"Ground truth at b=1024: α_t = {ALPHA_T_GSM8K_8B.get(1024, 'unknown')}")

    results = fit_and_predict(train_budgets, train_alpha, test_budgets)

    print("\n--- Fit Results ---")
    best_rmse = float('inf')
    best_name = None
    for name, res in results.items():
        if "error" in res:
            print(f"  {name}: FAILED ({res['error']})")
            continue
        print(f"  {name}:")
        print(f"    params: {[f'{p:.4g}' for p in res['params']]}")
        print(f"    train RMSE: {res['train_rmse']:.4f}")
        print(f"    predictions: {res['predictions']}")
        if res['train_rmse'] < best_rmse:
            best_rmse = res['train_rmse']
            best_name = name

    print(f"\nBest model: {best_name} (train RMSE = {best_rmse:.4f})")
    if best_name and 1024 in results[best_name]['predictions']:
        pred_1024 = results[best_name]['predictions'][1024]
        true_1024 = ALPHA_T_GSM8K_8B.get(1024)
        if true_1024 is not None:
            err = abs(pred_1024 - true_1024)
            print(f"Test at b=1024: predicted {pred_1024:.3f}, actual {true_1024:.3f}, err={err:.3f}")

    # Use best α_t fit to predict Acc_think(b) via decomposition
    print("\n" + "=" * 70)
    print("Out-of-sample Acc_think prediction using decomposition framework")
    print("=" * 70)

    # Use α_c ≈ 0.990 (from b=512), F_L values
    alpha_c = 0.99
    if best_name:
        params = results[best_name]['params']
        f_map = {"logistic": logistic, "power": power, "linear": linear}
        f = f_map[best_name]

        for b in [1024, 2048]:
            if b not in F_L_GSM8K_8B:
                continue
            alpha_t_pred = float(f(np.array([b]), *params)[0])
            F_L = F_L_GSM8K_8B[b]
            acc_pred = F_L * alpha_c + (1 - F_L) * alpha_t_pred
            print(f"  b={b}: F_L={F_L}, α_c={alpha_c}, α_t(pred)={alpha_t_pred:.3f} → Acc(pred)={acc_pred:.3f}")

    # Save JSON
    with open("results/analysis/alpha_curve_fit.json", "w") as fp:
        json.dump({"gsm8k_8b": results, "best_model": best_name}, fp, indent=2)
    print("\nSaved: results/analysis/alpha_curve_fit.json")


if __name__ == "__main__":
    import os
    os.makedirs("results/analysis", exist_ok=True)
    main()
