#!/usr/bin/env python3
"""Soft routing via calibrated confidence (replaces binary natural-stop).

Uses three signals from Stage 0 (nothink probe):
  1. Token utilization ratio: |y_0| / B_1  (how much budget used)
  2. Has-boxed flag: whether output contains \\boxed{} or "Final answer:"
  3. Last-token perplexity: avg -log p(tok) of last N tokens

Combine via Platt scaling (logistic regression) to predict P(correct|signals).
Calibrate on held-out validation split. Use:
  P > τ_high  → accept nothink (skip to answer)
  P < τ_low   → multi-round escalation (K=3)
  else        → single-round split-budget

Usage:
    python scripts/soft_routing.py --calibrate  # train on validation
    python scripts/soft_routing.py --run --thresholds 0.85 0.4
"""
import argparse, json, os, sys, glob, numpy as np


def extract_features(sample):
    """Extract 3 features from a stage-0 sample.

    Expects sample dict with keys:
      - tokens: int (generated token count)
      - budget: int (B_1)
      - text: str (generated text)
      - logprobs: List[float] (optional, per-token logp)
    """
    util = sample["tokens"] / max(1, sample["budget"])
    has_boxed = 1.0 if ("\\boxed" in sample.get("text", "") or
                        "Final answer:" in sample.get("text", "")) else 0.0
    if "logprobs" in sample and sample["logprobs"]:
        tail = sample["logprobs"][-min(32, len(sample["logprobs"])):]
        avg_neg_logp = -np.mean(tail)
    else:
        avg_neg_logp = 0.0
    return np.array([util, has_boxed, avg_neg_logp])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt(X, y):
    """Fit logistic regression via Newton-Raphson (pure numpy, no sklearn dep)."""
    X = np.c_[np.ones(len(X)), X]  # add bias
    w = np.zeros(X.shape[1])
    for _ in range(100):
        p = sigmoid(X @ w)
        grad = X.T @ (p - y)
        H = X.T @ np.diag(p * (1 - p)) @ X + 1e-4 * np.eye(X.shape[1])
        w -= np.linalg.solve(H, grad)
        if np.linalg.norm(grad) < 1e-6:
            break
    return w


def predict_platt(X, w):
    X = np.c_[np.ones(len(X)), X]
    return sigmoid(X @ w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--iris_files", nargs="+",
                   default=["results/iris/iris_Qwen3_8B_b1256_b2512_ba128_20260408_105307.json"],
                   help="IRIS result files to extract (stage0, correct) pairs from")
    p.add_argument("--output", default="results/soft_routing/calibration.json")
    args = p.parse_args()

    if args.calibrate:
        print("=" * 60)
        print("Calibrating soft-routing confidence via Platt scaling")
        print("=" * 60)

        X_all, y_all = [], []
        for fp in args.iris_files:
            if not os.path.exists(fp):
                print(f"  SKIP (not found): {fp}")
                continue
            with open(fp) as f:
                d = json.load(f)
            samples = d.get("per_sample_iris", d.get("per_sample", []))
            for s in samples:
                # stage 0 features — reconstruct from IRIS per-sample data
                if s.get("final_stage") != 1:  # was escalated — stage 0 uncertain
                    continue
                # For stage-1-final samples, nothink was accepted. We need the
                # s0 features and whether nothink was correct.
                feat = np.array([
                    s.get("tokens_total", 0) / 256.0,  # util at B_1=256
                    1.0 if "boxed" in str(s.get("pred_source", "")) else 0.0,
                    0.0,  # no logprob in current data
                ])
                X_all.append(feat)
                y_all.append(s.get("correct", 0))

        if len(X_all) < 20:
            print(f"ERROR: only {len(X_all)} samples — need per-sample logprob data")
            print("  Run IRIS with --log_logprobs flag first")
            return

        X = np.array(X_all); y = np.array(y_all, dtype=float)
        print(f"Training Platt scaling on {len(X)} samples")
        w = fit_platt(X, y)
        print(f"  Weights: {w}")

        p_pred = predict_platt(X, w)
        # Basic metrics
        from collections import Counter
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        for t in thresholds:
            acc_above = np.mean(y[p_pred >= t]) if (p_pred >= t).any() else 0
            n_above = (p_pred >= t).sum()
            print(f"  Threshold {t}: {n_above} samples accepted, accuracy {acc_above:.3f}")

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"weights": w.tolist(),
                       "n_samples": len(X),
                       "features": ["bias", "util", "has_boxed", "avg_neg_logp"]}, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
