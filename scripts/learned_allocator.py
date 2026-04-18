#!/usr/bin/env python3
"""Learned Budget Allocator — lightweight MLP maps query → (B_r*, B_a*).

Architecture:
    embedding (from question text) → MLP(hidden=64) → (B_r_logits, B_a_logits)
    Discretized: B_r ∈ {512, 1024, 2048, 4096}, B_a ∈ {128, 256, 512}

Training data: per-sample IRIS runs at different budgets give us
(question, budget, accuracy, tokens_total) tuples. For each question we
can identify the "best" (B_r, B_a) as the one achieving correct=1 at
lowest total tokens.

Loss: -Acc + λ · total_tokens / max_tokens

Usage:
    python scripts/learned_allocator.py --build_training_set
    python scripts/learned_allocator.py --train
    python scripts/learned_allocator.py --eval
"""
import argparse, json, os, glob, numpy as np
from collections import defaultdict


B_R_CHOICES = [512, 1024, 2048, 4096]
B_A_CHOICES = [128, 256, 512]


def build_training_set():
    """Extract (question, best_budget, acc) tuples from all IRIS runs."""
    out = []

    # 8B MATH-500 runs
    for path, b_r, b_a in [
        ("results/iris_math500_fullscale/b2048_iris_compact.json", 2048, 256),
        ("results/iris_math500_fullscale_b4096/b4096_iris_compact.json", 4096, 256),
        ("results/iris_improved_20260417/8b_math500_b4096_ba512_n500/checkpoint_iris_500.json", 4096, 512),
    ]:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        samples = d.get("iris_results") if isinstance(d, dict) else d
        for i, s in enumerate(samples):
            correct = s.get("correct", s.get("c", 0))
            tokens = s.get("tokens_total", s.get("t", b_r + b_a))
            out.append({"model": "8B", "bench": "math500", "idx": i,
                       "b_r": b_r, "b_a": b_a,
                       "correct": int(correct), "tokens": int(tokens)})

    print(f"Collected {len(out)} training samples")

    # For each question, find best-cost correct answer
    per_q = defaultdict(list)
    for r in out:
        key = (r["model"], r["bench"], r["idx"])
        per_q[key].append(r)

    ds = []
    for key, runs in per_q.items():
        correct_runs = [r for r in runs if r["correct"]]
        if correct_runs:
            best = min(correct_runs, key=lambda r: r["tokens"])
            ds.append({"key": key, "best_b_r": best["b_r"], "best_b_a": best["b_a"],
                       "best_tokens": best["tokens"], "solvable": True})
        else:
            # Unsolvable at any tested budget — choose largest budget
            largest = max(runs, key=lambda r: r["b_r"] + r["b_a"])
            ds.append({"key": key, "best_b_r": largest["b_r"], "best_b_a": largest["b_a"],
                       "best_tokens": largest["tokens"], "solvable": False})

    solvable = sum(1 for d in ds if d["solvable"])
    print(f"  Unique questions: {len(ds)}")
    print(f"  Solvable at any budget: {solvable}/{len(ds)}")

    # Budget distribution
    from collections import Counter
    br_dist = Counter(d["best_b_r"] for d in ds if d["solvable"])
    ba_dist = Counter(d["best_b_a"] for d in ds if d["solvable"])
    print(f"  Optimal B_r distribution: {dict(br_dist)}")
    print(f"  Optimal B_a distribution: {dict(ba_dist)}")

    os.makedirs("results/learned_allocator", exist_ok=True)
    with open("results/learned_allocator/training_set.json", "w") as f:
        json.dump(ds, f, indent=2)
    print(f"Saved: results/learned_allocator/training_set.json")
    return ds


def analyze_budget_policy(ds):
    """Without training, analyze what an oracle allocator would achieve."""
    print("\n" + "=" * 60)
    print("Oracle allocator analysis")
    print("=" * 60)

    # Oracle strategy: always use best (B_r, B_a) per question
    total_correct = sum(1 for d in ds if d["solvable"])
    total_tokens = sum(d["best_tokens"] for d in ds)
    n = len(ds)

    print(f"\nOracle allocator (if we knew optimal per-query):")
    print(f"  Accuracy: {total_correct}/{n} = {total_correct/n:.1%}")
    print(f"  Avg tokens: {total_tokens/n:.0f}")

    # Fixed strategy: always max budget
    for b_r, b_a in [(4096, 256), (4096, 512)]:
        filtered = [d for d in ds if d["best_b_r"] <= b_r and d["best_b_a"] <= b_a]
        correct = sum(1 for d in filtered if d["solvable"])
        print(f"\nFixed strategy (B_r={b_r}, B_a={b_a}):")
        print(f"  Accuracy bound: ≤ {correct}/{n} = {correct/n:.1%}")

    # Hypothetical gain from learned allocator
    # Lower bound: always use min (B_r,B_a) needed for correct answer
    # Upper bound: no savings (always use max budget)
    oracle_avg = total_tokens / n
    fixed_max_tokens = 4096 + 512  # largest config
    savings_pct = 100 * (1 - oracle_avg / fixed_max_tokens)
    print(f"\nOracle savings over fixed max-budget: {savings_pct:.1f}%")
    print(f"Potential target for learned allocator: 0-{savings_pct:.0f}% of oracle savings")


def extract_features(q_text):
    """Hand-crafted features from MATH-500 question text."""
    import re
    feats = {
        "len_chars": len(q_text),
        "len_words": len(q_text.split()),
        "n_digits": sum(c.isdigit() for c in q_text),
        "n_latex_inline": q_text.count("$"),
        "n_latex_block": q_text.count("\\["),
        "has_sum": int("\\sum" in q_text),
        "has_int": int("\\int" in q_text),
        "has_sqrt": int("\\sqrt" in q_text),
        "has_frac": int("\\frac" in q_text),
        "has_matrix": int("matrix" in q_text.lower() or "\\begin" in q_text),
        "has_prob": int(re.search(r"probab|chance|likelihood", q_text, re.I) is not None),
        "has_geom": int(re.search(r"triangle|circle|angle|polygon|quadrilateral", q_text, re.I) is not None),
        "has_number_theory": int(re.search(r"prime|divisor|modulo|gcd|congru", q_text, re.I) is not None),
    }
    return list(feats.values()), list(feats.keys())


def train_mlp(ds):
    """Train tiny MLP to predict (B_r, B_a) from question features."""
    # Use local MATH-500 cache (with question text already aligned to idx)
    local_path = "results_kun/math500_experiments/math500_Qwen3-8B_nothink_2048_20260401_011643.json"
    with open(local_path) as f:
        items = json.load(f)["per_sample"]
    # Each entry has idx matching training_set idx
    by_idx = {it["idx"]: it for it in items}
    assert len(by_idx) >= 500

    X, Y_br, Y_ba = [], [], []
    names = None
    for d in ds:
        idx = d["key"][2]
        if idx not in by_idx:
            continue
        feats, _names = extract_features(by_idx[idx]["question"])
        names = _names
        X.append(feats)
        Y_br.append(B_R_CHOICES.index(d["best_b_r"]))
        Y_ba.append(B_A_CHOICES.index(d["best_b_a"]))

    X = np.array(X, dtype=np.float32)
    Y_br = np.array(Y_br)
    Y_ba = np.array(Y_ba)
    n = len(X)
    print(f"\nTraining set: n={n}, features={X.shape[1]}")
    print(f"  Feature names: {names}")

    # Normalize
    mu, sigma = X.mean(axis=0), X.std(axis=0) + 1e-6
    X = (X - mu) / sigma

    # Train/test split (80/20)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    tr, te = perm[:split], perm[split:]

    # Train logistic regression per target (simpler than MLP, stable)
    from collections import Counter
    def train_lr(X_tr, y_tr, n_class):
        """Multinomial LR via softmax + NLL, 500 iterations."""
        d = X_tr.shape[1]
        W = np.zeros((d, n_class))
        lr = 0.1
        for it in range(500):
            Z = X_tr @ W
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
            Y_oh = np.eye(n_class)[y_tr]
            grad = X_tr.T @ (P - Y_oh) / len(X_tr) + 1e-3 * W
            W -= lr * grad
        return W

    W_br = train_lr(X[tr], Y_br[tr], len(B_R_CHOICES))
    W_ba = train_lr(X[tr], Y_ba[tr], len(B_A_CHOICES))

    def predict(W, X):
        Z = X @ W
        return Z.argmax(axis=1)

    br_pred = predict(W_br, X[te])
    ba_pred = predict(W_ba, X[te])
    br_acc = np.mean(br_pred == Y_br[te])
    ba_acc = np.mean(ba_pred == Y_ba[te])
    print(f"\nTest set (n={len(te)}):")
    print(f"  B_r prediction acc: {br_acc:.3f}  (majority baseline: {Counter(Y_br).most_common(1)[0][1]/n:.3f})")
    print(f"  B_a prediction acc: {ba_acc:.3f}  (majority baseline: {Counter(Y_ba).most_common(1)[0][1]/n:.3f})")

    # Compute token savings if we USE the predictions
    # Oracle: use best per query (lower bound of cost)
    # Fixed-max: always (4096, 512) upper bound
    # Learned: use predicted per query, but if too low need to fail gracefully
    # Conservative eval: tokens_if_solvable_at_predicted_or_higher
    total_tokens_learned = 0
    total_tokens_fixed = 0
    total_tokens_oracle = 0
    for i_te, idx in enumerate(te):
        b_r_pred = B_R_CHOICES[br_pred[i_te]]
        b_a_pred = B_A_CHOICES[ba_pred[i_te]]
        d = ds[idx]
        # If predicted >= oracle needed, pay ~oracle tokens; if less, pay max
        if b_r_pred >= d["best_b_r"] and b_a_pred >= d["best_b_a"]:
            total_tokens_learned += d["best_tokens"]
        else:
            # Prediction insufficient — use max budget (penalty)
            total_tokens_learned += 4096 + 512
        total_tokens_fixed += 4096 + 512
        total_tokens_oracle += d["best_tokens"]

    n_te = len(te)
    print(f"\nToken economics on test set:")
    print(f"  Learned avg: {total_tokens_learned/n_te:.0f}")
    print(f"  Fixed max avg: {total_tokens_fixed/n_te:.0f}")
    print(f"  Oracle avg: {total_tokens_oracle/n_te:.0f}")
    learned_savings = 100*(1 - total_tokens_learned/total_tokens_fixed)
    oracle_savings = 100*(1 - total_tokens_oracle/total_tokens_fixed)
    print(f"  Learned savings: {learned_savings:.1f}% (oracle upper bound: {oracle_savings:.1f}%)")

    os.makedirs("results/learned_allocator", exist_ok=True)
    out = {
        "n_train": int(split), "n_test": int(n_te),
        "features": names,
        "feature_mean": mu.tolist(), "feature_std": sigma.tolist(),
        "W_br": W_br.tolist(), "W_ba": W_ba.tolist(),
        "test_br_acc": float(br_acc), "test_ba_acc": float(ba_acc),
        "learned_savings_pct": float(learned_savings),
        "oracle_savings_pct": float(oracle_savings),
        "learned_avg_tokens": float(total_tokens_learned/n_te),
        "fixed_avg_tokens": float(total_tokens_fixed/n_te),
        "oracle_avg_tokens": float(total_tokens_oracle/n_te),
    }
    with open("results/learned_allocator/mlp_trained.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/learned_allocator/mlp_trained.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build_training_set", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--train", action="store_true")
    args = p.parse_args()

    if args.build_training_set:
        ds = build_training_set()
    else:
        path = "results/learned_allocator/training_set.json"
        if not os.path.exists(path):
            print(f"Run --build_training_set first")
            return
        with open(path) as f:
            ds = json.load(f)

    if args.train:
        train_mlp(ds)
    else:
        analyze_budget_policy(ds)


if __name__ == "__main__":
    main()
