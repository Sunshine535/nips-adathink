#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


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


def stable_hash(token: str, mod: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % mod


def parse_num(s: str):
    if s is None:
        return None
    t = str(s).strip().replace(",", "")
    if not t:
        return None
    if "/" in t:
        p = t.split("/")
        if len(p) == 2:
            try:
                den = float(p[1])
                if den == 0:
                    return None
                return float(p[0]) / den
            except Exception:
                return None
    try:
        return float(t)
    except Exception:
        return None


def detect_budgets(fieldnames: List[str]) -> List[int]:
    budgets = []
    for k in fieldnames:
        if k.startswith("fixed_") and k.endswith("_correct"):
            m = re.match(r"fixed_(\d+)_correct", k)
            if m:
                budgets.append(int(m.group(1)))
    budgets = sorted(set(budgets))
    if not budgets:
        raise RuntimeError("No fixed budget columns found.")
    return budgets


def utility(row: Dict[str, str], budget: int, lambda_cost: float, norm_tokens: float) -> float:
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / norm_tokens)


def best_budget_label(row: Dict[str, str], budgets: List[int], lambda_cost: float, norm_tokens: float) -> int:
    best_i = 0
    best_u = utility(row, budgets[0], lambda_cost, norm_tokens)
    for i in range(1, len(budgets)):
        u = utility(row, budgets[i], lambda_cost, norm_tokens)
        if u > best_u:
            best_u = u
            best_i = i
    return best_i


def featurize_row(
    row: Dict[str, str],
    hash_dim_question: int,
    hash_dim_raw: int,
    min_budget: int,
    max_budget: int,
) -> Dict[int, float]:
    x: Dict[int, float] = {}

    for tok in WORD_RE.findall((row.get("question") or "").lower()):
        idx = stable_hash(tok, hash_dim_question)
        x[idx] = x.get(idx, 0.0) + 1.0

    raw = row.get(f"fixed_{min_budget}_raw") or ""
    base = hash_dim_question
    for tok in WORD_RE.findall(raw.lower()):
        idx = base + stable_hash(tok, hash_dim_raw)
        x[idx] = x.get(idx, 0.0) + 1.0

    tail = base + hash_dim_raw
    tok_min = to_float(row.get(f"fixed_{min_budget}_tokens", 0.0))
    pred_min = row.get(f"fixed_{min_budget}_pred") or ""
    pred_val = parse_num(pred_min)

    x[tail + 0] = 1.0
    x[tail + 1] = min(tok_min / max(1.0, float(max_budget)), 2.0)
    x[tail + 2] = min(len(raw) / 2500.0, 2.0)
    x[tail + 3] = min(len(NUM_RE.findall(raw)) / 25.0, 2.0)
    x[tail + 4] = 0.0 if pred_val is None else min(math.log10(abs(pred_val) + 1.0) / 6.0, 2.0)
    x[tail + 5] = 1.0 if ("final answer" in raw.lower()) else 0.0
    x[tail + 6] = 1.0 if row.get(f"fixed_{min_budget}_pred_source", "") == "projection" else 0.0
    x[tail + 7] = to_float(row.get(f"fixed_{min_budget}_projection_tokens", 0.0)) / 64.0

    norm = math.sqrt(sum(v * v for v in x.values()))
    if norm > 0:
        for k in list(x.keys()):
            x[k] = x[k] / norm
    return x


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def train_binary_logistic(
    xs: List[Dict[int, float]],
    ys: List[int],
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> Tuple[Dict[int, float], float]:
    rnd = random.Random(seed)
    w: Dict[int, float] = {}
    b = 0.0
    order = list(range(len(xs)))

    for _ in range(epochs):
        rnd.shuffle(order)
        for i in order:
            x = xs[i]
            y = float(ys[i])
            s = b
            for fid, val in x.items():
                s += w.get(fid, 0.0) * val
            p = sigmoid(s)
            g = p - y
            b -= lr * g
            for fid, val in x.items():
                old = w.get(fid, 0.0)
                newv = old - lr * (g * val + l2 * old)
                if abs(newv) < 1e-12:
                    if fid in w:
                        del w[fid]
                else:
                    w[fid] = newv
    return w, b


def predict_proba(x: Dict[int, float], w: Dict[int, float], b: float) -> float:
    s = b
    for fid, val in x.items():
        s += w.get(fid, 0.0) * val
    return sigmoid(s)


def split_inner(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    train, val = [], []
    for r in rows:
        idx = to_int(r.get("idx", 0))
        if idx % 5 == 0:
            val.append(r)
        else:
            train.append(r)
    if not val:
        k = max(1, len(rows) // 5)
        val = rows[-k:]
        train = rows[:-k]
    return train, val


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def eval_policy(
    rows: List[Dict[str, str]],
    xs: List[Dict[int, float]],
    budgets: List[int],
    mean_tokens: Dict[int, float],
    models: Dict[int, Tuple[Dict[int, float], float]],
    decision_lambda: float,
    eval_lambda: float,
    norm_tokens: float,
) -> Dict:
    n = max(1, len(rows))
    acc = 0.0
    tok = 0.0
    util = 0.0
    match_best = 0
    out_rows = []

    for r, x in zip(rows, xs):
        best_b = budgets[0]
        best_score = -1e18
        prob_by_b = {}
        for b in budgets:
            w, bias = models[b]
            p = predict_proba(x, w, bias)
            prob_by_b[b] = p
            score = p - decision_lambda * (mean_tokens[b] / norm_tokens)
            if score > best_score:
                best_score = score
                best_b = b

        c = to_int(r.get(f"fixed_{best_b}_correct", 0))
        t = to_float(r.get(f"fixed_{best_b}_tokens", 0.0))
        u = c - eval_lambda * (t / norm_tokens)
        acc += c
        tok += t
        util += u

        oracle_b = budgets[best_budget_label(r, budgets, eval_lambda, norm_tokens)]
        if best_b == oracle_b:
            match_best += 1

        out_rows.append(
            {
                "idx": r.get("idx", ""),
                "chosen_budget": best_b,
                "best_budget": oracle_b,
                "correct": c,
                "tokens": t,
                "utility": u,
                "p_correct_128": prob_by_b.get(128, 0.0),
                "p_correct_256": prob_by_b.get(256, 0.0),
                "p_correct_512": prob_by_b.get(512, 0.0),
            }
        )

    return {
        "accuracy": acc / n,
        "avg_tokens": tok / n,
        "avg_utility": util / n,
        "action_match_best_rate": match_best / n,
        "rows": out_rows,
    }


def eval_fixed(rows, budget, lambda_cost, norm_tokens):
    n = max(1, len(rows))
    acc = sum(to_int(r.get(f"fixed_{budget}_correct", 0)) for r in rows) / n
    toks = sum(to_float(r.get(f"fixed_{budget}_tokens", 0.0)) for r in rows) / n
    util = acc - lambda_cost * (toks / norm_tokens)
    return {"accuracy": acc, "avg_tokens": toks, "avg_utility": util}


def eval_oracle(rows, budgets, lambda_cost, norm_tokens):
    n = max(1, len(rows))
    acc = toks = util = 0.0
    for r in rows:
        b = budgets[best_budget_label(r, budgets, lambda_cost, norm_tokens)]
        acc += to_int(r.get(f"fixed_{b}_correct", 0))
        toks += to_float(r.get(f"fixed_{b}_tokens", 0.0))
        util += utility(r, b, lambda_cost, norm_tokens)
    return {"accuracy": acc / n, "avg_tokens": toks / n, "avg_utility": util / n}


def make_features(
    rows: List[Dict[str, str]],
    budgets: List[int],
    hash_dim_question: int,
    hash_dim_raw: int,
) -> List[Dict[int, float]]:
    min_budget = budgets[0]
    max_budget = budgets[-1]
    return [
        featurize_row(r, hash_dim_question, hash_dim_raw, min_budget, max_budget)
        for r in rows
    ]


def main():
    ap = argparse.ArgumentParser(description="Value-based budget controller with leave-one-csv-out evaluation")
    ap.add_argument("--input_csvs", nargs="+", required=True)
    ap.add_argument("--eval_lambda", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument("--target_budget", type=int, default=256)
    ap.add_argument("--budget_penalty", type=float, default=0.0)
    ap.add_argument("--hash_dim_question", type=int, default=2048)
    ap.add_argument("--hash_dim_raw", type=int, default=1024)
    ap.add_argument("--epochs_grid", type=str, default="20,40")
    ap.add_argument("--lr_grid", type=str, default="0.05,0.15")
    ap.add_argument("--l2_grid", type=str, default="1e-5,1e-4")
    ap.add_argument(
        "--decision_lambda_grid",
        type=str,
        default="0.00,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70",
    )
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument("--output_csv", type=str, default="")
    args = ap.parse_args()

    datasets = []
    budgets = None
    for p in args.input_csvs:
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f"No rows in {p}")
        b = detect_budgets(list(rows[0].keys()))
        budgets = b if budgets is None else budgets
        if b != budgets:
            raise RuntimeError(f"Budget mismatch in {p}: {b} vs {budgets}")
        datasets.append((p, rows))

    if args.target_budget not in budgets:
        raise RuntimeError(f"target_budget={args.target_budget} not in budgets={budgets}")

    epochs_grid = parse_int_list(args.epochs_grid)
    lr_grid = parse_float_list(args.lr_grid)
    l2_grid = parse_float_list(args.l2_grid)
    decision_lambda_grid = parse_float_list(args.decision_lambda_grid)

    folds = []
    all_rows = []

    for i in range(len(datasets)):
        test_name, test_rows = datasets[i]
        train_rows = []
        for j, (_, rows) in enumerate(datasets):
            if j != i:
                train_rows.extend(rows)

        inner_train, inner_val = split_inner(train_rows)
        x_inner_train = make_features(inner_train, budgets, args.hash_dim_question, args.hash_dim_raw)
        x_inner_val = make_features(inner_val, budgets, args.hash_dim_question, args.hash_dim_raw)

        target_tok_inner = sum(
            to_float(r.get(f"fixed_{args.target_budget}_tokens", 0.0)) for r in inner_val
        ) / max(1, len(inner_val))

        best_cfg = None
        best_score = -1e18
        cfg_id = 0

        for ep in epochs_grid:
            for lr in lr_grid:
                for l2 in l2_grid:
                    models = {}
                    for bi, b in enumerate(budgets):
                        y = [to_int(r.get(f"fixed_{b}_correct", 0)) for r in inner_train]
                        models[b] = train_binary_logistic(
                            x_inner_train,
                            y,
                            epochs=ep,
                            lr=lr,
                            l2=l2,
                            seed=args.seed + i * 10000 + cfg_id * 100 + bi,
                        )
                    mean_tokens = {
                        b: sum(to_float(r.get(f"fixed_{b}_tokens", 0.0)) for r in inner_train)
                        / max(1, len(inner_train))
                        for b in budgets
                    }
                    for dlam in decision_lambda_grid:
                        ev = eval_policy(
                            inner_val,
                            x_inner_val,
                            budgets,
                            mean_tokens,
                            models,
                            decision_lambda=dlam,
                            eval_lambda=args.eval_lambda,
                            norm_tokens=args.norm_tokens,
                        )
                        score = ev["avg_utility"]
                        if args.budget_penalty > 0.0:
                            score -= (
                                args.budget_penalty
                                * abs(ev["avg_tokens"] - target_tok_inner)
                                / args.norm_tokens
                            )
                        if score > best_score:
                            best_score = score
                            best_cfg = {
                                "epochs": ep,
                                "lr": lr,
                                "l2": l2,
                                "decision_lambda": dlam,
                            }
                    cfg_id += 1

        x_train = make_features(train_rows, budgets, args.hash_dim_question, args.hash_dim_raw)
        x_test = make_features(test_rows, budgets, args.hash_dim_question, args.hash_dim_raw)
        mean_tokens_train = {
            b: sum(to_float(r.get(f"fixed_{b}_tokens", 0.0)) for r in train_rows) / max(1, len(train_rows))
            for b in budgets
        }

        models = {}
        for bi, b in enumerate(budgets):
            y = [to_int(r.get(f"fixed_{b}_correct", 0)) for r in train_rows]
            models[b] = train_binary_logistic(
                x_train,
                y,
                epochs=best_cfg["epochs"],
                lr=best_cfg["lr"],
                l2=best_cfg["l2"],
                seed=args.seed + i * 20000 + bi,
            )

        learned = eval_policy(
            test_rows,
            x_test,
            budgets,
            mean_tokens_train,
            models,
            decision_lambda=best_cfg["decision_lambda"],
            eval_lambda=args.eval_lambda,
            norm_tokens=args.norm_tokens,
        )
        fixed = {str(b): eval_fixed(test_rows, b, args.eval_lambda, args.norm_tokens) for b in budgets}
        oracle = eval_oracle(test_rows, budgets, args.eval_lambda, args.norm_tokens)

        folds.append(
            {
                "test_csv": test_name,
                "test_size": len(test_rows),
                "selected_hparams": best_cfg,
                "target_budget": args.target_budget,
                "budget_penalty": args.budget_penalty,
                "mean_tokens_train": mean_tokens_train,
                "learned": {
                    k: learned[k]
                    for k in ["accuracy", "avg_tokens", "avg_utility", "action_match_best_rate"]
                },
                "fixed": fixed,
                "oracle": oracle,
            }
        )

        for rr in learned["rows"]:
            rr["test_csv"] = test_name
            all_rows.append(rr)

    macro = {"learned": {}, "fixed": {}, "oracle": {}}
    macro["learned"]["accuracy"] = sum(f["learned"]["accuracy"] for f in folds) / len(folds)
    macro["learned"]["avg_tokens"] = sum(f["learned"]["avg_tokens"] for f in folds) / len(folds)
    macro["learned"]["avg_utility"] = sum(f["learned"]["avg_utility"] for f in folds) / len(folds)
    macro["learned"]["action_match_best_rate"] = (
        sum(f["learned"]["action_match_best_rate"] for f in folds) / len(folds)
    )
    for b in budgets:
        macro["fixed"][str(b)] = {
            "accuracy": sum(f["fixed"][str(b)]["accuracy"] for f in folds) / len(folds),
            "avg_tokens": sum(f["fixed"][str(b)]["avg_tokens"] for f in folds) / len(folds),
            "avg_utility": sum(f["fixed"][str(b)]["avg_utility"] for f in folds) / len(folds),
        }
    macro["oracle"]["accuracy"] = sum(f["oracle"]["accuracy"] for f in folds) / len(folds)
    macro["oracle"]["avg_tokens"] = sum(f["oracle"]["avg_tokens"] for f in folds) / len(folds)
    macro["oracle"]["avg_utility"] = sum(f["oracle"]["avg_utility"] for f in folds) / len(folds)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_json = (
        args.output_json
        if args.output_json
        else f"methods/01_adathink/results/value_controller_evalLam{str(args.eval_lambda).replace('.', 'p')}_{ts}.json"
    )
    out_csv = (
        args.output_csv
        if args.output_csv
        else f"methods/01_adathink/results/value_controller_rows_evalLam{str(args.eval_lambda).replace('.', 'p')}_{ts}.csv"
    )

    summary = {
        "meta": {
            "timestamp_utc": ts,
            "input_csvs": args.input_csvs,
            "budgets": budgets,
            "eval_lambda": args.eval_lambda,
            "norm_tokens": args.norm_tokens,
            "target_budget": args.target_budget,
            "budget_penalty": args.budget_penalty,
            "hash_dim_question": args.hash_dim_question,
            "hash_dim_raw": args.hash_dim_raw,
            "epochs_grid": epochs_grid,
            "lr_grid": lr_grid,
            "l2_grid": l2_grid,
            "decision_lambda_grid": decision_lambda_grid,
            "seed": args.seed,
            "protocol": "leave-one-csv-out with inner hyperparam selection",
            "objective": "predict per-budget correctness then select by value-cost score",
        },
        "folds": folds,
        "macro_mean": macro,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_csv",
                "idx",
                "chosen_budget",
                "best_budget",
                "correct",
                "tokens",
                "utility",
                "p_correct_128",
                "p_correct_256",
                "p_correct_512",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
