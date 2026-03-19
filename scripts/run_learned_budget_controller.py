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


def per_budget_utility(row: Dict[str, str], budget: int, lambda_cost: float, norm_tokens: float) -> float:
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / norm_tokens)


def best_budget_label(row: Dict[str, str], budgets: List[int], lambda_cost: float, norm_tokens: float) -> int:
    best_i = 0
    best_u = per_budget_utility(row, budgets[0], lambda_cost, norm_tokens)
    for i in range(1, len(budgets)):
        u = per_budget_utility(row, budgets[i], lambda_cost, norm_tokens)
        if u > best_u:
            best_u = u
            best_i = i
    return best_i


def featurize_row(
    row: Dict[str, str],
    hash_dim_question: int,
    hash_dim_raw: int,
    budgets: List[int],
) -> Dict[int, float]:
    x: Dict[int, float] = {}

    # Hashed bag-of-words from question
    for tok in WORD_RE.findall((row.get("question") or "").lower()):
        idx = stable_hash(tok, hash_dim_question)
        x[idx] = x.get(idx, 0.0) + 1.0

    # Hashed bag-of-words from 128-step raw text (available after first stage)
    raw128 = row.get("fixed_128_raw") or ""
    base = hash_dim_question
    for tok in WORD_RE.findall(raw128.lower()):
        idx = base + stable_hash(tok, hash_dim_raw)
        x[idx] = x.get(idx, 0.0) + 1.0

    # Dense features in tail slots
    tail = base + hash_dim_raw
    toks128 = to_float(row.get("fixed_128_tokens", 0.0))
    x[tail + 0] = 1.0
    x[tail + 1] = min(toks128 / max(1.0, float(budgets[-1])), 2.0)
    x[tail + 2] = min(len(raw128) / 2000.0, 2.0)
    x[tail + 3] = min(len(NUM_RE.findall(raw128)) / 20.0, 2.0)
    pred = row.get("fixed_128_pred") or ""
    pv = parse_num(pred)
    x[tail + 4] = 0.0 if pv is None else min(math.log10(abs(pv) + 1.0) / 6.0, 2.0)
    x[tail + 5] = 1.0 if ("final answer" in raw128.lower()) else 0.0
    x[tail + 6] = 1.0 if row.get("fixed_128_pred_source", "") == "projection" else 0.0
    x[tail + 7] = to_float(row.get("fixed_128_projection_tokens", 0.0)) / 64.0

    # L2 normalize sparse vector
    norm = math.sqrt(sum(v * v for v in x.values()))
    if norm > 0:
        for k in list(x.keys()):
            x[k] = x[k] / norm
    return x


def softmax(scores: List[float]) -> List[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def train_softmax(
    xs: List[Dict[int, float]],
    ys: List[int],
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> Tuple[List[Dict[int, float]], List[float]]:
    rnd = random.Random(seed)
    ws: List[Dict[int, float]] = [dict() for _ in range(n_classes)]
    bs: List[float] = [0.0 for _ in range(n_classes)]
    order = list(range(len(xs)))

    for _ in range(epochs):
        rnd.shuffle(order)
        for i in order:
            x = xs[i]
            y = ys[i]
            scores = []
            for c in range(n_classes):
                w = ws[c]
                s = bs[c]
                for fid, val in x.items():
                    s += w.get(fid, 0.0) * val
                scores.append(s)
            ps = softmax(scores)
            for c in range(n_classes):
                g = ps[c] - (1.0 if c == y else 0.0)
                bs[c] -= lr * g
                w = ws[c]
                for fid, val in x.items():
                    old = w.get(fid, 0.0)
                    newv = old - lr * (g * val + l2 * old)
                    if abs(newv) < 1e-12:
                        if fid in w:
                            del w[fid]
                    else:
                        w[fid] = newv
    return ws, bs


def predict_class(x: Dict[int, float], ws: List[Dict[int, float]], bs: List[float]) -> int:
    scores = []
    for c in range(len(ws)):
        s = bs[c]
        w = ws[c]
        for fid, val in x.items():
            s += w.get(fid, 0.0) * val
        scores.append(s)
    best = max(range(len(scores)), key=lambda i: scores[i])
    return best


def eval_policy(rows, xs, ws, bs, budgets, lambda_cost, norm_tokens):
    n = max(1, len(rows))
    acc = 0.0
    toks = 0.0
    util = 0.0
    match_best = 0
    out_rows = []

    for r, x in zip(rows, xs):
        pred_label = predict_class(x, ws, bs)
        chosen_budget = budgets[pred_label]
        best_label = best_budget_label(r, budgets, lambda_cost, norm_tokens)
        match_best += 1 if pred_label == best_label else 0

        c = to_int(r.get(f"fixed_{chosen_budget}_correct", 0))
        t = to_float(r.get(f"fixed_{chosen_budget}_tokens", 0.0))
        u = c - lambda_cost * (t / norm_tokens)
        acc += c
        toks += t
        util += u

        out_rows.append(
            {
                "idx": r.get("idx", ""),
                "chosen_budget": chosen_budget,
                "best_budget": budgets[best_label],
                "correct": c,
                "tokens": t,
                "utility": u,
            }
        )

    return {
        "accuracy": acc / n,
        "avg_tokens": toks / n,
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
        best_b = budgets[0]
        best_u = per_budget_utility(r, best_b, lambda_cost, norm_tokens)
        for b in budgets[1:]:
            u = per_budget_utility(r, b, lambda_cost, norm_tokens)
            if u > best_u:
                best_u = u
                best_b = b
        acc += to_int(r.get(f"fixed_{best_b}_correct", 0))
        toks += to_float(r.get(f"fixed_{best_b}_tokens", 0.0))
        util += best_u
    return {"accuracy": acc / n, "avg_tokens": toks / n, "avg_utility": util / n}


def read_rows(csv_path: str):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Learned budget controller from per-sample traces")
    ap.add_argument(
        "--input_csvs",
        nargs="+",
        default=[
            "methods/01_adathink/results/per_sample_Qwen3.5_27B_20260227_150649.csv",
            "methods/01_adathink/results/per_sample_Qwen3.5_27B_20260227_152431.csv",
            "methods/01_adathink/results/per_sample_Qwen3.5_27B_20260227_154356.csv",
        ],
    )
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument("--hash_dim_question", type=int, default=2048)
    ap.add_argument("--hash_dim_raw", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--output_dir", type=str, default="methods/01_adathink/results")
    args = ap.parse_args()

    input_rows = []
    budgets = None
    for p in args.input_csvs:
        rows = read_rows(p)
        input_rows.append((p, rows))
        b = detect_budgets(list(rows[0].keys()))
        budgets = b if budgets is None else budgets
        if b != budgets:
            raise RuntimeError(f"Budget mismatch: {p} has {b}, expected {budgets}")

    n_classes = len(budgets)
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    folds = []
    global_rows = []

    for test_i in range(len(input_rows)):
        test_name, test_rows = input_rows[test_i]
        train_rows = []
        for i, (_, rows) in enumerate(input_rows):
            if i != test_i:
                train_rows.extend(rows)

        x_train = [
            featurize_row(r, args.hash_dim_question, args.hash_dim_raw, budgets)
            for r in train_rows
        ]
        y_train = [
            best_budget_label(r, budgets, args.lambda_cost, args.norm_tokens)
            for r in train_rows
        ]
        ws, bs = train_softmax(
            x_train,
            y_train,
            n_classes=n_classes,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed + test_i,
        )

        x_test = [featurize_row(r, args.hash_dim_question, args.hash_dim_raw, budgets) for r in test_rows]
        learned = eval_policy(
            test_rows, x_test, ws, bs, budgets, args.lambda_cost, args.norm_tokens
        )
        fixed = {str(b): eval_fixed(test_rows, b, args.lambda_cost, args.norm_tokens) for b in budgets}
        oracle = eval_oracle(test_rows, budgets, args.lambda_cost, args.norm_tokens)

        fold = {
            "test_csv": test_name,
            "test_size": len(test_rows),
            "learned": {
                k: learned[k]
                for k in ["accuracy", "avg_tokens", "avg_utility", "action_match_best_rate"]
            },
            "fixed": fixed,
            "oracle": oracle,
        }
        folds.append(fold)

        for rr in learned["rows"]:
            rr["test_csv"] = test_name
            global_rows.append(rr)

    # Macro average across folds
    def mean_of(path):
        keys = path.split(".")
        vals = []
        for f in folds:
            cur = f
            for k in keys:
                cur = cur[k]
            vals.append(cur)
        return sum(vals) / max(1, len(vals))

    summary = {
        "meta": {
            "timestamp_utc": ts,
            "input_csvs": args.input_csvs,
            "budgets": budgets,
            "lambda_cost": args.lambda_cost,
            "norm_tokens": args.norm_tokens,
            "hash_dim_question": args.hash_dim_question,
            "hash_dim_raw": args.hash_dim_raw,
            "epochs": args.epochs,
            "lr": args.lr,
            "l2": args.l2,
            "seed": args.seed,
            "protocol": "leave-one-csv-out",
        },
        "folds": folds,
        "macro_mean": {
            "learned": {
                "accuracy": mean_of("learned.accuracy"),
                "avg_tokens": mean_of("learned.avg_tokens"),
                "avg_utility": mean_of("learned.avg_utility"),
                "action_match_best_rate": mean_of("learned.action_match_best_rate"),
            },
            "fixed": {
                str(b): {
                    "accuracy": sum(f["fixed"][str(b)]["accuracy"] for f in folds) / len(folds),
                    "avg_tokens": sum(f["fixed"][str(b)]["avg_tokens"] for f in folds) / len(folds),
                    "avg_utility": sum(f["fixed"][str(b)]["avg_utility"] for f in folds) / len(folds),
                }
                for b in budgets
            },
            "oracle": {
                "accuracy": mean_of("oracle.accuracy"),
                "avg_tokens": mean_of("oracle.avg_tokens"),
                "avg_utility": mean_of("oracle.avg_utility"),
            },
        },
    }

    lam_tag = str(args.lambda_cost).replace(".", "p")
    out_json = os.path.join(args.output_dir, f"learned_controller_lam{lam_tag}_{ts}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    out_csv = os.path.join(args.output_dir, f"learned_controller_rows_lam{lam_tag}_{ts}.csv")
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
            ],
        )
        writer.writeheader()
        writer.writerows(global_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
