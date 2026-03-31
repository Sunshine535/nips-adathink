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
        m = re.match(r"fixed_(\d+)_correct", k)
        if m:
            budgets.append(int(m.group(1)))
    budgets = sorted(set(budgets))
    if not budgets:
        raise RuntimeError("No budget columns found.")
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


def softmax(scores: List[float]) -> List[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return [e / z for e in exps]


def featurize_row(
    row: Dict[str, str],
    min_budget: int,
    max_budget: int,
    hash_dim_question: int,
    hash_dim_raw: int,
) -> Dict[int, float]:
    x: Dict[int, float] = {}

    for tok in WORD_RE.findall((row.get("question") or "").lower()):
        idx = stable_hash(tok, hash_dim_question)
        x[idx] = x.get(idx, 0.0) + 1.0

    raw_key = f"fixed_{min_budget}_raw"
    raw_text = row.get(raw_key) or ""
    base = hash_dim_question
    for tok in WORD_RE.findall(raw_text.lower()):
        idx = base + stable_hash(tok, hash_dim_raw)
        x[idx] = x.get(idx, 0.0) + 1.0

    tail = base + hash_dim_raw
    tok_val = to_float(row.get(f"fixed_{min_budget}_tokens", 0.0))
    pred_val = parse_num(row.get(f"fixed_{min_budget}_pred", ""))

    x[tail + 0] = 1.0
    x[tail + 1] = min(tok_val / max(1.0, float(max_budget)), 2.0)
    x[tail + 2] = min(len(raw_text) / 2500.0, 2.0)
    x[tail + 3] = min(len(NUM_RE.findall(raw_text)) / 25.0, 2.0)
    x[tail + 4] = 0.0 if pred_val is None else min(math.log10(abs(pred_val) + 1.0) / 6.0, 2.0)
    x[tail + 5] = 1.0 if ("final answer" in raw_text.lower()) else 0.0
    x[tail + 6] = 1.0 if row.get(f"fixed_{min_budget}_pred_source", "") == "projection" else 0.0
    x[tail + 7] = to_float(row.get(f"fixed_{min_budget}_projection_tokens", 0.0)) / 64.0

    norm = math.sqrt(sum(v * v for v in x.values()))
    if norm > 0:
        for k in list(x.keys()):
            x[k] = x[k] / norm
    return x


def train_expected_utility(
    xs: List[Dict[int, float]],
    us: List[List[float]],
    ts: List[List[float]],
    target_tokens: float,
    cost_weight: float,
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
    norm_tokens: float,
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
            u_vec = us[i]
            t_vec = ts[i]

            scores = []
            for c in range(n_classes):
                w = ws[c]
                s = bs[c]
                for fid, val in x.items():
                    s += w.get(fid, 0.0) * val
                scores.append(s)
            p = softmax(scores)

            exp_u = sum(p[c] * u_vec[c] for c in range(n_classes))
            exp_t = sum(p[c] * t_vec[c] for c in range(n_classes))

            excess = max(0.0, exp_t - target_tokens)
            cost_coeff = 0.0
            if cost_weight > 0.0 and excess > 0.0:
                cost_coeff = 2.0 * cost_weight * excess / max(1.0, norm_tokens * norm_tokens)

            grads = [0.0 for _ in range(n_classes)]
            for c in range(n_classes):
                g = p[c] * (exp_u - u_vec[c])
                if cost_coeff > 0.0:
                    g += cost_coeff * p[c] * (t_vec[c] - exp_t)
                grads[c] = g

            for c in range(n_classes):
                g = grads[c]
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
    return max(range(len(scores)), key=lambda i: scores[i])


def eval_policy(
    rows: List[Dict[str, str]],
    xs: List[Dict[int, float]],
    ws: List[Dict[int, float]],
    bs: List[float],
    budgets: List[int],
    lambda_cost: float,
    norm_tokens: float,
) -> Dict:
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


def build_training_mats(
    rows: List[Dict[str, str]],
    budgets: List[int],
    lambda_cost: float,
    norm_tokens: float,
    hash_dim_question: int,
    hash_dim_raw: int,
) -> Tuple[List[Dict[int, float]], List[List[float]], List[List[float]]]:
    min_budget = budgets[0]
    max_budget = budgets[-1]
    xs = []
    us = []
    ts = []
    for r in rows:
        xs.append(featurize_row(r, min_budget, max_budget, hash_dim_question, hash_dim_raw))
        us.append([per_budget_utility(r, b, lambda_cost, norm_tokens) for b in budgets])
        ts.append([to_float(r.get(f"fixed_{b}_tokens", 0.0)) for b in budgets])
    return xs, us, ts


def main():
    ap = argparse.ArgumentParser(description="Parametric budget controller with constrained expected-utility training")
    ap.add_argument("--input_csvs", nargs="+", required=True)
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument("--hash_dim_question", type=int, default=2048)
    ap.add_argument("--hash_dim_raw", type=int, default=1024)
    ap.add_argument("--target_budget", type=int, default=0, help="0 means middle budget")
    ap.add_argument("--epochs_grid", type=str, default="30,50")
    ap.add_argument("--lr_grid", type=str, default="0.1,0.2")
    ap.add_argument("--l2_grid", type=str, default="1e-4,5e-4")
    ap.add_argument("--cost_weight_grid", type=str, default="0.0,0.5")
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

    target_budget = args.target_budget
    if target_budget == 0:
        target_budget = budgets[len(budgets) // 2]
    if target_budget not in budgets:
        raise RuntimeError(f"target_budget={target_budget} not in budgets={budgets}")

    epochs_grid = parse_int_list(args.epochs_grid)
    lr_grid = parse_float_list(args.lr_grid)
    l2_grid = parse_float_list(args.l2_grid)
    cw_grid = parse_float_list(args.cost_weight_grid)

    folds = []
    all_rows = []
    for i in range(len(datasets)):
        test_name, test_rows = datasets[i]
        train_rows = []
        for j, (_, rows) in enumerate(datasets):
            if j != i:
                train_rows.extend(rows)

        inner_train, inner_val = split_inner(train_rows)
        x_tr, u_tr, t_tr = build_training_mats(
            inner_train,
            budgets,
            args.lambda_cost,
            args.norm_tokens,
            args.hash_dim_question,
            args.hash_dim_raw,
        )
        x_val, _, _ = build_training_mats(
            inner_val,
            budgets,
            args.lambda_cost,
            args.norm_tokens,
            args.hash_dim_question,
            args.hash_dim_raw,
        )

        inner_target_tokens = sum(
            to_float(r.get(f"fixed_{target_budget}_tokens", 0.0)) for r in inner_train
        ) / max(1, len(inner_train))

        best_cfg = None
        best_u = -1e18
        cfg_id = 0
        for ep in epochs_grid:
            for lr in lr_grid:
                for l2 in l2_grid:
                    for cw in cw_grid:
                        ws, bs = train_expected_utility(
                            x_tr,
                            u_tr,
                            t_tr,
                            target_tokens=inner_target_tokens,
                            cost_weight=cw,
                            n_classes=len(budgets),
                            epochs=ep,
                            lr=lr,
                            l2=l2,
                            norm_tokens=args.norm_tokens,
                            seed=args.seed + 1000 * i + cfg_id,
                        )
                        ev = eval_policy(
                            inner_val,
                            x_val,
                            ws,
                            bs,
                            budgets,
                            args.lambda_cost,
                            args.norm_tokens,
                        )
                        if ev["avg_utility"] > best_u:
                            best_u = ev["avg_utility"]
                            best_cfg = {"epochs": ep, "lr": lr, "l2": l2, "cost_weight": cw}
                        cfg_id += 1

        x_train, u_train, t_train = build_training_mats(
            train_rows,
            budgets,
            args.lambda_cost,
            args.norm_tokens,
            args.hash_dim_question,
            args.hash_dim_raw,
        )
        x_test, _, _ = build_training_mats(
            test_rows,
            budgets,
            args.lambda_cost,
            args.norm_tokens,
            args.hash_dim_question,
            args.hash_dim_raw,
        )

        train_target_tokens = sum(
            to_float(r.get(f"fixed_{target_budget}_tokens", 0.0)) for r in train_rows
        ) / max(1, len(train_rows))

        ws, bs = train_expected_utility(
            x_train,
            u_train,
            t_train,
            target_tokens=train_target_tokens,
            cost_weight=best_cfg["cost_weight"],
            n_classes=len(budgets),
            epochs=best_cfg["epochs"],
            lr=best_cfg["lr"],
            l2=best_cfg["l2"],
            norm_tokens=args.norm_tokens,
            seed=args.seed + 2000 * i + 7,
        )

        learned = eval_policy(
            test_rows,
            x_test,
            ws,
            bs,
            budgets,
            args.lambda_cost,
            args.norm_tokens,
        )
        fixed = {str(b): eval_fixed(test_rows, b, args.lambda_cost, args.norm_tokens) for b in budgets}
        oracle = eval_oracle(test_rows, budgets, args.lambda_cost, args.norm_tokens)

        folds.append(
            {
                "test_csv": test_name,
                "test_size": len(test_rows),
                "selected_hparams": best_cfg,
                "target_budget": target_budget,
                "target_tokens_train": train_target_tokens,
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
    lam_tag = str(args.lambda_cost).replace(".", "p")
    out_json = (
        args.output_json
        if args.output_json
        else f"results/param_controller_lam{lam_tag}_{ts}.json"
    )
    out_csv = (
        args.output_csv
        if args.output_csv
        else f"results/param_controller_rows_lam{lam_tag}_{ts}.csv"
    )

    summary = {
        "meta": {
            "timestamp_utc": ts,
            "input_csvs": args.input_csvs,
            "budgets": budgets,
            "lambda_cost": args.lambda_cost,
            "norm_tokens": args.norm_tokens,
            "target_budget": target_budget,
            "hash_dim_question": args.hash_dim_question,
            "hash_dim_raw": args.hash_dim_raw,
            "epochs_grid": epochs_grid,
            "lr_grid": lr_grid,
            "l2_grid": l2_grid,
            "cost_weight_grid": cw_grid,
            "seed": args.seed,
            "protocol": "leave-one-csv-out with inner hyperparam selection",
            "objective": "maximize expected utility with soft budget-constraint penalty",
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
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
