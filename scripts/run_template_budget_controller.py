#!/usr/bin/env python3
import argparse
import csv
import json
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


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


def utility(row: Dict[str, str], budget: int, lambda_cost: float, norm_tokens: float) -> float:
    c = to_int(row.get(f"fixed_{budget}_correct", 0))
    t = to_float(row.get(f"fixed_{budget}_tokens", 0.0))
    return c - lambda_cost * (t / norm_tokens)


def question_tokens(row: Dict[str, str]) -> List[str]:
    return WORD_RE.findall((row.get("question") or "").lower())


def make_key(row: Dict[str, str], mode: str) -> str:
    toks = question_tokens(row)
    if mode == "first1":
        return "|".join(toks[:1]) if toks else "_"
    if mode == "first2":
        return "|".join(toks[:2]) if toks else "_"
    if mode == "first3":
        return "|".join(toks[:3]) if toks else "_"
    if mode == "first4":
        return "|".join(toks[:4]) if toks else "_"
    if mode == "first2_lenbin":
        base = "|".join(toks[:2]) if toks else "_"
        return f"{len(toks)//6}|{base}"
    if mode == "first3_lenbin":
        base = "|".join(toks[:3]) if toks else "_"
        return f"{len(toks)//6}|{base}"
    raise ValueError(f"unknown mode: {mode}")


def build_policy(rows: List[Dict[str, str]], budgets: List[int], mode: str, lambda_cost: float, norm_tokens: float):
    stats: Dict[str, Dict[int, List[float]]] = {}
    for r in rows:
        k = make_key(r, mode)
        if k not in stats:
            stats[k] = {b: [0.0, 0] for b in budgets}
        for b in budgets:
            stats[k][b][0] += utility(r, b, lambda_cost, norm_tokens)
            stats[k][b][1] += 1

    mapping: Dict[str, int] = {}
    for k, st in stats.items():
        best_b = budgets[0]
        best_u = st[best_b][0] / max(1, st[best_b][1])
        for b in budgets[1:]:
            u = st[b][0] / max(1, st[b][1])
            if u > best_u:
                best_u = u
                best_b = b
        mapping[k] = best_b

    default_budget = budgets[0]
    best_global = -1e18
    for b in budgets:
        avg_u = sum(utility(r, b, lambda_cost, norm_tokens) for r in rows) / max(1, len(rows))
        if avg_u > best_global:
            best_global = avg_u
            default_budget = b

    return mapping, default_budget


def evaluate(rows, budgets, mode, mapping, default_budget, lambda_cost, norm_tokens):
    n = max(1, len(rows))
    acc = toks = util_sum = 0.0
    match_best = reuse = 0
    out_rows = []

    for r in rows:
        k = make_key(r, mode)
        b = mapping.get(k, default_budget)
        if k in mapping:
            reuse += 1

        best_b = max(budgets, key=lambda bb: utility(r, bb, lambda_cost, norm_tokens))
        if b == best_b:
            match_best += 1

        c = to_int(r.get(f"fixed_{b}_correct", 0))
        t = to_float(r.get(f"fixed_{b}_tokens", 0.0))
        u = utility(r, b, lambda_cost, norm_tokens)
        acc += c
        toks += t
        util_sum += u

        out_rows.append(
            {
                "idx": r.get("idx", ""),
                "mode": mode,
                "key": k,
                "chosen_budget": b,
                "best_budget": best_b,
                "correct": c,
                "tokens": t,
                "utility": u,
            }
        )

    return {
        "accuracy": acc / n,
        "avg_tokens": toks / n,
        "avg_utility": util_sum / n,
        "action_match_best_rate": match_best / n,
        "reuse_rate": reuse / n,
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
        b = max(budgets, key=lambda bb: utility(r, bb, lambda_cost, norm_tokens))
        acc += to_int(r.get(f"fixed_{b}_correct", 0))
        toks += to_float(r.get(f"fixed_{b}_tokens", 0.0))
        util += utility(r, b, lambda_cost, norm_tokens)
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


def main():
    ap = argparse.ArgumentParser(description="Template budget controller with leave-one-csv-out evaluation")
    ap.add_argument(
        "--input_csvs",
        nargs="+",
        default=[],
    )
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--norm_tokens", type=float, default=512.0)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["first1", "first2", "first3", "first4", "first2_lenbin", "first3_lenbin"],
    )
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

    folds = []
    all_rows = []
    for i in range(len(datasets)):
        test_name, test_rows = datasets[i]
        train_rows = []
        for j, (_, rows) in enumerate(datasets):
            if j != i:
                train_rows.extend(rows)

        inner_train, inner_val = split_inner(train_rows)

        best_mode = None
        best_u = -1e18
        for mode in args.modes:
            mapping, default_budget = build_policy(inner_train, budgets, mode, args.lambda_cost, args.norm_tokens)
            ev = evaluate(
                inner_val, budgets, mode, mapping, default_budget, args.lambda_cost, args.norm_tokens
            )
            if ev["avg_utility"] > best_u:
                best_u = ev["avg_utility"]
                best_mode = mode

        mapping, default_budget = build_policy(train_rows, budgets, best_mode, args.lambda_cost, args.norm_tokens)
        learned = evaluate(
            test_rows, budgets, best_mode, mapping, default_budget, args.lambda_cost, args.norm_tokens
        )
        fixed = {str(b): eval_fixed(test_rows, b, args.lambda_cost, args.norm_tokens) for b in budgets}
        oracle = eval_oracle(test_rows, budgets, args.lambda_cost, args.norm_tokens)

        folds.append(
            {
                "test_csv": test_name,
                "test_size": len(test_rows),
                "selected_mode": best_mode,
                "learned": {
                    k: learned[k]
                    for k in [
                        "accuracy",
                        "avg_tokens",
                        "avg_utility",
                        "action_match_best_rate",
                        "reuse_rate",
                    ]
                },
                "fixed": fixed,
                "oracle": oracle,
            }
        )

        for rr in learned["rows"]:
            rr["test_csv"] = test_name
            all_rows.append(rr)

    macro = {
        "learned": {},
        "fixed": {},
        "oracle": {},
    }
    macro["learned"]["accuracy"] = sum(f["learned"]["accuracy"] for f in folds) / len(folds)
    macro["learned"]["avg_tokens"] = sum(f["learned"]["avg_tokens"] for f in folds) / len(folds)
    macro["learned"]["avg_utility"] = sum(f["learned"]["avg_utility"] for f in folds) / len(folds)
    macro["learned"]["action_match_best_rate"] = (
        sum(f["learned"]["action_match_best_rate"] for f in folds) / len(folds)
    )
    macro["learned"]["reuse_rate"] = sum(f["learned"]["reuse_rate"] for f in folds) / len(folds)

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
        else f"results/template_controller_lam{lam_tag}_{ts}.json"
    )
    out_csv = (
        args.output_csv
        if args.output_csv
        else f"results/template_controller_rows_lam{lam_tag}_{ts}.csv"
    )

    summary = {
        "meta": {
            "timestamp_utc": ts,
            "input_csvs": args.input_csvs,
            "budgets": budgets,
            "lambda_cost": args.lambda_cost,
            "norm_tokens": args.norm_tokens,
            "modes": args.modes,
            "protocol": "leave-one-csv-out with inner mode selection",
        },
        "folds": folds,
        "macro_mean": macro,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "test_csv",
                "idx",
                "mode",
                "key",
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
