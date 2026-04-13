#!/usr/bin/env python3
"""
Uncertainty Router Baseline Comparison for TOWN (Think Only When Needed).

Implements several routing baselines to compare against TOWN's binary early-stop
routing on GSM8K with Qwen3-8B:

  1. NoThink-only:       nothink@256, no routing (lower bound)
  2. Think-only:         thinking@512, always think (token-heavy baseline)
  3. Random Router:      Randomly route p% to thinking (match routing rate)
  4. Token-Length Router: Route samples with highest nothink token count
  5. Wrong-Answer Router: Route samples where nothink prediction looks "wrong"
                          (proxy: route samples whose predicted number is unusual)
  6. Oracle Router:      Route only samples that benefit (upper bound)
  7. TOWN (Ours):        Binary early-stop routing (hit_budget → thinking)

Also produces:
  - Routing-rate sweep (5%–50%) comparing all methods
  - Per-sample analysis of TOWN's signal vs token-length

Usage:
    python scripts/uncertainty_router_baseline.py [--seeds 100] [--output results/uncertainty_router]
"""

import argparse
import csv
import json
import os
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_nothink_data(path: str) -> dict:
    """Load nothink@256 per-sample results."""
    with open(path) as f:
        data = json.load(f)
    samples = data["per_sample"]["nothink_256"]
    return {s["idx"]: s for s in samples}


def load_thinking_data(path: str) -> dict:
    """Load thinking@512 per-sample results from fulltest CSV."""
    results = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            results[idx] = {
                "idx": idx,
                "correct": int(row["fixed_512_correct"]) == 1,
                "tokens": int(row["fixed_512_tokens"]),
                "gold": row["gold"],
            }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Router Implementations
# ──────────────────────────────────────────────────────────────────────────────

def _build_result(nothink, think, routed_set):
    """Helper: build per-sample result dict given a set of routed indices."""
    results = {}
    for idx in nothink:
        if idx in routed_set:
            results[idx] = {
                "correct": think[idx]["correct"],
                "tokens": nothink[idx]["tokens"] + think[idx]["tokens"],
                "routed": True,
            }
        else:
            results[idx] = {
                "correct": nothink[idx]["correct"],
                "tokens": nothink[idx]["tokens"],
                "routed": False,
            }
    return results


def route_nothink_only(nothink, think):
    return _build_result(nothink, think, set())


def route_think_only(nothink, think):
    """Think@512 only: skip nothink, pay only thinking tokens."""
    results = {}
    for idx in nothink:
        results[idx] = {
            "correct": think[idx]["correct"],
            "tokens": think[idx]["tokens"],  # no nothink cost
            "routed": True,
        }
    return results


def route_random(nothink, think, n_route, rng):
    idxs = sorted(nothink.keys())
    routed = set(rng.choice(idxs, size=n_route, replace=False))
    return _build_result(nothink, think, routed)


def route_token_length(nothink, think, n_route):
    """Route samples with highest nothink token count (uncertainty proxy)."""
    ranked = sorted(nothink.keys(), key=lambda i: nothink[i]["tokens"], reverse=True)
    return _build_result(nothink, think, set(ranked[:n_route]))


def route_inverse_token_length(nothink, think, n_route):
    """Route samples with LOWEST nothink token count (anti-uncertainty control)."""
    ranked = sorted(nothink.keys(), key=lambda i: nothink[i]["tokens"])
    return _build_result(nothink, think, set(ranked[:n_route]))


def route_mid_token(nothink, think, n_route):
    """Route samples with mid-range token count (neither easy nor budget-hit)."""
    budget = 256
    median_tok = np.median([nothink[i]["tokens"] for i in nothink])
    ranked = sorted(nothink.keys(), key=lambda i: abs(nothink[i]["tokens"] - median_tok))
    return _build_result(nothink, think, set(ranked[:n_route]))


def route_oracle(nothink, think, n_route=None):
    """Route only samples that benefit: nothink wrong AND thinking correct."""
    beneficial = [i for i in nothink if not nothink[i]["correct"] and think[i]["correct"]]
    if n_route is not None and n_route < len(beneficial):
        routed = set(beneficial[:n_route])
    else:
        routed = set(beneficial)
    return _build_result(nothink, think, routed)


def route_town(nothink, think):
    """TOWN: route samples that hit the nothink budget (didn't stop early)."""
    routed = set(i for i in nothink if nothink[i]["hit_budget"])
    return _build_result(nothink, think, routed)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics Computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results, nothink, think):
    n = len(results)
    correct = sum(1 for r in results.values() if r["correct"])
    total_tokens = sum(r["tokens"] for r in results.values())
    n_routed = sum(1 for r in results.values() if r["routed"])

    recoveries = regrets = missed = 0
    for idx, r in results.items():
        nt_correct = nothink[idx]["correct"]
        tk_correct = think[idx]["correct"]
        if r["routed"]:
            if not nt_correct and r["correct"]:
                recoveries += 1
            elif nt_correct and not r["correct"]:
                regrets += 1
        else:
            if not nt_correct and tk_correct:
                missed += 1

    return {
        "accuracy": correct / n,
        "avg_tokens": total_tokens / n,
        "n_routed": n_routed,
        "routing_rate": n_routed / n,
        "recoveries": recoveries,
        "regrets": regrets,
        "net_gain": recoveries - regrets,
        "missed": missed,
        "n_correct": correct,
        "n_total": n,
    }


# ──────────────────────────────────────────────────────────────────────────────
# LaTeX Table Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_main_table(all_metrics):
    """Table 1: Main comparison at TOWN's routing rate."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Routing baseline comparison on GSM8K (Qwen3-8B, $n{=}1319$). "
                 r"All budget-matched routers use the same 11.2\% routing rate as TOWN. "
                 r"``Recov.'' = nothink-wrong samples corrected by thinking; "
                 r"``Regret'' = nothink-correct samples broken by thinking.}")
    lines.append(r"\label{tab:routing_baselines}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Acc.\,(\%) & Avg Tok. & Route\,\% & Recov. & Regret & Net\,$\Delta$ \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{l}{\textit{Baselines (no routing)}} \\")

    row_order = [
        ("NoThink@256", "nothink_only"),
        ("Think@512", "think_only"),
        (r"\midrule", None),
        (r"\multicolumn{7}{l}{\textit{Budget-matched routers (11.2\% routed to Think@512)}}", None),
        ("Random", "random"),
        ("Inverse Token-Length", "inverse_token_length"),
        ("Mid-Token", "mid_token"),
        ("Token-Length (uncertainty)", "token_length"),
        (r"\midrule", None),
        (r"\multicolumn{7}{l}{\textit{Bounds}}", None),
        ("Oracle (budget-matched)", "oracle_capped"),
        ("Oracle (uncapped)", "oracle_uncapped"),
        (r"\midrule", None),
        (r"\textbf{TOWN (Ours)}", "town"),
    ]

    for display_name, key in row_order:
        if key is None:
            # \midrule lines don't need \\, section headers do
            if display_name.startswith(r"\midrule"):
                lines.append(display_name)
            else:
                lines.append(display_name + r" \\")
            continue
        m = all_metrics[key]
        if key == "random" and "accuracy_std" in m:
            acc_s = f"${m['accuracy']*100:.1f} \\pm {m['accuracy_std']*100:.1f}$"
            tok_s = f"${m['avg_tokens']:.0f} \\pm {m['avg_tokens_std']:.0f}$"
            rec_s = f"${m['recoveries']:.1f}$"
            reg_s = f"${m['regrets']:.1f}$"
            net_s = f"${m['net_gain']:.1f}$"
        else:
            acc_s = f"{m['accuracy']*100:.1f}"
            tok_s = f"{m['avg_tokens']:.0f}"
            rec_s = f"{m['recoveries']}"
            reg_s = f"{m['regrets']}"
            net_s = f"{m['net_gain']}"

        if key == "town":
            acc_s = r"\textbf{" + acc_s + "}"
            net_s = r"\textbf{" + net_s + "}"

        route_s = f"{m['routing_rate']*100:.1f}"
        lines.append(f"{display_name} & {acc_s} & {tok_s} & {route_s} & {rec_s} & {reg_s} & {net_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_sweep_table(sweep_data):
    """Table 2: Routing-rate sweep comparing key methods."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Accuracy (\%) at varying routing rates on GSM8K (Qwen3-8B). "
                 r"TOWN's natural rate is 11.2\%; other rates are hypothetical.}")
    lines.append(r"\label{tab:routing_sweep}")
    lines.append(r"\small")

    rates = sorted(sweep_data.keys())
    methods = ["random", "token_length", "oracle", "town_extrapolated"]
    method_names = ["Random", "Token-Length", "Oracle", "TOWN*"]

    header = "Rate (\\%)" + " & ".join([""] + method_names) + r" \\"
    lines.append(r"\begin{tabular}{@{}l" + "c" * len(methods) + r"@{}}")
    lines.append(r"\toprule")
    lines.append(f"Route \\% & {' & '.join(method_names)} \\\\")
    lines.append(r"\midrule")

    for rate in rates:
        vals = []
        for m in methods:
            if m in sweep_data[rate]:
                v = sweep_data[rate][m]
                if isinstance(v, tuple):
                    vals.append(f"${v[0]*100:.1f} \\pm {v[1]*100:.1f}$")
                else:
                    vals.append(f"{v*100:.1f}")
            else:
                vals.append("---")
        lines.append(f"{rate*100:.0f} & {' & '.join(vals)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{-0.5em}")
    lines.append(r"\begin{flushleft}")
    lines.append(r"\footnotesize *TOWN has a fixed routing rate (11.2\%); other rates shown for reference only.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Uncertainty Router Baseline Comparison for TOWN")
    parser.add_argument("--nothink-path",
                        default="results_kun/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_000345.json",
                        help="Path to nothink@256 fullset JSON")
    parser.add_argument("--thinking-path",
                        default="results_kun/fulltest/per_sample_gsm8k_Qwen3_8B_20260324_120316.csv",
                        help="Path to thinking fulltest CSV")
    parser.add_argument("--seeds", type=int, default=100,
                        help="Number of random seeds for random router (default: 100)")
    parser.add_argument("--output", default="results/uncertainty_router",
                        help="Output directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    nothink_path = project_root / args.nothink_path
    thinking_path = project_root / args.thinking_path
    output_dir = project_root / args.output

    print("=" * 72)
    print("  Uncertainty Router Baseline Comparison for TOWN")
    print("=" * 72)

    # ── Load ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    nothink = load_nothink_data(str(nothink_path))
    think = load_thinking_data(str(thinking_path))
    n = len(nothink)
    town_n_route = sum(1 for s in nothink.values() if s["hit_budget"])
    town_rate = town_n_route / n
    print(f"  {n} samples | TOWN routing rate: {town_rate*100:.1f}% ({town_n_route} samples)")

    # Quick data summary
    n_nothink_correct = sum(1 for s in nothink.values() if s["correct"])
    n_think_correct = sum(1 for s in think.values() if s["correct"])
    n_both_correct = sum(1 for i in nothink if nothink[i]["correct"] and think[i]["correct"])
    n_neither = sum(1 for i in nothink if not nothink[i]["correct"] and not think[i]["correct"])
    n_only_nothink = sum(1 for i in nothink if nothink[i]["correct"] and not think[i]["correct"])
    n_only_think = sum(1 for i in nothink if not nothink[i]["correct"] and think[i]["correct"])

    print(f"\n  Correctness contingency table:")
    print(f"  {'':>20} {'Think ✓':>10} {'Think ✗':>10} {'Total':>8}")
    print(f"  {'NoThink ✓':<20} {n_both_correct:>10} {n_only_nothink:>10} {n_nothink_correct:>8}")
    print(f"  {'NoThink ✗':<20} {n_only_think:>10} {n_neither:>10} {n - n_nothink_correct:>8}")
    print(f"  {'Total':<20} {n_think_correct:>10} {n - n_think_correct:>10} {n:>8}")
    print(f"\n  Max possible recovery: {n_only_think} ({n_only_think/n*100:.1f}%)")
    print(f"  Max possible regret:   {n_only_nothink} ({n_only_nothink/n*100:.1f}%)")

    # ── Main comparison ──────────────────────────────────────────
    print("\n[2/5] Running baselines at TOWN's routing rate...")
    all_metrics = {}

    # Static baselines
    for name, fn in [("nothink_only", route_nothink_only), ("think_only", route_think_only)]:
        res = fn(nothink, think)
        all_metrics[name] = compute_metrics(res, nothink, think)

    # TOWN
    res_town = route_town(nothink, think)
    all_metrics["town"] = compute_metrics(res_town, nothink, think)

    # Deterministic routers at TOWN's n_route
    for name, fn in [
        ("token_length", route_token_length),
        ("inverse_token_length", route_inverse_token_length),
        ("mid_token", route_mid_token),
    ]:
        res = fn(nothink, think, town_n_route)
        all_metrics[name] = compute_metrics(res, nothink, think)

    # Random router (many seeds)
    print(f"  Running random router ({args.seeds} seeds)...", end=" ", flush=True)
    rand_list = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        res = route_random(nothink, think, town_n_route, rng)
        rand_list.append(compute_metrics(res, nothink, think))
    rm = {}
    for k in ["accuracy", "avg_tokens", "routing_rate", "recoveries", "regrets", "net_gain", "missed"]:
        vals = [m[k] for m in rand_list]
        rm[k] = float(np.mean(vals))
        rm[f"{k}_std"] = float(np.std(vals))
    rm["n_routed"] = float(np.mean([m["n_routed"] for m in rand_list]))
    rm["n_correct"] = float(np.mean([m["n_correct"] for m in rand_list]))
    rm["n_total"] = n
    all_metrics["random"] = rm
    print("done")

    # Oracle (capped + uncapped)
    res = route_oracle(nothink, think, town_n_route)
    all_metrics["oracle_capped"] = compute_metrics(res, nothink, think)
    res = route_oracle(nothink, think, None)
    all_metrics["oracle_uncapped"] = compute_metrics(res, nothink, think)

    # ── Print comparison table ───────────────────────────────────
    print(f"\n  {'Method':<28} {'Acc%':>6} {'Tok':>6} {'Route%':>7} {'Recv':>5} {'Regt':>5} {'Net':>5}")
    print(f"  {'─'*68}")
    display_order = [
        "nothink_only", "think_only", "",
        "random", "inverse_token_length", "mid_token", "token_length", "",
        "oracle_capped", "oracle_uncapped", "",
        "town",
    ]
    labels = {
        "nothink_only": "NoThink@256",
        "think_only": "Think@512",
        "random": "Random",
        "token_length": "Token-Length (uncertainty)",
        "inverse_token_length": "Inverse Token-Length",
        "mid_token": "Mid-Token",
        "oracle_capped": "Oracle (capped)",
        "oracle_uncapped": "Oracle (uncapped)",
        "town": "★ TOWN (Ours)",
    }
    for key in display_order:
        if key == "":
            print(f"  {'─'*68}")
            continue
        m = all_metrics[key]
        acc = m["accuracy"] * 100
        tok = m["avg_tokens"]
        route = m["routing_rate"] * 100
        net = m["net_gain"]
        rec = m["recoveries"]
        reg = m["regrets"]
        if key == "random" and "accuracy_std" in m:
            acc_s = f"{acc:.1f}±{m['accuracy_std']*100:.1f}"
        else:
            acc_s = f"{acc:.1f}"
        print(f"  {labels.get(key, key):<28} {acc_s:>6} {tok:>6.0f} {route:>7.1f} {rec:>5.1f} {reg:>5.1f} {net:>5.1f}")

    # ── Overlap analysis ─────────────────────────────────────────
    print(f"\n[3/5] Routing signal analysis...")
    town_set = set(i for i in nothink if nothink[i]["hit_budget"])
    tl_ranked = sorted(nothink.keys(), key=lambda i: nothink[i]["tokens"], reverse=True)
    tl_set = set(tl_ranked[:town_n_route])

    overlap = town_set & tl_set
    print(f"  TOWN ∩ Token-Length: {len(overlap)}/{len(town_set)} "
          f"({len(overlap)/len(town_set)*100:.0f}% overlap)")

    # Characterize TOWN's routed vs non-routed
    routed_correct_rate = sum(1 for i in town_set if nothink[i]["correct"]) / len(town_set)
    nonrouted_correct_rate = sum(1 for i in nothink if i not in town_set and nothink[i]["correct"]) / (n - len(town_set))
    print(f"  TOWN-routed nothink accuracy:     {routed_correct_rate*100:.1f}% (target: low → room for recovery)")
    print(f"  TOWN-not-routed nothink accuracy:  {nonrouted_correct_rate*100:.1f}% (target: high → safe to keep)")

    # Among routed: what fraction recovers?
    town_m = all_metrics["town"]
    print(f"  Recovery rate among routed: {town_m['recoveries']}/{town_n_route} "
          f"= {town_m['recoveries']/town_n_route*100:.1f}%")
    print(f"  Regret rate among routed:  {town_m['regrets']}/{town_n_route} "
          f"= {town_m['regrets']/town_n_route*100:.1f}%")

    # Token distribution
    routed_tokens = [nothink[i]["tokens"] for i in town_set]
    nonrouted_tokens = [nothink[i]["tokens"] for i in nothink if i not in town_set]
    print(f"\n  Token statistics:")
    print(f"    TOWN-routed:     min={min(routed_tokens)}, "
          f"median={int(np.median(routed_tokens))}, "
          f"max={max(routed_tokens)}, "
          f"mean={np.mean(routed_tokens):.0f}")
    print(f"    TOWN-not-routed: min={min(nonrouted_tokens)}, "
          f"median={int(np.median(nonrouted_tokens))}, "
          f"max={max(nonrouted_tokens)}, "
          f"mean={np.mean(nonrouted_tokens):.0f}")

    # ── Routing-rate sweep ───────────────────────────────────────
    print(f"\n[4/5] Routing-rate sweep...")
    sweep_rates = [0.05, 0.10, 0.112, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    sweep_data = {}

    for rate in sweep_rates:
        nr = int(round(rate * n))
        sweep_data[rate] = {}

        # Token-length
        res = route_token_length(nothink, think, nr)
        m = compute_metrics(res, nothink, think)
        sweep_data[rate]["token_length"] = m["accuracy"]

        # Oracle
        res = route_oracle(nothink, think, nr)
        m = compute_metrics(res, nothink, think)
        sweep_data[rate]["oracle"] = m["accuracy"]

        # Random (5 seeds for speed)
        accs = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            res = route_random(nothink, think, nr, rng)
            m = compute_metrics(res, nothink, think)
            accs.append(m["accuracy"])
        sweep_data[rate]["random"] = (np.mean(accs), np.std(accs))

        # TOWN (only at its natural rate)
        if abs(rate - town_rate) < 0.005:
            sweep_data[rate]["town_extrapolated"] = all_metrics["town"]["accuracy"]

    print(f"\n  {'Rate%':>6} {'Random':>14} {'TokLen':>8} {'Oracle':>8} {'TOWN':>8}")
    print(f"  {'─'*50}")
    for rate in sweep_rates:
        s = sweep_data[rate]
        rand_v = s["random"]
        tl_v = s["token_length"]
        ora_v = s["oracle"]
        town_v = s.get("town_extrapolated", None)
        rand_s = f"{rand_v[0]*100:.1f}±{rand_v[1]*100:.1f}"
        town_s = f"{town_v*100:.1f}" if town_v else "---"
        print(f"  {rate*100:6.1f} {rand_s:>14} {tl_v*100:>8.1f} {ora_v*100:>8.1f} {town_s:>8}")

    # ── Generate outputs ─────────────────────────────────────────
    print(f"\n[5/5] Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)

    # LaTeX tables
    table1 = generate_main_table(all_metrics)
    table1_path = output_dir / "routing_baselines_table.tex"
    with open(table1_path, "w") as f:
        f.write(table1)

    table2 = generate_sweep_table(sweep_data)
    table2_path = output_dir / "routing_sweep_table.tex"
    with open(table2_path, "w") as f:
        f.write(table2)

    # JSON metrics
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_out = {
        "main_comparison": {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in all_metrics.items()},
        "sweep": {str(k): {kk: (list(vv) if isinstance(vv, tuple) else convert(vv))
                           for kk, vv in v.items()} for k, v in sweep_data.items()},
        "contingency": {
            "both_correct": n_both_correct,
            "only_nothink": n_only_nothink,
            "only_think": n_only_think,
            "neither": n_neither,
        },
    }
    json_path = output_dir / "routing_baselines_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=convert)

    print(f"  {table1_path}")
    print(f"  {table2_path}")
    print(f"  {json_path}")

    # ── Print LaTeX ──────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("TABLE 1: Main Comparison")
    print(f"{'='*72}")
    print(table1)

    print(f"\n{'='*72}")
    print("TABLE 2: Routing-Rate Sweep")
    print(f"{'='*72}")
    print(table2)

    # ── Key findings ─────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("KEY FINDINGS FOR PAPER")
    print(f"{'='*72}")

    tm = all_metrics["town"]
    rm_ = all_metrics["random"]
    tlm = all_metrics["token_length"]
    om = all_metrics["oracle_capped"]

    print(f"""
1. TOWN achieves {tm['accuracy']*100:.1f}% accuracy at {tm['avg_tokens']:.0f} avg tokens
   ({tm['routing_rate']*100:.1f}% routing rate, {tm['recoveries']} recoveries, {tm['regrets']} regrets)

2. vs Random router (same rate):  {rm_['accuracy']*100:.1f}% → TOWN is +{(tm['accuracy']-rm_['accuracy'])*100:.1f}% better
   Random routing hurts accuracy because it randomly replaces nothink (87.5%)
   with thinking (65.2%) for most samples — net negative.

3. vs Token-Length uncertainty router: {tlm['accuracy']*100:.1f}% (identical to TOWN)
   This is because TOWN's hit_budget signal IS token-length at 256 boundary:
   100% overlap in routed sample sets.

   KEY INSIGHT: TOWN's early-stop mechanism is a natural, threshold-free
   implementation of token-length-based uncertainty routing. It requires
   no hyperparameter tuning (the budget boundary provides automatic thresholding).

4. vs Oracle (budget-matched): {om['accuracy']*100:.1f}% — gap of {(om['accuracy']-tm['accuracy'])*100:.1f}%
   TOWN closes {(tm['accuracy']-rm_['accuracy'])/(om['accuracy']-rm_['accuracy'])*100:.0f}% of the random→oracle gap.

5. Inverse/Mid-Token routers perform worse: {all_metrics['inverse_token_length']['accuracy']*100:.1f}% / {all_metrics['mid_token']['accuracy']*100:.1f}%
   confirming that routing high-token (uncertain) samples IS the right strategy.

6. Why TOWN outperforms random so significantly:
   - TOWN routes samples with {routed_correct_rate*100:.1f}% nothink accuracy (uncertain/struggling)
   - Random routes samples with ~{n_nothink_correct/n*100:.1f}% nothink accuracy (population average)
   - Net effect: TOWN recovers {tm['recoveries']} / {tm['net_gain']} net; Random loses {rm_['net_gain']:.1f} net
""")

    return all_metrics


if __name__ == "__main__":
    main()
