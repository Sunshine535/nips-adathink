#!/usr/bin/env python3
"""Analyze all Reasoning Speculation experiment results.

Reads JSON files from results/speculation/, produces:
  1. Summary table (accuracy, tokens, route distribution) for all configs
  2. Accuracy vs Tokens scatter plot (Pareto frontier)
  3. Route distribution bar chart
  4. K / probe_budget / threshold ablation tables
  5. Per-route accuracy breakdown
  6. Token efficiency analysis

Output: results/paper_assets/speculation_*.{png,csv,tex}

Usage:
  python scripts/analyze_speculation_results.py \
    --results_dir results/speculation \
    --output_dir results/paper_assets
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_results(results_dir: str) -> List[dict]:
    """Load all JSON result files."""
    files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not files:
        print(f"ERROR: No JSON files found in {results_dir}")
        sys.exit(1)

    all_data = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            data["_file"] = os.path.basename(f)
            all_data.append(data)
    print(f"Loaded {len(all_data)} result files from {results_dir}")
    return all_data


def get_config(data: dict) -> dict:
    """Extract config from either 'config' or 'meta' key."""
    return data.get("config", {}) or data.get("meta", {})


def extract_config_tag(data: dict) -> str:
    """Create a human-readable config tag."""
    cfg = get_config(data)
    k = cfg.get("k_paths", "?")
    probe = cfg.get("probe_budget", "?")
    med = cfg.get("medium_budget", "?")
    hard = cfg.get("hard_budget", "?")
    et = cfg.get("easy_threshold", 0.75)
    mt = cfg.get("medium_threshold", 0.5)
    n = cfg.get("n_samples", "?")
    seed = cfg.get("seed", "?")

    tag = f"K{k}_p{probe}_m{med}_h{hard}"
    if et != 0.75 or mt != 0.5:
        tag += f"_et{et}_mt{mt}"
    tag += f"_n{n}_s{seed}"
    return tag


# ──────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ──────────────────────────────────────────────────────────────────────────────

def build_summary_table(all_data: List[dict]) -> List[dict]:
    """Build summary table with all methods from all experiments."""
    rows = []

    for data in all_data:
        cfg = get_config(data)
        tag = extract_config_tag(data)
        n = cfg.get("n_samples", 0)

        # Reasoning Speculation main result
        spec = data.get("reasoning_speculation", {})
        if spec:
            # Compute route distribution from per_sample data
            per_sample = data.get("per_sample", [])
            from collections import Counter as _Counter
            route_counts = _Counter(s.get("route", "unknown") for s in per_sample)
            total_n = len(per_sample) or n or 1

            # Compute per-route accuracy
            route_accs_computed = {}
            for route in route_counts:
                route_samples = [s for s in per_sample if s.get("route") == route]
                if route_samples:
                    route_accs_computed[route] = sum(1 for s in route_samples if s.get("correct")) / len(route_samples)

            rows.append({
                "config": tag,
                "method": "ReasonSpec",
                "accuracy": spec.get("accuracy", 0),
                "avg_tokens": spec.get("avg_tokens", 0),
                "n_samples": total_n,
                "k_paths": cfg.get("k_paths"),
                "probe_budget": cfg.get("probe_budget"),
                "medium_budget": cfg.get("medium_budget"),
                "hard_budget": cfg.get("hard_budget"),
                "easy_threshold": cfg.get("easy_threshold", 0.75),
                "medium_threshold": cfg.get("medium_threshold", 0.5),
                "seed": cfg.get("seed"),
                "easy_pct": route_counts.get("easy", 0) / total_n,
                "medium_pct": route_counts.get("medium", 0) / total_n,
                "hard_pct": route_counts.get("hard", 0) / total_n,
                "easy_acc": route_accs_computed.get("easy"),
                "medium_acc": route_accs_computed.get("medium"),
                "hard_acc": route_accs_computed.get("hard"),
            })

        # Baselines
        baselines = data.get("baselines", {})
        for bname, bdata in baselines.items():
            rows.append({
                "config": tag,
                "method": bname,
                "accuracy": bdata.get("accuracy", 0),
                "avg_tokens": bdata.get("avg_tokens", 0),
                "n_samples": n,
                "k_paths": cfg.get("k_paths"),
                "probe_budget": cfg.get("probe_budget"),
                "medium_budget": cfg.get("medium_budget"),
                "hard_budget": cfg.get("hard_budget"),
                "easy_threshold": cfg.get("easy_threshold", 0.75),
                "medium_threshold": cfg.get("medium_threshold", 0.5),
                "seed": cfg.get("seed"),
            })

    return rows


def print_summary(rows: List[dict], output_dir: str):
    """Print and save summary table."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    # Group by config
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    lines = []
    for config, methods in sorted(by_config.items()):
        lines.append(f"\n{'─' * 80}")
        lines.append(f"Config: {config}")
        lines.append(f"{'─' * 80}")
        header = f"{'Method':<30} {'Accuracy':>10} {'Avg Tokens':>12} {'Δ vs Fixed-512':>15}"
        lines.append(header)
        lines.append("-" * 70)

        fixed512_acc = None
        for m in methods:
            if "fixed_512" in m["method"].lower():
                fixed512_acc = m["accuracy"]

        for m in sorted(methods, key=lambda x: -x["accuracy"]):
            delta = ""
            if fixed512_acc is not None and m["method"] != "fixed_512":
                d = m["accuracy"] - fixed512_acc
                delta = f"{d:+.1%}"
            acc_str = f"{m['accuracy']:.1%}" if m['accuracy'] else "N/A"
            tok_str = f"{m['avg_tokens']:.1f}" if m['avg_tokens'] else "N/A"
            line = f"{m['method']:<30} {acc_str:>10} {tok_str:>12} {delta:>15}"
            lines.append(line)

            # Route breakdown for ReasonSpec
            if m["method"] == "ReasonSpec" and m.get("easy_pct") is not None:
                route_line = (f"  → Routes: easy={m['easy_pct']:.0%}, "
                             f"med={m['medium_pct']:.0%}, hard={m['hard_pct']:.0%}")
                lines.append(route_line)
                if m.get("easy_acc") is not None:
                    acc_line = (f"  → Route acc: easy={m['easy_acc']:.1%}, "
                               f"med={m.get('medium_acc', 0):.1%}")
                    if m.get("hard_acc") is not None:
                        acc_line += f", hard={m['hard_acc']:.1%}"
                    lines.append(acc_line)

    text = "\n".join(lines)
    print(text)

    # Save
    out_path = os.path.join(output_dir, "speculation_summary.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nSaved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_tokens(rows: List[dict], output_dir: str):
    """Scatter plot: accuracy vs avg tokens for all methods."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Separate ReasonSpec and baselines
    spec_points = [(r["avg_tokens"], r["accuracy"], r["config"])
                   for r in rows if r["method"] == "ReasonSpec" and r["avg_tokens"]]
    baseline_points = defaultdict(list)
    for r in rows:
        if r["method"] != "ReasonSpec" and r["avg_tokens"]:
            baseline_points[r["method"]].append((r["avg_tokens"], r["accuracy"]))

    # Color map for baselines
    baseline_colors = {
        "fixed_128": "#2196F3",
        "fixed_256": "#4CAF50",
        "fixed_512": "#FF9800",
        "sc_4x128": "#9C27B0",
    }
    baseline_markers = {
        "fixed_128": "s",
        "fixed_256": "D",
        "fixed_512": "^",
        "sc_4x128": "v",
    }

    # Plot baselines
    for bname, points in baseline_points.items():
        tokens = [p[0] for p in points]
        accs = [p[1] for p in points]
        color = baseline_colors.get(bname, "#666666")
        marker = baseline_markers.get(bname, "o")
        ax.scatter(tokens, accs, c=color, marker=marker, s=80, alpha=0.7,
                  label=bname.replace("_", " ").title(), zorder=3, edgecolors="white", linewidth=0.5)

    # Plot ReasonSpec variants
    if spec_points:
        tokens = [p[0] for p in spec_points]
        accs = [p[1] for p in spec_points]
        labels = [p[2] for p in spec_points]
        scatter = ax.scatter(tokens, accs, c="#E91E63", marker="*", s=200, alpha=0.9,
                           label="ReasonSpec (ours)", zorder=5, edgecolors="black", linewidth=0.5)

        # Annotate ReasonSpec points
        for t, a, l in zip(tokens, accs, labels):
            # Extract short label
            short = l.split("_n")[0]  # remove n_samples and seed
            ax.annotate(short, (t, a), textcoords="offset points",
                       xytext=(8, 5), fontsize=7, alpha=0.8)

    ax.set_xlabel("Average Tokens per Sample")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reasoning Speculation: Accuracy vs Token Cost")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    out_path = os.path.join(output_dir, "speculation_accuracy_vs_tokens.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_route_distribution(rows: List[dict], output_dir: str):
    """Stacked bar chart of route distributions across configs."""
    spec_rows = [r for r in rows if r["method"] == "ReasonSpec"
                 and r.get("easy_pct") is not None]
    if not spec_rows:
        print("No ReasonSpec data for route distribution plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    configs = [r["config"].split("_n")[0] for r in spec_rows]
    easy = [r["easy_pct"] * 100 for r in spec_rows]
    medium = [r["medium_pct"] * 100 for r in spec_rows]
    hard = [r["hard_pct"] * 100 for r in spec_rows]

    x = np.arange(len(configs))
    width = 0.6

    ax.bar(x, easy, width, label="Easy (consensus vote)", color="#4CAF50", alpha=0.85)
    ax.bar(x, medium, width, bottom=easy, label="Medium (extend)", color="#FF9800", alpha=0.85)
    ax.bar(x, hard, width, bottom=[e+m for e, m in zip(easy, medium)],
           label="Hard (deliberation)", color="#F44336", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Route Distribution Across Configurations")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    out_path = os.path.join(output_dir, "speculation_route_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ablation_heatmap(rows: List[dict], output_dir: str):
    """Heatmap showing accuracy for different K × probe_budget combinations."""
    spec_rows = [r for r in rows if r["method"] == "ReasonSpec"
                 and r.get("k_paths") and r.get("probe_budget")]
    if len(spec_rows) < 2:
        print("Not enough data for ablation heatmap")
        return

    # Group by (k, probe)
    by_kp = {}
    for r in spec_rows:
        key = (r["k_paths"], r["probe_budget"])
        if key not in by_kp or r["n_samples"] > by_kp[key]["n_samples"]:
            by_kp[key] = r  # keep largest n_samples run

    ks = sorted(set(k for k, p in by_kp.keys()))
    probes = sorted(set(p for k, p in by_kp.keys()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy heatmap
    acc_matrix = np.full((len(ks), len(probes)), np.nan)
    tok_matrix = np.full((len(ks), len(probes)), np.nan)
    for (k, p), r in by_kp.items():
        ki = ks.index(k)
        pi = probes.index(p)
        acc_matrix[ki, pi] = r["accuracy"] * 100
        tok_matrix[ki, pi] = r["avg_tokens"]

    for idx, (matrix, title, fmt, cmap) in enumerate([
        (acc_matrix, "Accuracy (%)", ".1f", "RdYlGn"),
        (tok_matrix, "Avg Tokens", ".0f", "RdYlGn_r"),
    ]):
        ax = axes[idx]
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(probes)))
        ax.set_xticklabels(probes)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_xlabel("Probe Budget")
        ax.set_ylabel("K (paths)")
        ax.set_title(title)

        for i in range(len(ks)):
            for j in range(len(probes)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                           fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("K × Probe Budget Ablation", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "speculation_ablation_heatmap.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_latex_table(rows: List[dict], output_dir: str):
    """Generate LaTeX table for paper."""
    spec_rows = [r for r in rows if r["method"] == "ReasonSpec" and r["n_samples"] >= 200]
    baseline_rows = [r for r in rows if r["method"] != "ReasonSpec" and r["n_samples"] >= 200]

    # Deduplicate baselines (keep first occurrence)
    seen_baselines = set()
    unique_baselines = []
    for r in baseline_rows:
        if r["method"] not in seen_baselines:
            seen_baselines.add(r["method"])
            unique_baselines.append(r)
    baseline_rows = unique_baselines

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reasoning Speculation vs Baselines on GSM8K (Qwen3-8B, $n=200$).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & K & Probe & Accuracy (\%) & Avg Tokens & Easy (\%) & Hard (\%) \\",
        r"\midrule",
    ]

    # Baselines
    for r in sorted(baseline_rows, key=lambda x: x["avg_tokens"]):
        lines.append(
            f"{r['method'].replace('_', '-')} & -- & -- & "
            f"{r['accuracy']*100:.1f} & {r['avg_tokens']:.0f} & -- & -- \\\\"
        )

    lines.append(r"\midrule")

    # ReasonSpec variants
    best_acc = max((r["accuracy"] for r in spec_rows), default=0)
    for r in sorted(spec_rows, key=lambda x: x["accuracy"], reverse=True):
        bold = r["accuracy"] == best_acc
        acc_str = f"\\textbf{{{r['accuracy']*100:.1f}}}" if bold else f"{r['accuracy']*100:.1f}"
        easy_str = f"{r.get('easy_pct', 0)*100:.0f}" if r.get("easy_pct") is not None else "--"
        hard_str = f"{r.get('hard_pct', 0)*100:.0f}" if r.get("hard_pct") is not None else "--"
        lines.append(
            f"RS(K={r['k_paths']},p={r['probe_budget']}) & "
            f"{r['k_paths']} & {r['probe_budget']} & "
            f"{acc_str} & {r['avg_tokens']:.0f} & {easy_str} & {hard_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)

    out_path = os.path.join(output_dir, "speculation_main_table.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"Saved: {out_path}")
    print(tex)


def compute_token_efficiency(rows: List[dict], output_dir: str):
    """Compute token efficiency metrics."""
    print("\n" + "=" * 80)
    print("TOKEN EFFICIENCY ANALYSIS")
    print("=" * 80)

    # For each config, compare ReasonSpec to Fixed baselines
    by_config = defaultdict(dict)
    for r in rows:
        by_config[r["config"]][r["method"]] = r

    lines = []
    for config, methods in sorted(by_config.items()):
        if "ReasonSpec" not in methods:
            continue
        rs = methods["ReasonSpec"]
        lines.append(f"\nConfig: {config}")
        lines.append(f"  ReasonSpec: acc={rs['accuracy']:.1%}, tokens={rs['avg_tokens']:.0f}")

        # Compare to each fixed baseline
        for bname in ["fixed_128", "fixed_256", "fixed_512"]:
            if bname in methods:
                bl = methods[bname]
                acc_delta = rs["accuracy"] - bl["accuracy"]
                tok_delta = rs["avg_tokens"] - bl["avg_tokens"]
                tok_ratio = rs["avg_tokens"] / bl["avg_tokens"] if bl["avg_tokens"] > 0 else float("inf")
                lines.append(
                    f"  vs {bname}: Δacc={acc_delta:+.1%}, "
                    f"Δtok={tok_delta:+.0f} ({tok_ratio:.1f}×)"
                )

        # Compare to SC baseline
        for bname in methods:
            if "sc_" in bname:
                sc = methods[bname]
                acc_delta = rs["accuracy"] - sc["accuracy"]
                tok_delta = rs["avg_tokens"] - sc["avg_tokens"]
                lines.append(
                    f"  vs {bname}: Δacc={acc_delta:+.1%}, Δtok={tok_delta:+.0f}"
                )

    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(output_dir, "speculation_token_efficiency.txt")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nSaved to {out_path}")


def identify_best_config(rows: List[dict]):
    """Identify the best ReasonSpec configuration."""
    spec_rows = [r for r in rows if r["method"] == "ReasonSpec" and r["n_samples"] >= 100]
    if not spec_rows:
        print("\nNo ReasonSpec results with n>=100 found")
        return None

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION ANALYSIS")
    print("=" * 80)

    # Rank by accuracy
    ranked = sorted(spec_rows, key=lambda x: -x["accuracy"])
    print("\nRanked by accuracy:")
    for i, r in enumerate(ranked[:10]):
        print(f"  #{i+1}: {r['config']}")
        print(f"       acc={r['accuracy']:.1%}, tokens={r['avg_tokens']:.0f}, "
              f"easy={r.get('easy_pct', 0):.0%}, hard={r.get('hard_pct', 0):.0%}")

    # Find Pareto-optimal: best accuracy at each token budget level
    print("\nPareto frontier (accuracy vs tokens):")
    pareto = []
    for r in sorted(spec_rows + [r2 for r2 in rows if r2["method"] != "ReasonSpec"],
                    key=lambda x: x["avg_tokens"]):
        if not pareto or r["accuracy"] > pareto[-1]["accuracy"]:
            pareto.append(r)
            print(f"  {r['method']}: acc={r['accuracy']:.1%}, tokens={r['avg_tokens']:.0f}")

    return ranked[0] if ranked else None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Reasoning Speculation results")
    parser.add_argument("--results_dir", default="results/speculation",
                       help="Directory with JSON result files")
    parser.add_argument("--output_dir", default="results/paper_assets",
                       help="Directory for output figures and tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load
    all_data = load_all_results(args.results_dir)

    # Build summary
    rows = build_summary_table(all_data)
    if not rows:
        print("No results to analyze!")
        return

    # Analysis
    print_summary(rows, args.output_dir)
    compute_token_efficiency(rows, args.output_dir)
    best = identify_best_config(rows)

    # Plots
    plot_accuracy_vs_tokens(rows, args.output_dir)
    plot_route_distribution(rows, args.output_dir)
    plot_ablation_heatmap(rows, args.output_dir)

    # LaTeX
    generate_latex_table(rows, args.output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Outputs in: {args.output_dir}")
    print("=" * 80)

    if best:
        print(f"\n🏆 BEST CONFIG: {best['config']}")
        print(f"   acc={best['accuracy']:.1%}, tokens={best['avg_tokens']:.0f}")


if __name__ == "__main__":
    main()
