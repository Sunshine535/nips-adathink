#!/usr/bin/env python3
"""Visualize entropy dynamics from collect_entropy_dynamics.py output.

Generates figures for the paper:
  1. Entropy traces for easy/medium/hard problems (main figure)
  2. Entropy drop distribution: correct vs incorrect
  3. P(</think>) dynamics showing readiness signal
  4. Token savings potential at different entropy thresholds

Usage:
    python scripts/plot_entropy_dynamics.py \
        --input results/entropy_dynamics/entropy_dynamics_Qwen3_8B_b512_n200_*.json \
        --output_dir results/paper_figures/entropy/
"""

import argparse
import json
import glob
import logging
import os
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Defer matplotlib import to allow headless servers
def get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_results(input_path: str) -> Dict:
    """Load entropy dynamics results, supporting glob patterns."""
    if "*" in input_path:
        files = sorted(glob.glob(input_path))
        if not files:
            raise FileNotFoundError(f"No files matching {input_path}")
        input_path = files[-1]  # Use most recent
        log.info(f"Using: {input_path}")
    with open(input_path, "r") as f:
        return json.load(f)


def classify_difficulty(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Classify samples into easy/medium/hard based on tokens consumed."""
    valid = [r for r in results if "error" not in r and r.get("n_tokens", 0) > 0]
    if not valid:
        return {"easy": [], "medium": [], "hard": []}

    tokens = [r["n_tokens"] for r in valid]
    p33 = np.percentile(tokens, 33)
    p66 = np.percentile(tokens, 66)

    groups = {"easy": [], "medium": [], "hard": []}
    for r in valid:
        t = r["n_tokens"]
        if t <= p33:
            groups["easy"].append(r)
        elif t <= p66:
            groups["medium"].append(r)
        else:
            groups["hard"].append(r)
    return groups


def plot_entropy_traces(groups: Dict[str, List[Dict]], output_dir: str, budget: int):
    """Figure 1: Entropy traces for easy/medium/hard problems.

    Shows the per-token entropy H_t over time, with shaded bands for each
    difficulty group. Highlights the "saturation point" where entropy drops.
    """
    plt = get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728"}
    titles = {"easy": "Easy (bottom 33%)", "medium": "Medium", "hard": "Hard (top 33%)"}

    for ax, (group_name, samples) in zip(axes, groups.items()):
        traces = [s.get("entropy_trace", []) for s in samples if s.get("entropy_trace")]
        if not traces:
            ax.set_title(titles[group_name])
            continue

        # Plot individual traces (light)
        for trace in traces[:20]:  # Limit to 20 for readability
            ax.plot(range(len(trace)), trace, alpha=0.15, color=colors[group_name], linewidth=0.5)

        # Compute mean trace (pad shorter traces with NaN)
        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t

        mean_trace = np.nanmean(padded, axis=0)
        std_trace = np.nanstd(padded, axis=0)

        x = np.arange(max_len)
        ax.plot(x, mean_trace, color=colors[group_name], linewidth=2, label="Mean")
        ax.fill_between(x, mean_trace - std_trace, mean_trace + std_trace,
                       alpha=0.2, color=colors[group_name])

        # Mark think_end positions
        think_ends = [s.get("think_end_position") for s in samples
                     if s.get("think_end_position") is not None]
        if think_ends:
            mean_end = np.mean(think_ends)
            ax.axvline(x=mean_end, color="gray", linestyle="--", alpha=0.7,
                      label=f"Avg </think> @ {mean_end:.0f}")

        ax.set_title(titles[group_name], fontsize=12)
        ax.set_xlabel("Token position")
        if ax == axes[0]:
            ax.set_ylabel("Next-token entropy (nats)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, budget)

    fig.suptitle(f"Per-token Entropy Dynamics During Thinking (budget={budget})",
                fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_entropy_traces_by_difficulty.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_entropy_drop_distribution(results: List[Dict], output_dir: str):
    """Figure 2: Entropy drop ratio distribution — correct vs incorrect."""
    plt = get_plt()
    valid = [r for r in results if "stats" in r and "error" not in r]
    correct = [r["stats"]["entropy_drop_ratio"] for r in valid if r["correct"] == 1]
    incorrect = [r["stats"]["entropy_drop_ratio"] for r in valid if r["correct"] == 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.5, 1.0, 30)
    ax.hist(correct, bins=bins, alpha=0.6, color="#2ca02c", label=f"Correct (n={len(correct)})", density=True)
    ax.hist(incorrect, bins=bins, alpha=0.6, color="#d62728", label=f"Incorrect (n={len(incorrect)})", density=True)

    ax.set_xlabel("Entropy Drop Ratio (early → late thinking)")
    ax.set_ylabel("Density")
    ax.set_title("Entropy Drop Distribution: Correct vs Incorrect", fontweight="bold")
    ax.legend()
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # Add effect size annotation
    if correct and incorrect:
        mean_c = np.mean(correct)
        mean_i = np.mean(incorrect)
        pooled_std = np.sqrt((np.var(correct) + np.var(incorrect)) / 2)
        cohens_d = (mean_c - mean_i) / pooled_std if pooled_std > 0 else 0
        ax.annotate(f"Cohen's d = {cohens_d:.2f}\n"
                   f"Correct mean = {mean_c:.3f}\n"
                   f"Incorrect mean = {mean_i:.3f}",
                   xy=(0.98, 0.98), xycoords="axes fraction",
                   ha="right", va="top", fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_entropy_drop_correct_vs_incorrect.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_think_end_probability(results: List[Dict], output_dir: str, budget: int):
    """Figure 3: P(</think>) dynamics — model's readiness signal."""
    plt = get_plt()
    valid = [r for r in results if r.get("think_end_prob_trace") and "error" not in r]

    # Separate natural stop vs truncated
    ns_traces = [r["think_end_prob_trace"] for r in valid if r.get("natural_stop")]
    trunc_traces = [r["think_end_prob_trace"] for r in valid if not r.get("natural_stop")]

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, traces, color in [
        (f"Natural stop (n={len(ns_traces)})", ns_traces, "#2ca02c"),
        (f"Truncated (n={len(trunc_traces)})", trunc_traces, "#d62728"),
    ]:
        if not traces:
            continue
        max_len = max(len(t) for t in traces)
        padded = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            padded[i, :len(t)] = t
        mean_trace = np.nanmean(padded, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean_trace, color=color, linewidth=2, label=label)
        std_trace = np.nanstd(padded, axis=0)
        ax.fill_between(x, np.maximum(mean_trace - std_trace, 0),
                       mean_trace + std_trace, alpha=0.15, color=color)

    ax.set_xlabel("Token position")
    ax.set_ylabel("P(</think>)")
    ax.set_title("Model's Readiness-to-Answer Signal", fontweight="bold")
    ax.legend()
    ax.set_xlim(0, budget)
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_think_end_probability.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_savings_frontier(results: List[Dict], output_dir: str):
    """Figure 4: Accuracy vs token savings at different entropy thresholds.

    Shows a Pareto frontier: for each tau_h threshold, what fraction of
    tokens are saved vs what accuracy would result if we stopped early.
    """
    plt = get_plt()
    valid = [r for r in results if r.get("entropy_trace") and "error" not in r
             and r.get("think_end_position") is not None]

    if not valid:
        log.warning("Not enough valid samples for savings frontier plot")
        return

    thresholds = np.arange(0.5, 5.0, 0.25)
    results_per_thresh = []

    for tau in thresholds:
        # For each sample, find the first token position where chunk entropy < tau
        chunk_size = 32
        stopped_early = 0
        total_savings = 0
        total_tokens = 0
        correct_at_stop = 0
        n_valid = 0

        for r in valid:
            trace = r["entropy_trace"]
            n_tok = len(trace)
            total_tokens += n_tok
            n_valid += 1

            # Find stop point
            stop_pos = n_tok  # Default: no early stop
            for c_start in range(0, n_tok, chunk_size):
                c_end = min(c_start + chunk_size, n_tok)
                chunk_mean = np.mean(trace[c_start:c_end])
                if chunk_mean < tau and c_start >= chunk_size * 2:  # min 2 chunks
                    stop_pos = c_end
                    break

            if stop_pos < n_tok:
                stopped_early += 1
                total_savings += (n_tok - stop_pos)

            # Estimate accuracy: if we stop early, do we have enough info?
            # Heuristic: if we stop after think_end_position, we have the answer
            think_end = r.get("think_end_position", n_tok + 1)
            if stop_pos >= think_end:
                correct_at_stop += r["correct"]
            else:
                # Stopped before think_end — assume same as truncated accuracy
                correct_at_stop += r["correct"] * 0.45  # Estimated truncated accuracy

        if n_valid > 0:
            results_per_thresh.append({
                "tau": float(tau),
                "savings_pct": total_savings / total_tokens * 100 if total_tokens > 0 else 0,
                "stop_rate": stopped_early / n_valid * 100,
                "est_accuracy": correct_at_stop / n_valid * 100,
            })

    if not results_per_thresh:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    taus = [r["tau"] for r in results_per_thresh]
    savings = [r["savings_pct"] for r in results_per_thresh]
    accs = [r["est_accuracy"] for r in results_per_thresh]

    ax1.plot(taus, savings, "b-o", linewidth=2, markersize=4, label="Token savings (%)")
    ax2.plot(taus, accs, "r-s", linewidth=2, markersize=4, label="Est. accuracy (%)")

    ax1.set_xlabel("Entropy threshold (tau_H)", fontsize=11)
    ax1.set_ylabel("Token savings (%)", color="blue", fontsize=11)
    ax2.set_ylabel("Estimated accuracy (%)", color="red", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")

    ax1.set_title("IRIS Savings Frontier: Accuracy vs Token Efficiency", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    path = os.path.join(output_dir, "fig_iris_savings_frontier.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_combined_go_nogo(data: Dict, output_dir: str):
    """Combined GO/NO-GO summary figure for the pilot report."""
    plt = get_plt()
    go_no_go = data.get("go_no_go", {})
    results = data.get("results", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Entropy trace (average correct vs incorrect)
    ax = axes[0, 0]
    correct_traces = [r["entropy_trace"] for r in results
                     if r.get("entropy_trace") and r["correct"] == 1]
    incorrect_traces = [r["entropy_trace"] for r in results
                       if r.get("entropy_trace") and r["correct"] == 0]

    for label, traces, color in [
        (f"Correct (n={len(correct_traces)})", correct_traces, "#2ca02c"),
        (f"Incorrect (n={len(incorrect_traces)})", incorrect_traces, "#d62728"),
    ]:
        if traces:
            max_len = max(len(t) for t in traces)
            padded = np.full((len(traces), max_len), np.nan)
            for i, t in enumerate(traces):
                padded[i, :len(t)] = t
            mean = np.nanmean(padded, axis=0)
            ax.plot(range(len(mean)), mean, color=color, linewidth=2, label=label)

    ax.set_title("Mean Entropy: Correct vs Incorrect", fontweight="bold")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Entropy (nats)")
    ax.legend()

    # Panel 2: Entropy drop distribution
    ax = axes[0, 1]
    drops_c = [r["stats"]["entropy_drop_ratio"] for r in results
               if "stats" in r and r["correct"] == 1 and "error" not in r]
    drops_i = [r["stats"]["entropy_drop_ratio"] for r in results
               if "stats" in r and r["correct"] == 0 and "error" not in r]
    if drops_c or drops_i:
        bins = np.linspace(-0.5, 1.0, 25)
        if drops_c:
            ax.hist(drops_c, bins=bins, alpha=0.6, color="#2ca02c", label="Correct", density=True)
        if drops_i:
            ax.hist(drops_i, bins=bins, alpha=0.6, color="#d62728", label="Incorrect", density=True)
    ax.set_title("Entropy Drop Ratio", fontweight="bold")
    ax.set_xlabel("Drop ratio")
    ax.legend()

    # Panel 3: Late thinking entropy
    ax = axes[1, 0]
    late_c = [r["stats"]["late_thinking_entropy"] for r in results
              if "stats" in r and r["correct"] == 1 and "error" not in r]
    late_i = [r["stats"]["late_thinking_entropy"] for r in results
              if "stats" in r and r["correct"] == 0 and "error" not in r]
    if late_c or late_i:
        bins = np.linspace(0, 8, 25)
        if late_c:
            ax.hist(late_c, bins=bins, alpha=0.6, color="#2ca02c", label="Correct", density=True)
        if late_i:
            ax.hist(late_i, bins=bins, alpha=0.6, color="#d62728", label="Incorrect", density=True)
    ax.set_title("Late Thinking Entropy (last chunk)", fontweight="bold")
    ax.set_xlabel("Entropy (nats)")
    ax.legend()

    # Panel 4: GO/NO-GO summary box
    ax = axes[1, 1]
    ax.axis("off")
    decision = go_no_go.get("decision", "UNKNOWN")
    color = "#2ca02c" if decision == "GO" else "#d62728"

    text_lines = [
        f"DECISION: {decision}",
        "",
        f"Accuracy: {go_no_go.get('accuracy', 0):.1%}",
        f"Natural stop rate: {go_no_go.get('natural_stop_rate', 0):.1%}",
        "",
        "Entropy drop (early → late):",
        f"  Overall: {go_no_go.get('entropy_drop', {}).get('overall_mean', 0):.3f}",
        f"  Correct: {go_no_go.get('entropy_drop', {}).get('correct_mean', 0):.3f}",
        f"  Incorrect: {go_no_go.get('entropy_drop', {}).get('incorrect_mean', 0):.3f}",
        "",
        "Late thinking entropy:",
        f"  Correct: {go_no_go.get('late_thinking_entropy', {}).get('correct_mean', 0):.4f}",
        f"  Incorrect: {go_no_go.get('late_thinking_entropy', {}).get('incorrect_mean', 0):.4f}",
    ]

    ax.text(0.5, 0.95, "\n".join(text_lines), transform=ax.transAxes,
           fontsize=11, verticalalignment="top", horizontalalignment="center",
           fontfamily="monospace",
           bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.15))

    fig.suptitle("IRIS GO/NO-GO Pilot Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_go_nogo_summary.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize entropy dynamics for IRIS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to entropy_dynamics JSON (supports glob)")
    parser.add_argument("--output_dir", type=str, default="results/paper_figures/entropy")
    parser.add_argument("--budget", type=int, default=512,
                        help="Token budget used in the experiment")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    data = load_results(args.input)
    results = data.get("results", [])
    log.info(f"Loaded {len(results)} samples")

    # Classify difficulty
    groups = classify_difficulty(results)
    for k, v in groups.items():
        log.info(f"  {k}: {len(v)} samples")

    # Generate all figures
    plot_entropy_traces(groups, args.output_dir, args.budget)
    plot_entropy_drop_distribution(results, args.output_dir)
    plot_think_end_probability(results, args.output_dir, args.budget)
    plot_savings_frontier(results, args.output_dir)
    plot_combined_go_nogo(data, args.output_dir)

    log.info(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
