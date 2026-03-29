#!/usr/bin/env python3
"""
Crossover analysis: identify the token budget where thinking surpasses nothink.

Reads JSON result files from a directory (default: results/crossover/),
separates nothink vs thinking results, computes per-budget accuracy,
identifies the crossover point, and generates a publication-quality figure
plus a LaTeX table.

Usage:
    # From JSON files
    python scripts/analyze_crossover.py --input-dir results/crossover/

    # With manual data points (format: mode@budget=accuracy, optionally :tokens)
    python scripts/analyze_crossover.py \
        --manual-data \
            nothink@128=85.0 nothink@256=88.5 nothink@512=90.2 \
            nothink@1024=91.0 nothink@2048=91.1 nothink@4096=91.2 \
            thinking@128=3.0 thinking@256=18.0 thinking@512=18.3 \
            thinking@1024=72.5 thinking@2048=89.0 thinking@4096=93.8

    # Mix: JSON dir + manual overrides
    python scripts/analyze_crossover.py \
        --input-dir results/crossover/ \
        --manual-data thinking@128=3.0:50 thinking@256=18.0:120
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
BUDGETS = [128, 256, 512, 1024, 2048, 4096]
MODES = ["nothink", "thinking"]

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def read_json(path: Path) -> dict:
    """Read a single JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def extract_fields(data: dict) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[float], Optional[int]]:
    """
    Extract (mode, budget, accuracy, avg_tokens, n_samples) from a JSON dict,
    handling alternate key names.

    Returns (mode, budget, accuracy, avg_tokens, n_samples).
    Any field may be None if not found.
    """
    # --- mode ---
    mode = data.get("mode", None)
    if mode is None:
        enable = data.get("enable_thinking", None)
        if enable is not None:
            mode = "thinking" if enable else "nothink"

    # --- budget ---
    budget = data.get("budget", None)
    if budget is None:
        budget = data.get("max_new_tokens", None)
    if budget is not None:
        budget = int(budget)

    # --- accuracy ---
    accuracy = data.get("accuracy", None)
    if accuracy is not None:
        accuracy = float(accuracy)

    # --- avg_tokens ---
    avg_tokens = data.get("avg_tokens", None)
    if avg_tokens is None:
        avg_tokens = data.get("avg_output_tokens", None)
    if avg_tokens is not None:
        avg_tokens = float(avg_tokens)

    # --- n_samples ---
    n_samples = data.get("n_samples", None)
    if n_samples is None:
        n_samples = data.get("total_samples", None)
    if n_samples is not None:
        n_samples = int(n_samples)

    return mode, budget, accuracy, avg_tokens, n_samples


def infer_mode_from_filename(fname: str) -> Optional[str]:
    """Try to infer mode from filename patterns."""
    fname_lower = fname.lower()
    if "nothink" in fname_lower or "no_think" in fname_lower or "no-think" in fname_lower:
        return "nothink"
    if "thinking" in fname_lower or "think" in fname_lower:
        return "thinking"
    return None


def infer_budget_from_filename(fname: str) -> Optional[int]:
    """Try to infer budget from filename patterns like 'budget_512' or 'b512'."""
    import re
    # Try patterns: budget_512, budget512, b_512, b512, _512_, tokens_512
    patterns = [
        r"budget[_-]?(\d+)",
        r"b[_-]?(\d+)",
        r"tokens[_-]?(\d+)",
        r"_(\d+)_",
        r"_(\d+)\.",
    ]
    for pat in patterns:
        m = re.search(pat, fname)
        if m:
            val = int(m.group(1))
            if val in BUDGETS:
                return val
    return None


def load_from_directory(input_dir: Path) -> Dict[str, Dict[int, dict]]:
    """
    Scan a directory for JSON files and aggregate results.

    Returns:
        {mode: {budget: {"accuracy": float, "avg_tokens": float, "n_samples": int}}}
    """
    results: Dict[str, Dict[int, dict]] = {"nothink": {}, "thinking": {}}
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        log.warning(f"No JSON files found in {input_dir}")
        return results

    log.info(f"Found {len(json_files)} JSON files in {input_dir}")

    for jf in json_files:
        try:
            data = read_json(jf)
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Skipping {jf.name}: {e}")
            continue

        mode, budget, accuracy, avg_tokens, n_samples = extract_fields(data)

        # Fall back to filename inference
        if mode is None:
            mode = infer_mode_from_filename(jf.name)
        if budget is None:
            budget = infer_budget_from_filename(jf.name)

        if mode is None:
            log.warning(f"Cannot determine mode for {jf.name}, skipping")
            continue
        if budget is None:
            log.warning(f"Cannot determine budget for {jf.name}, skipping")
            continue
        if accuracy is None:
            log.warning(f"No accuracy in {jf.name}, skipping")
            continue

        if mode not in results:
            log.warning(f"Unknown mode '{mode}' in {jf.name}, skipping")
            continue

        entry = {"accuracy": accuracy}
        if avg_tokens is not None:
            entry["avg_tokens"] = avg_tokens
        if n_samples is not None:
            entry["n_samples"] = n_samples

        # If duplicate budget, keep the one with more samples (or later file)
        if budget in results[mode]:
            existing = results[mode][budget]
            existing_n = existing.get("n_samples", 0)
            new_n = entry.get("n_samples", 0)
            if new_n > existing_n:
                results[mode][budget] = entry
                log.info(f"  Updated {mode}@{budget} from {jf.name} (n={new_n})")
            else:
                log.info(f"  Kept existing {mode}@{budget}, skipped {jf.name}")
        else:
            results[mode][budget] = entry
            log.info(f"  Loaded {mode}@{budget}: acc={accuracy:.4f} from {jf.name}")

    return results


def parse_manual_data(tokens: List[str]) -> Dict[str, Dict[int, dict]]:
    """
    Parse manual data points.

    Format: mode@budget=accuracy[:avg_tokens]
    Examples:
        thinking@128=3.0
        nothink@512=90.2:350
    """
    results: Dict[str, Dict[int, dict]] = {"nothink": {}, "thinking": {}}

    for tok in tokens:
        try:
            left, right = tok.split("=")
            mode, budget_str = left.split("@")
            budget = int(budget_str)
            mode = mode.strip().lower()

            # Parse accuracy and optional avg_tokens
            if ":" in right:
                acc_str, tok_str = right.split(":")
                accuracy = float(acc_str) / 100.0  # Interpret as percentage
                avg_tokens = float(tok_str)
            else:
                accuracy = float(right) / 100.0
                avg_tokens = None

            if mode not in results:
                log.warning(f"Unknown mode '{mode}' in manual data '{tok}', skipping")
                continue

            entry = {"accuracy": accuracy}
            if avg_tokens is not None:
                entry["avg_tokens"] = avg_tokens

            results[mode][budget] = entry
            log.info(f"  Manual: {mode}@{budget} = {accuracy:.4f}")

        except (ValueError, KeyError) as e:
            log.error(f"Failed to parse manual data '{tok}': {e}")
            continue

    return results


def merge_results(
    base: Dict[str, Dict[int, dict]],
    override: Dict[str, Dict[int, dict]],
) -> Dict[str, Dict[int, dict]]:
    """Merge two result dicts; override takes precedence."""
    merged: Dict[str, Dict[int, dict]] = {}
    for mode in MODES:
        merged[mode] = {}
        if mode in base:
            merged[mode].update(base[mode])
        if mode in override:
            for budget, entry in override[mode].items():
                merged[mode][budget] = entry
    return merged


def find_crossover(
    results: Dict[str, Dict[int, dict]],
    budgets: List[int],
) -> Optional[int]:
    """
    Find the lowest budget where thinking accuracy >= nothink accuracy.

    Both modes must have data at that budget.
    """
    for b in sorted(budgets):
        if b in results["thinking"] and b in results["nothink"]:
            think_acc = results["thinking"][b]["accuracy"]
            nothink_acc = results["nothink"][b]["accuracy"]
            if think_acc >= nothink_acc:
                return b
    return None


# ──────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────

def generate_figure(
    results: Dict[str, Dict[int, dict]],
    crossover_budget: Optional[int],
    output_path: Path,
    annotate_budgets: Optional[List[int]] = None,
) -> None:
    """Generate publication-quality crossover figure."""
    # --- Publication style ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.usetex": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Collect budgets that have data for at least one mode
    all_budgets = sorted(
        set(results["nothink"].keys()) | set(results["thinking"].keys())
    )

    for mode, color, marker, ls in [
        ("nothink", "#1f77b4", "s", "-"),   # blue, square
        ("thinking", "#ff7f0e", "o", "-"),   # orange, circle
    ]:
        budgets_m = sorted(results[mode].keys())
        accs = [results[mode][b]["accuracy"] * 100 for b in budgets_m]
        label = "NoThink" if mode == "nothink" else "Thinking"
        ax.plot(
            budgets_m, accs,
            color=color, marker=marker, markersize=7,
            linewidth=2.0, linestyle=ls, label=label,
            zorder=3,
        )

    # --- Crossover vertical line ---
    if crossover_budget is not None:
        ax.axvline(
            x=crossover_budget, color="#2ca02c", linestyle="--",
            linewidth=1.5, alpha=0.8, zorder=2,
            label=f"Crossover ({crossover_budget} tokens)",
        )

    # --- Annotate gap at selected budgets ---
    if annotate_budgets is None:
        # Auto-select: first, crossover, last
        annotate_budgets = []
        if all_budgets:
            annotate_budgets.append(all_budgets[0])
            if crossover_budget and crossover_budget not in annotate_budgets:
                annotate_budgets.append(crossover_budget)
            if all_budgets[-1] not in annotate_budgets:
                annotate_budgets.append(all_budgets[-1])

    for b in annotate_budgets:
        if b in results["nothink"] and b in results["thinking"]:
            nt_acc = results["nothink"][b]["accuracy"] * 100
            th_acc = results["thinking"][b]["accuracy"] * 100
            gap = th_acc - nt_acc
            mid_y = (nt_acc + th_acc) / 2
            sign = "+" if gap >= 0 else ""
            ax.annotate(
                f"{sign}{gap:.1f}%",
                xy=(b, mid_y),
                xytext=(12, 0),
                textcoords="offset points",
                fontsize=9,
                color="#555555",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8),
            )

    # --- Axes ---
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xticks(all_budgets if all_budgets else BUDGETS)

    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Accuracy (%)")

    # Y-axis: nice range
    all_accs = []
    for mode in MODES:
        for b, entry in results[mode].items():
            all_accs.append(entry["accuracy"] * 100)
    if all_accs:
        y_min = max(0, min(all_accs) - 5)
        y_max = min(100, max(all_accs) + 5)
        ax.set_ylim(y_min, y_max)

    # Grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2)

    # Legend
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#cccccc")

    # No title (for paper)
    fig.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    log.info(f"Figure saved to {output_path}")

    # Also save PNG for quick preview
    png_path = output_path.with_suffix(".png")
    fig.savefig(str(png_path), format="png", bbox_inches="tight")
    log.info(f"Preview saved to {png_path}")

    plt.close(fig)


# ──────────────────────────────────────────────────────────
# LaTeX table
# ──────────────────────────────────────────────────────────

def generate_latex_table(
    results: Dict[str, Dict[int, dict]],
    crossover_budget: Optional[int],
) -> str:
    """Generate a LaTeX table for the paper."""
    all_budgets = sorted(
        set(results["nothink"].keys()) | set(results["thinking"].keys())
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Crossover analysis: NoThink vs.\ Thinking accuracy across token budgets on GSM8K.}")
    lines.append(r"\label{tab:crossover}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r cc cc c}")
    lines.append(r"\toprule")
    lines.append(r"Budget & \multicolumn{2}{c}{NoThink} & \multicolumn{2}{c}{Thinking} & Gap \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r" & Acc (\%) & Tokens & Acc (\%) & Tokens & ($\Delta$\%) \\")
    lines.append(r"\midrule")

    for b in all_budgets:
        nt = results["nothink"].get(b, {})
        th = results["thinking"].get(b, {})

        nt_acc_str = f"{nt['accuracy'] * 100:.1f}" if "accuracy" in nt else "--"
        nt_tok_str = f"{nt['avg_tokens']:.0f}" if "avg_tokens" in nt else "--"
        th_acc_str = f"{th['accuracy'] * 100:.1f}" if "accuracy" in th else "--"
        th_tok_str = f"{th['avg_tokens']:.0f}" if "avg_tokens" in th else "--"

        if "accuracy" in nt and "accuracy" in th:
            gap = (th["accuracy"] - nt["accuracy"]) * 100
            sign = "+" if gap >= 0 else ""
            gap_str = f"{sign}{gap:.1f}"
        else:
            gap_str = "--"

        # Bold the crossover row
        if b == crossover_budget:
            row = rf"\textbf{{{b}}} & \textbf{{{nt_acc_str}}} & {nt_tok_str} & \textbf{{{th_acc_str}}} & {th_tok_str} & \textbf{{{gap_str}}} \\"
        else:
            row = rf"{b} & {nt_acc_str} & {nt_tok_str} & {th_acc_str} & {th_tok_str} & {gap_str} \\"

        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if crossover_budget is not None:
        lines.append(rf"\vspace{{2pt}}")
        lines.append(rf"\raggedright\footnotesize\textit{{Crossover at {crossover_budget} tokens (bold). Gap = Thinking $-$ NoThink.}}")

    lines.append(r"\end{table}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────

def print_summary(
    results: Dict[str, Dict[int, dict]],
    crossover_budget: Optional[int],
) -> None:
    """Print a human-readable summary to stdout."""
    all_budgets = sorted(
        set(results["nothink"].keys()) | set(results["thinking"].keys())
    )

    print()
    print("=" * 72)
    print("  Crossover Analysis: NoThink vs Thinking")
    print("=" * 72)

    header = f"{'Budget':>8}  {'NoThink Acc':>12}  {'NoThink Tok':>12}  {'Think Acc':>10}  {'Think Tok':>10}  {'Gap':>8}"
    print(header)
    print("-" * len(header))

    for b in all_budgets:
        nt = results["nothink"].get(b, {})
        th = results["thinking"].get(b, {})

        nt_acc = f"{nt['accuracy'] * 100:.1f}%" if "accuracy" in nt else "--"
        nt_tok = f"{nt['avg_tokens']:.0f}" if "avg_tokens" in nt else "--"
        th_acc = f"{th['accuracy'] * 100:.1f}%" if "accuracy" in th else "--"
        th_tok = f"{th['avg_tokens']:.0f}" if "avg_tokens" in th else "--"

        if "accuracy" in nt and "accuracy" in th:
            gap = (th["accuracy"] - nt["accuracy"]) * 100
            sign = "+" if gap >= 0 else ""
            gap_str = f"{sign}{gap:.1f}%"
        else:
            gap_str = "--"

        marker = " <-- crossover" if b == crossover_budget else ""
        print(f"{b:>8}  {nt_acc:>12}  {nt_tok:>12}  {th_acc:>10}  {th_tok:>10}  {gap_str:>8}{marker}")

    print()
    if crossover_budget is not None:
        print(f"Crossover point: {crossover_budget} tokens "
              f"(lowest budget where Thinking >= NoThink)")
    else:
        print("No crossover found in the available data.")
    print()


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crossover analysis: find where thinking surpasses nothink",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results/crossover/",
        help="Directory containing crossover result JSON files (default: results/crossover/)",
    )
    parser.add_argument(
        "--manual-data",
        nargs="+",
        default=None,
        metavar="MODE@BUDGET=ACC[:TOKENS]",
        help="Manual data points. Format: mode@budget=accuracy[:avg_tokens]. "
             "Accuracy in %%. Example: thinking@128=3.0 nothink@512=90.2:350",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default="paper/fig_crossover_analysis.pdf",
        help="Output figure path (default: paper/fig_crossover_analysis.pdf)",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default=None,
        help="Output LaTeX table path (default: print to stdout)",
    )
    parser.add_argument(
        "--annotate-budgets",
        nargs="+",
        type=int,
        default=None,
        help="Budget values to annotate with gap on figure (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    # --- Load data ---
    results: Dict[str, Dict[int, dict]] = {"nothink": {}, "thinking": {}}

    # From directory
    input_dir = Path(args.input_dir)
    if input_dir.exists() and input_dir.is_dir():
        dir_results = load_from_directory(input_dir)
        results = merge_results(results, dir_results)
    else:
        if args.manual_data is None:
            log.warning(
                f"Input directory {input_dir} does not exist and no --manual-data provided."
            )

    # From manual data (overrides directory data)
    if args.manual_data is not None:
        manual_results = parse_manual_data(args.manual_data)
        results = merge_results(results, manual_results)

    # --- Validate ---
    total_points = sum(len(v) for v in results.values())
    if total_points == 0:
        log.error("No data points loaded. Provide --input-dir or --manual-data.")
        sys.exit(1)

    log.info(
        f"Loaded {len(results['nothink'])} nothink + "
        f"{len(results['thinking'])} thinking data points"
    )

    # --- Analysis ---
    crossover_budget = find_crossover(results, BUDGETS)

    # --- Console output ---
    print_summary(results, crossover_budget)

    # --- Figure ---
    output_fig = Path(args.output_fig)
    generate_figure(results, crossover_budget, output_fig, args.annotate_budgets)

    # --- LaTeX table ---
    latex_table = generate_latex_table(results, crossover_budget)
    print("\n% ──── LaTeX table (copy into paper) ────")
    print(latex_table)
    print()

    if args.output_table is not None:
        table_path = Path(args.output_table)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(latex_table + "\n")
        log.info(f"LaTeX table saved to {table_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
