#!/usr/bin/env python3
"""
integrate_bbh_results.py
========================
Process BBH experiment output and generate LaTeX tables for the paper.

Reads the summary JSON produced by run_bbh_experiment.py and generates:
  1. Full appendix table (all tasks × budgets): tab:bbh-results
  2. Compact main-body table (overall aggregates):  tab:bbh-compact
  3. Key findings summary printed to stdout

Usage:
    # Basic (reads latest summary JSON from results dir)
    python scripts/integrate_bbh_results.py \
        --results_dir results/bbh_v2

    # Explicit JSON file
    python scripts/integrate_bbh_results.py \
        --summary_json results/bbh_v2/summary_bbh_Qwen3_8B_20260402.json

    # Custom output location
    python scripts/integrate_bbh_results.py \
        --results_dir results/bbh_v2 \
        --output paper/sections/table_bbh_results.tex \
        --output_compact paper/sections/table_bbh_compact.tex

    # With per-sample CSV for bootstrap CIs
    python scripts/integrate_bbh_results.py \
        --results_dir results/bbh_v2 \
        --csv results/bbh_v2/per_sample_bbh_Qwen3_8B_20260402.csv
"""

import argparse
import csv
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUDGETS = [256, 512, 1024, 2048]

# Human-readable task names for the table
TASK_DISPLAY_NAMES = {
    "boolean_expressions": "Boolean Expr.",
    "causal_judgement": "Causal Judg.",
    "date_understanding": "Date Under.",
    "logical_deduction_five_objects": "Logical Ded.",
    "tracking_shuffled_objects_three_objects": "Shuffled Obj.",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Find the latest file matching a glob pattern in a directory."""
    matches = sorted(glob.glob(os.path.join(directory, pattern)))
    if not matches:
        return None
    # Sort by modification time, return newest
    matches.sort(key=os.path.getmtime)
    return matches[-1]


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_low, ci_high) as fractions (not percentages).
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return float(values.mean()), lo, hi


def load_per_sample_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load the per-sample CSV into a list of dicts."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for k, v in row.items():
                if k.endswith("_correct") or k.endswith("_tokens") or k == "idx":
                    try:
                        row[k] = int(v)
                    except (ValueError, TypeError):
                        pass
                elif k.endswith("_latency_s"):
                    try:
                        row[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            records.append(row)
    return records


def compute_bootstrap_cis_from_csv(
    records: List[Dict[str, Any]],
    budgets: List[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Compute bootstrap CIs for overall accuracy at each (mode, budget).

    Returns:
        {mode: {budget_str: (mean, ci_lo, ci_hi)}}
    """
    cis: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for mode in ["nothink", "thinking"]:
        mode_recs = [r for r in records if r.get("mode") == mode]
        if not mode_recs:
            continue
        cis[mode] = {}
        for budget in budgets:
            col = f"fixed_{budget}_correct"
            vals = np.array([r[col] for r in mode_recs if col in r], dtype=float)
            if len(vals) > 0:
                mean, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, seed=seed)
                cis[mode][str(budget)] = (mean, lo, hi)
    return cis


def compute_per_task_bootstrap_cis(
    records: List[Dict[str, Any]],
    budgets: List[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]]:
    """Compute bootstrap CIs for per-task accuracy.

    Returns:
        {task: {mode: {budget_str: (mean, ci_lo, ci_hi)}}}
    """
    from collections import defaultdict
    task_groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        task_groups[r.get("bbh_task", "unknown")].append(r)

    result = {}
    for task_name, task_recs in task_groups.items():
        result[task_name] = {}
        for mode in ["nothink", "thinking"]:
            mode_recs = [r for r in task_recs if r.get("mode") == mode]
            if not mode_recs:
                continue
            result[task_name][mode] = {}
            for budget in budgets:
                col = f"fixed_{budget}_correct"
                vals = np.array([r[col] for r in mode_recs if col in r], dtype=float)
                if len(vals) > 0:
                    mean, lo, hi = bootstrap_ci(
                        vals, n_bootstrap=n_bootstrap, seed=seed,
                    )
                    result[task_name][mode][str(budget)] = (mean, lo, hi)
    return result


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------

def fmt_pct(val: float) -> str:
    """Format a fraction as percentage with 1 decimal."""
    return f"{val * 100:.1f}\\%"


def fmt_pct_with_ci(
    val: float,
    ci_lo: Optional[float] = None,
    ci_hi: Optional[float] = None,
) -> str:
    """Format a fraction as percentage, optionally with CI."""
    base = f"{val * 100:.1f}\\%"
    if ci_lo is not None and ci_hi is not None:
        base += f" \\ci{{{ci_lo * 100:.1f}}}{{{ci_hi * 100:.1f}}}"
    return base


def fmt_tax(val: float) -> str:
    """Format thinking tax (positive = thinking hurts)."""
    sign = "+" if val >= 0 else "$-$"
    abs_val = abs(val) * 100
    if val >= 0:
        return f"+{abs_val:.1f}"
    else:
        return f"$-${abs_val:.1f}"


def generate_full_table(
    summary: Dict[str, Any],
    budgets: List[int],
    overall_cis: Optional[Dict] = None,
    per_task_cis: Optional[Dict] = None,
) -> str:
    """Generate the full appendix table with all tasks and budgets.

    Rows: each BBH subtask + Overall
    Columns: Budget | Nothink Acc | Think Acc | Tax (NT−T) | Hit Budget (think)
    One row per (task, budget) combination.
    """
    per_task = summary.get("per_task", {})
    per_mode = summary.get("per_mode", {})
    thinking_tax = summary.get("thinking_tax", {})
    meta = summary.get("meta", {})

    n_total = meta.get("n_total_samples", "?")
    model_name = meta.get("model", "Qwen3-8B")
    model_short = model_name.split("/")[-1]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{\textbf{BBH thinking tax results} (%s, $n{=}%s$). "
                  % (model_short.replace("_", r"\_"), str(n_total)))
    lines.append(r"Accuracy for non-thinking (NT) and thinking (T) modes at each "
                  r"token budget across five BBH subtasks. "
                  r"Tax = NT acc $-$ T acc (positive means thinking hurts). "
                  r"Hit Budget = fraction of thinking-mode samples that exhausted the token budget.}")
    lines.append(r"\label{tab:bbh-results}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{ll rrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Task} & \textbf{Budget} & "
                  r"\textbf{NT Acc} & \textbf{T Acc} & "
                  r"\textbf{Tax} & \textbf{Hit Budget (T)} \\")
    lines.append(r"\midrule")

    # Sort tasks alphabetically
    task_names = sorted(per_task.keys())

    for t_idx, task_name in enumerate(task_names):
        task_data = per_task[task_name]
        display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
        task_n = task_data.get("n", "?")
        tax_data = task_data.get("thinking_tax", {})

        for b_idx, budget in enumerate(budgets):
            b_str = str(budget)

            # Nothink accuracy
            nt_acc = task_data.get("nothink", {}).get(b_str, {}).get("accuracy", None)
            # Thinking accuracy
            th_acc = task_data.get("thinking", {}).get(b_str, {}).get("accuracy", None)
            # Tax
            tax_val = tax_data.get(b_str, None)
            # Hit budget rate (thinking mode)
            hit_budget = task_data.get("thinking", {}).get(b_str, {}).get(
                "hit_budget_rate", None,
            )

            # Per-task CI if available
            nt_ci = None
            th_ci = None
            if per_task_cis and task_name in per_task_cis:
                nt_ci_data = per_task_cis.get(task_name, {}).get("nothink", {}).get(b_str)
                th_ci_data = per_task_cis.get(task_name, {}).get("thinking", {}).get(b_str)
                if nt_ci_data:
                    nt_ci = (nt_ci_data[1], nt_ci_data[2])
                if th_ci_data:
                    th_ci = (th_ci_data[1], th_ci_data[2])

            # Format task name only on first budget row
            task_col = f"{display_name} ($n$={task_n})" if b_idx == 0 else ""

            # Format values
            nt_str = fmt_pct(nt_acc) if nt_acc is not None else "---"
            th_str = fmt_pct(th_acc) if th_acc is not None else "---"
            tax_str = fmt_tax(tax_val) if tax_val is not None else "---"
            hb_str = fmt_pct(hit_budget) if hit_budget is not None else "---"

            lines.append(
                f"  {task_col} & {budget} & {nt_str} & {th_str} & {tax_str} & {hb_str} \\\\"
            )

        # Add midrule between tasks (but not after the last one)
        if t_idx < len(task_names) - 1:
            lines.append(r"  \midrule")

    # Overall row
    lines.append(r"  \midrule")
    for b_idx, budget in enumerate(budgets):
        b_str = str(budget)

        nt_overall = per_mode.get("nothink", {}).get(b_str, {}).get("accuracy", None)
        th_overall = per_mode.get("thinking", {}).get(b_str, {}).get("accuracy", None)
        tax_overall = thinking_tax.get(b_str, {}).get("thinking_tax", None)
        hb_overall = per_mode.get("thinking", {}).get(b_str, {}).get("hit_budget_rate", None)

        # Bootstrap CIs for overall
        nt_ci_str = ""
        th_ci_str = ""
        if overall_cis:
            nt_ci_data = overall_cis.get("nothink", {}).get(b_str)
            th_ci_data = overall_cis.get("thinking", {}).get(b_str)
        else:
            nt_ci_data = None
            th_ci_data = None

        task_col = r"\textbf{Overall}" if b_idx == 0 else ""

        if nt_overall is not None:
            if nt_ci_data:
                nt_str = fmt_pct_with_ci(nt_overall, nt_ci_data[1], nt_ci_data[2])
            else:
                nt_str = fmt_pct(nt_overall)
        else:
            nt_str = "---"

        if th_overall is not None:
            if th_ci_data:
                th_str = fmt_pct_with_ci(th_overall, th_ci_data[1], th_ci_data[2])
            else:
                th_str = fmt_pct(th_overall)
        else:
            th_str = "---"

        tax_str = fmt_tax(tax_overall) if tax_overall is not None else "---"
        hb_str = fmt_pct(hb_overall) if hb_overall is not None else "---"

        lines.append(
            f"  {task_col} & {budget} & {nt_str} & {th_str} & {tax_str} & {hb_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_compact_table(
    summary: Dict[str, Any],
    budgets: List[int],
    overall_cis: Optional[Dict] = None,
) -> str:
    """Generate a compact main-body table with overall aggregates only.

    One row per budget: Budget | NT Acc (CI) | T Acc (CI) | Tax | Hit Budget (T)
    """
    per_mode = summary.get("per_mode", {})
    thinking_tax = summary.get("thinking_tax", {})
    meta = summary.get("meta", {})

    n_total = meta.get("n_total_samples", "?")
    model_name = meta.get("model", "Qwen3-8B")
    model_short = model_name.split("/")[-1]
    n_tasks = len(meta.get("tasks", []))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Thinking tax on non-mathematical reasoning} "
        r"(BBH, %s, %d subtasks, $n{=}%s$). "
        % (model_short.replace("_", r"\_"), n_tasks, str(n_total))
    )
    lines.append(
        r"Non-thinking mode consistently outperforms thinking mode at constrained budgets, "
        r"confirming the thinking tax generalises beyond mathematical benchmarks.}"
    )
    lines.append(r"\label{tab:bbh-compact}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r rr rr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Budget} & \textbf{NT Acc} & \textbf{T Acc} & "
        r"\textbf{Tax (pp)} & \textbf{Hit Budget (T)} \\"
    )
    lines.append(r"\midrule")

    for budget in budgets:
        b_str = str(budget)

        nt_acc = per_mode.get("nothink", {}).get(b_str, {}).get("accuracy", None)
        th_acc = per_mode.get("thinking", {}).get(b_str, {}).get("accuracy", None)
        tax_info = thinking_tax.get(b_str, {})
        tax_val = tax_info.get("thinking_tax", None)
        hb = per_mode.get("thinking", {}).get(b_str, {}).get("hit_budget_rate", None)

        # CIs
        nt_ci_data = overall_cis.get("nothink", {}).get(b_str) if overall_cis else None
        th_ci_data = overall_cis.get("thinking", {}).get(b_str) if overall_cis else None

        if nt_acc is not None:
            if nt_ci_data:
                nt_str = fmt_pct_with_ci(nt_acc, nt_ci_data[1], nt_ci_data[2])
            else:
                nt_str = fmt_pct(nt_acc)
        else:
            nt_str = "---"

        if th_acc is not None:
            if th_ci_data:
                th_str = fmt_pct_with_ci(th_acc, th_ci_data[1], th_ci_data[2])
            else:
                th_str = fmt_pct(th_acc)
        else:
            th_str = "---"

        tax_str = fmt_tax(tax_val) if tax_val is not None else "---"
        hb_str = fmt_pct(hb) if hb is not None else "---"

        lines.append(f"  {budget} & {nt_str} & {th_str} & {tax_str} & {hb_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Key findings analysis
# ---------------------------------------------------------------------------

def print_key_findings(
    summary: Dict[str, Any],
    budgets: List[int],
    overall_cis: Optional[Dict] = None,
) -> None:
    """Print key findings for the paper text to stdout."""
    per_task = summary.get("per_task", {})
    thinking_tax = summary.get("thinking_tax", {})
    meta = summary.get("meta", {})

    print("\n" + "=" * 72)
    print("KEY FINDINGS FOR PAPER TEXT")
    print("=" * 72)

    # 1. Overall thinking tax at each budget
    print("\n--- Overall Thinking Tax ---")
    for budget in budgets:
        b_str = str(budget)
        tax_info = thinking_tax.get(b_str, {})
        if tax_info:
            nt = tax_info.get("nothink_acc", 0)
            th = tax_info.get("thinking_acc", 0)
            tax = tax_info.get("thinking_tax", 0)

            ci_note = ""
            if overall_cis:
                nt_ci = overall_cis.get("nothink", {}).get(b_str)
                th_ci = overall_cis.get("thinking", {}).get(b_str)
                if nt_ci:
                    ci_note += f"  NT 95% CI: [{nt_ci[1]*100:.1f}, {nt_ci[2]*100:.1f}]"
                if th_ci:
                    ci_note += f"  T 95% CI: [{th_ci[1]*100:.1f}, {th_ci[2]*100:.1f}]"

            sign = "+" if tax >= 0 else ""
            print(
                f"  Budget {budget:>5d}: NT={nt*100:.1f}%  T={th*100:.1f}%  "
                f"Tax={sign}{tax*100:.1f}pp{ci_note}"
            )

    # 2. Which tasks show the largest tax
    print("\n--- Per-Task Thinking Tax (largest first) ---")
    task_taxes = []
    for task_name, task_data in per_task.items():
        tax_data = task_data.get("thinking_tax", {})
        if tax_data:
            # Average tax across budgets
            taxes = [v for v in tax_data.values() if v is not None]
            avg_tax = sum(taxes) / len(taxes) if taxes else 0
            # Tax at largest budget
            largest_b = str(max(budgets))
            tax_at_largest = tax_data.get(largest_b, 0)
            task_taxes.append((task_name, avg_tax, tax_at_largest, tax_data))

    task_taxes.sort(key=lambda x: x[1], reverse=True)
    for task_name, avg_tax, tax_at_largest, tax_dict in task_taxes:
        display = TASK_DISPLAY_NAMES.get(task_name, task_name)
        detail = "  ".join(
            f"@{b}: {'+' if tax_dict.get(str(b), 0) >= 0 else ''}{tax_dict.get(str(b), 0)*100:.1f}pp"
            for b in budgets if str(b) in tax_dict
        )
        print(f"  {display:<20s}  avg={avg_tax*100:+.1f}pp  ({detail})")

    # 3. Does the tax persist on non-mathematical tasks?
    print("\n--- Generalisation Finding ---")
    positive_tax_count = 0
    total_entries = 0
    for task_name, task_data in per_task.items():
        tax_data = task_data.get("thinking_tax", {})
        for b_str, val in tax_data.items():
            total_entries += 1
            if val is not None and val > 0:
                positive_tax_count += 1

    print(
        f"  Thinking tax is positive (thinking hurts) in {positive_tax_count}/{total_entries} "
        f"({positive_tax_count/max(1,total_entries)*100:.0f}%) of task-budget combinations."
    )

    # Overall summary sentence
    overall_taxes = [
        thinking_tax.get(str(b), {}).get("thinking_tax", None)
        for b in budgets
    ]
    overall_taxes = [t for t in overall_taxes if t is not None]
    if overall_taxes:
        min_tax = min(overall_taxes)
        max_tax = max(overall_taxes)
        all_positive = all(t > 0 for t in overall_taxes)
        if all_positive:
            print(
                f"\n  ✓ The thinking tax is consistently positive across all budgets "
                f"({min_tax*100:.1f}–{max_tax*100:.1f} pp), confirming that the phenomenon "
                f"generalises beyond mathematical reasoning to diverse BBH tasks."
            )
        else:
            crossover_budgets = [
                budgets[i] for i, t in enumerate(overall_taxes) if t <= 0
            ]
            print(
                f"\n  ! The thinking tax turns negative (thinking helps) at budgets: "
                f"{crossover_budgets}. The crossover pattern observed on GSM8K "
                f"also appears on BBH."
            )

    # 4. Summary statistics for easy reference
    print("\n--- Copy-Paste Statistics ---")
    n_tasks = len(meta.get("tasks", []))
    n_total = meta.get("n_total_samples", "?")
    model = meta.get("model", "?")
    print(f"  Model: {model}")
    print(f"  Tasks: {n_tasks}, Total samples: {n_total}")
    if overall_taxes:
        mid_budget = budgets[len(budgets) // 2] if len(budgets) > 1 else budgets[0]
        mid_tax = thinking_tax.get(str(mid_budget), {}).get("thinking_tax", None)
        if mid_tax is not None:
            print(f"  Headline: thinking tax = {mid_tax*100:+.1f} pp at budget {mid_budget}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process BBH experiment results and generate LaTeX tables.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing BBH results (searches for latest summary JSON).",
    )
    parser.add_argument(
        "--summary_json", type=str, default=None,
        help="Explicit path to summary JSON file (overrides --results_dir search).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to per-sample CSV for bootstrap CIs. "
             "If not given, searches --results_dir for the latest one.",
    )
    parser.add_argument(
        "--output", type=str, default="paper/sections/table_bbh_results.tex",
        help="Output path for the full appendix table.",
    )
    parser.add_argument(
        "--output_compact", type=str, default="paper/sections/table_bbh_compact.tex",
        help="Output path for the compact main-body table.",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=BUDGETS,
        help=f"Token budgets to include in tables (default: {BUDGETS}).",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=10000,
        help="Number of bootstrap resamples for CIs (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print tables to stdout instead of writing files.",
    )
    args = parser.parse_args()

    # --- Resolve summary JSON path ---
    summary_path = args.summary_json
    if summary_path is None:
        if args.results_dir is None:
            # Default: try common locations
            candidates = [
                PROJECT_ROOT / "results" / "bbh_v2",
                PROJECT_ROOT / "results" / "bbh",
                PROJECT_ROOT / "results_kun" / "bbh_v2",
            ]
            for cand in candidates:
                if cand.is_dir():
                    args.results_dir = str(cand)
                    log.info(f"Auto-detected results dir: {args.results_dir}")
                    break
            if args.results_dir is None:
                log.error(
                    "No --results_dir or --summary_json given, and no default "
                    "results directory found. Aborting."
                )
                sys.exit(1)

        summary_path = find_latest_file(args.results_dir, "summary_bbh_*.json")
        if summary_path is None:
            log.error(
                f"No summary JSON files matching 'summary_bbh_*.json' found in "
                f"{args.results_dir}. Aborting."
            )
            sys.exit(1)

    log.info(f"Loading summary: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # --- Resolve per-sample CSV (optional, for bootstrap CIs) ---
    csv_path = args.csv
    if csv_path is None and args.results_dir:
        csv_path = find_latest_file(args.results_dir, "per_sample_bbh_*.csv")
        if csv_path:
            log.info(f"Auto-detected per-sample CSV: {csv_path}")

    # --- Compute bootstrap CIs if CSV is available ---
    overall_cis = None
    per_task_cis = None
    if csv_path and os.path.isfile(csv_path):
        log.info(f"Loading per-sample CSV for bootstrap CIs: {csv_path}")
        records = load_per_sample_csv(csv_path)
        log.info(f"  Loaded {len(records)} records")

        log.info(f"  Computing bootstrap CIs (n_bootstrap={args.n_bootstrap})...")
        overall_cis = compute_bootstrap_cis_from_csv(
            records, args.budgets,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        per_task_cis = compute_per_task_bootstrap_cis(
            records, args.budgets,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        log.info("  Done.")
    else:
        log.warning(
            "No per-sample CSV found; tables will not include bootstrap CIs. "
            "Pass --csv to enable."
        )

    # --- Filter budgets to those actually present in summary ---
    available_budgets_str = set(summary.get("per_mode", {}).get("nothink", {}).keys())
    available_budgets_str |= set(summary.get("per_mode", {}).get("thinking", {}).keys())
    effective_budgets = [b for b in args.budgets if str(b) in available_budgets_str]

    if not effective_budgets:
        log.error(
            f"None of the requested budgets {args.budgets} found in summary. "
            f"Available: {sorted(available_budgets_str)}. Aborting."
        )
        sys.exit(1)

    if set(effective_budgets) != set(args.budgets):
        missing = set(args.budgets) - set(effective_budgets)
        log.warning(f"Budgets {missing} not found in summary, skipping them.")

    log.info(f"Generating tables for budgets: {effective_budgets}")

    # --- Generate tables ---
    full_table = generate_full_table(
        summary, effective_budgets,
        overall_cis=overall_cis,
        per_task_cis=per_task_cis,
    )
    compact_table = generate_compact_table(
        summary, effective_budgets,
        overall_cis=overall_cis,
    )

    # --- Output ---
    if args.dry_run:
        print("\n% ===== FULL TABLE (appendix) =====")
        print(full_table)
        print("\n% ===== COMPACT TABLE (main body) =====")
        print(compact_table)
    else:
        # Full table
        out_full = PROJECT_ROOT / args.output
        os.makedirs(out_full.parent, exist_ok=True)
        with open(out_full, "w", encoding="utf-8") as f:
            f.write(full_table + "\n")
        log.info(f"Wrote full table:    {out_full}")

        # Compact table
        out_compact = PROJECT_ROOT / args.output_compact
        os.makedirs(out_compact.parent, exist_ok=True)
        with open(out_compact, "w", encoding="utf-8") as f:
            f.write(compact_table + "\n")
        log.info(f"Wrote compact table: {out_compact}")

    # --- Print key findings ---
    print_key_findings(summary, effective_budgets, overall_cis=overall_cis)


if __name__ == "__main__":
    main()
