#!/usr/bin/env python3
"""
generate_final_paper_data.py
============================
Consolidate all AdaThink experiment results into a unified dataset,
generate crossover analysis figure/table, model-size scaling table,
and print summary statistics.

Data sources (in priority order, later overrides earlier):
  1. Hardcoded confirmed data points (known ground truth)
  2. results_kun/  (local sync of server results)
  3. results/crossover/  (crossover experiment JSONs)
  4. results/fulltest_27b_nothink/ (27B nothink JSONs)
  5. --extra-json  (arbitrary additional JSON files)

Output:
  - paper/fig_crossover_analysis.pdf        (crossover figure)
  - results/paper_figures/table_crossover.tex (LaTeX crossover table)
  - results/paper_figures/table_model_size_scaling_updated.tex
  - results/consolidated_results.json        (all data in one place)

Usage:
    python scripts/generate_final_paper_data.py
    python scripts/generate_final_paper_data.py --dry-run
    python scripts/generate_final_paper_data.py --extra-json results/new_run.json
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
BUDGETS = [128, 256, 512, 1024, 2048, 4096]
MODES = ["nothink", "thinking"]
MODELS = ["8B", "27B"]

# Resolve project root relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Hardcoded confirmed data (ground truth, highest priority as seed defaults)
# These are confirmed full GSM8K (n=1319) results.
# ---------------------------------------------------------------------------
CONFIRMED_DATA: List[Dict[str, Any]] = [
    # === 8B nothink ===
    {
        "model": "8B", "mode": "nothink", "budget": 128,
        "accuracy": 0.508, "avg_tokens": 113.0,
        "early_stop_rate": 0.394, "n_samples": 1319,
        "source_file": "hardcoded:nothink_baseline_fullset_128.json",
    },
    {
        "model": "8B", "mode": "nothink", "budget": 256,
        "accuracy": 0.875, "avg_tokens": 146.0,
        "early_stop_rate": 0.888, "n_samples": 1319,
        "source_file": "hardcoded:nothink_baseline_fullset_complete.json",
    },
    # === 8B thinking ===
    {
        "model": "8B", "mode": "thinking", "budget": 128,
        "accuracy": 0.030, "avg_tokens": 128.0,
        "early_stop_rate": 0.0, "n_samples": 1319,
        "source_file": "hardcoded:fulltest/summary_gsm8k_Qwen3_8B.json",
    },
    {
        "model": "8B", "mode": "thinking", "budget": 256,
        "accuracy": 0.180, "avg_tokens": 255.0,
        "early_stop_rate": 0.014, "n_samples": 1319,
        "source_file": "hardcoded:fulltest/summary_gsm8k_Qwen3_8B.json",
    },
    {
        "model": "8B", "mode": "thinking", "budget": 512,
        "accuracy": 0.652, "avg_tokens": 460.0,
        "early_stop_rate": 0.374, "n_samples": 1319,
        "source_file": "hardcoded:fulltest/summary_gsm8k_Qwen3_8B.json",
    },
    # === 27B thinking ===
    {
        "model": "27B", "mode": "thinking", "budget": 128,
        "accuracy": 0.0356, "avg_tokens": 144.0,
        "early_stop_rate": 0.0, "n_samples": 1319,
        "source_file": "hardcoded:fulltest_27b/summary_gsm8k_Qwen3.5_27B.json",
    },
    {
        "model": "27B", "mode": "thinking", "budget": 256,
        "accuracy": 0.0788, "avg_tokens": 272.0,
        "early_stop_rate": 0.0, "n_samples": 1319,
        "source_file": "hardcoded:fulltest_27b/summary_gsm8k_Qwen3.5_27B.json",
    },
    {
        "model": "27B", "mode": "thinking", "budget": 512,
        "accuracy": 0.1835, "avg_tokens": 528.0,
        "early_stop_rate": 0.007, "n_samples": 1319,
        "source_file": "hardcoded:fulltest_27b/summary_gsm8k_Qwen3.5_27B.json",
    },
]

# Model name -> model tag mapping
MODEL_TAG_MAP = {
    "Qwen/Qwen3-8B": "8B",
    "Qwen/Qwen3.5-8B": "8B",
    "Qwen3-8B": "8B",
    "Qwen3.5-8B": "8B",
    "Qwen/Qwen3.5-27B": "27B",
    "Qwen3.5-27B": "27B",
    "Qwen/Qwen3-27B": "27B",
}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------
def make_record(
    model: str,
    mode: str,
    budget: int,
    accuracy: float,
    avg_tokens: float = float("nan"),
    early_stop_rate: float = float("nan"),
    n_samples: int = 0,
    source_file: str = "",
) -> Dict[str, Any]:
    """Create a canonical data record."""
    return {
        "model": model,
        "mode": mode,
        "budget": int(budget),
        "accuracy": float(accuracy),
        "avg_tokens": float(avg_tokens),
        "early_stop_rate": float(early_stop_rate),
        "n_samples": int(n_samples),
        "source_file": str(source_file),
    }


def record_key(r: Dict[str, Any]) -> Tuple[str, str, int]:
    """Unique key for a data record: (model, mode, budget)."""
    return (r["model"], r["mode"], r["budget"])


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def infer_model_tag(data: dict, filename: str) -> Optional[str]:
    """Infer '8B' or '27B' from JSON content or filename."""
    # From meta.model
    for key_path in [("meta", "model"), ("model",)]:
        d = data
        for k in key_path:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                d = None
                break
        if d and isinstance(d, str):
            for pattern, tag in MODEL_TAG_MAP.items():
                if pattern.lower() in d.lower():
                    return tag
    # From filename
    fname = filename.lower()
    if "27b" in fname:
        return "27B"
    if "8b" in fname:
        return "8B"
    return None


def infer_mode(data: dict, filename: str) -> Optional[str]:
    """Infer 'nothink' or 'thinking' from JSON content or filename."""
    # Explicit field
    mode = data.get("mode")
    if mode in MODES:
        return mode

    # enable_thinking
    enable = data.get("enable_thinking")
    if enable is None and "meta" in data:
        enable = data["meta"].get("enable_thinking")
    if enable is not None:
        return "thinking" if enable else "nothink"

    # method field
    method = data.get("method") or (data.get("meta", {}).get("method") or "")
    if "nothink" in method.lower():
        return "nothink"

    # Filename inference
    fname = filename.lower()
    if "nothink" in fname or "no_think" in fname or "no-think" in fname:
        return "nothink"
    if "thinking" in fname or "think" in fname:
        return "thinking"

    return None


def infer_budget_from_filename(fname: str) -> Optional[int]:
    """Infer budget from filename patterns."""
    patterns = [
        r"budget[_-]?(\d+)",
        r"_b(\d+)[_.]",
        r"tokens[_-]?(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, fname, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if val in BUDGETS:
                return val
    return None


# ---------------------------------------------------------------------------
# Loaders for different JSON formats
# ---------------------------------------------------------------------------
def load_summary_json(fpath: Path, benchmark_filter: str = "gsm8k") -> List[Dict[str, Any]]:
    """
    Load a summary JSON file (format: meta + fixed + adaptive).
    This is the format of fulltest summary files.
    Only loads results matching benchmark_filter (default: gsm8k).
    """
    records = []
    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"  Cannot read {fpath.name}: {e}")
        return records

    # Filter by benchmark
    meta = data.get("meta", {})
    benchmark = meta.get("benchmark", "")
    if benchmark_filter and benchmark and benchmark.lower() != benchmark_filter.lower():
        log.info(f"  Skipping {fpath.name}: benchmark={benchmark} (want {benchmark_filter})")
        return records

    model_tag = infer_model_tag(data, fpath.name)
    enable_thinking = meta.get("enable_thinking", True)  # default thinking
    mode = "thinking" if enable_thinking else "nothink"
    n_samples = meta.get("n_samples", 0)

    # Parse fixed budget results
    fixed = data.get("fixed", {})
    for budget_str, entry in fixed.items():
        try:
            budget = int(budget_str)
        except ValueError:
            continue
        acc = entry.get("accuracy")
        if acc is None:
            continue
        avg_tok = entry.get("avg_tokens", float("nan"))
        early_stop = entry.get("early_stop_rate", float("nan"))
        if model_tag:
            records.append(make_record(
                model=model_tag, mode=mode, budget=budget,
                accuracy=acc, avg_tokens=avg_tok,
                early_stop_rate=early_stop, n_samples=n_samples,
                source_file=str(fpath),
            ))

    return records


def load_nothink_baseline_json(fpath: Path) -> List[Dict[str, Any]]:
    """
    Load a nothink_baseline JSON (format: meta + results.{nothink_B, thinking_B}).
    """
    records = []
    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"  Cannot read {fpath.name}: {e}")
        return records

    model_tag = infer_model_tag(data, fpath.name)
    meta = data.get("meta", {})
    n_samples = meta.get("n_samples", 0)

    results_block = data.get("results", {})
    for key, entry in results_block.items():
        # key format: nothink_128, thinking_256, etc.
        m = re.match(r"(nothink|thinking)_(\d+)", key)
        if not m:
            continue
        mode = m.group(1)
        budget = int(m.group(2))
        acc = entry.get("accuracy")
        if acc is None:
            continue
        avg_tok = entry.get("avg_tokens", float("nan"))
        early_stop = entry.get("early_stop_rate", float("nan"))
        if model_tag:
            records.append(make_record(
                model=model_tag, mode=mode, budget=budget,
                accuracy=acc, avg_tokens=avg_tok,
                early_stop_rate=early_stop, n_samples=n_samples,
                source_file=str(fpath),
            ))

    return records


def load_flat_json(fpath: Path) -> List[Dict[str, Any]]:
    """
    Load a flat JSON file with top-level keys: mode, budget, accuracy, etc.
    Also handles crossover-style results.
    """
    records = []
    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"  Cannot read {fpath.name}: {e}")
        return records

    model_tag = infer_model_tag(data, fpath.name)
    mode = infer_mode(data, fpath.name)
    n_samples = data.get("n_samples") or data.get("total_samples") or 0

    # Case 1: flat with mode/budget/accuracy at top level
    budget = data.get("budget") or data.get("max_new_tokens")
    acc = data.get("accuracy")
    if mode and budget and acc is not None and model_tag:
        avg_tok = data.get("avg_tokens", data.get("avg_output_tokens", float("nan")))
        early_stop = data.get("early_stop_rate", float("nan"))
        records.append(make_record(
            model=model_tag, mode=mode, budget=int(budget),
            accuracy=float(acc), avg_tokens=float(avg_tok) if avg_tok else float("nan"),
            early_stop_rate=float(early_stop) if early_stop else float("nan"),
            n_samples=int(n_samples),
            source_file=str(fpath),
        ))

    return records


def load_any_json(fpath: Path, benchmark_filter: str = "gsm8k") -> List[Dict[str, Any]]:
    """
    Try all known JSON formats in sequence. Return parsed records.
    Only includes results from matching benchmark (default: gsm8k).
    """
    records = []

    try:
        with open(fpath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"  Cannot read {fpath.name}: {e}")
        return records

    # Detect format
    if "fixed" in data and "meta" in data:
        records = load_summary_json(fpath, benchmark_filter=benchmark_filter)
    elif "results" in data and isinstance(data.get("results"), dict):
        first_key = next(iter(data["results"]), "")
        if re.match(r"(nothink|thinking)_\d+", first_key):
            records = load_nothink_baseline_json(fpath)
        else:
            records = load_flat_json(fpath)
    else:
        records = load_flat_json(fpath)

    return records


# ---------------------------------------------------------------------------
# Directory scanner
# ---------------------------------------------------------------------------
def scan_directory(dirpath: Path) -> List[Dict[str, Any]]:
    """Recursively scan a directory for JSON files and extract records."""
    all_records = []
    if not dirpath.exists():
        log.info(f"  Directory does not exist: {dirpath}")
        return all_records

    json_files = sorted(dirpath.rglob("*.json"))
    if not json_files:
        log.info(f"  No JSON files in {dirpath}")
        return all_records

    log.info(f"  Scanning {len(json_files)} JSON files in {dirpath}")
    for jf in json_files:
        recs = load_any_json(jf)
        if recs:
            all_records.extend(recs)

    return all_records


# ---------------------------------------------------------------------------
# Data consolidation
# ---------------------------------------------------------------------------
def consolidate(
    record_lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Merge multiple record lists. Later lists override earlier ones
    for the same (model, mode, budget) key. Among duplicates,
    prefer records with larger n_samples.
    """
    registry: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for records in record_lists:
        for r in records:
            key = record_key(r)
            if key not in registry:
                registry[key] = r
            else:
                existing = registry[key]
                # Prefer larger n_samples, or later record as tiebreak
                if r["n_samples"] >= existing["n_samples"]:
                    registry[key] = r

    # Sort by model, mode, budget
    result = sorted(
        registry.values(),
        key=lambda r: (r["model"], r["mode"], r["budget"]),
    )
    return result


# ---------------------------------------------------------------------------
# Crossover analysis
# ---------------------------------------------------------------------------
def find_crossover(
    data: List[Dict[str, Any]],
    model: str = "8B",
) -> Optional[int]:
    """
    Find the lowest budget where thinking accuracy >= nothink accuracy
    for a given model.
    """
    nothink = {r["budget"]: r["accuracy"] for r in data
               if r["model"] == model and r["mode"] == "nothink"}
    thinking = {r["budget"]: r["accuracy"] for r in data
                if r["model"] == model and r["mode"] == "thinking"}

    for b in sorted(BUDGETS):
        if b in thinking and b in nothink:
            if thinking[b] >= nothink[b]:
                return b
    return None


def interpolate_crossover(
    data: List[Dict[str, Any]],
    model: str = "8B",
) -> Optional[float]:
    """
    Estimate crossover budget via log-linear interpolation between the
    last budget where nothink > thinking and the first where thinking >= nothink.
    Returns None if crossover cannot be estimated.
    """
    nothink = {r["budget"]: r["accuracy"] for r in data
               if r["model"] == model and r["mode"] == "nothink"}
    thinking = {r["budget"]: r["accuracy"] for r in data
                if r["model"] == model and r["mode"] == "thinking"}

    common = sorted(set(nothink.keys()) & set(thinking.keys()))
    if len(common) < 2:
        return None

    # Find transition pair
    for i in range(len(common) - 1):
        b_lo, b_hi = common[i], common[i + 1]
        gap_lo = thinking[b_lo] - nothink[b_lo]
        gap_hi = thinking[b_hi] - nothink[b_hi]
        if gap_lo < 0 and gap_hi >= 0:
            # Log-linear interpolation
            log_lo = np.log2(b_lo)
            log_hi = np.log2(b_hi)
            # gap_lo + t * (gap_hi - gap_lo) = 0  =>  t = -gap_lo / (gap_hi - gap_lo)
            t = -gap_lo / (gap_hi - gap_lo) if (gap_hi - gap_lo) != 0 else 0.5
            log_cross = log_lo + t * (log_hi - log_lo)
            return 2 ** log_cross

    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def generate_crossover_figure(
    data: List[Dict[str, Any]],
    model: str,
    crossover_budget: Optional[int],
    crossover_interp: Optional[float],
    output_path: Path,
) -> None:
    """Generate publication-quality crossover figure."""
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

    COLOR_NOTHINK = "#1f77b4"
    COLOR_THINKING = "#ff7f0e"
    COLOR_CROSSOVER = "#2ca02c"

    for mode, color, marker, label in [
        ("nothink", COLOR_NOTHINK, "s", "NoThink"),
        ("thinking", COLOR_THINKING, "o", "Thinking"),
    ]:
        subset = [r for r in data if r["model"] == model and r["mode"] == mode]
        if not subset:
            continue
        subset.sort(key=lambda r: r["budget"])
        budgets_m = [r["budget"] for r in subset]
        accs = [r["accuracy"] * 100 for r in subset]
        ax.plot(
            budgets_m, accs,
            color=color, marker=marker, markersize=7,
            linewidth=2.0, linestyle="-", label=label,
            zorder=3,
        )
        # Data labels
        for b, a in zip(budgets_m, accs):
            offset_y = 3 if a < 50 else -5
            ax.annotate(
                f"{a:.1f}%",
                xy=(b, a),
                xytext=(0, offset_y),
                textcoords="offset points",
                fontsize=8, ha="center", va="bottom" if offset_y > 0 else "top",
                color=color, alpha=0.85,
            )

    # Crossover annotation
    cross_x = crossover_interp or (crossover_budget if crossover_budget else None)
    if cross_x is not None:
        ax.axvline(
            x=cross_x, color=COLOR_CROSSOVER, linestyle="--",
            linewidth=1.5, alpha=0.8, zorder=2,
        )
        # Find y-position for annotation (midpoint of available range)
        all_accs = [r["accuracy"] * 100 for r in data if r["model"] == model]
        mid_y = (min(all_accs) + max(all_accs)) / 2 if all_accs else 50
        label_text = f"Crossover\n~{cross_x:.0f} tokens"
        ax.annotate(
            label_text,
            xy=(cross_x, mid_y),
            xytext=(15, 20),
            textcoords="offset points",
            fontsize=9, color=COLOR_CROSSOVER,
            ha="left", va="center",
            arrowprops=dict(
                arrowstyle="->", color=COLOR_CROSSOVER,
                lw=1.2, connectionstyle="arc3,rad=0.2",
            ),
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor=COLOR_CROSSOVER, alpha=0.9,
            ),
        )

    # Axes
    all_budgets = sorted({r["budget"] for r in data if r["model"] == model})
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    if all_budgets:
        ax.set_xticks(all_budgets)

    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Accuracy (%)")

    # Y range
    all_accs = [r["accuracy"] * 100 for r in data if r["model"] == model]
    if all_accs:
        y_min = max(0, min(all_accs) - 5)
        y_max = min(100, max(all_accs) + 8)
        ax.set_ylim(y_min, y_max)

    # Grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2)

    # Legend
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format="pdf", bbox_inches="tight")
    log.info(f"Figure saved: {output_path}")

    png_path = output_path.with_suffix(".png")
    fig.savefig(str(png_path), format="png", bbox_inches="tight")
    log.info(f"Preview saved: {png_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX tables
# ---------------------------------------------------------------------------
def generate_crossover_table(
    data: List[Dict[str, Any]],
    model: str,
    crossover_budget: Optional[int],
) -> str:
    """Generate LaTeX crossover table: budget x mode -> accuracy."""
    nothink = {r["budget"]: r for r in data
               if r["model"] == model and r["mode"] == "nothink"}
    thinking = {r["budget"]: r for r in data
                if r["model"] == model and r["mode"] == "thinking"}

    all_budgets = sorted(set(nothink.keys()) | set(thinking.keys()))

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Crossover analysis: NoThink vs.\ Thinking accuracy} "
        rf"on GSM8K ({model} model). "
        r"The crossover point marks where Thinking first matches NoThink.}"
    )
    lines.append(r"\label{tab:crossover}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{r cc cc c}")
    lines.append(r"\toprule")
    lines.append(
        r"Budget & \multicolumn{2}{c}{NoThink} & "
        r"\multicolumn{2}{c}{Thinking} & Gap \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r" & Acc (\%) & Tokens & Acc (\%) & Tokens & ($\Delta$\%) \\")
    lines.append(r"\midrule")

    for b in all_budgets:
        nt = nothink.get(b)
        th = thinking.get(b)

        nt_acc = f"{nt['accuracy'] * 100:.1f}" if nt else "--"
        nt_tok = f"{nt['avg_tokens']:.0f}" if (nt and not np.isnan(nt["avg_tokens"])) else "--"
        th_acc = f"{th['accuracy'] * 100:.1f}" if th else "--"
        th_tok = f"{th['avg_tokens']:.0f}" if (th and not np.isnan(th["avg_tokens"])) else "--"

        if nt and th:
            gap = (th["accuracy"] - nt["accuracy"]) * 100
            sign = "+" if gap >= 0 else ""
            gap_str = f"{sign}{gap:.1f}"
        else:
            gap_str = "--"

        if b == crossover_budget:
            row = (
                rf"\textbf{{{b}}} & \textbf{{{nt_acc}}} & {nt_tok} "
                rf"& \textbf{{{th_acc}}} & {th_tok} & \textbf{{{gap_str}}} \\"
            )
        else:
            row = rf"{b} & {nt_acc} & {nt_tok} & {th_acc} & {th_tok} & {gap_str} \\"

        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    if crossover_budget is not None:
        lines.append(r"\vspace{2pt}")
        lines.append(
            rf"\raggedright\footnotesize\textit{{Crossover at {crossover_budget} "
            r"tokens (bold). Gap = Thinking $-$ NoThink.}}"
        )

    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_model_size_table(
    data: List[Dict[str, Any]],
) -> str:
    """
    Generate updated model-size scaling table, including 27B nothink row
    if available.
    """
    # Gather data by (model, mode) -> {budget: record}
    index: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    for r in data:
        key = (r["model"], r["mode"])
        if key not in index:
            index[key] = {}
        index[key][r["budget"]] = r

    # Reference: 8B nothink (for computing thinking tax)
    nothink_8b = index.get(("8B", "nothink"), {})

    # Budgets to show
    budgets_show = [128, 256, 512, 1024, 2048, 4096]
    # Only show budgets that have at least one data point
    budgets_show = [
        b for b in budgets_show
        if any(b in index.get(k, {}) for k in index)
    ]

    has_27b_nothink = ("27B", "nothink") in index and len(index[("27B", "nothink")]) > 0

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Thinking Tax scales with model size.} "
        r"Larger models have longer reasoning chains that are more severely "
        r"truncated at matched budgets. "
        r"Tax = NoThink$_\text{8B}$ accuracy $-$ Think accuracy.}"
    )
    lines.append(r"\label{tab:model-size-scaling-updated}")
    lines.append(r"\small")

    # Columns: Budget | 8B Think | 27B Think | (27B NoThink if available) | 27B vs 8B
    if has_27b_nothink:
        ncols = 5
        lines.append(r"\begin{tabular}{r rr r r}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Budget} & \textbf{8B Think} & \textbf{27B Think} "
            r"& \textbf{27B NoThink} & \textbf{27B vs 8B} \\"
        )
    else:
        ncols = 4
        lines.append(r"\begin{tabular}{r rr r}")
        lines.append(r"\toprule")
        lines.append(
            r"\textbf{Budget} & \textbf{8B Think} & \textbf{27B Think} "
            r"& \textbf{27B vs 8B} \\"
        )

    lines.append(r"\midrule")

    for b in budgets_show:
        think_8b = index.get(("8B", "thinking"), {}).get(b)
        think_27b = index.get(("27B", "thinking"), {}).get(b)
        nothink_27b_rec = index.get(("27B", "nothink"), {}).get(b)

        t8 = f"{think_8b['accuracy'] * 100:.1f}\\%" if think_8b else "--"
        t27 = f"{think_27b['accuracy'] * 100:.1f}\\%" if think_27b else "--"

        if think_8b and think_27b:
            diff = (think_27b["accuracy"] - think_8b["accuracy"]) * 100
            sign = "+" if diff >= 0 else ""
            diff_str = f"${sign}{diff:.1f}$pp"
        else:
            diff_str = "--"

        if has_27b_nothink:
            nt27 = f"{nothink_27b_rec['accuracy'] * 100:.1f}\\%" if nothink_27b_rec else "--"
            row = rf"{b} & {t8} & {t27} & {nt27} & {diff_str} \\"
        else:
            row = rf"{b} & {t8} & {t27} & {diff_str} \\"

        lines.append(row)

    # Thinking Tax section
    lines.append(r"\midrule")
    tax_cols = ncols
    lines.append(
        rf"\multicolumn{{{tax_cols}}}{{l}}"
        r"{\textit{Thinking Tax (NoThink$_\text{8B}$ $-$ Think)}} \\"
    )
    lines.append(r"\midrule")

    for b in budgets_show:
        nt8_rec = nothink_8b.get(b)
        think_8b = index.get(("8B", "thinking"), {}).get(b)
        think_27b = index.get(("27B", "thinking"), {}).get(b)

        if nt8_rec and think_8b:
            tax_8b = (nt8_rec["accuracy"] - think_8b["accuracy"]) * 100
            tax_8b_str = f"+{tax_8b:.1f}pp"
        else:
            tax_8b_str = "--"
            tax_8b = None

        if nt8_rec and think_27b:
            tax_27b = (nt8_rec["accuracy"] - think_27b["accuracy"]) * 100
            tax_27b_str = f"+{tax_27b:.1f}pp"
        else:
            tax_27b_str = "--"
            tax_27b = None

        if tax_8b is not None and tax_27b is not None and tax_8b > 0:
            ratio = tax_27b / tax_8b
            ratio_str = rf"\textbf{{{ratio:.1f}$\times$ larger}}"
        else:
            ratio_str = "---"

        if has_27b_nothink:
            row = rf"{b} & {tax_8b_str} & {tax_27b_str} & -- & {ratio_str} \\"
        else:
            row = rf"{b} & {tax_8b_str} & {tax_27b_str} & {ratio_str} \\"

        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(data: List[Dict[str, Any]]) -> None:
    """Print human-readable summary to stdout."""
    print()
    print("=" * 80)
    print("  AdaThink Consolidated Results Summary")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Group by model
    models_found = sorted({r["model"] for r in data})
    for model in models_found:
        subset = [r for r in data if r["model"] == model]
        modes_found = sorted({r["mode"] for r in subset})

        print(f"\n--- Model: {model} ---")
        header = (
            f"{'Mode':>10}  {'Budget':>6}  {'Accuracy':>10}  {'AvgTokens':>10}  "
            f"{'EarlyStop':>10}  {'N':>6}  {'Source':>40}"
        )
        print(header)
        print("-" * len(header))

        for mode in modes_found:
            mode_data = sorted(
                [r for r in subset if r["mode"] == mode],
                key=lambda r: r["budget"],
            )
            for r in mode_data:
                acc_str = f"{r['accuracy'] * 100:.1f}%"
                tok_str = f"{r['avg_tokens']:.0f}" if not np.isnan(r["avg_tokens"]) else "--"
                es_str = (
                    f"{r['early_stop_rate'] * 100:.1f}%"
                    if not np.isnan(r["early_stop_rate"]) else "--"
                )
                src = os.path.basename(r["source_file"])[:40]
                print(
                    f"{mode:>10}  {r['budget']:>6}  {acc_str:>10}  {tok_str:>10}  "
                    f"{es_str:>10}  {r['n_samples']:>6}  {src:>40}"
                )

    # Crossover analysis per model
    print("\n--- Crossover Analysis ---")
    for model in models_found:
        cross = find_crossover(data, model)
        cross_interp = interpolate_crossover(data, model)
        if cross is not None:
            print(f"  {model}: Crossover at budget={cross}")
        elif cross_interp is not None:
            print(f"  {model}: Estimated crossover at ~{cross_interp:.0f} tokens (interpolated)")
        else:
            nothink_budgets = [r["budget"] for r in data if r["model"] == model and r["mode"] == "nothink"]
            thinking_budgets = [r["budget"] for r in data if r["model"] == model and r["mode"] == "thinking"]
            if not nothink_budgets:
                print(f"  {model}: No nothink data available, crossover cannot be determined")
            elif not thinking_budgets:
                print(f"  {model}: No thinking data available")
            else:
                print(f"  {model}: No crossover found in available budgets {sorted(set(nothink_budgets) | set(thinking_budgets))}")

    # Data completeness
    print("\n--- Data Completeness ---")
    for model in models_found:
        for mode in MODES:
            budgets_have = sorted({
                r["budget"] for r in data
                if r["model"] == model and r["mode"] == mode
            })
            budgets_missing = [b for b in BUDGETS if b not in budgets_have]
            status = "COMPLETE" if not budgets_missing else f"Missing: {budgets_missing}"
            print(f"  {model:>4} {mode:>10}: have {budgets_have}  {status}")

    # Total records
    print(f"\nTotal data points: {len(data)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate all AdaThink experiment results and generate "
            "paper-ready figures, tables, and a unified JSON dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-kun",
        type=str,
        default=str(PROJECT_ROOT / "results_kun"),
        help="Path to results_kun/ directory (default: %(default)s)",
    )
    parser.add_argument(
        "--crossover-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "crossover"),
        help="Path to results/crossover/ directory (default: %(default)s)",
    )
    parser.add_argument(
        "--nothink-27b-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "fulltest_27b_nothink"),
        help="Path to 27B nothink results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--extra-json",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Additional JSON files to include (overrides earlier data)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="8B",
        choices=MODELS,
        help="Model for crossover figure (default: 8B)",
    )
    parser.add_argument(
        "--output-fig",
        type=str,
        default=str(PROJECT_ROOT / "paper" / "fig_crossover_analysis.pdf"),
        help="Crossover figure output path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default=str(PROJECT_ROOT / "results" / "paper_figures" / "table_crossover.tex"),
        help="Crossover LaTeX table output (default: %(default)s)",
    )
    parser.add_argument(
        "--output-scaling-table",
        type=str,
        default=str(
            PROJECT_ROOT / "results" / "paper_figures"
            / "table_model_size_scaling_updated.tex"
        ),
        help="Model-size scaling LaTeX table output (default: %(default)s)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "results" / "consolidated_results.json"),
        help="Consolidated JSON output (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan and print what was found; do not write any files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Phase 1: Collect data ────────────────────────────────────────────
    log.info("Phase 1: Collecting data from all sources")

    # Layer 0: hardcoded confirmed data (lowest layer, will be overridden)
    log.info("  [0] Loading hardcoded confirmed data points")
    hardcoded_records = list(CONFIRMED_DATA)
    log.info(f"      {len(hardcoded_records)} records")

    # Layer 1: results_kun/
    log.info(f"  [1] Scanning {args.results_kun}")
    results_kun_dir = Path(args.results_kun)
    kun_records = []
    # Scan known subdirectories
    for subdir_name in ["fulltest", "fulltest_27b"]:
        subdir = results_kun_dir / subdir_name
        if subdir.exists():
            recs = scan_directory(subdir)
            kun_records.extend(recs)
            log.info(f"      {subdir_name}: {len(recs)} records")
    # Scan top-level nothink baselines
    for jf in sorted(results_kun_dir.glob("nothink_baseline_*.json")):
        recs = load_any_json(jf)
        kun_records.extend(recs)
        if recs:
            log.info(f"      {jf.name}: {len(recs)} records")
    log.info(f"      Total from results_kun: {len(kun_records)} records")

    # Layer 2: results/crossover/
    log.info(f"  [2] Scanning {args.crossover_dir}")
    crossover_records = scan_directory(Path(args.crossover_dir))
    log.info(f"      {len(crossover_records)} records")

    # Layer 3: results/fulltest_27b_nothink/
    log.info(f"  [3] Scanning {args.nothink_27b_dir}")
    nothink_27b_records = scan_directory(Path(args.nothink_27b_dir))
    log.info(f"      {len(nothink_27b_records)} records")

    # Layer 4: extra JSON files
    extra_records = []
    if args.extra_json:
        log.info(f"  [4] Loading {len(args.extra_json)} extra JSON files")
        for jp in args.extra_json:
            jpath = Path(jp)
            if jpath.exists():
                recs = load_any_json(jpath)
                extra_records.extend(recs)
                log.info(f"      {jpath.name}: {len(recs)} records")
            else:
                log.warning(f"      File not found: {jp}")

    # ── Phase 2: Consolidate ─────────────────────────────────────────────
    log.info("Phase 2: Consolidating data (later layers override earlier)")
    all_data = consolidate([
        hardcoded_records,   # lowest priority
        kun_records,
        crossover_records,
        nothink_27b_records,
        extra_records,       # highest priority
    ])
    log.info(f"  Final dataset: {len(all_data)} unique (model, mode, budget) records")

    # ── Phase 3: Analysis ────────────────────────────────────────────────
    log.info("Phase 3: Crossover analysis")
    crossover_budget = find_crossover(all_data, args.model)
    crossover_interp = interpolate_crossover(all_data, args.model)

    if crossover_budget:
        log.info(f"  Exact crossover at budget={crossover_budget}")
    elif crossover_interp:
        log.info(f"  Interpolated crossover at ~{crossover_interp:.0f} tokens")
    else:
        log.info("  No crossover found in available data")

    # ── Phase 4: Print summary ───────────────────────────────────────────
    print_summary(all_data)

    if args.dry_run:
        log.info("Dry-run mode: no files written.")
        return

    # ── Phase 5: Generate outputs ────────────────────────────────────────
    log.info("Phase 5: Generating output files")

    # 5a: Crossover figure
    log.info(f"  [fig] {args.output_fig}")
    generate_crossover_figure(
        all_data,
        model=args.model,
        crossover_budget=crossover_budget,
        crossover_interp=crossover_interp,
        output_path=Path(args.output_fig),
    )

    # 5b: Crossover LaTeX table
    log.info(f"  [table] {args.output_table}")
    table_tex = generate_crossover_table(all_data, args.model, crossover_budget)
    table_path = Path(args.output_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(table_tex + "\n")
    log.info(f"  Table saved: {table_path}")

    # 5c: Model-size scaling table
    log.info(f"  [table] {args.output_scaling_table}")
    scaling_tex = generate_model_size_table(all_data)
    scaling_path = Path(args.output_scaling_table)
    scaling_path.parent.mkdir(parents=True, exist_ok=True)
    scaling_path.write_text(scaling_tex + "\n")
    log.info(f"  Scaling table saved: {scaling_path}")

    # 5d: Consolidated JSON
    log.info(f"  [json] {args.output_json}")
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    output_obj = {
        "generated_at": datetime.now().isoformat(),
        "n_records": len(all_data),
        "crossover": {
            "model": args.model,
            "exact_budget": crossover_budget,
            "interpolated_budget": (
                round(crossover_interp, 1) if crossover_interp else None
            ),
        },
        "records": all_data,
    }
    # Replace NaN with None for valid JSON
    def sanitize_for_json(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        return obj

    output_obj = sanitize_for_json(output_obj)
    with open(json_path, "w") as f:
        json.dump(output_obj, f, indent=2, default=str)
    log.info(f"  JSON saved: {json_path}")

    # Print table to stdout as well
    print("\n% ──── Crossover Table (LaTeX) ────")
    print(table_tex)
    print("\n% ──── Model-Size Scaling Table (LaTeX) ────")
    print(scaling_tex)
    print()

    log.info("Done. All outputs written successfully.")


if __name__ == "__main__":
    main()
