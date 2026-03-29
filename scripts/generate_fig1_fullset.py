#!/usr/bin/env python3
"""
Generate Figure 1: NoThink vs Thinking at matched token budgets.

Full GSM8K (n=1319) data with Qwen3-8B.
Produces a grouped bar chart showing accuracy and average token usage,
with "thinking tax" gap annotations.

Usage:
    python scripts/generate_fig1_fullset.py
Output:
    results/paper_figures/fig_nothink_vs_thinking_fullset.pdf
"""

import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Data ─────────────────────────────────────────────────────────────────────
# Full GSM8K n=1319 (Qwen3-8B) unless noted
BUDGETS = [128, 256, 512]

FULLSET = {
    "nothink": {
        128: {"acc": 50.8, "tok": 113},
        256: {"acc": 87.5, "tok": 146},
        512: {"acc": 94.0, "tok": 145, "note": "†"},  # 200-sample
    },
    "thinking": {
        128: {"acc": 3.0,  "tok": 128},
        256: {"acc": 18.0, "tok": 255},
        512: {"acc": 65.2, "tok": 460},
    },
}

# 200-sample reference (not plotted, kept for record)
SAMPLE200 = {
    "nothink": {128: {"acc": 54.5, "tok": 111}, 256: {"acc": 89.0, "tok": 140}, 512: {"acc": 94.0, "tok": 145}},
    "thinking": {128: {"acc": 2.0,  "tok": 128}, 256: {"acc": 22.0, "tok": 255}, 512: {"acc": 66.5, "tok": 442}},
}


def make_figure(outpath: str) -> None:
    # ── Style ────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,       # TrueType in PDF
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    n_budgets = len(BUDGETS)
    x = np.arange(n_budgets)
    bar_w = 0.32
    gap = 0.04  # half-gap between the two bars in a group

    # Colors — colorblind-friendly
    c_nothink = "#4878CF"   # steel blue
    c_think   = "#E8752A"   # burnt orange

    nothink_accs = [FULLSET["nothink"][b]["acc"] for b in BUDGETS]
    think_accs   = [FULLSET["thinking"][b]["acc"] for b in BUDGETS]
    nothink_toks = [FULLSET["nothink"][b]["tok"] for b in BUDGETS]
    think_toks   = [FULLSET["thinking"][b]["tok"] for b in BUDGETS]

    bars_nt = ax.bar(x - bar_w / 2 - gap, nothink_accs, bar_w,
                     label="NoThink", color=c_nothink, edgecolor="white", linewidth=0.5, zorder=3)
    bars_tk = ax.bar(x + bar_w / 2 + gap, think_accs, bar_w,
                     label="Thinking", color=c_think, edgecolor="white", linewidth=0.5, zorder=3)

    # ── Annotations: avg tokens on each bar ──────────────────────────────
    def annotate_bar(bar, tok_val, note=""):
        """Place avg-token text inside/above each bar."""
        bx = bar.get_x() + bar.get_width() / 2
        by = bar.get_height()
        label = f"{tok_val} tok{note}"
        # Place inside if bar is tall enough, else above
        if by > 18:
            ax.text(bx, by - 3, label, ha="center", va="top",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)
        else:
            ax.text(bx, by + 1.5, label, ha="center", va="bottom",
                    fontsize=7.5, color="black", zorder=4)

    for i, b in enumerate(BUDGETS):
        note_nt = FULLSET["nothink"][b].get("note", "")
        annotate_bar(bars_nt[i], nothink_toks[i], note_nt)
        annotate_bar(bars_tk[i], think_toks[i])

    # ── Thinking-tax gap arrows ──────────────────────────────────────────
    for i, b in enumerate(BUDGETS):
        gap_val = nothink_accs[i] - think_accs[i]
        if gap_val <= 0:
            continue
        x_mid = x[i]
        y_top = nothink_accs[i]
        y_bot = think_accs[i]
        y_mid = (y_top + y_bot) / 2

        # Draw a bracket / double-arrow between the two bar tops
        ax.annotate(
            "", xy=(x_mid, y_top), xytext=(x_mid, y_bot),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0,
                            shrinkA=2, shrinkB=2),
            zorder=5,
        )
        # Gap label
        ax.text(x_mid + 0.01, y_mid, f"Δ{gap_val:.1f}%",
                ha="center", va="center", fontsize=8, color="#333333",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=6)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xlabel("Token Budget", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))

    # Light grid
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#cccccc",
              fontsize=10, handlelength=1.2)

    # Title
    ax.set_title("NoThink vs Thinking at Matched Budgets\n(Qwen3-8B, GSM8K full set $n$=1319)",
                 fontsize=11.5, pad=10)

    # Footnote for dagger
    fig.text(0.13, 0.01,
             "† nothink@512 uses 200-sample estimate (full-set run pending).",
             fontsize=7.5, color="#666666", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    logger.info("Saved figure → %s", outpath)

    # Also save PNG for quick preview
    png_path = outpath.replace(".pdf", ".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    logger.info("Saved preview → %s", png_path)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 1: NoThink vs Thinking")
    parser.add_argument("--outpath", type=str,
                        default="results/paper_figures/fig_nothink_vs_thinking_fullset.pdf",
                        help="Output PDF path")
    args = parser.parse_args()

    # Resolve relative path from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    outpath = os.path.join(project_root, args.outpath) if not os.path.isabs(args.outpath) else args.outpath

    logger.info("Generating Figure 1 (NoThink vs Thinking) ...")
    logger.info("  Budgets: %s", BUDGETS)
    for mode in ["nothink", "thinking"]:
        for b in BUDGETS:
            d = FULLSET[mode][b]
            note = d.get("note", "")
            logger.info("  %s@%d: acc=%.1f%%, tok=%d %s", mode, b, d["acc"], d["tok"], note)

    make_figure(outpath)
    logger.info("Done.")


if __name__ == "__main__":
    main()
