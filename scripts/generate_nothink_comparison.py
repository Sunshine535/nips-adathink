#!/usr/bin/env python3
"""Generate publication-quality figure: Non-thinking vs Thinking mode comparison.

Key finding: Non-thinking mode (TOWN) is dramatically more efficient than
thinking mode at matched token budgets on Qwen3-8B / GSM8K.

Outputs:
  results/paper_figures/fig_nothink_vs_thinking.pdf
  results/paper_figures/fig_nothink_vs_thinking.png
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

# ---------------------------------------------------------------------------
# Style — NeurIPS-quality, serif, 300 DPI
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 13,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data — Qwen3-8B, GSM8K, 200 samples, seed=42
# ---------------------------------------------------------------------------
# Non-thinking mode (TOWN)
NOTHINK_BUDGETS    = np.array([32,   64,   128,   256,   512])
NOTHINK_ACC        = np.array([3.0,  12.0, 54.5,  89.0,  94.0])
NOTHINK_AVG_TOK    = np.array([32,   64,   111,   140,   145])
NOTHINK_EARLY_STOP = np.array([0.0,  2.0,  43.5,  92.0,  99.5])

# Thinking mode (FINAL data from recovery Phase R1)
THINK_BUDGETS   = np.array([128,  256,   512])
THINK_ACC       = np.array([2.0,  22.0,  66.5])   # FINAL data from recovery Phase R1
THINK_AVG_TOK   = np.array([128,  255,   442])
THINK_EARLY_STOP = np.array([0.0,  0.0,  10.0])   # rough estimate for 512

# Colours
C_NOTHINK = "#27ae60"   # green
C_THINK   = "#c0392b"   # red
C_SHADE   = "#e8f5e9"   # light green for "thinking tax"
C_PARETO  = "#1565c0"   # blue for Pareto
C_TOWN    = "#ff8f00"   # amber for TOWN operating point

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotate_gap(ax, x, y_top, y_bot, label, offset_x=0, fontsize=9):
    """Draw a double-headed arrow + label between two points at the same x."""
    ax.annotate(
        "", xy=(x, y_top), xytext=(x, y_bot),
        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.2,
                        shrinkA=2, shrinkB=2),
    )
    mid_y = (y_top + y_bot) / 2
    ax.text(x + offset_x, mid_y, label,
            ha="left", va="center", fontsize=fontsize,
            color="0.2", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7",
                      alpha=0.9))


def pareto_frontier(xs, ys):
    """Return indices on the Pareto frontier (lower x, higher y is better)."""
    order = np.argsort(xs)
    xs_s, ys_s = xs[order], ys[order]
    frontier_idx = [0]
    best_y = ys_s[0]
    for i in range(1, len(xs_s)):
        if ys_s[i] > best_y:
            frontier_idx.append(i)
            best_y = ys_s[i]
    return order[frontier_idx]


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(outdir: str):
    os.makedirs(outdir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.32)

    # =======================================================================
    # LEFT — Accuracy vs Token Budget (log scale)
    # =======================================================================
    ax1.plot(NOTHINK_BUDGETS, NOTHINK_ACC,
             "o-", color=C_NOTHINK, lw=2.2, ms=8, zorder=5,
             label="Non-thinking (TOWN)")
    ax1.plot(THINK_BUDGETS, THINK_ACC,
             "s--", color=C_THINK, lw=2.2, ms=8, zorder=5,
             label="Thinking")

    # Shade the "thinking tax" region between the two curves at shared budgets
    shared_budgets = np.array([128, 256, 512])
    nothink_at_shared = np.array([54.5, 89.0, 94.0])
    think_at_shared   = np.array([2.0,  22.0, 66.5])
    ax1.fill_between(shared_budgets, think_at_shared, nothink_at_shared,
                     color=C_SHADE, alpha=0.55, zorder=1,
                     label="Efficiency gap")

    # Annotate the huge gap at budget=256
    _annotate_gap(ax1, 256, 89.0, 22.0,
                  " 63 pp gap\n at budget 256",
                  offset_x=12, fontsize=9)

    # Mark the placeholder point
    ax1.annotate("", xy=(256, 22), fontsize=8,  # was placeholder, now final
                 color=C_THINK, alpha=0.7,
                 xytext=(180, 16), ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_THINK, alpha=0.5,
                                 lw=0.8))

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(NOTHINK_BUDGETS)
    ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax1.set_xlim(24, 700)
    ax1.set_ylim(-2, 102)
    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a)  Accuracy vs Token Budget")
    ax1.legend(loc="upper left", frameon=True)

    # =======================================================================
    # RIGHT — Accuracy vs Actual Tokens Used (efficiency view)
    # =======================================================================
    # Combine all operating points
    all_tok = np.concatenate([NOTHINK_AVG_TOK, THINK_AVG_TOK])
    all_acc = np.concatenate([NOTHINK_ACC, THINK_ACC])
    all_labels = (["NT@{}".format(b) for b in NOTHINK_BUDGETS] +
                  ["T@{}".format(b) for b in THINK_BUDGETS])
    all_colors = [C_NOTHINK] * len(NOTHINK_BUDGETS) + [C_THINK] * len(THINK_BUDGETS)

    # Scatter all points
    ax2.scatter(NOTHINK_AVG_TOK, NOTHINK_ACC, c=C_NOTHINK, s=90,
                marker="o", zorder=5, edgecolors="white", linewidths=0.8,
                label="Non-thinking (TOWN)")
    ax2.scatter(THINK_AVG_TOK, THINK_ACC, c=C_THINK, s=90,
                marker="s", zorder=5, edgecolors="white", linewidths=0.8,
                label="Thinking")

    # Label each point
    nudge = {
        "NT@32":  (6, -9),
        "NT@64":  (6, -9),
        "NT@128": (6, -9),
        "NT@256": (-50, -12),
        "NT@512": (6, 5),
        "T@128":  (6, 5),
        "T@256":  (6, -9),
        "T@512":  (-18, -12),
    }
    for tok, acc, lbl, col in zip(all_tok, all_acc, all_labels, all_colors):
        dx, dy = nudge.get(lbl, (5, 3))
        ax2.annotate(lbl, (tok, acc), fontsize=8.5, color=col,
                     fontweight="semibold",
                     textcoords="offset points", xytext=(dx, dy))

    # Pareto frontier (from all points, lower tok + higher acc is better)
    pf_idx = pareto_frontier(all_tok, all_acc)
    pf_tok = all_tok[pf_idx]
    pf_acc = all_acc[pf_idx]
    sort_ord = np.argsort(pf_tok)
    pf_tok, pf_acc = pf_tok[sort_ord], pf_acc[sort_ord]
    # Extend to axis edges for visual clarity
    ax2.plot(pf_tok, pf_acc, "-", color=C_PARETO, lw=1.5, alpha=0.6,
             zorder=3, label="Pareto frontier")
    ax2.fill_between(pf_tok, pf_acc, alpha=0.07, color=C_PARETO, zorder=1)

    # Highlight TOWN operating point (NT@256)
    town_tok, town_acc = 140, 89.0
    ax2.scatter([town_tok], [town_acc], s=260, facecolors="none",
                edgecolors=C_TOWN, linewidths=2.5, zorder=6)
    ax2.annotate("TOWN\noperating point",
                 xy=(town_tok, town_acc),
                 xytext=(town_tok + 130, town_acc - 35),
                 fontsize=9.5, fontweight="bold", color=C_TOWN,
                 ha="center",
                 arrowprops=dict(arrowstyle="-|>", color=C_TOWN,
                                 lw=1.5, shrinkB=8))

    # Draw efficiency comparison arrow: T@512 → NT@256
    ax2.annotate(
        "",
        xy=(140, 89.0), xytext=(442, 66.5),
        arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.3,
                        linestyle="dashed", shrinkA=10, shrinkB=14),
    )
    ax2.text(310, 72, "3.2× fewer tokens\n+22.5 pp accuracy",
             fontsize=8.5, ha="center", va="top", color="0.3",
             fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                       alpha=0.85))

    ax2.set_xlim(0, 520)
    ax2.set_ylim(-2, 102)
    ax2.set_xlabel("Average Tokens Actually Used")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b)  Accuracy vs Actual Token Usage")
    ax2.legend(loc="lower right", frameon=True)

    # =======================================================================
    # Save
    # =======================================================================
    pdf_path = os.path.join(outdir, "fig_nothink_vs_thinking.pdf")
    png_path = os.path.join(outdir, "fig_nothink_vs_thinking.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    log.info("Saved  %s", pdf_path)
    log.info("Saved  %s", png_path)
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate non-thinking vs thinking comparison figure.")
    parser.add_argument("--outdir", type=str,
                        default="results/paper_figures",
                        help="Output directory for figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    log.info("Generating nothink vs thinking comparison figure …")
    pdf, png = make_figure(args.outdir)
    log.info("Done. Files:\n  %s\n  %s", pdf, png)


if __name__ == "__main__":
    main()
