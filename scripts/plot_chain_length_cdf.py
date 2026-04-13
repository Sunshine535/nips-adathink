#!/usr/bin/env python3
"""
Plot chain-length CDF F_L(b) for 8B and 9B on GSM8K.
Shows stochastic dominance: 9B chains are longer → F_L shifts right → higher truncation → larger tax.
Uses per-sample data with right-censoring correction via Kaplan-Meier.
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'figure.figsize': (5.5, 4.0),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

def load_chain_lengths(filepath, budget_key):
    """Load per-sample chain lengths and censoring flags from JSON."""
    with open(filepath) as f:
        data = json.load(f)
    ps = data["per_sample"]
    if isinstance(ps, dict):
        samples = ps[budget_key]
    else:
        samples = ps  # flat list (e.g., 9B single-budget files)
    lengths = [s["tokens"] for s in samples]
    censored = [s["hit_budget"] for s in samples]  # True = right-censored
    return np.array(lengths), np.array(censored)

def kaplan_meier_cdf(lengths, censored):
    """Compute Kaplan-Meier estimate of CDF from right-censored data."""
    n = len(lengths)
    # Sort by length
    order = np.argsort(lengths)
    lengths_sorted = lengths[order]
    censored_sorted = censored[order]

    # Kaplan-Meier survival function
    unique_times = np.unique(lengths_sorted[~censored_sorted])
    survival = 1.0
    times = [0]
    survivals = [1.0]

    for t in unique_times:
        at_risk = np.sum(lengths_sorted >= t)
        events = np.sum((lengths_sorted == t) & (~censored_sorted))
        if at_risk > 0:
            survival *= (1 - events / at_risk)
        times.append(t)
        survivals.append(survival)

    times = np.array(times)
    survivals = np.array(survivals)
    cdf = 1 - survivals
    return times, cdf

def empirical_cdf(lengths):
    """Simple empirical CDF (no censoring)."""
    sorted_lengths = np.sort(lengths)
    n = len(sorted_lengths)
    cdf = np.arange(1, n + 1) / n
    return sorted_lengths, cdf

# Paths
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 8B: thinking@2048, n=1319 (92.8% natural stops)
path_8b = os.path.join(base, "results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260408_103622.json")

# 9B: thinking@2048, n=1319 (56.3% natural stops)
path_9b = os.path.join(base, "results_kun/thinking_hf/qwen35_9b_2048/thinking_2048_20260401_085348.json")

fig, ax = plt.subplots()

# Budget range for x-axis
budgets = np.arange(0, 2200, 10)

for path, key, label, color, ls in [
    (path_8b, "thinking_2048", "Qwen3-8B", '#3498DB', '-'),
    (path_9b, "thinking_2048", "Qwen3.5-9B", '#E74C3C', '--'),
]:
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping {label}")
        continue

    lengths, censored = load_chain_lengths(path, key)
    n_total = len(lengths)
    n_censored = censored.sum()
    n_natural = n_total - n_censored
    print(f"{label}: n={n_total}, natural_stops={n_natural} ({n_natural/n_total*100:.1f}%), "
          f"censored={n_censored} ({n_censored/n_total*100:.1f}%)")
    print(f"  Chain lengths: mean={lengths[~censored].mean():.0f}, "
          f"median={np.median(lengths[~censored]):.0f}, "
          f"p95={np.percentile(lengths[~censored], 95):.0f}")

    if n_censored > 0:
        times, cdf = kaplan_meier_cdf(lengths, censored)
    else:
        times, cdf = empirical_cdf(lengths)

    ax.plot(times, cdf, color=color, linestyle=ls, linewidth=2, label=label)

# Add vertical lines at key budgets
for b, ls_b in [(256, ':'), (512, ':'), (1024, ':')]:
    ax.axvline(x=b, color='gray', linestyle=ls_b, alpha=0.3, linewidth=0.8)
    ax.text(b + 15, 0.02, f'b={b}', fontsize=7.5, color='gray', alpha=0.7)

# Add horizontal arrow showing stochastic dominance
ax.annotate('', xy=(1400, 0.55), xytext=(700, 0.55),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))
ax.text(1050, 0.58, 'Stochastic\ndominance', fontsize=8, color='gray',
        alpha=0.6, ha='center')

ax.set_xlabel('Chain length $L$ (tokens)')
ax.set_ylabel('$F_L(b) = \\Pr(L \\leq b)$')
ax.set_title('Chain-Length CDF: Larger Models Have Longer Chains')
ax.legend(loc='lower right')
ax.set_xlim(0, 2100)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.2)

plt.savefig(os.path.join(base, 'paper/fig_chain_length_cdf.pdf'))
plt.savefig(os.path.join(base, 'paper/fig_chain_length_cdf.png'))
print("\nSaved: paper/fig_chain_length_cdf.pdf")
