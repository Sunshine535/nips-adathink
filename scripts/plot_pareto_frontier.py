#!/usr/bin/env python3
"""
Plot Pareto frontier: accuracy vs. average tokens for all methods on MATH-500.
Generates fig_pareto_frontier.pdf for the paper.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (5.5, 4.0),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Data points: (avg_tokens, accuracy, label)
# From paper tables (experiments_final.tex, compute-matched table)

nothink = [
    (418, 47.5, 'nothink@512'),
    (600, 59.8, 'nothink@1024'),
]

think = [
    (256, 3.0, 'think@256'),
    (477, 6.2, 'think@512'),
    (1024, 18.0, 'think@1024'),
    (1978, 44.0, 'think@2048'),
]

town = [
    (961, 49.0, 'TOWN@1024'),
    (1590, 55.0, 'TOWN@2048'),
    (2565, 71.8, 'TOWN@4096'),
]

iris = [
    (987, 62.5, 'IRIS@1024'),
    (1573, 67.2, 'IRIS@2048'),
    (2401, 74.0, 'IRIS@4096'),
]

fig, ax = plt.subplots()

# Plot each method
for data, color, marker, label, zorder in [
    (think, '#E67E22', 'v', 'Think', 2),
    (nothink, '#3498DB', 'o', 'Nothink', 3),
    (town, '#27AE60', 's', 'TOWN', 4),
    (iris, '#E74C3C', '*', 'IRIS (split-budget)', 5),
]:
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    ms = 10 if marker == '*' else 7
    ax.scatter(xs, ys, c=color, marker=marker, s=ms**2, label=label,
               zorder=zorder, edgecolors='white', linewidth=0.5)
    # Connect points with thin line
    ax.plot(xs, ys, c=color, alpha=0.4, linewidth=1.2, zorder=zorder-1)

# Draw Pareto frontier (connecting Pareto-optimal points)
# Pareto-optimal: nothink@512 (418, 47.5), nothink@1024 (600, 59.8),
# IRIS@1024 (987, 62.5), IRIS@2048 (1573, 67.2), IRIS@4096 (2401, 74.0)
pareto_x = [418, 600, 987, 1573, 2401]
pareto_y = [47.5, 59.8, 62.5, 67.2, 74.0]
ax.plot(pareto_x, pareto_y, 'k--', alpha=0.3, linewidth=1.5, zorder=1,
        label='Pareto frontier')

# Annotate key points
offsets = {
    'nothink@512': (-10, 8),
    'nothink@1024': (-10, 8),
    'think@256': (8, -5),
    'think@1024': (8, 2),
    'think@2048': (8, 2),
    'TOWN@2048': (8, -10),
    'TOWN@4096': (8, -10),
    'IRIS@1024': (-60, -14),
    'IRIS@2048': (-60, -14),
    'IRIS@4096': (8, 4),
}
for data_list in [nothink, think, town, iris]:
    for x, y, label in data_list:
        if label in offsets:
            dx, dy = offsets[label]
            short = label.replace('nothink', 'NT').replace('think', 'T')
            ax.annotate(short, (x, y), xytext=(dx, dy),
                       textcoords='offset points', fontsize=7.5, alpha=0.7)

ax.set_xlabel('Average tokens per sample')
ax.set_ylabel('Accuracy (%)')
ax.set_title('MATH-500 Accuracy vs. Token Cost (Qwen3-8B)')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(0, 82)
ax.set_xlim(0, 2800)
ax.grid(True, alpha=0.2)

plt.savefig('paper/fig_pareto_frontier.pdf')
plt.savefig('paper/fig_pareto_frontier.png')
print("Saved: paper/fig_pareto_frontier.pdf")
