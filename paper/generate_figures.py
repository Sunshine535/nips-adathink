"""Generate all paper figures from experiment results."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 8.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

RESULTS_DIR = os.environ.get('RESULTS_DIR', '../results')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'fixed': '#7f8c8d',
    'adaptive': '#e67e22',
    'template': '#2ecc71',
    'parametric': '#3498db',
    'value': '#9b59b6',
    'oracle': '#e74c3c',
    'sc': '#1abc9c',
}
MARKERS = {
    'fixed': 's',
    'adaptive': 'D',
    'template': 'o',
    'parametric': '^',
    'value': 'v',
    'oracle': '*',
}


def fig1_pareto_curves():
    """Pareto frontier: accuracy vs tokens across benchmarks (27B)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    benchmarks = [
        {
            'name': 'GSM8K (27B)',
            'fixed': [(158.3, 0.337), (286.3, 0.462), (542.7, 0.487)],
            'fixed_labels': ['128', '256', '512'],
            'adaptive': [(542.5, 0.508)],
            'template': [(380.1, 0.604)],
            'parametric': [(480.1, 0.570)],
            'oracle': [(238.9, 0.648)],
        },
        {
            'name': 'MATH500 (27B)',
            'fixed': [(1972.5, 0.226), (3570.4, 0.371), (5976.4, 0.541)],
            'fixed_labels': ['2048', '4096', '8192'],
            'adaptive': [(5066.6, 0.485)],
            'template': [(6181.9, 0.491)],
            'parametric': [(7061.0, 0.525)],
            'oracle': [(3053.1, 0.582)],
        },
        {
            'name': 'BBH (27B)',
            'fixed': [(988.9, 0.247), (1624.1, 0.482), (2239.0, 0.600)],
            'fixed_labels': ['1024', '2048', '4096'],
            'adaptive': [(1792.9, 0.515)],
            'template': [(2583.1, 0.596)],
            'parametric': [(2810.7, 0.588)],
            'oracle': [(1381.7, 0.607)],
        },
    ]

    for ax, bm in zip(axes, benchmarks):
        fx, fy = zip(*bm['fixed'])
        ax.plot(fx, fy, '-s', color=COLORS['fixed'], markersize=7,
                label='Fixed budget', zorder=3)
        for (x, y), lbl in zip(bm['fixed'], bm['fixed_labels']):
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(0, 8), fontsize=7, ha='center', color=COLORS['fixed'])

        for key, label in [('adaptive', 'Naive adaptive'),
                           ('template', 'Template ctrl.'),
                           ('parametric', 'Parametric ctrl.'),
                           ('oracle', 'Oracle')]:
            if key in bm:
                pts = bm[key]
                x, y = zip(*pts)
                ax.scatter(x, y, marker=MARKERS.get(key, 'o'), s=70,
                           color=COLORS[key], label=label, zorder=4, edgecolors='white', linewidths=0.5)

        ax.set_xlabel('End-to-end tokens')
        ax.set_ylabel('Accuracy')
        ax.set_title(bm['name'])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_pareto_curves.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_pareto_curves.png'))
    plt.close()
    print('Generated fig1_pareto_curves')


def fig2_pareto_8b():
    """Pareto frontier for 8B models."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    benchmarks_8b = [
        {
            'name': 'MATH500 (8B)',
            'fixed': [(519.1, 0.006), (1030.0, 0.086), (1972.1, 0.436)],
            'fixed_labels': ['512', '1024', '2048'],
            'adaptive': [(1968.0, 0.425)],
            'template': [(1841.3, 0.375)],
            'parametric': [(2321.6, 0.436)],
            'oracle': [(1048.0, 0.439)],
        },
        {
            'name': 'BBH (8B)',
            'fixed': [(254.7, 0.036), (478.0, 0.171), (773.1, 0.325)],
            'fixed_labels': ['256', '512', '1024'],
            'adaptive': [(767.9, 0.321)],
            'template': [(787.5, 0.293)],
            'parametric': [(783.2, 0.304)],
            'oracle': [(368.0, 0.346)],
        },
    ]

    for ax, bm in zip(axes, benchmarks_8b):
        fx, fy = zip(*bm['fixed'])
        ax.plot(fx, fy, '-s', color=COLORS['fixed'], markersize=7,
                label='Fixed budget', zorder=3)
        for (x, y), lbl in zip(bm['fixed'], bm['fixed_labels']):
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(0, 8), fontsize=7, ha='center', color=COLORS['fixed'])

        for key, label in [('adaptive', 'Naive adaptive'),
                           ('template', 'Template ctrl.'),
                           ('parametric', 'Parametric ctrl.'),
                           ('oracle', 'Oracle')]:
            if key in bm:
                pts = bm[key]
                x, y = zip(*pts)
                ax.scatter(x, y, marker=MARKERS.get(key, 'o'), s=70,
                           color=COLORS[key], label=label, zorder=4,
                           edgecolors='white', linewidths=0.5)

        ax.set_xlabel('End-to-end tokens')
        ax.set_ylabel('Accuracy')
        ax.set_title(bm['name'])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc='lower right', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_pareto_8b.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_pareto_8b.png'))
    plt.close()
    print('Generated fig2_pareto_8b')


def fig3_ablation_heatmap():
    """Ablation heatmap: utility across variants and benchmarks."""
    variants = ['Full', 'Halting-only', 'No-branch', 'Max-only', 'Mid-only']
    benchmarks = ['GSM8K\n27B', 'MATH500\n27B', 'BBH\n27B', 'MATH500\n8B', 'BBH\n8B']

    utility = np.array([
        [0.541, 0.409, 0.524, 0.270, 0.213],
        [0.440, 0.427, 0.530, 0.292, 0.226],
        [0.467, 0.294, 0.416, -0.005, 0.077],
        [0.308, 0.432, 0.518, 0.292, 0.212],
        [0.400, 0.305, 0.423, 0.011, 0.101],
    ])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(utility, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.6)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, fontsize=9)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=9)

    for i in range(len(variants)):
        for j in range(len(benchmarks)):
            val = utility[i, j]
            color = 'white' if val < 0.1 or val > 0.45 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if i == 0 else 'normal')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Utility ($\\lambda=0.15$)', fontsize=9)
    ax.set_title('Controller Ablation: Utility Across Settings', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_ablation_heatmap.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_ablation_heatmap.png'))
    plt.close()
    print('Generated fig3_ablation_heatmap')


def fig4_penalty_sweep():
    """Value controller penalty sweep on GSM8K-8B."""
    penalties = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2]
    delta_acc = [0.136, 0.046, 0.046, 0.046, 0.046, 0.046]
    delta_tokens = [86.7, 11.7, 11.7, 11.7, 11.7, 11.7]
    delta_utility = [0.110, 0.043, 0.043, 0.043, 0.043, 0.043]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(penalties, [d*100 for d in delta_acc], '-o', color=COLORS['template'],
                  markersize=7, label='$\\Delta$Acc (%)', zorder=3)
    l2 = ax2.plot(penalties, delta_tokens, '-^', color=COLORS['value'],
                  markersize=7, label='$\\Delta$Tokens', zorder=3)

    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xlabel('Penalty ($\\rho$)')
    ax1.set_ylabel('$\\Delta$Accuracy (%)', color=COLORS['template'])
    ax2.set_ylabel('$\\Delta$Tokens', color=COLORS['value'])
    ax1.tick_params(axis='y', labelcolor=COLORS['template'])
    ax2.tick_params(axis='y', labelcolor=COLORS['value'])

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)

    ax1.set_title('Value Controller Penalty Sweep (GSM8K-8B)')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_penalty_sweep.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_penalty_sweep.png'))
    plt.close()
    print('Generated fig4_penalty_sweep')


def fig5_significance_forest():
    """Forest plot of paired deltas with CIs across all settings."""
    settings = [
        ('GSM8K-27B', 0.142, 0.120, 0.165),
        ('MATH500-27B', 0.121, 0.088, 0.153),
        ('BBH-27B', 0.113, 0.088, 0.140),
        ('MATH500-8B', 0.289, 0.242, 0.339),
        ('BBH-8B', 0.121, 0.075, 0.168),
    ]

    fig, ax = plt.subplots(figsize=(6, 3))
    y_pos = list(range(len(settings)))

    for i, (name, mean, lo, hi) in enumerate(settings):
        color = COLORS['template'] if '27B' in name else COLORS['parametric']
        ax.errorbar(mean * 100, i, xerr=[[mean*100-lo*100], [hi*100-mean*100]],
                     fmt='o', color=color, markersize=8, capsize=4, capthick=1.5,
                     elinewidth=1.5, zorder=3)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=1.5, color='red', linestyle=':', alpha=0.4, label='Success criterion (+1.5%)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels([s[0] for s in settings])
    ax.set_xlabel('$\\Delta$Accuracy (%) vs. matched fixed baseline')
    ax.set_title('Template Controller: Paired Accuracy Gains')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_significance_forest.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_significance_forest.png'))
    plt.close()
    print('Generated fig5_significance_forest')


if __name__ == '__main__':
    fig1_pareto_curves()
    fig2_pareto_8b()
    fig3_ablation_heatmap()
    fig4_penalty_sweep()
    fig5_significance_forest()
    print(f'\nAll figures saved to {OUTPUT_DIR}/')
