#!/usr/bin/env python3
"""Generate core paper figures for 'Natural Stop as Confidence Oracle'.

Figures:
1. Accuracy vs Token Budget (scaling curve)
2. Natural-stop rate and accuracy by budget
3. Difficulty distribution (easy/medium/hard/impossible pie)
4. Confidence calibration: stop vs hit-budget accuracy
5. Token waste analysis: where do wasted tokens go?
6. Selective prediction: accuracy-coverage curve
7. Cross-model/cross-benchmark comparison (placeholder)
"""

import argparse
import csv
import json
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'easy': '#2ecc71',
    'medium': '#f39c12',
    'hard': '#e74c3c',
    'impossible': '#95a5a6',
    'stop': '#27ae60',
    'hit': '#c0392b',
    'fixed': '#3498db',
    'adaptive': '#e67e22',
}


def load_fulltest_csv(csv_path):
    """Load the full GSM8K per-sample CSV."""
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def get_int(r, key, default=0):
    v = r.get(key)
    if v is None or v == '' or v == 'None':
        return default
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return default


def is_true(val):
    if val is None:
        return False
    return str(val).strip().lower() in ('1', 'true', 'yes')


def categorize_samples(rows):
    """Categorize each sample by difficulty."""
    categories = []
    for r in rows:
        c128 = is_true(r.get("fixed_128_correct"))
        c256 = is_true(r.get("fixed_256_correct"))
        c512 = is_true(r.get("fixed_512_correct"))
        if c128:
            categories.append('easy')
        elif c256:
            categories.append('medium')
        elif c512:
            categories.append('hard')
        else:
            categories.append('impossible')
    return categories


def get_main_tokens(r, budget):
    """Get main tokens (excluding projection)."""
    total = get_int(r, f"fixed_{budget}_tokens", budget)
    if is_true(r.get(f"fixed_{budget}_projection_used")):
        proj = get_int(r, f"fixed_{budget}_projection_tokens", 0)
        return total - proj
    return total


def fig1_accuracy_vs_budget(rows, out_dir):
    """Figure 1: Accuracy vs token budget."""
    budgets = [128, 256, 512]
    accs = []
    avg_toks = []
    for b in budgets:
        acc = sum(1 for r in rows if is_true(r.get(f"fixed_{b}_correct"))) / len(rows)
        tok = sum(get_int(r, f"fixed_{b}_tokens", b) for r in rows) / len(rows)
        accs.append(acc)
        avg_toks.append(tok)

    fig, ax1 = plt.subplots()
    ax1.plot(budgets, [a*100 for a in accs], 'o-', color=COLORS['fixed'], linewidth=2, markersize=8, label='Accuracy')
    ax1.set_xlabel('Token Budget')
    ax1.set_ylabel('Accuracy (%)', color=COLORS['fixed'])
    ax1.set_ylim(0, 100)
    ax1.set_xticks(budgets)
    ax1.tick_params(axis='y', labelcolor=COLORS['fixed'])

    ax2 = ax1.twinx()
    ax2.bar([b+20 for b in budgets], avg_toks, width=40, alpha=0.3, color=COLORS['adaptive'], label='Avg Tokens')
    ax2.set_ylabel('Avg Tokens Used', color=COLORS['adaptive'])
    ax2.tick_params(axis='y', labelcolor=COLORS['adaptive'])

    fig.suptitle('Accuracy and Token Usage vs Budget\n(Qwen3-8B, GSM8K, n=1319)', fontsize=13)
    fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.85))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_accuracy_vs_budget.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig1_accuracy_vs_budget.png'))
    plt.close(fig)
    print("Generated: fig1_accuracy_vs_budget")


def fig2_natural_stop_oracle(rows, out_dir):
    """Figure 2: Natural stop as confidence oracle."""
    budgets = [128, 256, 512]
    data = {'budget': [], 'stop_rate': [], 'stop_acc': [], 'hit_acc': []}

    for b in budgets:
        tokens = [get_main_tokens(r, b) for r in rows]
        threshold = b * 0.95

        es = [(r, t) for r, t in zip(rows, tokens) if t < threshold]
        hb = [(r, t) for r, t in zip(rows, tokens) if t >= threshold]

        data['budget'].append(b)
        data['stop_rate'].append(len(es) / len(rows))
        data['stop_acc'].append(
            sum(1 for r, _ in es if is_true(r.get(f"fixed_{b}_correct"))) / max(len(es), 1)
        )
        data['hit_acc'].append(
            sum(1 for r, _ in hb if is_true(r.get(f"fixed_{b}_correct"))) / max(len(hb), 1)
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stop rate by budget
    ax1.bar(range(len(budgets)), [r*100 for r in data['stop_rate']],
            color=COLORS['stop'], alpha=0.8)
    ax1.set_xticks(range(len(budgets)))
    ax1.set_xticklabels([f'Budget={b}' for b in budgets])
    ax1.set_ylabel('Natural Stop Rate (%)')
    ax1.set_title('Natural Early-Stop Rate by Budget')
    for i, v in enumerate(data['stop_rate']):
        ax1.text(i, v*100+1, f'{v:.0%}', ha='center', fontweight='bold')

    # Right: accuracy comparison
    x = np.arange(len(budgets))
    width = 0.35
    ax2.bar(x - width/2, [a*100 for a in data['stop_acc']], width,
            label='Natural Stop', color=COLORS['stop'], alpha=0.8)
    ax2.bar(x + width/2, [a*100 for a in data['hit_acc']], width,
            label='Hit Budget', color=COLORS['hit'], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Budget={b}' for b in budgets])
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy: Natural Stop vs Hit Budget')
    ax2.legend()
    ax2.set_ylim(0, 105)

    for i in range(len(budgets)):
        ax2.text(i - width/2, data['stop_acc'][i]*100+1.5,
                 f'{data["stop_acc"][i]:.0%}', ha='center', fontsize=10, fontweight='bold')
        ax2.text(i + width/2, data['hit_acc'][i]*100+1.5,
                 f'{data["hit_acc"][i]:.0%}', ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Natural Stopping as a Free Confidence Oracle', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_natural_stop_oracle.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig2_natural_stop_oracle.png'))
    plt.close(fig)
    print("Generated: fig2_natural_stop_oracle")


def fig3_difficulty_distribution(rows, out_dir):
    """Figure 3: Difficulty distribution pie chart."""
    cats = categorize_samples(rows)
    counts = Counter(cats)
    n = len(rows)

    labels = ['Easy\n(correct@128)', 'Medium\n(need 256)', 'Hard\n(need 512)', 'Impossible\n(wrong at all)']
    sizes = [counts.get('easy', 0), counts.get('medium', 0),
             counts.get('hard', 0), counts.get('impossible', 0)]
    colors = [COLORS['easy'], COLORS['medium'], COLORS['hard'], COLORS['impossible']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75, textprops={'fontsize': 11}
    )
    for t in autotexts:
        t.set_fontweight('bold')
    ax1.set_title(f'Question Difficulty Distribution\n(n={n})')

    # Stacked bar: where do tokens go?
    total_tok = {}
    for cat in ['easy', 'medium', 'hard', 'impossible']:
        cat_rows = [r for r, c in zip(rows, cats) if c == cat]
        total_tok[cat] = sum(get_int(r, "fixed_512_tokens", 512) for r in cat_rows)

    total = sum(total_tok.values())
    labels2 = ['Easy', 'Medium', 'Hard', 'Impossible']
    tok_vals = [total_tok[c] for c in ['easy', 'medium', 'hard', 'impossible']]
    tok_pcts = [v/total for v in tok_vals]

    bars = ax2.bar(labels2, [v/1000 for v in tok_vals], color=colors, alpha=0.8)
    ax2.set_ylabel('Total Tokens (×1000)')
    ax2.set_title('Token Allocation by Difficulty\n(Fixed-512 budget)')
    for bar, pct in zip(bars, tok_pcts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
                 f'{pct:.0%}', ha='center', fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_difficulty_distribution.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig3_difficulty_distribution.png'))
    plt.close(fig)
    print("Generated: fig3_difficulty_distribution")


def fig4_selective_prediction(rows, out_dir):
    """Figure 4: Selective prediction accuracy-coverage curve."""
    # At Fixed-512, sort samples by "confidence" = early-stop vs hit-budget
    tokens_512 = [get_main_tokens(r, 512) for r in rows]
    correct_512 = [is_true(r.get("fixed_512_correct")) for r in rows]

    # Sort by token count (ascending = most confident first)
    combined = sorted(zip(tokens_512, correct_512))

    # Compute cumulative accuracy at each coverage level
    coverages = []
    accuracies = []
    total_correct = 0
    for i, (tok, corr) in enumerate(combined):
        if corr:
            total_correct += 1
        coverage = (i + 1) / len(combined)
        accuracy = total_correct / (i + 1)
        if (i + 1) % 10 == 0 or i == len(combined) - 1:
            coverages.append(coverage)
            accuracies.append(accuracy)

    fig, ax = plt.subplots()
    ax.plot([c*100 for c in coverages], [a*100 for a in accuracies],
            color=COLORS['fixed'], linewidth=2)

    # Mark the natural-stop threshold
    threshold = 512 * 0.95
    stop_count = sum(1 for t in tokens_512 if t < threshold)
    stop_coverage = stop_count / len(rows)
    stop_correct = sum(1 for t, c in zip(tokens_512, correct_512) if t < threshold and c)
    stop_acc = stop_correct / max(stop_count, 1)

    ax.axvline(x=stop_coverage*100, color='gray', linestyle='--', alpha=0.7)
    ax.plot(stop_coverage*100, stop_acc*100, 'o', color=COLORS['stop'], markersize=12,
            label=f'Natural Stop: {stop_acc:.0%} acc @ {stop_coverage:.0%} coverage')

    # Full coverage point
    full_acc = sum(correct_512) / len(rows)
    ax.plot(100, full_acc*100, 's', color=COLORS['hit'], markersize=10,
            label=f'Full: {full_acc:.0%} acc @ 100% coverage')

    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Selective Prediction: Accuracy vs Coverage\n(Token count as confidence proxy)')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_selective_prediction.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig4_selective_prediction.png'))
    plt.close(fig)
    print("Generated: fig4_selective_prediction")


def fig5_token_waste(rows, out_dir):
    """Figure 5: Token waste waterfall chart."""
    cats = categorize_samples(rows)
    n = len(rows)

    # For each category, compute: total tokens spent at Fixed-512
    waste_data = {}
    for cat in ['easy', 'medium', 'hard', 'impossible']:
        cat_rows = [(r, c) for r, c in zip(rows, cats) if c == cat]
        total_tok = sum(get_int(r, "fixed_512_tokens", 512) for r, _ in cat_rows)

        # "Optimal" tokens: what's the minimum they need?
        if cat == 'easy':
            optimal = sum(get_int(r, "fixed_128_tokens", 128) for r, _ in cat_rows)
        elif cat == 'medium':
            optimal = sum(get_int(r, "fixed_256_tokens", 256) for r, _ in cat_rows)
        elif cat == 'hard':
            optimal = sum(get_int(r, "fixed_512_tokens", 512) for r, _ in cat_rows)
        else:  # impossible
            optimal = 0  # ideally skip entirely

        waste_data[cat] = {
            'count': len(cat_rows),
            'total_tok': total_tok,
            'optimal_tok': optimal,
            'waste': total_tok - optimal,
        }

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ['Easy', 'Medium', 'Hard', 'Impossible']
    useful = [waste_data[c]['optimal_tok']/1000 for c in ['easy', 'medium', 'hard', 'impossible']]
    wasted = [waste_data[c]['waste']/1000 for c in ['easy', 'medium', 'hard', 'impossible']]

    x = np.arange(len(labels))
    width = 0.6

    p1 = ax.bar(x, useful, width, label='Useful Tokens', color=COLORS['stop'], alpha=0.8)
    p2 = ax.bar(x, wasted, width, bottom=useful, label='Wasted Tokens', color=COLORS['hit'], alpha=0.5)

    ax.set_ylabel('Total Tokens (×1000)')
    ax.set_title('Token Waste Analysis\n(Fixed-512 vs Oracle Allocation)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Annotate waste percentages
    for i, (u, w) in enumerate(zip(useful, wasted)):
        if w > 0:
            pct = w / (u + w)
            ax.text(i, u + w/2, f'{pct:.0%}\nwaste', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

    total_waste = sum(wasted)
    total_all = sum(u+w for u, w in zip(useful, wasted))
    ax.text(0.95, 0.95, f'Total waste: {total_waste/(total_all)*100:.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_token_waste.pdf'))
    fig.savefig(os.path.join(out_dir, 'fig5_token_waste.png'))
    plt.close(fig)
    print("Generated: fig5_token_waste")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--csv", default="results_kun/fulltest/per_sample_gsm8k_Qwen3_8B_20260324_120316.csv")
    parser.add_argument("--output_dir", default="results/paper_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.csv}...")
    rows = load_fulltest_csv(args.csv)
    print(f"Loaded {len(rows)} samples")

    fig1_accuracy_vs_budget(rows, args.output_dir)
    fig2_natural_stop_oracle(rows, args.output_dir)
    fig3_difficulty_distribution(rows, args.output_dir)
    fig4_selective_prediction(rows, args.output_dir)
    fig5_token_waste(rows, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
