"""Analyze per-question error patterns for failure taxonomy."""
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.environ.get('RESULTS_DIR', '../results')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_controller_rows(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    return pd.read_csv(path)


def error_taxonomy_gsm8k():
    """Classify errors for GSM8K-27B template controller."""
    df = load_controller_rows('template_controller_rows_lam0p15_20260228_23seed.csv')
    if df is None:
        return {}

    n = len(df)
    correct = df['correct'].sum()
    incorrect = n - correct

    budget_dist = df['chosen_budget'].value_counts().sort_index()
    best_dist = df['best_budget'].value_counts().sort_index()

    under_budget = ((df['chosen_budget'] < df['best_budget']) & (df['correct'] == 0)).sum()
    over_budget = ((df['chosen_budget'] > df['best_budget']) & (df['correct'] == 0)).sum()
    matched_wrong = ((df['chosen_budget'] == df['best_budget']) & (df['correct'] == 0)).sum()
    matched_right = ((df['chosen_budget'] == df['best_budget']) & (df['correct'] == 1)).sum()

    oracle_correct = 0
    for _, row in df.iterrows():
        bb = row['best_budget']
        if row['correct'] == 1:
            oracle_correct += 1
        elif bb != row['chosen_budget']:
            oracle_correct += 1

    results = {
        'benchmark': 'GSM8K-27B',
        'n_total': n,
        'n_correct': int(correct),
        'n_incorrect': int(incorrect),
        'accuracy': round(correct / n, 4),
        'budget_distribution': {str(k): int(v) for k, v in budget_dist.items()},
        'best_budget_distribution': {str(k): int(v) for k, v in best_dist.items()},
        'error_categories': {
            'under_budget_wrong': int(under_budget),
            'over_budget_wrong': int(over_budget),
            'matched_budget_wrong': int(matched_wrong),
            'matched_budget_right': int(matched_right),
        },
        'error_fractions': {
            'under_budget': round(under_budget / max(incorrect, 1), 4),
            'over_budget': round(over_budget / max(incorrect, 1), 4),
            'matched_wrong': round(matched_wrong / max(incorrect, 1), 4),
        }
    }
    return results


def error_taxonomy_cross(filename, name, budgets):
    """Classify errors for any benchmark controller rows."""
    df = load_controller_rows(filename)
    if df is None:
        return {}

    n = len(df)
    correct = df['correct'].sum()
    incorrect = n - correct

    budget_dist = df['chosen_budget'].value_counts().sort_index()

    under_budget = ((df['chosen_budget'] < df['best_budget']) & (df['correct'] == 0)).sum()
    over_budget = ((df['chosen_budget'] > df['best_budget']) & (df['correct'] == 0)).sum()
    matched_wrong = ((df['chosen_budget'] == df['best_budget']) & (df['correct'] == 0)).sum()

    return {
        'benchmark': name,
        'n_total': n,
        'n_correct': int(correct),
        'n_incorrect': int(incorrect),
        'accuracy': round(correct / n, 4),
        'budget_distribution': {str(k): int(v) for k, v in budget_dist.items()},
        'error_categories': {
            'under_budget_wrong': int(under_budget),
            'over_budget_wrong': int(over_budget),
            'matched_budget_wrong': int(matched_wrong),
        },
        'error_fractions': {
            'under_budget': round(under_budget / max(incorrect, 1), 4),
            'over_budget': round(over_budget / max(incorrect, 1), 4),
            'matched_wrong': round(matched_wrong / max(incorrect, 1), 4),
        }
    }


def plot_error_taxonomy(all_results):
    """Plot error taxonomy across benchmarks."""
    benchmarks = []
    under = []
    over = []
    intrinsic = []

    for r in all_results:
        if not r:
            continue
        benchmarks.append(r['benchmark'])
        ef = r['error_fractions']
        under.append(ef['under_budget'] * 100)
        over.append(ef['over_budget'] * 100)
        intrinsic.append(ef['matched_wrong'] * 100)

    if not benchmarks:
        print("No data for error taxonomy plot")
        return

    x = np.arange(len(benchmarks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(x - width, under, width, label='Under-budget\n(needed more tokens)', color='#e74c3c', alpha=0.85)
    ax.bar(x, over, width, label='Over-budget\n(wasted tokens)', color='#f39c12', alpha=0.85)
    ax.bar(x + width, intrinsic, width, label='Intrinsic error\n(correct budget, still wrong)', color='#7f8c8d', alpha=0.85)

    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Fraction of errors (%)')
    ax.set_title('Error Taxonomy: Why Does the Controller Fail?')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=15, ha='right')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_error_taxonomy.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_error_taxonomy.png'))
    plt.close()
    print('Generated fig6_error_taxonomy')


def plot_budget_allocation(all_results):
    """Plot budget allocation distribution across benchmarks."""
    fig, axes = plt.subplots(1, len(all_results), figsize=(3.2 * len(all_results), 3))
    if len(all_results) == 1:
        axes = [axes]

    for ax, r in zip(axes, all_results):
        if not r:
            continue
        bd = r['budget_distribution']
        budgets = sorted(bd.keys(), key=lambda x: float(x))
        counts = [bd[b] for b in budgets]
        total = sum(counts)
        fracs = [c / total * 100 for c in counts]

        colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(budgets)]
        ax.bar(range(len(budgets)), fracs, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels(budgets)
        ax.set_ylabel('Questions (%)')
        ax.set_xlabel('Assigned budget')
        ax.set_title(r['benchmark'])
        ax.grid(True, axis='y', alpha=0.3)

        for i, (f, c) in enumerate(zip(fracs, counts)):
            ax.text(i, f + 1, f'{f:.0f}%', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_budget_allocation.pdf'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_budget_allocation.png'))
    plt.close()
    print('Generated fig7_budget_allocation')


if __name__ == '__main__':
    gsm8k = error_taxonomy_gsm8k()
    math500 = error_taxonomy_cross(
        'template_controller_rows_math500_27b_20260320_160051.csv', 'MATH500-27B',
        [2048, 4096, 8192])
    bbh = error_taxonomy_cross(
        'template_controller_rows_bbh_27b_20260320_160051.csv', 'BBH-27B',
        [1024, 2048, 4096])
    math500_8b = error_taxonomy_cross(
        'template_controller_rows_math500_8b_20260320_160051.csv', 'MATH500-8B',
        [512, 1024, 2048])
    bbh_8b = error_taxonomy_cross(
        'template_controller_rows_bbh_8b_20260320_160051.csv', 'BBH-8B',
        [256, 512, 1024])

    all_results = [r for r in [gsm8k, math500, bbh, math500_8b, bbh_8b] if r]

    report = {'error_taxonomies': all_results}
    with open(os.path.join(OUTPUT_DIR, 'error_taxonomy_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved with {len(all_results)} benchmarks")

    for r in all_results:
        print(f"\n--- {r['benchmark']} ---")
        print(f"  Accuracy: {r['accuracy']}")
        print(f"  Budget dist: {r['budget_distribution']}")
        print(f"  Error fracs: {r['error_fractions']}")

    plot_error_taxonomy(all_results)
    plot_budget_allocation(all_results)
