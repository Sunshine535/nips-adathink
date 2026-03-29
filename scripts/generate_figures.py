#!/usr/bin/env python3
"""Generate figures for overthinking paper."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

Path('results/figures').mkdir(exist_ok=True)

# Load data
with open('results/overthinking_mechanism_analysis.json') as f:
    data = json.load(f)

# Figure 1: Overthinking rates
fig, ax = plt.subplots(figsize=(6, 4))
categories = ['Overthinking\n(256✓→512✗)', 'Stable\n(256✓→512✓)', 'Improved\n(256✗→512✓)']
counts = [
    data['summary']['overthinking_count'],
    data['summary']['stable_correct_count'],
    data['summary']['improved_count']
]
colors = ['#d62728', '#2ca02c', '#1f77b4']
ax.bar(categories, counts, color=colors, alpha=0.8)
ax.set_ylabel('Number of Samples')
ax.set_title('Distribution of Compute Scaling Outcomes')
plt.tight_layout()
plt.savefig('results/figures/overthinking_distribution.pdf')
print("✓ Saved: overthinking_distribution.pdf")

# Figure 2: Question length comparison
fig, ax = plt.subplots(figsize=(6, 4))
features = data['feature_analysis']
lengths = [
    features['overthinking']['avg_question_len'],
    features['stable']['avg_question_len'],
    features['improved']['avg_question_len']
]
ax.bar(categories, lengths, color=colors, alpha=0.8)
ax.set_ylabel('Average Question Length (words)')
ax.set_title('Question Length by Outcome Type')
ax.axhline(y=np.mean(lengths), color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('results/figures/question_length_comparison.pdf')
print("✓ Saved: question_length_comparison.pdf")

print("\nFigures generated in results/figures/")
