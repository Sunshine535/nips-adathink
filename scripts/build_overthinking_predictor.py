#!/usr/bin/env python3
"""Build predictor for overthinking risk based on question features."""
import csv, json, re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

def extract_features(question):
    """Extract features from question text."""
    words = question.split()
    numbers = [float(x) for x in re.findall(r'\d+\.?\d*', question)]

    return {
        'q_len': len(words),
        'num_count': len(numbers),
        'max_num': max(numbers) if numbers else 0,
        'avg_num': np.mean(numbers) if numbers else 0,
        'has_fraction': 1 if '/' in question else 0,
        'has_percent': 1 if '%' in question else 0
    }

def main():
    # Load data
    samples = []
    for csv_path in Path('results').glob('per_sample_Qwen3.5_27B_*.csv'):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                c256 = int(float(row.get('fixed_256_correct', 0)))
                c512 = int(float(row.get('fixed_512_correct', 0)))
                if c256 == 1:  # Only consider samples correct at 256
                    samples.append({
                        'question': row.get('question', ''),
                        'overthink': 1 if c512 == 0 else 0
                    })

    # Extract features
    X = []
    y = []
    for s in samples:
        feat = extract_features(s['question'])
        X.append([feat['q_len'], feat['num_count'], feat['max_num'],
                  feat['has_fraction'], feat['has_percent']])
        y.append(s['overthink'])

    X = np.array(X)
    y = np.array(y)

    # Train predictor
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)

    print(f"Overthinking Predictor:")
    print(f"  Samples: {len(y)}, Overthinking rate: {y.mean():.2%}")
    print(f"  Accuracy: {clf.score(X, y):.2%}")
    print(f"\nFeature importance:")
    for i, name in enumerate(['q_len', 'num_count', 'max_num', 'has_fraction', 'has_percent']):
        print(f"  {name}: {clf.coef_[0][i]:.3f}")

if __name__ == '__main__':
    from pathlib import Path
    main()
