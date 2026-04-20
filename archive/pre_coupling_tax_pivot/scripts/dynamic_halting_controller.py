#!/usr/bin/env python3
"""
Dynamic Halting Controller: Learn when to stop based on reasoning progress.
Key innovation: Uses reasoning dynamics, not question features.
"""
import csv, json, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def extract_dynamic_features(row, budget):
    """Extract features from reasoning process at given budget."""
    correct = int(float(row.get(f'fixed_{budget}_correct', 0)))
    tokens = float(row.get(f'fixed_{budget}_tokens', 0))
    raw = row.get(f'fixed_{budget}_raw', '').lower()

    # Feature 1: Token utilization rate
    utilization = tokens / budget

    # Feature 2: Answer confidence (has definitive markers)
    confidence_markers = ['final answer', 'therefore', 'thus', 'answer is']
    confidence = sum(1 for m in confidence_markers if m in raw) / len(confidence_markers)

    # Feature 3: Reasoning coherence (no repetition)
    words = raw.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
    else:
        unique_ratio = 1.0

    return [utilization, confidence, unique_ratio], correct

def build_halting_policy(csv_files):
    """Train policy: should we stop at budget B or continue?"""
    X_128, y_128 = [], []
    X_256, y_256 = [], []

    for csv_path in csv_files:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                # At 128: should we stop or continue to 256?
                feat_128, c128 = extract_dynamic_features(row, 128)
                c256 = int(float(row.get('fixed_256_correct', 0)))

                # Label: STOP if already correct, CONTINUE if can improve
                should_stop_at_128 = (c128 == 1 and c256 == 1)  # Stable correct
                X_128.append(feat_128)
                y_128.append(1 if should_stop_at_128 else 0)

                # At 256: should we stop or continue to 512?
                feat_256, c256_val = extract_dynamic_features(row, 256)
                c512 = int(float(row.get('fixed_512_correct', 0)))

                should_stop_at_256 = (c256_val == 1 and c512 == 0)  # Overthinking risk
                X_256.append(feat_256)
                y_256.append(1 if should_stop_at_256 else 0)

    # Train two classifiers
    clf_128 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_256 = RandomForestClassifier(n_estimators=100, random_state=42)

    clf_128.fit(X_128, y_128)
    clf_256.fit(X_256, y_256)

    return clf_128, clf_256

def evaluate_dynamic_halting(csv_files, clf_128, clf_256):
    """Evaluate dynamic halting policy."""
    total = correct = tokens_used = 0

    for csv_path in csv_files:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                total += 1

                # Start at 128
                feat_128, c128 = extract_dynamic_features(row, 128)
                if clf_128.predict([feat_128])[0] == 1:  # STOP
                    correct += c128
                    tokens_used += float(row.get('fixed_128_tokens', 0))
                    continue

                # Continue to 256
                feat_256, c256 = extract_dynamic_features(row, 256)
                if clf_256.predict([feat_256])[0] == 1:  # STOP
                    correct += c256
                    tokens_used += float(row.get('fixed_256_tokens', 0))
                    continue

                # Continue to 512
                c512 = int(float(row.get('fixed_512_correct', 0)))
                correct += c512
                tokens_used += float(row.get('fixed_512_tokens', 0))

    return {
        'accuracy': correct / total,
        'avg_tokens': tokens_used / total,
        'utility': (correct / total) - 0.15 * (tokens_used / total / 1000)
    }

if __name__ == '__main__':
    from pathlib import Path
    csv_files = list(Path('results').glob('per_sample_Qwen3.5_27B_*.csv'))[:10]

    print("Training dynamic halting policy...")
    clf_128, clf_256 = build_halting_policy(csv_files)

    print("\nEvaluating...")
    result = evaluate_dynamic_halting(csv_files, clf_128, clf_256)

    print(f"\nDynamic Halting Results:")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Avg tokens: {result['avg_tokens']:.1f}")
    print(f"  Utility: {result['utility']:.4f}")
