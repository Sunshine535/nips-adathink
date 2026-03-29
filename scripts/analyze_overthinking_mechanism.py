#!/usr/bin/env python3
"""
Analyze overthinking mechanism: identify samples where higher budget hurts accuracy.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

def extract_numbers(text):
    """Extract all numbers from text for complexity analysis."""
    return [float(x) for x in re.findall(r'\d+\.?\d*', text)]

def analyze_overthinking_samples(csv_files, output_json):
    """Identify and analyze samples where 256->512 causes accuracy drop."""
    overthinking_samples = []
    correct_at_both = []
    improved_samples = []

    for csv_path in csv_files:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = row.get('idx', '')
                question = row.get('question', '')

                # Get correctness at different budgets
                c128 = int(float(row.get('fixed_128_correct', 0)))
                c256 = int(float(row.get('fixed_256_correct', 0)))
                c512 = int(float(row.get('fixed_512_correct', 0)))

                # Get raw outputs
                raw256 = row.get('fixed_256_raw', '')
                raw512 = row.get('fixed_512_raw', '')

                # Overthinking: correct at 256, wrong at 512
                if c256 == 1 and c512 == 0:
                    overthinking_samples.append({
                        'idx': idx,
                        'question': question,
                        'question_len': len(question.split()),
                        'num_count': len(extract_numbers(question)),
                        'raw_256': raw256[:500],
                        'raw_512': raw512[:500],
                        'c128': c128, 'c256': c256, 'c512': c512
                    })

                # Correct at both (stable)
                elif c256 == 1 and c512 == 1:
                    correct_at_both.append({
                        'idx': idx,
                        'question_len': len(question.split()),
                        'num_count': len(extract_numbers(question))
                    })

                # Improved: wrong at 256, correct at 512
                elif c256 == 0 and c512 == 1:
                    improved_samples.append({
                        'idx': idx,
                        'question_len': len(question.split()),
                        'num_count': len(extract_numbers(question))
                    })

    # Statistics
    total = len(overthinking_samples) + len(correct_at_both) + len(improved_samples)

    result = {
        'summary': {
            'total_samples': total,
            'overthinking_count': len(overthinking_samples),
            'overthinking_rate': len(overthinking_samples) / total if total > 0 else 0,
            'stable_correct_count': len(correct_at_both),
            'improved_count': len(improved_samples)
        },
        'overthinking_samples': overthinking_samples[:50],  # Top 50
        'feature_analysis': {
            'overthinking': {
                'avg_question_len': sum(s['question_len'] for s in overthinking_samples) / max(1, len(overthinking_samples)),
                'avg_num_count': sum(s['num_count'] for s in overthinking_samples) / max(1, len(overthinking_samples))
            },
            'stable': {
                'avg_question_len': sum(s['question_len'] for s in correct_at_both) / max(1, len(correct_at_both)),
                'avg_num_count': sum(s['num_count'] for s in correct_at_both) / max(1, len(correct_at_both))
            },
            'improved': {
                'avg_question_len': sum(s['question_len'] for s in improved_samples) / max(1, len(improved_samples)),
                'avg_num_count': sum(s['num_count'] for s in improved_samples) / max(1, len(improved_samples))
            }
        }
    }

    with open(output_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Overthinking Analysis:")
    print(f"  Total samples: {total}")
    print(f"  Overthinking (256✓→512✗): {len(overthinking_samples)} ({100*len(overthinking_samples)/total:.1f}%)")
    print(f"  Stable correct: {len(correct_at_both)} ({100*len(correct_at_both)/total:.1f}%)")
    print(f"  Improved (256✗→512✓): {len(improved_samples)} ({100*len(improved_samples)/total:.1f}%)")
    print(f"\nFeature comparison:")
    print(f"  Overthinking - Avg Q len: {result['feature_analysis']['overthinking']['avg_question_len']:.1f}")
    print(f"  Stable - Avg Q len: {result['feature_analysis']['stable']['avg_question_len']:.1f}")
    print(f"  Improved - Avg Q len: {result['feature_analysis']['improved']['avg_question_len']:.1f}")

if __name__ == '__main__':
    import sys
    csv_files = sys.argv[1:-1]
    output = sys.argv[-1]
    analyze_overthinking_samples(csv_files, output)
