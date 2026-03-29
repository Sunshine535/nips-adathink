#!/usr/bin/env python3
"""Generate case study: manually analyze top overthinking samples."""
import json
import csv
from pathlib import Path

def main():
    # Load overthinking analysis
    with open('results/overthinking_mechanism_analysis.json') as f:
        data = json.load(f)

    samples = data['overthinking_samples'][:20]  # Top 20

    print("=== Manual Case Study Template ===\n")
    print("Analyze these 20 overthinking cases:\n")

    for i, s in enumerate(samples, 1):
        print(f"\n--- Case {i} ---")
        print(f"Question: {s['question'][:150]}...")
        print(f"Length: {s['question_len']} words, Numbers: {s['num_count']}")
        print(f"Status: 128={'✓' if s['c128']==1 else '✗'}, 256=✓, 512=✗")
        print(f"\nOutput@256: {s['raw_256'][:200]}")
        print(f"\nOutput@512: {s['raw_512'][:200]}")
        print(f"\nPattern: [TODO: error_propagation / self_contradiction / attention_dilution / other]")
        print("-" * 80)

if __name__ == '__main__':
    main()
