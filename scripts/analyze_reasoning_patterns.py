#!/usr/bin/env python3
"""
Analyze token-level patterns in overthinking: what goes wrong in longer reasoning?
"""
import json
import re
from collections import Counter

def extract_reasoning_steps(text):
    """Extract reasoning steps from output."""
    steps = []
    # Split by common delimiters
    for line in text.split('\n'):
        line = line.strip()
        if line and len(line) > 10:
            steps.append(line)
    return steps

def detect_error_patterns(raw_256, raw_512):
    """Detect what changes between correct (256) and wrong (512) reasoning."""
    steps_256 = extract_reasoning_steps(raw_256)
    steps_512 = extract_reasoning_steps(raw_512)

    # Pattern 1: Self-contradiction
    contradiction = 0
    if len(steps_512) > len(steps_256):
        # Check if later steps contradict earlier ones
        for i, step in enumerate(steps_512[len(steps_256):]):
            if any(word in step.lower() for word in ['wait', 'actually', 'no', 'mistake', 'wrong']):
                contradiction = 1
                break

    # Pattern 2: Repetition/loops
    if len(steps_512) > 5:
        step_texts = [s.lower()[:50] for s in steps_512]
        repetition_rate = 1 - len(set(step_texts)) / len(step_texts)
    else:
        repetition_rate = 0

    # Pattern 3: Length explosion
    length_ratio = len(raw_512) / max(1, len(raw_256))

    return {
        'has_contradiction': contradiction,
        'repetition_rate': repetition_rate,
        'length_ratio': length_ratio,
        'extra_steps': len(steps_512) - len(steps_256)
    }

def main():
    with open('results/overthinking_mechanism_analysis.json', 'r') as f:
        data = json.load(f)

    patterns = []
    for sample in data['overthinking_samples']:
        pattern = detect_error_patterns(sample['raw_256'], sample['raw_512'])
        pattern['idx'] = sample['idx']
        patterns.append(pattern)

    # Aggregate statistics
    result = {
        'pattern_statistics': {
            'contradiction_rate': sum(p['has_contradiction'] for p in patterns) / len(patterns),
            'avg_repetition': sum(p['repetition_rate'] for p in patterns) / len(patterns),
            'avg_length_ratio': sum(p['length_ratio'] for p in patterns) / len(patterns),
            'avg_extra_steps': sum(p['extra_steps'] for p in patterns) / len(patterns)
        },
        'samples_with_patterns': patterns[:20]
    }

    with open('results/reasoning_pattern_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("Reasoning Pattern Analysis:")
    print(f"  Contradiction rate: {result['pattern_statistics']['contradiction_rate']:.2%}")
    print(f"  Avg repetition: {result['pattern_statistics']['avg_repetition']:.2%}")
    print(f"  Avg length ratio (512/256): {result['pattern_statistics']['avg_length_ratio']:.2f}x")
    print(f"  Avg extra steps: {result['pattern_statistics']['avg_extra_steps']:.1f}")

if __name__ == '__main__':
    main()
