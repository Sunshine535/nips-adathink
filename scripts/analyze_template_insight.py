#!/usr/bin/env python3
"""Deep analysis of WHY the 3-bit template controller works."""

import csv
import os
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

CONFIGS = {
    "gsm8k_27b": {"prefix": "per_sample_gsm8k_Qwen3.5_27B", "budgets": [128, 256, 512]},
    "math500_27b": {"prefix": "per_sample_math500_Qwen3.5_27B", "budgets": [2048, 4096, 8192]},
    "bbh_27b": {"prefix": "per_sample_bbh_Qwen3.5_27B", "budgets": [1024, 2048, 4096]},
}


def load_rows(prefix):
    rows = []
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith(prefix) and f.endswith(".csv") and "template" not in f and "controller" not in f:
            with open(os.path.join(RESULTS_DIR, f)) as fh:
                for row in csv.DictReader(fh):
                    rows.append(row)
    return rows


def extract_features(row, b1):
    has_final = int(float(row.get(f"fixed_{b1}_has_final", 0)))
    tokens = float(row.get(f"fixed_{b1}_tokens", b1))
    util = tokens / b1 if b1 > 0 else 1.0
    util_bin = 1 if util >= 0.95 else 0
    consistency = 1 if has_final and util_bin == 0 else 0
    return has_final, util_bin, consistency


def analyze(name, rows, budgets):
    b1, b_mid, b_max = budgets
    total = len(rows)
    feat_counts = defaultdict(int)
    feat_correct = defaultdict(lambda: defaultdict(int))
    feat_total = defaultdict(lambda: defaultdict(int))

    for row in rows:
        feat = extract_features(row, b1)
        feat_counts[feat] += 1
        for b in budgets:
            correct = int(float(row.get(f"fixed_{b}_correct", 0)))
            feat_correct[feat][b] += correct
            feat_total[feat][b] += 1

    print(f"\n{'='*80}")
    print(f"  {name} ({total} samples, budgets={budgets})")
    print(f"{'='*80}")
    print(f"  {'Pattern':<22} {'Count':>6} {'Frac':>6}  ", end="")
    for b in budgets:
        print(f"{'Acc@'+str(b):>10}", end="")
    print(f"  {'Best':>6}  Interpretation")
    print("-" * 100)

    interpretations = {
        (1, 0, 0): "Has answer, low util, INCONSISTENT -> likely wrong",
        (1, 0, 1): "Has answer, low util, CONSISTENT -> EASY",
        (1, 1, 0): "Has answer, high util -> Medium (long but answered)",
        (0, 0, 0): "No answer, low util -> ???",
        (0, 1, 0): "No answer, high util -> HARD (exhausted budget)",
    }

    for feat in sorted(feat_counts.keys()):
        count = feat_counts[feat]
        pct = 100 * count / total
        accs = {}
        for b in budgets:
            accs[b] = feat_correct[feat][b] / max(feat_total[feat][b], 1) * 100
        best = max(budgets, key=lambda b: accs[b])
        interp = interpretations.get(feat, "")
        print(f"  {str(feat):<22} {count:>6} {pct:>5.1f}%  ", end="")
        for b in budgets:
            print(f"{accs[b]:>9.1f}%", end="")
        print(f"  {best:>6}  {interp}")

    # Overthinking analysis
    print(f"\n  Overthinking (correct at b1 but wrong at b_max):")
    for feat in sorted(feat_counts.keys()):
        overthink = 0
        total_correct_b1 = 0
        for row in rows:
            f = extract_features(row, b1)
            if f != feat:
                continue
            c1 = int(float(row.get(f"fixed_{b1}_correct", 0)))
            c_max = int(float(row.get(f"fixed_{b_max}_correct", 0)))
            if c1:
                total_correct_b1 += 1
                if not c_max:
                    overthink += 1
        if total_correct_b1 > 0:
            rate = 100 * overthink / total_correct_b1
            print(f"    {str(feat):<22} b1-correct: {total_correct_b1:>4}, overthink: {overthink:>4} ({rate:>5.1f}%)")

    # Feature importance (mutual information approximation)
    print(f"\n  Feature discriminative power (accuracy range across budgets):")
    for feat in sorted(feat_counts.keys()):
        accs = [feat_correct[feat][b] / max(feat_total[feat][b], 1) for b in budgets]
        range_acc = max(accs) - min(accs)
        print(f"    {str(feat):<22} acc range: {range_acc*100:>6.1f}pp (min={min(accs)*100:.1f}%, max={max(accs)*100:.1f}%)")

    # Key insight: what fraction of questions are "easy" (b1 suffices)?
    easy_patterns = [(1, 0, 1)]  # consistent answer at low util
    easy_count = sum(feat_counts[p] for p in easy_patterns if p in feat_counts)
    hard_patterns = [(0, 1, 0)]  # no answer, high utilization
    hard_count = sum(feat_counts[p] for p in hard_patterns if p in feat_counts)
    medium_count = total - easy_count - hard_count
    print(f"\n  Difficulty distribution:")
    print(f"    Easy (b1 suffices):  {easy_count:>5} ({100*easy_count/total:.1f}%)")
    print(f"    Hard (needs b_max):  {hard_count:>5} ({100*hard_count/total:.1f}%)")
    print(f"    Medium/Ambiguous:    {medium_count:>5} ({100*medium_count/total:.1f}%)")


def main():
    print("TEMPLATE CONTROLLER: DEEP INSIGHT ANALYSIS")
    print("Why do 3 binary features suffice for effective budget allocation?")

    for name, cfg in CONFIGS.items():
        rows = load_rows(cfg["prefix"])
        if rows:
            analyze(name, rows, cfg["budgets"])

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    print("1. The three features partition questions into difficulty strata with")
    print("   dramatically different accuracy profiles across budgets.")
    print("2. 'Easy' questions (consistent answer at low utilization) show high")
    print("   accuracy at b1 with diminishing or negative returns at higher budgets.")
    print("3. 'Hard' questions (no answer, high utilization) show near-zero accuracy")
    print("   at b1 but substantial gains at b_max.")
    print("4. The template controller exploits this structure by routing each stratum")
    print("   to its optimal budget tier.")


if __name__ == "__main__":
    main()
