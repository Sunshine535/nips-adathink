#!/usr/bin/env python3
"""
Router Baseline Analysis for TOWN Paper
========================================
Compares different routing strategies for nothink→thinking escalation:
  1. TOWN (token-length router): route samples that hit nothink budget
  2. Answer-presence router: route samples without parseable answer
  3. Random router: route K% of samples randomly
  4. Oracle router: route samples where thinking@512 is correct but nothink@256 is wrong
  5. Token-length threshold sweep: vary the token-length cutoff

All routers use the same budget: nothink@256 → thinking@512 for routed samples.
"""

import json
import os
import sys
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
NOTHINK_256_PATH = "results/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_000345.json"
THINKING_512_PATH = "results/thinking_hf/hf_512/nothink_baseline_Qwen3-8B_gsm8k_20260330_125514.json"
FULLTEST_CSV_PATH = "results/fulltest/per_sample_gsm8k_Qwen3_8B_20260324_120316.csv"
THINKING_1024_PATH = "results/thinking_hf/hf_1024/nothink_baseline_Qwen3-8B_gsm8k_20260330_125513.json"
THINKING_2048_PATH = "results/thinking_hf/hf_2048/nothink_baseline_Qwen3-8B_gsm8k_20260330_125514.json"
OUTPUT_DIR = "results/router_baselines"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading data...")

with open(NOTHINK_256_PATH) as f:
    nothink_data = json.load(f)
nothink_samples = nothink_data["per_sample"]["nothink_256"]  # 1319 samples

with open(THINKING_512_PATH) as f:
    thinking_512_data = json.load(f)
thinking_512_samples = thinking_512_data["per_sample"]["thinking_512"]

with open(THINKING_1024_PATH) as f:
    thinking_1024_data = json.load(f)
thinking_1024_samples = thinking_1024_data["per_sample"]["thinking_1024"]

with open(THINKING_2048_PATH) as f:
    thinking_2048_data = json.load(f)
thinking_2048_samples = thinking_2048_data["per_sample"]["thinking_2048"]

N = len(nothink_samples)
assert N == len(thinking_512_samples) == len(thinking_1024_samples) == len(thinking_2048_samples) == 1319

# ── Build per-sample lookup ───────────────────────────────────────────────
# Index by sample idx
nothink = {s["idx"]: s for s in nothink_samples}
think512 = {s["idx"]: s for s in thinking_512_samples}
think1024 = {s["idx"]: s for s in thinking_1024_samples}
think2048 = {s["idx"]: s for s in thinking_2048_samples}

# Verify indices match
assert set(nothink.keys()) == set(think512.keys())
print(f"Loaded {N} samples, indices matched.")

# ── Helper: Evaluate Router ───────────────────────────────────────────────
def evaluate_router(routed_indices, think_budget="512", label=""):
    """
    Given a set of sample indices to route from nothink@256 → thinking@{budget},
    compute the combined accuracy and average tokens.
    """
    think_map = {"512": think512, "1024": think1024, "2048": think2048}
    think_data = think_map[think_budget]

    correct = 0
    total_tokens = 0
    routed_set = set(routed_indices)
    n_routed = len(routed_set)

    for idx in range(N):
        if idx in routed_set:
            # Use thinking result
            s = think_data[idx]
            correct += int(s["correct"])
            total_tokens += s["tokens"]
        else:
            # Use nothink result
            s = nothink[idx]
            correct += int(s["correct"])
            total_tokens += s["tokens"]

    accuracy = correct / N
    avg_tokens = total_tokens / N

    # Breakdown: accuracy on routed vs non-routed
    routed_correct = sum(int(think_data[i]["correct"]) for i in routed_set) if routed_set else 0
    nonrouted_correct = sum(int(nothink[i]["correct"]) for i in range(N) if i not in routed_set)
    routed_acc = routed_correct / n_routed if n_routed > 0 else float("nan")
    nonrouted_acc = nonrouted_correct / (N - n_routed) if (N - n_routed) > 0 else float("nan")

    # Compute token overhead vs pure nothink
    nothink_only_tokens = sum(nothink[i]["tokens"] for i in range(N)) / N
    token_overhead = avg_tokens - nothink_only_tokens

    return {
        "label": label,
        "n_routed": n_routed,
        "route_rate": n_routed / N,
        "accuracy": accuracy,
        "avg_tokens": avg_tokens,
        "token_overhead": token_overhead,
        "routed_accuracy": routed_acc,
        "nonrouted_accuracy": nonrouted_acc,
        "think_budget": think_budget,
    }


# ══════════════════════════════════════════════════════════════════════════
# 1. BASELINES (no routing)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("1. BASELINES (no routing)")
print("="*70)

nothink_acc = sum(int(nothink[i]["correct"]) for i in range(N)) / N
nothink_avg_tok = sum(nothink[i]["tokens"] for i in range(N)) / N
think512_acc = sum(int(think512[i]["correct"]) for i in range(N)) / N
think512_avg_tok = sum(think512[i]["tokens"] for i in range(N)) / N
think1024_acc = sum(int(think1024[i]["correct"]) for i in range(N)) / N
think1024_avg_tok = sum(think1024[i]["tokens"] for i in range(N)) / N
think2048_acc = sum(int(think2048[i]["correct"]) for i in range(N)) / N
think2048_avg_tok = sum(think2048[i]["tokens"] for i in range(N)) / N

print(f"  Nothink@256:   acc={nothink_acc:.4f} ({nothink_acc*100:.2f}%), avg_tok={nothink_avg_tok:.1f}")
print(f"  Thinking@512:  acc={think512_acc:.4f} ({think512_acc*100:.2f}%), avg_tok={think512_avg_tok:.1f}")
print(f"  Thinking@1024: acc={think1024_acc:.4f} ({think1024_acc*100:.2f}%), avg_tok={think1024_avg_tok:.1f}")
print(f"  Thinking@2048: acc={think2048_acc:.4f} ({think2048_acc*100:.2f}%), avg_tok={think2048_avg_tok:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 2. TOWN ROUTER (token-length / hit_budget)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. TOWN ROUTER (route samples that hit nothink@256 budget)")
print("="*70)

town_routed = [i for i in range(N) if nothink[i]["hit_budget"]]
town_result = evaluate_router(town_routed, "512", "TOWN (hit_budget → think@512)")
print(f"  Routed: {town_result['n_routed']}/{N} ({town_result['route_rate']*100:.1f}%)")
print(f"  Combined acc: {town_result['accuracy']*100:.2f}%")
print(f"  Avg tokens:   {town_result['avg_tokens']:.1f} (overhead: +{town_result['token_overhead']:.1f})")
print(f"  Routed acc:   {town_result['routed_accuracy']*100:.1f}% | Non-routed acc: {town_result['nonrouted_accuracy']*100:.1f}%")

# Also evaluate TOWN with higher thinking budgets
town_1024 = evaluate_router(town_routed, "1024", "TOWN (hit_budget → think@1024)")
town_2048 = evaluate_router(town_routed, "2048", "TOWN (hit_budget → think@2048)")
print(f"\n  TOWN → think@1024: acc={town_1024['accuracy']*100:.2f}%, avg_tok={town_1024['avg_tokens']:.1f}")
print(f"  TOWN → think@2048: acc={town_2048['accuracy']*100:.2f}%, avg_tok={town_2048['avg_tokens']:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 3. ANSWER-PRESENCE ROUTER
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. ANSWER-PRESENCE ROUTER (route samples with pred_source != 'boxed')")
print("="*70)

# pred_source='boxed' means a \\boxed{} answer was found
# pred_source='last_number' means we fell back to extracting the last number
# Samples without a clear boxed answer are less confident
no_boxed = [i for i in range(N) if nothink[i]["pred_source"] != "boxed"]
has_boxed = [i for i in range(N) if nothink[i]["pred_source"] == "boxed"]
print(f"  boxed: {len(has_boxed)}, non-boxed: {len(no_boxed)}")
# Since almost all are last_number (1311), this isn't very selective
# Instead, check has_final flag
no_final = [i for i in range(N) if not nothink[i]["has_final"]]
has_final_idx = [i for i in range(N) if nothink[i]["has_final"]]
print(f"  has_final: {len(has_final_idx)}, no_final: {len(no_final)}")

# For nothink mode, most answers come via last_number extraction
# The real signal is whether the prediction looks "parseable"
# Let's check: samples where pred is None, empty, or non-numeric
unparseable = [i for i in range(N) if nothink[i]["pred"] is None or nothink[i]["pred"].strip() == ""]
print(f"  Unparseable predictions: {len(unparseable)}")

# Since pred_source is not discriminative enough (1311/1319 are last_number),
# let's use a different signal: hit_budget + token length

# ══════════════════════════════════════════════════════════════════════════
# 4. TOKEN-LENGTH THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("4. TOKEN-LENGTH THRESHOLD SWEEP")
print("   Route samples with token_count >= threshold to thinking@512")
print("="*70)

# Sorted token counts
token_counts = [(i, nothink[i]["tokens"]) for i in range(N)]

# Sweep thresholds from low to high
thresholds = list(range(50, 260, 10))  # 50, 60, ..., 250, 256
thresholds.append(256)
thresholds = sorted(set(thresholds))

sweep_results = []
for thresh in thresholds:
    routed = [i for i in range(N) if nothink[i]["tokens"] >= thresh]
    res = evaluate_router(routed, "512", f"token_length >= {thresh}")
    sweep_results.append({
        "threshold": thresh,
        **res
    })
    if thresh in [100, 150, 200, 250, 256]:
        print(f"  thresh={thresh:3d}: routed={res['n_routed']:4d} ({res['route_rate']*100:5.1f}%), "
              f"acc={res['accuracy']*100:.2f}%, avg_tok={res['avg_tokens']:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 5. ROUTE-TOP-K% SWEEP (by token length, descending)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("5. ROUTE-TOP-K% SWEEP (route K% of samples with longest token counts)")
print("="*70)

# Sort by token count descending
sorted_by_tokens = sorted(range(N), key=lambda i: nothink[i]["tokens"], reverse=True)

topk_results = []
percentages = [1, 2, 3, 5, 7, 10, 11.2, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

for pct in percentages:
    k = int(round(N * pct / 100))
    k = min(k, N)
    routed = sorted_by_tokens[:k]
    res = evaluate_router(routed, "512", f"top-{pct}% by token length")
    topk_results.append({
        "route_pct": pct,
        "k": k,
        **res
    })
    print(f"  top-{pct:5.1f}%: k={k:4d}, acc={res['accuracy']*100:.2f}%, "
          f"avg_tok={res['avg_tokens']:.1f}, routed_acc={res['routed_accuracy']*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 6. RANDOM ROUTER (averaged over seeds)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("6. RANDOM ROUTER (average over 100 seeds)")
print("="*70)

N_SEEDS = 100
random_results = []

for pct in [5, 10, 11.2, 15, 20, 30, 50]:
    k = int(round(N * pct / 100))
    accs = []
    toks = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        routed = rng.sample(range(N), k)
        res = evaluate_router(routed, "512")
        accs.append(res["accuracy"])
        toks.append(res["avg_tokens"])

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    mean_tok = np.mean(toks)

    random_results.append({
        "route_pct": pct,
        "k": k,
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "mean_avg_tokens": float(mean_tok),
        "ci95_low": float(mean_acc - 1.96 * std_acc),
        "ci95_high": float(mean_acc + 1.96 * std_acc),
    })
    print(f"  random-{pct:5.1f}%: k={k:4d}, acc={mean_acc*100:.2f}% ± {std_acc*100:.2f}%, avg_tok={mean_tok:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 7. ORACLE ROUTER (best possible)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("7. ORACLE ROUTER (route when thinking@512 correct AND nothink@256 wrong)")
print("="*70)

# Samples where nothink wrong but thinking right
oracle_routed = [i for i in range(N)
                 if not nothink[i]["correct"] and think512[i]["correct"]]
# Samples where nothink wrong and thinking also wrong → don't route, no help
both_wrong = [i for i in range(N)
              if not nothink[i]["correct"] and not think512[i]["correct"]]
# Samples where nothink right but thinking wrong → must NOT route
regress = [i for i in range(N)
           if nothink[i]["correct"] and not think512[i]["correct"]]

oracle_result = evaluate_router(oracle_routed, "512", "Oracle (route only beneficial)")
print(f"  Oracle routed: {len(oracle_routed)} ({len(oracle_routed)/N*100:.1f}%)")
print(f"  Both wrong (unhelpable by think@512): {len(both_wrong)}")
print(f"  Would regress if routed: {len(regress)}")
print(f"  Oracle acc: {oracle_result['accuracy']*100:.2f}%")
print(f"  Oracle avg_tok: {oracle_result['avg_tokens']:.1f}")

# Perfect oracle: route wrong→right, avoid right→wrong
# Upper bound accuracy
nothink_correct_count = sum(int(nothink[i]["correct"]) for i in range(N))
upper_bound = (nothink_correct_count + len(oracle_routed)) / N
print(f"  Upper bound acc (oracle routing): {upper_bound*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════
# 8. COMPOSITE SIGNALS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("8. COMPOSITE SIGNALS")
print("="*70)

# Signal 1: hit_budget (TOWN)
# Signal 2: high token count (close to budget but didn't hit)
# Signal 3: non-boxed answer

# Hybrid: hit_budget OR token >= 240 (catches borderline cases)
for soft_thresh in [200, 210, 220, 230, 240, 250]:
    hybrid = [i for i in range(N) if nothink[i]["hit_budget"] or nothink[i]["tokens"] >= soft_thresh]
    res = evaluate_router(hybrid, "512", f"TOWN + token>={soft_thresh}")
    print(f"  TOWN ∪ token>={soft_thresh}: routed={res['n_routed']:4d} ({res['route_rate']*100:5.1f}%), "
          f"acc={res['accuracy']*100:.2f}%, avg_tok={res['avg_tokens']:.1f}")

# ══════════════════════════════════════════════════════════════════════════
# 9. QUALITY OF ROUTING SIGNAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("9. ROUTING SIGNAL QUALITY ANALYSIS")
print("="*70)

# For TOWN router: precision and recall of identifying "needs thinking" samples
# "needs thinking" = nothink wrong AND thinking right
needs_thinking = set(oracle_routed)
town_set = set(town_routed)

if needs_thinking:
    tp = len(town_set & needs_thinking)
    fp = len(town_set - needs_thinking)
    fn = len(needs_thinking - town_set)
    tn = N - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  TOWN signal quality for 'needs thinking' detection:")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

# Breakdown of TOWN routed samples
print(f"\n  TOWN routed samples ({len(town_routed)} total):")
town_nothink_correct = sum(int(nothink[i]["correct"]) for i in town_routed)
town_think_correct = sum(int(think512[i]["correct"]) for i in town_routed)
print(f"    Nothink correct: {town_nothink_correct} ({town_nothink_correct/len(town_routed)*100:.1f}%)")
print(f"    Think@512 correct: {town_think_correct} ({town_think_correct/len(town_routed)*100:.1f}%)")
town_improved = sum(1 for i in town_routed if not nothink[i]["correct"] and think512[i]["correct"])
town_regressed = sum(1 for i in town_routed if nothink[i]["correct"] and not think512[i]["correct"])
town_both_correct = sum(1 for i in town_routed if nothink[i]["correct"] and think512[i]["correct"])
town_both_wrong = sum(1 for i in town_routed if not nothink[i]["correct"] and not think512[i]["correct"])
print(f"    Improved:    {town_improved} (nothink✗→think✓)")
print(f"    Regressed:   {town_regressed} (nothink✓→think✗)")
print(f"    Both correct:{town_both_correct}")
print(f"    Both wrong:  {town_both_wrong}")
print(f"    Net gain:    {town_improved - town_regressed} samples")

# ══════════════════════════════════════════════════════════════════════════
# 10. TOKEN DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("10. TOKEN DISTRIBUTION ANALYSIS")
print("="*70)

# Token length distribution by correctness
correct_tokens = [nothink[i]["tokens"] for i in range(N) if nothink[i]["correct"]]
wrong_tokens = [nothink[i]["tokens"] for i in range(N) if not nothink[i]["correct"]]

print(f"  Correct samples ({len(correct_tokens)}):")
print(f"    mean={np.mean(correct_tokens):.1f}, median={np.median(correct_tokens):.1f}, "
      f"std={np.std(correct_tokens):.1f}")
print(f"  Wrong samples ({len(wrong_tokens)}):")
print(f"    mean={np.mean(wrong_tokens):.1f}, median={np.median(wrong_tokens):.1f}, "
      f"std={np.std(wrong_tokens):.1f}")

# Token length histogram (10 bins)
bins = list(range(0, 270, 32))  # 0-32, 32-64, ..., 224-256
print(f"\n  Token distribution (correct vs wrong):")
print(f"  {'Bin':>10s} | {'Correct':>8s} | {'Wrong':>8s} | {'Wrong%':>7s}")
print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
for lo, hi in zip(bins, bins[1:] + [257]):
    c = sum(1 for t in correct_tokens if lo <= t < hi)
    w = sum(1 for t in wrong_tokens if lo <= t < hi)
    total = c + w
    wpct = w / total * 100 if total > 0 else 0
    print(f"  {lo:3d}-{hi:3d}   | {c:8d} | {w:8d} | {wpct:6.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# 11. EFFICIENCY COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("11. EFFICIENCY COMPARISON TABLE (for paper)")
print("="*70)

# All at ~11.2% route rate for fair comparison
town_at_11 = town_result  # exactly 11.2%

# Random at 11.2%
k_11 = int(round(N * 0.112))
random_11_accs = []
for seed in range(N_SEEDS):
    rng = random.Random(seed)
    routed = rng.sample(range(N), k_11)
    res = evaluate_router(routed, "512")
    random_11_accs.append(res["accuracy"])
random_11_mean = np.mean(random_11_accs)
random_11_std = np.std(random_11_accs)

# Top-11.2% by token length (same as TOWN since hit_budget = max tokens)
topk_11 = sorted_by_tokens[:k_11]
topk_11_res = evaluate_router(topk_11, "512", "top-11.2% by tokens")

# Oracle at 11.2% (route the 11.2% that benefit most)
# Sort oracle candidates by: nothink_wrong AND think_right
oracle_candidates = sorted(
    range(N),
    key=lambda i: (
        # Priority 1: nothink wrong, thinking right (definitely route)
        -(not nothink[i]["correct"] and think512[i]["correct"]),
        # Priority 2: both wrong (no benefit)
        -(not nothink[i]["correct"] and not think512[i]["correct"]),
        # Priority 3: both correct (no harm but wasteful)
        -(nothink[i]["correct"] and think512[i]["correct"]),
    )
)
oracle_topk = oracle_candidates[:k_11]
oracle_topk_res = evaluate_router(oracle_topk, "512", "oracle-11.2%")

print(f"\n  {'Method':<30s} | {'Route%':>6s} | {'Accuracy':>8s} | {'Δ vs nothink':>12s} | {'Avg Tok':>8s}")
print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}")

rows = [
    ("Nothink@256 (no routing)", 0, nothink_acc, 0, nothink_avg_tok),
    ("Random@11.2%", 11.2, random_11_mean, random_11_mean - nothink_acc, None),
    ("TOWN (hit_budget)", town_result['route_rate']*100, town_result['accuracy'],
     town_result['accuracy'] - nothink_acc, town_result['avg_tokens']),
    ("Top-11.2% by tokens", 11.2, topk_11_res['accuracy'],
     topk_11_res['accuracy'] - nothink_acc, topk_11_res['avg_tokens']),
    ("Oracle@11.2%", 11.2, oracle_topk_res['accuracy'],
     oracle_topk_res['accuracy'] - nothink_acc, oracle_topk_res['avg_tokens']),
    ("Oracle (all beneficial)", len(oracle_routed)/N*100, oracle_result['accuracy'],
     oracle_result['accuracy'] - nothink_acc, oracle_result['avg_tokens']),
    ("Thinking@512 (all routed)", 100, think512_acc,
     think512_acc - nothink_acc, think512_avg_tok),
]

for name, pct, acc, delta, tok in rows:
    tok_str = f"{tok:.1f}" if tok is not None else "—"
    print(f"  {name:<30s} | {pct:5.1f}% | {acc*100:7.2f}% | {delta*100:+11.2f}% | {tok_str:>8s}")

# ══════════════════════════════════════════════════════════════════════════
# 12. KEY PAPER NUMBERS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("12. KEY PAPER NUMBERS")
print("="*70)

gain_over_nothink = (town_result['accuracy'] - nothink_acc) * 100
gain_over_random = (town_result['accuracy'] - random_11_mean) * 100
oracle_gap = (oracle_topk_res['accuracy'] - town_result['accuracy']) * 100
efficiency = (town_result['accuracy'] - nothink_acc) / (think512_acc - nothink_acc) * 100 if think512_acc != nothink_acc else 0

print(f"  TOWN accuracy:           {town_result['accuracy']*100:.2f}%")
print(f"  TOWN route rate:         {town_result['route_rate']*100:.1f}%")
print(f"  TOWN gain over nothink:  +{gain_over_nothink:.2f}pp")
print(f"  TOWN gain over random:   +{gain_over_random:.2f}pp")
print(f"  TOWN gap to oracle@11%:  {oracle_gap:.2f}pp")
print(f"  TOWN efficiency:         {efficiency:.1f}% of full-thinking gain, using {town_result['route_rate']*100:.1f}% budget")
print(f"  Token overhead:          +{town_result['token_overhead']:.1f} tokens/sample ({town_result['token_overhead']/nothink_avg_tok*100:.1f}% overhead)")

# Net improvement breakdown
print(f"\n  TOWN routing breakdown ({town_result['n_routed']} samples):")
print(f"    Improved (✗→✓):  {town_improved}")
print(f"    Regressed (✓→✗): {town_regressed}")
print(f"    Net gain:        {town_improved - town_regressed} correct answers")
print(f"    Net acc gain:    +{(town_improved - town_regressed)/N*100:.2f}pp")

# ══════════════════════════════════════════════════════════════════════════
# 13. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "meta": {
        "timestamp": timestamp,
        "model": "Qwen/Qwen3-8B",
        "benchmark": "gsm8k",
        "n_samples": N,
        "nothink_budget": 256,
        "thinking_budget": 512,
        "nothink_source": NOTHINK_256_PATH,
        "thinking_source": THINKING_512_PATH,
    },
    "baselines": {
        "nothink_256": {"accuracy": nothink_acc, "avg_tokens": nothink_avg_tok},
        "thinking_512": {"accuracy": think512_acc, "avg_tokens": think512_avg_tok},
        "thinking_1024": {"accuracy": think1024_acc, "avg_tokens": think1024_avg_tok},
        "thinking_2048": {"accuracy": think2048_acc, "avg_tokens": think2048_avg_tok},
    },
    "town_router": {
        "512": {k: v for k, v in town_result.items()},
        "1024": {k: v for k, v in town_1024.items()},
        "2048": {k: v for k, v in town_2048.items()},
    },
    "token_length_sweep": sweep_results,
    "topk_sweep": topk_results,
    "random_router": random_results,
    "oracle": {
        "all_beneficial": {
            "n_routed": len(oracle_routed),
            "accuracy": oracle_result["accuracy"],
            "avg_tokens": oracle_result["avg_tokens"],
        },
        "at_11pct": {
            "n_routed": k_11,
            "accuracy": oracle_topk_res["accuracy"],
            "avg_tokens": oracle_topk_res["avg_tokens"],
        }
    },
    "signal_quality": {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "town_improved": town_improved,
        "town_regressed": town_regressed,
        "town_net_gain": town_improved - town_regressed,
        "both_wrong": len(both_wrong),
        "would_regress": len(regress),
    },
    "composite_signals": [],
    "paper_numbers": {
        "town_accuracy": town_result["accuracy"],
        "town_route_rate": town_result["route_rate"],
        "town_gain_over_nothink_pp": gain_over_nothink,
        "town_gain_over_random_pp": gain_over_random,
        "town_gap_to_oracle_pp": oracle_gap,
        "town_efficiency_pct": efficiency,
        "token_overhead": town_result["token_overhead"],
    }
}

# Add composite signals
for soft_thresh in [200, 210, 220, 230, 240, 250]:
    hybrid = [i for i in range(N) if nothink[i]["hit_budget"] or nothink[i]["tokens"] >= soft_thresh]
    res = evaluate_router(hybrid, "512", f"TOWN + token>={soft_thresh}")
    output["composite_signals"].append({
        "threshold": soft_thresh,
        **res,
    })

output_path = os.path.join(OUTPUT_DIR, f"router_baselines_{timestamp}.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n  Results saved to: {output_path}")

# Also save a latest symlink-like copy
latest_path = os.path.join(OUTPUT_DIR, "router_baselines_latest.json")
with open(latest_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"  Latest copy at:  {latest_path}")

print("\n✅ Router baseline analysis complete!")
