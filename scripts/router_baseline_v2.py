#!/usr/bin/env python3
"""
Router Baseline Analysis v2 — Multi-Budget TOWN Analysis
=========================================================
Key finding from v1: thinking@512 (57.5% acc) is too low-budget, causing TOWN
to HURT accuracy. We need to analyze TOWN with think@1024 and think@2048.

This script produces the definitive router comparison table for the paper.
"""

import json
import os
import random
import numpy as np
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
NOTHINK_256 = "results/nothink_fullset/nothink_baseline_Qwen3-8B_gsm8k_20260329_000345.json"
THINK_PATHS = {
    512:  "results/thinking_hf/hf_512/nothink_baseline_Qwen3-8B_gsm8k_20260330_125514.json",
    1024: "results/thinking_hf/hf_1024/nothink_baseline_Qwen3-8B_gsm8k_20260330_125513.json",
    2048: "results/thinking_hf/hf_2048/nothink_baseline_Qwen3-8B_gsm8k_20260330_125514.json",
}
OUTPUT_DIR = "results/router_baselines"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SEEDS = 200  # for random baselines

# ── Load Data ──────────────────────────────────────────────────────────────
print("Loading data...")
with open(NOTHINK_256) as f:
    nd = json.load(f)
nothink = {s["idx"]: s for s in nd["per_sample"]["nothink_256"]}
N = len(nothink)
assert N == 1319

think = {}
for budget, path in THINK_PATHS.items():
    with open(path) as f:
        td = json.load(f)
    think[budget] = {s["idx"]: s for s in td["per_sample"][f"thinking_{budget}"]}
    assert len(think[budget]) == N

print(f"Loaded {N} samples with thinking budgets: {sorted(THINK_PATHS.keys())}")

# ── Precompute nothink stats ─────────────────────────────────────────────
nothink_acc = sum(int(nothink[i]["correct"]) for i in range(N)) / N
nothink_avg_tok = sum(nothink[i]["tokens"] for i in range(N)) / N
town_indices = [i for i in range(N) if nothink[i]["hit_budget"]]
town_set = set(town_indices)
sorted_by_tokens = sorted(range(N), key=lambda i: nothink[i]["tokens"], reverse=True)

# ── Evaluate Router ──────────────────────────────────────────────────────
def eval_router(routed_indices, budget):
    """Return (accuracy, avg_tokens, improved, regressed, net_gain)."""
    td = think[budget]
    rs = set(routed_indices)
    correct = 0
    total_tok = 0
    improved = 0
    regressed = 0
    for i in range(N):
        if i in rs:
            correct += int(td[i]["correct"])
            total_tok += td[i]["tokens"]
            if not nothink[i]["correct"] and td[i]["correct"]:
                improved += 1
            if nothink[i]["correct"] and not td[i]["correct"]:
                regressed += 1
        else:
            correct += int(nothink[i]["correct"])
            total_tok += nothink[i]["tokens"]
    return {
        "accuracy": correct / N,
        "avg_tokens": total_tok / N,
        "n_routed": len(rs),
        "route_rate": len(rs) / N,
        "improved": improved,
        "regressed": regressed,
        "net_gain": improved - regressed,
    }

# ══════════════════════════════════════════════════════════════════════════
# TABLE 1: Pure baselines
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 1: PURE BASELINES (no routing)")
print("="*80)
print(f"  {'Mode':<20s} | {'Accuracy':>8s} | {'Avg Tok':>8s}")
print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}")
print(f"  {'Nothink@256':<20s} | {nothink_acc*100:7.2f}% | {nothink_avg_tok:7.1f}")
for b in sorted(think.keys()):
    td = think[b]
    acc = sum(int(td[i]["correct"]) for i in range(N)) / N
    avg = sum(td[i]["tokens"] for i in range(N)) / N
    print(f"  {'Thinking@'+str(b):<20s} | {acc*100:7.2f}% | {avg:7.1f}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 2: TOWN Router across thinking budgets
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 2: TOWN ROUTER (route hit_budget samples to different thinking budgets)")
print(f"  TOWN routes {len(town_indices)}/{N} = {len(town_indices)/N*100:.1f}% of samples")
print("="*80)
print(f"  {'Think Budget':>12s} | {'Accuracy':>8s} | {'Delta':>7s} | {'Avg Tok':>8s} | {'Improved':>8s} | {'Regressed':>9s} | {'Net':>4s}")
print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*4}")

town_multi = {}
for b in sorted(think.keys()):
    r = eval_router(town_indices, b)
    town_multi[b] = r
    delta = r["accuracy"] - nothink_acc
    print(f"  {'think@'+str(b):>12s} | {r['accuracy']*100:7.2f}% | {delta*100:+6.2f}% | {r['avg_tokens']:7.1f} | {r['improved']:8d} | {r['regressed']:9d} | {r['net_gain']:+4d}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 3: Router Comparison at matched route rates
# For each thinking budget, compare TOWN vs Random vs Oracle
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 3: ROUTER COMPARISON (TOWN vs Random vs Oracle, matched route rate)")
print("="*80)

for b in sorted(think.keys()):
    td = think[b]
    think_full_acc = sum(int(td[i]["correct"]) for i in range(N)) / N

    # TOWN
    town_r = eval_router(town_indices, b)

    # Random at same rate
    k = len(town_indices)
    rand_accs = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        routed = rng.sample(range(N), k)
        rand_accs.append(eval_router(routed, b)["accuracy"])
    rand_mean = np.mean(rand_accs)
    rand_std = np.std(rand_accs)

    # Oracle at same rate: prioritize routing nothink-wrong & think-right
    oracle_candidates = sorted(
        range(N),
        key=lambda i: (
            -(not nothink[i]["correct"] and td[i]["correct"]),   # best: fix wrong
            -(not nothink[i]["correct"] and not td[i]["correct"]),  # neutral: both wrong
            -(nothink[i]["correct"] and td[i]["correct"]),         # neutral: both right
            # worst: nothink right but think wrong (regression)
        )
    )
    oracle_r = eval_router(oracle_candidates[:k], b)

    # Full oracle (route all beneficial)
    beneficial = [i for i in range(N) if not nothink[i]["correct"] and td[i]["correct"]]
    full_oracle_r = eval_router(beneficial, b)

    print(f"\n  --- Thinking Budget: {b} (full thinking acc: {think_full_acc*100:.2f}%) ---")
    print(f"  {'Method':<30s} | {'Route%':>6s} | {'Accuracy':>8s} | {'Delta':>7s} | {'Avg Tok':>8s}")
    print(f"  {'-'*30}-+-{'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
    print(f"  {'Nothink@256 (no route)':<30s} | {'0.0%':>6s} | {nothink_acc*100:7.2f}% | {'+0.00%':>7s} | {nothink_avg_tok:7.1f}")
    print(f"  {'Random@11.2%':<30s} | {k/N*100:5.1f}% | {rand_mean*100:7.2f}% | {(rand_mean-nothink_acc)*100:+6.2f}% | {'—':>8s}")
    print(f"  {'TOWN (hit_budget)':<30s} | {town_r['route_rate']*100:5.1f}% | {town_r['accuracy']*100:7.2f}% | {(town_r['accuracy']-nothink_acc)*100:+6.2f}% | {town_r['avg_tokens']:7.1f}")
    print(f"  {'Oracle@11.2%':<30s} | {k/N*100:5.1f}% | {oracle_r['accuracy']*100:7.2f}% | {(oracle_r['accuracy']-nothink_acc)*100:+6.2f}% | {oracle_r['avg_tokens']:7.1f}")
    print(f"  {'Oracle (all beneficial)':<30s} | {len(beneficial)/N*100:5.1f}% | {full_oracle_r['accuracy']*100:7.2f}% | {(full_oracle_r['accuracy']-nothink_acc)*100:+6.2f}% | {full_oracle_r['avg_tokens']:7.1f}")
    print(f"  {'Thinking@'+str(b)+' (all route)':<30s} | {'100%':>6s} | {think_full_acc*100:7.2f}% | {(think_full_acc-nothink_acc)*100:+6.2f}% | {sum(td[i]['tokens'] for i in range(N))/N:7.1f}")

    # TOWN advantage over random
    adv = (town_r['accuracy'] - rand_mean) * 100
    print(f"  >> TOWN advantage over random: {adv:+.2f}pp")
    print(f"  >> TOWN gap to oracle@11.2%:   {(oracle_r['accuracy']-town_r['accuracy'])*100:.2f}pp")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 4: Top-K% sweep with think@1024 (the sweet spot)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 4: TOP-K% SWEEP (route by token length → think@1024)")
print("="*80)

pcts = [1, 2, 3, 5, 7, 10, 11.2, 15, 20, 25, 30, 50, 100]
topk_1024 = []
print(f"  {'Route%':>7s} | {'k':>5s} | {'Accuracy':>8s} | {'Delta':>7s} | {'Avg Tok':>8s}")
print(f"  {'-'*7}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}")
for pct in pcts:
    k = min(int(round(N * pct / 100)), N)
    routed = sorted_by_tokens[:k]
    r = eval_router(routed, 1024)
    delta = r["accuracy"] - nothink_acc
    topk_1024.append({"pct": pct, "k": k, **r})
    print(f"  {pct:6.1f}% | {k:5d} | {r['accuracy']*100:7.2f}% | {delta*100:+6.2f}% | {r['avg_tokens']:7.1f}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 5: Signal quality analysis for each thinking budget
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 5: TOWN SIGNAL QUALITY (precision/recall for 'needs thinking')")
print("="*80)

signal_quality = {}
for b in sorted(think.keys()):
    td = think[b]
    needs_thinking = set(i for i in range(N) if not nothink[i]["correct"] and td[i]["correct"])
    tp = len(town_set & needs_thinking)
    fp = len(town_set - needs_thinking)
    fn = len(needs_thinking - town_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    signal_quality[b] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "needs_thinking": len(needs_thinking)}
    print(f"  think@{b}: needs_thinking={len(needs_thinking):3d}, TP={tp:3d}, FP={fp:3d}, FN={fn:3d}, "
          f"Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 6: Pareto analysis — accuracy vs token cost
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 6: PARETO ANALYSIS — accuracy vs average token cost")
print("="*80)

pareto_points = []
# Nothink baseline
pareto_points.append(("Nothink@256", nothink_acc, nothink_avg_tok))

# TOWN with each budget
for b in sorted(think.keys()):
    r = eval_router(town_indices, b)
    pareto_points.append((f"TOWN→think@{b}", r["accuracy"], r["avg_tokens"]))

# Full thinking baselines
for b in sorted(think.keys()):
    td = think[b]
    acc = sum(int(td[i]["correct"]) for i in range(N)) / N
    avg = sum(td[i]["tokens"] for i in range(N)) / N
    pareto_points.append((f"Thinking@{b}", acc, avg))

pareto_points.sort(key=lambda x: x[2])

print(f"  {'Method':<25s} | {'Accuracy':>8s} | {'Avg Tok':>8s} | {'Acc/$tok':>10s}")
print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
for name, acc, tok in pareto_points:
    eff = acc / tok * 1000  # accuracy per 1000 tokens
    print(f"  {name:<25s} | {acc*100:7.2f}% | {tok:7.1f} | {eff:9.2f}%/kt")

# ══════════════════════════════════════════════════════════════════════════
# TABLE 7: Token distribution of TOWN-routed samples by outcome
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("TABLE 7: TOWN ROUTING OUTCOME ANALYSIS (think@1024)")
print("="*80)

td = think[1024]
# Among routed samples
improved = [i for i in town_indices if not nothink[i]["correct"] and td[i]["correct"]]
regressed = [i for i in town_indices if nothink[i]["correct"] and not td[i]["correct"]]
both_right = [i for i in town_indices if nothink[i]["correct"] and td[i]["correct"]]
both_wrong = [i for i in town_indices if not nothink[i]["correct"] and not td[i]["correct"]]

print(f"  TOWN routes {len(town_indices)} samples → think@1024:")
print(f"    Improved  (nothink wrong → think right): {len(improved):3d} ({len(improved)/len(town_indices)*100:.1f}%)")
print(f"    Regressed (nothink right → think wrong): {len(regressed):3d} ({len(regressed)/len(town_indices)*100:.1f}%)")
print(f"    Both right:                              {len(both_right):3d} ({len(both_right)/len(town_indices)*100:.1f}%)")
print(f"    Both wrong:                              {len(both_wrong):3d} ({len(both_wrong)/len(town_indices)*100:.1f}%)")
print(f"    Net gain: {len(improved) - len(regressed):+d} correct answers = {(len(improved)-len(regressed))/N*100:+.2f}pp")

# Also for think@2048
td2 = think[2048]
improved2 = [i for i in town_indices if not nothink[i]["correct"] and td2[i]["correct"]]
regressed2 = [i for i in town_indices if nothink[i]["correct"] and not td2[i]["correct"]]
print(f"\n  TOWN routes {len(town_indices)} samples → think@2048:")
print(f"    Improved: {len(improved2):3d}, Regressed: {len(regressed2):3d}, Net: {len(improved2)-len(regressed2):+d} = {(len(improved2)-len(regressed2))/N*100:+.2f}pp")

# ══════════════════════════════════════════════════════════════════════════
# KEY PAPER NUMBERS (corrected)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("KEY PAPER NUMBERS")
print("="*80)

# Focus on think@1024 as the recommended configuration
r1024 = town_multi[1024]
k = len(town_indices)
rand_1024_accs = []
for seed in range(N_SEEDS):
    rng = random.Random(seed)
    routed = rng.sample(range(N), k)
    rand_1024_accs.append(eval_router(routed, 1024)["accuracy"])
rand_1024_mean = np.mean(rand_1024_accs)

print(f"\n  [Recommended config: nothink@256 → think@1024]")
print(f"  Nothink@256 baseline:  {nothink_acc*100:.2f}%")
print(f"  TOWN accuracy:         {r1024['accuracy']*100:.2f}% (+{(r1024['accuracy']-nothink_acc)*100:.2f}pp)")
print(f"  TOWN route rate:       {r1024['route_rate']*100:.1f}%")
print(f"  TOWN avg tokens:       {r1024['avg_tokens']:.1f} (overhead: +{r1024['avg_tokens']-nothink_avg_tok:.1f})")
print(f"  Random@11.2% baseline: {rand_1024_mean*100:.2f}%")
print(f"  TOWN vs random:        +{(r1024['accuracy']-rand_1024_mean)*100:.2f}pp")
print(f"  TOWN improved:         {r1024['improved']} samples")
print(f"  TOWN regressed:        {r1024['regressed']} samples")
print(f"  TOWN net gain:         {r1024['net_gain']:+d} ({r1024['net_gain']/N*100:+.2f}pp)")

# Efficiency: what fraction of possible gain does TOWN capture?
think1024_full_acc = sum(int(think[1024][i]["correct"]) for i in range(N)) / N
if think1024_full_acc > nothink_acc:
    eff = (r1024['accuracy'] - nothink_acc) / (think1024_full_acc - nothink_acc) * 100
    print(f"  Efficiency: {eff:.1f}% of think@1024 gain ({think1024_full_acc*100:.2f}%)")
    print(f"    ...using only {r1024['route_rate']*100:.1f}% routing → {eff/r1024['route_rate']*100:.0f}x leverage")

# think@2048 numbers
r2048 = town_multi[2048]
print(f"\n  [Alternative: nothink@256 → think@2048]")
print(f"  TOWN accuracy:         {r2048['accuracy']*100:.2f}% (+{(r2048['accuracy']-nothink_acc)*100:.2f}pp)")
print(f"  TOWN avg tokens:       {r2048['avg_tokens']:.1f}")
print(f"  Net gain:              {r2048['net_gain']:+d} samples")

# Why think@512 fails
r512 = town_multi[512]
print(f"\n  [Why think@512 fails]")
print(f"  TOWN→think@512 acc:    {r512['accuracy']*100:.2f}% ({(r512['accuracy']-nothink_acc)*100:+.2f}pp)")
print(f"  Improved: {r512['improved']}, Regressed: {r512['regressed']}, Net: {r512['net_gain']:+d}")
print(f"  Thinking@512 full acc: {sum(int(think[512][i]['correct']) for i in range(N))/N*100:.2f}% << nothink@256")
print(f"  → 512 tokens insufficient for thinking; 67.6% of hit_budget samples still wrong under think@512")

# ══════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = {
    "meta": {
        "timestamp": timestamp,
        "model": "Qwen/Qwen3-8B",
        "benchmark": "gsm8k",
        "n_samples": N,
        "nothink_budget": 256,
        "analysis_version": "v2_multi_budget",
    },
    "baselines": {
        "nothink_256": {"accuracy": nothink_acc, "avg_tokens": nothink_avg_tok},
        **{f"thinking_{b}": {
            "accuracy": sum(int(think[b][i]["correct"]) for i in range(N)) / N,
            "avg_tokens": sum(think[b][i]["tokens"] for i in range(N)) / N,
        } for b in sorted(think.keys())}
    },
    "town_router": {str(b): r for b, r in town_multi.items()},
    "topk_sweep_1024": topk_1024,
    "signal_quality": {str(b): v for b, v in signal_quality.items()},
    "pareto": [{"method": n, "accuracy": a, "avg_tokens": t} for n, a, t in pareto_points],
    "random_baselines": {
        "1024": {"mean": float(rand_1024_mean), "std": float(np.std(rand_1024_accs)), "n_seeds": N_SEEDS},
    },
    "routing_outcomes_1024": {
        "improved": len(improved),
        "regressed": len(regressed),
        "both_right": len(both_right),
        "both_wrong": len(both_wrong),
        "net_gain": len(improved) - len(regressed),
    },
}

out_path = os.path.join(OUTPUT_DIR, f"router_baselines_v2_{timestamp}.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

latest = os.path.join(OUTPUT_DIR, "router_baselines_v2_latest.json")
with open(latest, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Saved: {out_path}")
print(f"  Latest: {latest}")
print("\nDone!")
