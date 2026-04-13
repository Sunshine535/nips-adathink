#!/usr/bin/env python3
"""IRIS Batch-Level Budget Reallocation.

For production deployment with N queries and a total budget B_total:
  Phase 1: Run all queries through nothink probe (cheap)
  Phase 2: Identify hard queries (those that hit budget in Phase 1)
  Phase 3: Allocate remaining budget greedily via knapsack optimization

Formalization:
  max  Sum_i(Acc_i)
  s.t. Sum_i(T_i) <= B_total
  where T_i = tokens allocated to query i

Usage:
    # Fixed total budget for 1319 queries
    python scripts/run_iris_batch.py \
        --model Qwen/Qwen3-8B \
        --total_budget 200000 \
        --b1 256 --b2_max 512 --b_answer 128 \
        --seed 42

    # Average budget constraint
    python scripts/run_iris_batch.py \
        --model Qwen/Qwen3-8B \
        --avg_budget 200 \
        --seed 42
"""

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import shared utilities from run_iris
from run_iris import (
    load_model_and_tokenizer, build_prompt, generate_simple,
    generate_adaptive_thinking, generate_decoupled_answer,
    parse_prediction, is_correct, get_gold_from_gsm8k,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def load_gsm8k(n_samples: int, seed: int) -> List[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(idxs)
    selected = [ds[i] for i in idxs[:n_samples]]
    items = []
    for raw in selected:
        gold = get_gold_from_gsm8k(raw["answer"])
        items.append({"question": raw["question"], "gold": gold})
    return items


def knapsack_allocate(
    hard_indices: List[int],
    stage1_tokens: List[int],
    remaining_budget: int,
    b2_max: int,
    b_answer: int,
    n_priority_bins: int = 4,
) -> Dict[int, int]:
    """Greedy knapsack allocation for hard queries.

    Sorts hard queries by estimated difficulty (stage 1 tokens consumed),
    then allocates remaining budget greedily — queries closest to the
    nothink acceptance boundary get priority (highest marginal gain).

    Returns dict mapping query index -> allocated thinking budget.
    """
    if not hard_indices or remaining_budget <= 0:
        return {}

    # Sort hard queries by stage 1 tokens (ascending = closest to acceptance boundary)
    # These are the "easiest hard queries" — most likely to benefit from minimal thinking
    sorted_hard = sorted(hard_indices, key=lambda i: stage1_tokens[i])

    allocations = {}
    budget_left = remaining_budget

    # Priority tiers: allocate smaller budgets first to maximize coverage
    tier_budgets = []
    for tier in range(n_priority_bins):
        tier_budget = b2_max * (tier + 1) // n_priority_bins
        tier_budgets.append(min(tier_budget, b2_max))

    # Greedy: try to give each hard query the smallest effective budget
    for idx in sorted_hard:
        # Find the smallest tier budget that fits
        cost = tier_budgets[0] + b_answer  # Minimum: smallest tier + answer budget
        if budget_left >= cost:
            allocated = tier_budgets[0]
            allocations[idx] = allocated
            budget_left -= (allocated + b_answer)
        else:
            # Not enough budget — skip this query
            allocations[idx] = 0

    # Second pass: if budget remains, upgrade allocations
    for tier_i in range(1, len(tier_budgets)):
        for idx in sorted_hard:
            if idx not in allocations or allocations[idx] >= tier_budgets[tier_i]:
                continue
            upgrade_cost = tier_budgets[tier_i] - allocations[idx]
            if budget_left >= upgrade_cost:
                allocations[idx] = tier_budgets[tier_i]
                budget_left -= upgrade_cost

    return allocations


def main():
    parser = argparse.ArgumentParser(
        description="IRIS Batch-Level Budget Reallocation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--n_samples", type=int, default=99999)
    parser.add_argument("--seed", type=int, default=42)

    # Budget constraints (one of these)
    parser.add_argument("--total_budget", type=int, default=None,
                        help="Total token budget across all queries")
    parser.add_argument("--avg_budget", type=int, default=200,
                        help="Average token budget per query (default=200, matches TOWN)")

    # IRIS hyperparameters
    parser.add_argument("--b1", type=int, default=256)
    parser.add_argument("--b2_max", type=int, default=512)
    parser.add_argument("--b_answer", type=int, default=128)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--tau_h", type=float, default=1.5)
    parser.add_argument("--tau_s", type=float, default=50.0)

    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/iris_batch"
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_").replace("-", "_")

    # Load model and data
    model, tokenizer = load_model_and_tokenizer(args.model)
    items = load_gsm8k(args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} samples")

    # Compute total budget
    if args.total_budget is not None:
        total_budget = args.total_budget
    else:
        total_budget = args.avg_budget * n
    log.info(f"Total budget: {total_budget} ({total_budget/n:.0f} avg per query)")

    # ======= PHASE 1: Nothink probe all queries =======
    log.info("=" * 60)
    log.info("PHASE 1: Nothink probe (all queries)")
    log.info("=" * 60)

    phase1_results = []
    easy_indices = []
    hard_indices = []
    phase1_tokens_used = 0

    for i, item in enumerate(items):
        prompt = build_prompt(item["question"], tokenizer, enable_thinking=False)
        text, tokens, elapsed, hit_budget = generate_simple(
            model, tokenizer, prompt, max_new_tokens=args.b1, temperature=0.0,
        )
        pred, source = parse_prediction(text)
        correct = is_correct(pred, item["gold"])

        phase1_results.append({
            "idx": i,
            "tokens": tokens,
            "hit_budget": hit_budget,
            "pred": pred,
            "correct": int(correct),
        })
        phase1_tokens_used += tokens

        if hit_budget:
            hard_indices.append(i)
        else:
            easy_indices.append(i)

        if (i + 1) % 50 == 0 or i == n - 1:
            done = i + 1
            n_correct = sum(r["correct"] for r in phase1_results)
            log.info(
                f"  [{done}/{n}] acc={n_correct/done:.3f}  "
                f"easy={len(easy_indices)}  hard={len(hard_indices)}  "
                f"tokens_used={phase1_tokens_used}"
            )

    phase1_acc = sum(r["correct"] for r in phase1_results) / n
    log.info(f"\nPhase 1 done: acc={phase1_acc:.3f}, "
             f"easy={len(easy_indices)} ({len(easy_indices)/n:.1%}), "
             f"hard={len(hard_indices)} ({len(hard_indices)/n:.1%}), "
             f"tokens={phase1_tokens_used}")

    # ======= PHASE 2: Budget allocation =======
    remaining_budget = total_budget - phase1_tokens_used
    log.info(f"\nRemaining budget: {remaining_budget} "
             f"({remaining_budget/len(hard_indices):.0f} avg per hard query)"
             if hard_indices else "\nNo hard queries!")

    stage1_tokens = [r["tokens"] for r in phase1_results]
    allocations = knapsack_allocate(
        hard_indices, stage1_tokens, remaining_budget,
        b2_max=args.b2_max, b_answer=args.b_answer,
    )

    n_allocated = sum(1 for v in allocations.values() if v > 0)
    log.info(f"Allocated thinking budget to {n_allocated}/{len(hard_indices)} hard queries")

    # ======= PHASE 3: Run thinking for hard queries =======
    log.info("=" * 60)
    log.info("PHASE 3: Adaptive thinking for hard queries")
    log.info("=" * 60)

    phase3_results = {}
    phase3_tokens = 0
    phase3_correct = 0

    for j, idx in enumerate(hard_indices):
        alloc = allocations.get(idx, 0)
        if alloc <= 0:
            # No budget allocated — keep Phase 1 prediction
            phase3_results[idx] = {
                "upgraded": False,
                "pred": phase1_results[idx]["pred"],
                "correct": phase1_results[idx]["correct"],
                "extra_tokens": 0,
            }
            continue

        # Run adaptive thinking
        prompt_s2 = build_prompt(items[idx]["question"], tokenizer, enable_thinking=True)
        s2_result = generate_adaptive_thinking(
            model, tokenizer, prompt_s2,
            max_think_tokens=alloc,
            chunk_size=args.chunk_size,
            tau_h=args.tau_h,
            tau_s=args.tau_s,
        )

        # Check if thinking produced a natural stop with answer
        pred_final = None
        source_final = None
        extra_tokens = s2_result["n_tokens_used"]

        if s2_result["stop_reason"] == "natural_stop" and "</think>" in s2_result["full_text"]:
            after_think = s2_result["full_text"].split("</think>", 1)[1]
            pred_final, source_final = parse_prediction(after_think)

        if pred_final is None:
            # Decoupled answer generation
            answer_text, answer_tokens, _ = generate_decoupled_answer(
                model, tokenizer, items[idx]["question"],
                s2_result["thinking_text"], answer_budget=args.b_answer,
            )
            pred_final, source_final = parse_prediction(answer_text)
            extra_tokens += answer_tokens

        correct = is_correct(pred_final, items[idx]["gold"])
        if correct:
            phase3_correct += 1
        phase3_tokens += extra_tokens

        phase3_results[idx] = {
            "upgraded": True,
            "pred": pred_final,
            "correct": int(correct),
            "extra_tokens": extra_tokens,
            "stop_reason": s2_result["stop_reason"],
            "tokens_saved": s2_result["tokens_saved"],
        }

        if (j + 1) % 20 == 0 or j == len(hard_indices) - 1:
            done = j + 1
            upgraded = sum(1 for r in phase3_results.values() if r.get("upgraded"))
            log.info(f"  [{done}/{len(hard_indices)}] upgraded={upgraded}  "
                     f"extra_tokens={phase3_tokens}")

    # ======= FINAL RESULTS =======
    # Merge Phase 1 (easy) + Phase 3 (hard, possibly upgraded)
    final_correct = 0
    final_tokens = phase1_tokens_used + phase3_tokens
    final_results = []

    for i in range(n):
        if i in easy_indices:
            correct = phase1_results[i]["correct"]
            pred = phase1_results[i]["pred"]
            stage = "easy"
            tokens = phase1_results[i]["tokens"]
        else:
            r3 = phase3_results.get(i, {})
            correct = r3.get("correct", phase1_results[i]["correct"])
            pred = r3.get("pred", phase1_results[i]["pred"])
            stage = "hard_upgraded" if r3.get("upgraded") else "hard_skipped"
            tokens = phase1_results[i]["tokens"] + r3.get("extra_tokens", 0)

        if correct:
            final_correct += 1

        final_results.append({
            "idx": i,
            "gold": items[i]["gold"],
            "pred": pred,
            "correct": int(correct),
            "stage": stage,
            "tokens": tokens,
        })

    final_acc = final_correct / n
    final_avg_tokens = final_tokens / n

    # Comparison metrics
    phase1_only_acc = phase1_acc  # If we only used Phase 1

    # Recovery analysis
    recovered = 0
    regressed = 0
    for idx in hard_indices:
        p1_correct = phase1_results[idx]["correct"]
        p3_correct = phase3_results.get(idx, {}).get("correct", p1_correct)
        if not p1_correct and p3_correct:
            recovered += 1
        elif p1_correct and not p3_correct:
            regressed += 1

    summary = {
        "meta": {
            "script": "run_iris_batch.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "n_samples": n,
            "seed": args.seed,
            "total_budget": total_budget,
            "avg_budget": total_budget / n,
        },
        "config": {
            "b1": args.b1,
            "b2_max": args.b2_max,
            "b_answer": args.b_answer,
            "chunk_size": args.chunk_size,
            "tau_h": args.tau_h,
            "tau_s": args.tau_s,
        },
        "results": {
            "final_accuracy": round(final_acc, 4),
            "final_avg_tokens": round(final_avg_tokens, 2),
            "phase1_only_accuracy": round(phase1_only_acc, 4),
            "phase1_tokens": phase1_tokens_used,
            "phase3_extra_tokens": phase3_tokens,
            "total_tokens": final_tokens,
            "n_easy": len(easy_indices),
            "n_hard": len(hard_indices),
            "n_upgraded": sum(1 for r in phase3_results.values() if r.get("upgraded")),
            "recovered": recovered,
            "regressed": regressed,
            "net_gain": recovered - regressed,
        },
        "per_sample": final_results,
    }

    # Save
    out_fname = f"iris_batch_{model_tag}_avg{total_budget//n}_{timestamp}.json"
    out_path = os.path.join(args.output_dir, out_fname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\nSaved: {out_path}")

    # Print summary
    log.info("")
    log.info("=" * 70)
    log.info("IRIS BATCH REALLOCATION — RESULTS")
    log.info("=" * 70)
    log.info(f"  Model:               {args.model}")
    log.info(f"  Samples:             {n}")
    log.info(f"  Total budget:        {total_budget} ({total_budget/n:.0f} avg)")
    log.info(f"  Phase 1 accuracy:    {phase1_only_acc:.1%}")
    log.info(f"  Final accuracy:      {final_acc:.1%}")
    log.info(f"  Accuracy gain:       {(final_acc - phase1_only_acc)*100:+.1f}pp")
    log.info(f"  Avg tokens used:     {final_avg_tokens:.0f}")
    log.info(f"  Easy (S1 accepted):  {len(easy_indices)} ({len(easy_indices)/n:.1%})")
    log.info(f"  Hard (escalated):    {len(hard_indices)} ({len(hard_indices)/n:.1%})")
    log.info(f"  Upgraded:            {sum(1 for r in phase3_results.values() if r.get('upgraded'))}")
    log.info(f"  Recovered:           {recovered}")
    log.info(f"  Regressed:           {regressed}")
    log.info(f"  Net gain:            {recovered - regressed}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
