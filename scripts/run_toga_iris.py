#!/usr/bin/env python3
"""TOGA-IRIS: Theory-Guided Optimal Budget Allocation + IRIS cascade.

Operationalizes Proposition 6 from the paper: instead of fixed (B_r, B_a)
allocation, TOGA computes optimal splits based on difficulty tiers.

Key idea: The truncation-waste decomposition predicts accuracy from
chain-length CDF F_L and extraction success rate α_extract. By fitting F_L
from calibration data, we can compute the optimal (B_r, B_a) split for
any total budget B, maximizing: F_L(B_r)·α_c + (1-F_L(B_r))·α_extract(B_r, B_a).

Difficulty tiers use Stage 0 nothink output length as a proxy:
- Tier 1 (short nothink, easy): smaller B_r, larger B_a
- Tier 2 (medium nothink): balanced split
- Tier 3 (long nothink / hit budget, hard): max B_r, min B_a

Usage:
    python scripts/run_toga_iris.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --n_samples 500 \
        --b1 512 --b_total 2048 \
        --seed 42 --output_dir results/toga_iris
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_iris import (
    load_model_and_tokenizer,
    build_prompt,
    generate_simple,
    generate_adaptive_thinking,
    generate_decoupled_answer,
    parse_prediction,
    parse_prediction_dispatch,
    is_correct_dispatch,
    load_benchmark_data,
    model_input_device,
    run_iris_sample,
    run_town_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOGA: Compute tier-specific budget allocations
# ---------------------------------------------------------------------------
def compute_toga_allocations(
    b_total: int,
    b_answer_min: int = 64,
    b_answer_max: int = 512,
    n_tiers: int = 3,
) -> List[Dict]:
    """Compute TOGA budget allocations for each difficulty tier.

    Theory-guided heuristic based on Proposition 6:
    - Easy problems (short nothink) → need less reasoning, more answer budget
    - Hard problems (long/truncated nothink) → need max reasoning, min answer budget

    Returns list of tier configs: [{b_r, b_a, description}, ...]
    """
    allocations = []

    if n_tiers == 3:
        # Tier 1 (easy escalation): 60% reasoning, 40% answer
        b_a_1 = min(b_answer_max, max(b_answer_min, int(b_total * 0.35)))
        b_r_1 = b_total - b_a_1
        allocations.append({
            "tier": 1, "b_r": b_r_1, "b_a": b_a_1,
            "description": "easy_escalation",
        })

        # Tier 2 (medium): 80% reasoning, 20% answer (close to IRIS default)
        b_a_2 = min(b_answer_max, max(b_answer_min, int(b_total * 0.15)))
        b_r_2 = b_total - b_a_2
        allocations.append({
            "tier": 2, "b_r": b_r_2, "b_a": b_a_2,
            "description": "medium_difficulty",
        })

        # Tier 3 (hard): 90% reasoning, 10% answer (minimize answer budget)
        b_a_3 = max(b_answer_min, int(b_total * 0.08))
        b_r_3 = b_total - b_a_3
        allocations.append({
            "tier": 3, "b_r": b_r_3, "b_a": b_a_3,
            "description": "hard_problem",
        })

    return allocations


def classify_difficulty_tier(
    nothink_tokens: int,
    nothink_hit_budget: bool,
    b1: int,
    n_tiers: int = 3,
) -> int:
    """Classify a sample into difficulty tier based on Stage 0 output.

    Tier 1 (easy): natural stop with short output (<40% of budget)
    Tier 2 (medium): natural stop with medium output (40-80% of budget)
    Tier 3 (hard): hit budget or very long output (>80% of budget)
    """
    if nothink_hit_budget:
        return 3  # Hard — needed escalation and used full nothink budget

    ratio = nothink_tokens / b1 if b1 > 0 else 0
    if ratio < 0.40:
        return 1  # Easy — short nothink output
    elif ratio < 0.80:
        return 2  # Medium
    else:
        return 3  # Hard — near-budget nothink


# ---------------------------------------------------------------------------
# TOGA-IRIS pipeline
# ---------------------------------------------------------------------------
def run_toga_iris_sample(
    model,
    tokenizer,
    question: str,
    b1: int = 512,
    b_total: int = 2048,
    toga_allocations: List[Dict] = None,
    chunk_size: int = 32,
    tau_h: float = 1.5,
    tau_s: float = 50.0,
    min_chunks: int = 2,
    benchmark: str = "gsm8k",
) -> Dict:
    """Run TOGA-IRIS cascade with tier-specific budget allocation.

    Stage 0: nothink@B1 → natural stop → accept
    Classification: determine difficulty tier from Stage 0 output
    Stage 1: think@B_r(tier) — tier-specific reasoning budget
    Stage 2: decoupled answer@B_a(tier) — tier-specific answer budget
    """
    result = {}

    # ======= STAGE 0: Nothink triage =======
    prompt_s0 = build_prompt(question, tokenizer, enable_thinking=False, benchmark=benchmark)
    text_s0, tokens_s0, elapsed_s0, hit_budget_s0 = generate_simple(
        model, tokenizer, prompt_s0, max_new_tokens=b1, temperature=0.0,
    )
    pred_s0, source_s0 = parse_prediction_dispatch(text_s0, benchmark)

    result["stage0"] = {
        "text": text_s0,
        "tokens": tokens_s0,
        "elapsed": round(elapsed_s0, 4),
        "hit_budget": hit_budget_s0,
        "pred": pred_s0,
        "pred_source": source_s0,
    }

    if not hit_budget_s0:
        result["final_stage"] = 0
        result["pred"] = pred_s0
        result["pred_source"] = source_s0
        result["tokens_total"] = tokens_s0
        result["elapsed_total"] = round(elapsed_s0, 4)
        result["tier"] = 0  # Not escalated
        result["stage1"] = None
        result["stage2"] = None
        result["stop_reason"] = "stage0_natural_stop"
        return result

    # ======= DIFFICULTY CLASSIFICATION =======
    tier = classify_difficulty_tier(tokens_s0, hit_budget_s0, b1)
    tier_config = toga_allocations[tier - 1]  # 0-indexed
    b_r = tier_config["b_r"]
    b_a = tier_config["b_a"]

    result["tier"] = tier
    result["tier_config"] = {"b_r": b_r, "b_a": b_a}

    # ======= STAGE 1: Adaptive thinking with tier-specific budget =======
    prompt_s1 = build_prompt(question, tokenizer, enable_thinking=True, benchmark=benchmark)
    s1_result = generate_adaptive_thinking(
        model, tokenizer, prompt_s1,
        max_think_tokens=b_r,
        chunk_size=chunk_size,
        tau_h=tau_h,
        tau_s=tau_s,
        min_chunks=min_chunks,
    )

    result["stage1"] = {
        "tokens_generated": s1_result["n_tokens_generated"],
        "tokens_used": s1_result["n_tokens_used"],
        "stop_reason": s1_result["stop_reason"],
        "elapsed": s1_result["elapsed_s"],
    }

    # If thinking completed naturally with answer
    if s1_result["stop_reason"] == "natural_stop":
        if "</think>" in s1_result["full_text"]:
            after_think = s1_result["full_text"].split("</think>", 1)[1]
            pred_s1, source_s1 = parse_prediction_dispatch(after_think, benchmark)
            if pred_s1 is not None:
                result["final_stage"] = 1
                result["pred"] = pred_s1
                result["pred_source"] = f"s1_{source_s1}"
                result["tokens_total"] = tokens_s0 + s1_result["n_tokens_generated"]
                result["elapsed_total"] = round(elapsed_s0 + s1_result["elapsed_s"], 4)
                result["stage2"] = {"skipped": True, "reason": "s1_has_answer"}
                result["stop_reason"] = "stage1_complete"
                return result

    # ======= STAGE 2: Decoupled answer with tier-specific budget =======
    thinking_trace = s1_result["thinking_text"]
    answer_text, tokens_s2, elapsed_s2 = generate_decoupled_answer(
        model, tokenizer, question, thinking_trace,
        answer_budget=b_a, benchmark=benchmark,
    )
    pred_s2, source_s2 = parse_prediction_dispatch(answer_text, benchmark)

    result["stage2"] = {
        "text": answer_text,
        "tokens": tokens_s2,
        "elapsed": round(elapsed_s2, 4),
        "pred": pred_s2,
        "pred_source": source_s2,
    }

    result["final_stage"] = 2
    result["pred"] = pred_s2
    result["pred_source"] = f"s2_{source_s2}"
    result["tokens_total"] = tokens_s0 + s1_result["n_tokens_used"] + tokens_s2
    result["elapsed_total"] = round(elapsed_s0 + s1_result["elapsed_s"] + elapsed_s2, 4)
    result["stop_reason"] = f"stage2_after_{s1_result['stop_reason']}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="TOGA-IRIS: Theory-Guided Optimal Budget Allocation + IRIS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # TOGA parameters
    parser.add_argument("--b1", type=int, default=512, help="Stage 0 nothink budget")
    parser.add_argument("--b_total", type=int, default=2048,
                        help="Total budget for escalated samples (B_r + B_a)")
    parser.add_argument("--b_answer_min", type=int, default=64)
    parser.add_argument("--b_answer_max", type=int, default=512)

    # Entropy monitoring (passed through)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--tau_h", type=float, default=1.5)
    parser.add_argument("--tau_s", type=float, default=50.0)
    parser.add_argument("--min_chunks", type=int, default=2)

    # Baselines
    parser.add_argument("--run_baseline_iris", action="store_true", default=True)
    parser.add_argument("--iris_b_answer", type=int, default=256,
                        help="Fixed B_answer for baseline IRIS")
    parser.add_argument("--run_town", action="store_true", default=False)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/toga_iris"
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Compute TOGA allocations
    toga_allocations = compute_toga_allocations(
        b_total=args.b_total,
        b_answer_min=args.b_answer_min,
        b_answer_max=args.b_answer_max,
    )
    log.info("TOGA allocations:")
    for a in toga_allocations:
        log.info(f"  Tier {a['tier']} ({a['description']}): B_r={a['b_r']}, B_a={a['b_a']}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    items = load_benchmark_data(args.benchmark, args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} {args.benchmark} samples (seed={args.seed})")

    # ========= RUN TOGA-IRIS =========
    log.info(f"TOGA-IRIS: B1={args.b1}, B_total={args.b_total}")

    toga_results = []
    toga_correct = 0
    toga_total_tokens = 0
    toga_tier_dist = {}
    toga_tier_correct = {}

    for i, item in enumerate(items):
        result = run_toga_iris_sample(
            model, tokenizer, item["question"],
            b1=args.b1, b_total=args.b_total,
            toga_allocations=toga_allocations,
            chunk_size=args.chunk_size, tau_h=args.tau_h, tau_s=args.tau_s,
            min_chunks=args.min_chunks, benchmark=args.benchmark,
        )

        correct = is_correct_dispatch(result["pred"], item["gold"], args.benchmark)
        if correct:
            toga_correct += 1

        tier = result["tier"]
        toga_tier_dist[tier] = toga_tier_dist.get(tier, 0) + 1
        if correct:
            toga_tier_correct[tier] = toga_tier_correct.get(tier, 0) + 1

        toga_total_tokens += result["tokens_total"]

        row = {
            "idx": i,
            "gold": item["gold"],
            "correct": int(correct),
            "pred": result["pred"],
            "pred_source": result.get("pred_source", ""),
            "final_stage": result["final_stage"],
            "tokens_total": result["tokens_total"],
            "elapsed_total": result["elapsed_total"],
            "tier": result["tier"],
            "stop_reason": result["stop_reason"],
        }
        if result.get("tier_config"):
            row["b_r"] = result["tier_config"]["b_r"]
            row["b_a"] = result["tier_config"]["b_a"]

        toga_results.append(row)

        if (i + 1) % 20 == 0 or i == n - 1:
            done = i + 1
            acc = toga_correct / done
            avg_tok = toga_total_tokens / done
            log.info(
                f"  TOGA [{done}/{n}] acc={acc:.3f} avg_tok={avg_tok:.0f} "
                f"tiers={toga_tier_dist}"
            )

        if (i + 1) % args.checkpoint_every == 0:
            ckpt = {"meta": {"n_done": i + 1}, "toga_results": toga_results}
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_toga_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2, default=str)

    toga_acc = toga_correct / n if n > 0 else 0.0
    toga_avg_tokens = toga_total_tokens / n if n > 0 else 0.0

    # Per-tier accuracy
    log.info("\nPer-tier accuracy:")
    for tier in sorted(toga_tier_dist.keys()):
        tier_n = toga_tier_dist[tier]
        tier_c = toga_tier_correct.get(tier, 0)
        tier_acc = tier_c / tier_n if tier_n > 0 else 0.0
        if tier > 0:
            alloc = toga_allocations[tier - 1]
            log.info(f"  Tier {tier} ({alloc['description']}): "
                     f"n={tier_n} acc={tier_acc:.3f} B_r={alloc['b_r']} B_a={alloc['b_a']}")
        else:
            log.info(f"  Tier 0 (not escalated): n={tier_n} acc={tier_acc:.3f}")

    # ========= BASELINE IRIS =========
    iris_results = []
    iris_acc = 0.0
    iris_avg_tokens = 0.0
    if args.run_baseline_iris:
        iris_b2_max = args.b_total - args.iris_b_answer
        log.info(f"\nBaseline IRIS: B1={args.b1}, B2_max={iris_b2_max}, B_answer={args.iris_b_answer}")
        iris_correct = 0
        iris_total_tokens = 0

        for i, item in enumerate(items):
            result = run_iris_sample(
                model, tokenizer, item["question"],
                b1=args.b1, b2_max=iris_b2_max, b_answer=args.iris_b_answer,
                chunk_size=args.chunk_size, tau_h=args.tau_h, tau_s=args.tau_s,
                min_chunks=args.min_chunks, benchmark=args.benchmark,
            )
            correct = is_correct_dispatch(result["pred"], item["gold"], args.benchmark)
            if correct:
                iris_correct += 1
            iris_total_tokens += result["tokens_total"]
            iris_results.append({
                "idx": i, "gold": item["gold"], "correct": int(correct),
                "pred": result["pred"], "final_stage": result["final_stage"],
                "tokens_total": result["tokens_total"],
            })

            if (i + 1) % 20 == 0 or i == n - 1:
                done = i + 1
                log.info(f"  IRIS [{done}/{n}] acc={iris_correct/done:.3f}")

        iris_acc = iris_correct / n if n > 0 else 0.0
        iris_avg_tokens = iris_total_tokens / n if n > 0 else 0.0

    # ========= SUMMARY =========
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info(f"  TOGA-IRIS:  acc={toga_acc:.4f}  avg_tok={toga_avg_tokens:.0f}")
    if args.run_baseline_iris:
        log.info(f"  IRIS:       acc={iris_acc:.4f}  avg_tok={iris_avg_tokens:.0f}")
        log.info(f"  Delta:      {(toga_acc - iris_acc)*100:+.1f}pp")

    # Save
    summary = {
        "meta": {
            "script": "run_toga_iris.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "seed": args.seed,
        },
        "toga_config": {
            "b1": args.b1,
            "b_total": args.b_total,
            "b_answer_min": args.b_answer_min,
            "b_answer_max": args.b_answer_max,
            "allocations": toga_allocations,
        },
        "toga_results": {
            "accuracy": round(toga_acc, 4),
            "avg_tokens": round(toga_avg_tokens, 2),
            "tier_distribution": toga_tier_dist,
            "tier_accuracy": {
                str(k): round(toga_tier_correct.get(k, 0) / toga_tier_dist[k], 4)
                for k in toga_tier_dist if toga_tier_dist[k] > 0
            },
        },
        "per_sample_toga": toga_results,
    }

    if args.run_baseline_iris:
        summary["baseline_iris"] = {
            "accuracy": round(iris_acc, 4),
            "avg_tokens": round(iris_avg_tokens, 2),
        }

    out_path = os.path.join(
        args.output_dir,
        f"toga_iris_{args.benchmark}_b{args.b_total}_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nSaved to {out_path}")

    # Per-sample CSV
    csv_path = os.path.join(
        args.output_dir,
        f"per_sample_toga_{args.benchmark}_b{args.b_total}_{timestamp}.csv"
    )
    if toga_results:
        keys = list(toga_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(toga_results)
        log.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
