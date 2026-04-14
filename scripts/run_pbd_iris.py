#!/usr/bin/env python3
"""PBD-IRIS: Progressive Budget Doubling + IRIS cascade.

Instead of a single fixed thinking budget, PBD uses iterative deepening:
    Round 1: think@B_min (e.g., 512)
    Round 2: think@2*B_min (e.g., 1024)
    Round k: think@2^(k-1)*B_min, up to B_max
Each round checks for natural stop + answer extraction before escalating.
Total budget is capped.

Key insight: LogNormal chain-length distribution means many problems complete
reasoning at B_min or 2*B_min. Only truly hard problems need the full B_max.
This is iterative deepening search applied to LLM reasoning budgets.

Usage:
    python scripts/run_pbd_iris.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --n_samples 500 \
        --b1 512 --b_min 512 --b_max 4096 --b_answer 256 \
        --seed 42 --output_dir results/pbd_iris
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


def run_pbd_sample(
    model,
    tokenizer,
    question: str,
    b1: int = 256,
    b_min: int = 512,
    b_max: int = 4096,
    b_answer: int = 256,
    total_budget_cap: Optional[int] = None,
    benchmark: str = "gsm8k",
) -> Dict:
    """Run PBD-IRIS cascade: nothink triage → progressive budget doubling.

    Stage 0: nothink@B1 → natural stop → accept
    Rounds 1..K: think@(B_min * 2^(k-1)), doubling each round
        → natural stop with answer → accept
        → budget exhausted → decoupled answer extraction
            → if extraction confident → accept
            → else → next round
    """
    result = {"rounds": []}
    tokens_spent = 0
    elapsed_spent = 0.0

    # ======= STAGE 0: Nothink triage =======
    prompt_s0 = build_prompt(question, tokenizer, enable_thinking=False, benchmark=benchmark)
    text_s0, tokens_s0, elapsed_s0, hit_budget_s0 = generate_simple(
        model, tokenizer, prompt_s0, max_new_tokens=b1, temperature=0.0,
    )
    pred_s0, source_s0 = parse_prediction_dispatch(text_s0, benchmark)
    tokens_spent += tokens_s0
    elapsed_spent += elapsed_s0

    result["stage0"] = {
        "text": text_s0,
        "tokens": tokens_s0,
        "elapsed": round(elapsed_s0, 4),
        "hit_budget": hit_budget_s0,
        "pred": pred_s0,
        "pred_source": source_s0,
    }

    if not hit_budget_s0:
        result["final_stage"] = "s0"
        result["pred"] = pred_s0
        result["pred_source"] = source_s0
        result["tokens_total"] = tokens_spent
        result["elapsed_total"] = round(elapsed_spent, 4)
        result["n_rounds"] = 0
        result["stop_reason"] = "stage0_natural_stop"
        return result

    # ======= Progressive Budget Doubling =======
    # Compute round budgets: b_min, 2*b_min, 4*b_min, ..., up to b_max
    round_budgets = []
    b = b_min
    while b <= b_max:
        round_budgets.append(b)
        b *= 2
    if not round_budgets or round_budgets[-1] < b_max:
        round_budgets.append(b_max)

    best_pred = None
    best_source = None
    best_round = -1

    for k, b_think in enumerate(round_budgets):
        # Budget cap check
        if total_budget_cap is not None:
            remaining = total_budget_cap - tokens_spent - b_answer
            if remaining < b_min // 2:
                break  # Not enough budget for meaningful thinking
            b_think = min(b_think, remaining)

        # Generate thinking
        prompt_think = build_prompt(question, tokenizer, enable_thinking=True, benchmark=benchmark)
        text_think, tokens_think, elapsed_think, hit_budget_think = generate_simple(
            model, tokenizer, prompt_think, max_new_tokens=b_think, temperature=0.0,
        )
        tokens_spent += tokens_think
        elapsed_spent += elapsed_think

        round_info = {
            "round": k + 1,
            "b_think": b_think,
            "tokens_think": tokens_think,
            "hit_budget": hit_budget_think,
            "elapsed": round(elapsed_think, 4),
        }

        # Check if thinking completed naturally with answer
        if not hit_budget_think:
            # Natural stop — extract answer from thinking output
            # Check for answer after </think> tag
            if "</think>" in text_think:
                after_think = text_think.split("</think>", 1)[1]
                pred_direct, source_direct = parse_prediction_dispatch(after_think, benchmark)
                if pred_direct is not None:
                    round_info["pred"] = pred_direct
                    round_info["pred_source"] = f"direct_{source_direct}"
                    round_info["outcome"] = "natural_stop_with_answer"
                    result["rounds"].append(round_info)
                    best_pred = pred_direct
                    best_source = f"r{k+1}_direct_{source_direct}"
                    best_round = k + 1
                    break

            # Natural stop but no answer after </think> → decoupled extraction
            thinking_trace = text_think
            answer_text, tokens_ans, elapsed_ans = generate_decoupled_answer(
                model, tokenizer, question, thinking_trace,
                answer_budget=b_answer, benchmark=benchmark,
            )
            tokens_spent += tokens_ans
            elapsed_spent += elapsed_ans

            pred_ext, source_ext = parse_prediction_dispatch(answer_text, benchmark)
            round_info["pred"] = pred_ext
            round_info["pred_source"] = f"extract_{source_ext}"
            round_info["extract_tokens"] = tokens_ans
            round_info["outcome"] = "natural_stop_extracted"
            result["rounds"].append(round_info)
            best_pred = pred_ext
            best_source = f"r{k+1}_extract_{source_ext}"
            best_round = k + 1
            break  # Natural stop means reasoning completed

        # Budget exhausted — try decoupled answer extraction
        # Extract thinking trace (remove any answer portion)
        thinking_trace = text_think
        answer_text, tokens_ans, elapsed_ans = generate_decoupled_answer(
            model, tokenizer, question, thinking_trace,
            answer_budget=b_answer, benchmark=benchmark,
        )
        tokens_spent += tokens_ans
        elapsed_spent += elapsed_ans

        pred_ext, source_ext = parse_prediction_dispatch(answer_text, benchmark)
        round_info["pred"] = pred_ext
        round_info["pred_source"] = f"extract_{source_ext}"
        round_info["extract_tokens"] = tokens_ans

        if pred_ext is not None:
            # Got an answer — check confidence via extraction source
            # boxed/final_answer are more confident than last_number/fallback
            confident = source_ext in ("boxed", "final_answer", "hash")
            round_info["confident"] = confident
            round_info["outcome"] = "budget_exhausted_confident" if confident else "budget_exhausted_uncertain"

            # Always keep the latest answer as best (more reasoning = better)
            best_pred = pred_ext
            best_source = f"r{k+1}_extract_{source_ext}"
            best_round = k + 1

            # If last round, accept whatever we have
            if k == len(round_budgets) - 1:
                result["rounds"].append(round_info)
                break

            # If confident extraction, still try next round for possible improvement
            # (progressive deepening: more thinking might help even if current answer exists)
            result["rounds"].append(round_info)
            continue
        else:
            round_info["confident"] = False
            round_info["outcome"] = "budget_exhausted_no_answer"
            result["rounds"].append(round_info)
            continue

    # Finalize
    result["final_stage"] = f"r{best_round}" if best_round > 0 else "s0"
    result["pred"] = best_pred if best_pred is not None else pred_s0
    result["pred_source"] = best_source if best_source is not None else source_s0
    result["tokens_total"] = tokens_spent
    result["elapsed_total"] = round(elapsed_spent, 4)
    result["n_rounds"] = len(result["rounds"])
    result["stop_reason"] = result["rounds"][-1]["outcome"] if result["rounds"] else "no_rounds"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PBD-IRIS: Progressive Budget Doubling + IRIS cascade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # PBD hyperparameters
    parser.add_argument("--b1", type=int, default=256, help="Stage 0 nothink budget")
    parser.add_argument("--b_min", type=int, default=512, help="Minimum thinking budget (Round 1)")
    parser.add_argument("--b_max", type=int, default=4096, help="Maximum thinking budget (last round)")
    parser.add_argument("--b_answer", type=int, default=256, help="Decoupled answer budget")
    parser.add_argument("--total_budget_cap", type=int, default=None)

    # Baseline comparison
    parser.add_argument("--run_baseline_iris", action="store_true", default=True)
    parser.add_argument("--iris_b2_max", type=int, default=None,
                        help="IRIS baseline B2_max (default: same as b_max)")
    parser.add_argument("--run_town", action="store_true", default=False)
    parser.add_argument("--town_b2", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    if args.iris_b2_max is None:
        args.iris_b2_max = args.b_max
    if args.town_b2 is None:
        args.town_b2 = args.b_max

    if args.output_dir is None:
        args.output_dir = "results/pbd_iris"
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    model, tokenizer = load_model_and_tokenizer(args.model)
    items = load_benchmark_data(args.benchmark, args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} {args.benchmark} samples (seed={args.seed})")

    # ========= RUN PBD-IRIS =========
    log.info(f"PBD-IRIS: B1={args.b1}, B_min={args.b_min}, B_max={args.b_max}, "
             f"B_answer={args.b_answer}")

    pbd_results = []
    pbd_correct = 0
    pbd_total_tokens = 0
    pbd_round_dist = {}

    for i, item in enumerate(items):
        result = run_pbd_sample(
            model, tokenizer, item["question"],
            b1=args.b1, b_min=args.b_min, b_max=args.b_max,
            b_answer=args.b_answer, total_budget_cap=args.total_budget_cap,
            benchmark=args.benchmark,
        )

        correct = is_correct_dispatch(result["pred"], item["gold"], args.benchmark)
        if correct:
            pbd_correct += 1
        pbd_total_tokens += result["tokens_total"]

        nr = result["n_rounds"]
        pbd_round_dist[nr] = pbd_round_dist.get(nr, 0) + 1

        row = {
            "idx": i,
            "gold": item["gold"],
            "correct": int(correct),
            "pred": result["pred"],
            "pred_source": result.get("pred_source", ""),
            "final_stage": result["final_stage"],
            "tokens_total": result["tokens_total"],
            "elapsed_total": result["elapsed_total"],
            "n_rounds": result["n_rounds"],
            "stop_reason": result["stop_reason"],
        }
        pbd_results.append(row)

        if (i + 1) % 20 == 0 or i == n - 1:
            done = i + 1
            acc = pbd_correct / done
            avg_tok = pbd_total_tokens / done
            log.info(
                f"  PBD [{done}/{n}] acc={acc:.3f} avg_tok={avg_tok:.0f} "
                f"rounds={pbd_round_dist}"
            )

        if (i + 1) % args.checkpoint_every == 0:
            ckpt = {"meta": {"n_done": i + 1}, "pbd_results": pbd_results}
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_pbd_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2, default=str)

    pbd_acc = pbd_correct / n if n > 0 else 0.0
    pbd_avg_tokens = pbd_total_tokens / n if n > 0 else 0.0

    # ========= BASELINE IRIS =========
    iris_results = []
    iris_acc = 0.0
    iris_avg_tokens = 0.0
    if args.run_baseline_iris:
        log.info(f"\nBaseline IRIS: B1={args.b1}, B2_max={args.iris_b2_max}, B_answer={args.b_answer}")
        iris_correct = 0
        iris_total_tokens = 0

        for i, item in enumerate(items):
            result = run_iris_sample(
                model, tokenizer, item["question"],
                b1=args.b1, b2_max=args.iris_b2_max, b_answer=args.b_answer,
                benchmark=args.benchmark,
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

    # ========= TOWN (optional) =========
    town_acc = 0.0
    town_avg_tokens = 0.0
    if args.run_town:
        log.info(f"\nTOWN: B1={args.b1}, B2={args.town_b2}")
        town_correct = 0
        town_total_tokens = 0

        for i, item in enumerate(items):
            result = run_town_sample(
                model, tokenizer, item["question"],
                b1=args.b1, b2=args.town_b2, benchmark=args.benchmark,
            )
            correct = is_correct_dispatch(result["pred"], item["gold"], args.benchmark)
            if correct:
                town_correct += 1
            town_total_tokens += result["tokens_total"]

            if (i + 1) % 20 == 0 or i == n - 1:
                done = i + 1
                log.info(f"  TOWN [{done}/{n}] acc={town_correct/done:.3f}")

        town_acc = town_correct / n if n > 0 else 0.0
        town_avg_tokens = town_total_tokens / n if n > 0 else 0.0

    # ========= SUMMARY =========
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info(f"  PBD-IRIS:  acc={pbd_acc:.4f}  avg_tok={pbd_avg_tokens:.0f}  rounds={pbd_round_dist}")
    if args.run_baseline_iris:
        log.info(f"  IRIS:      acc={iris_acc:.4f}  avg_tok={iris_avg_tokens:.0f}")
        log.info(f"  Delta acc: {(pbd_acc - iris_acc)*100:+.1f}pp")
        if iris_avg_tokens > 0:
            log.info(f"  Delta tok: {(pbd_avg_tokens - iris_avg_tokens)/iris_avg_tokens*100:+.1f}%")
    if args.run_town:
        log.info(f"  TOWN:      acc={town_acc:.4f}  avg_tok={town_avg_tokens:.0f}")

    # Save
    summary = {
        "meta": {
            "script": "run_pbd_iris.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "seed": args.seed,
        },
        "pbd_config": {
            "b1": args.b1,
            "b_min": args.b_min,
            "b_max": args.b_max,
            "b_answer": args.b_answer,
            "total_budget_cap": args.total_budget_cap,
        },
        "pbd_results": {
            "accuracy": round(pbd_acc, 4),
            "avg_tokens": round(pbd_avg_tokens, 2),
            "round_distribution": pbd_round_dist,
        },
        "per_sample_pbd": pbd_results,
    }

    if args.run_baseline_iris:
        summary["baseline_iris"] = {
            "accuracy": round(iris_acc, 4),
            "avg_tokens": round(iris_avg_tokens, 2),
        }

    if args.run_town:
        summary["town_results"] = {
            "accuracy": round(town_acc, 4),
            "avg_tokens": round(town_avg_tokens, 2),
        }

    out_path = os.path.join(
        args.output_dir,
        f"pbd_iris_{args.benchmark}_bmax{args.b_max}_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nSaved to {out_path}")

    # Per-sample CSV
    csv_path = os.path.join(
        args.output_dir,
        f"per_sample_pbd_{args.benchmark}_bmax{args.b_max}_{timestamp}.csv"
    )
    if pbd_results:
        keys = list(pbd_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(pbd_results)
        log.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
