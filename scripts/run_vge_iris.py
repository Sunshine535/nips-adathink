#!/usr/bin/env python3
"""VGE-IRIS: Verification-Gated Escalation + IRIS cascade.

Extends IRIS with Stage 0.5 — a lightweight self-verification step
that catches false accepts from Stage 1 (natural-stop triage).

Pipeline:
    Stage 0: nothink@B1 → natural stop → candidate answer A0
    Stage 0.5 [NEW]: verify A0 with nothink@B_verify
        → "Yes" → accept A0
        → "No" / unclear → escalate (override natural stop)
    Stage 1-3: standard IRIS (think → decoupled answer)

Targets failure mode F1: 27% of IRIS errors are Stage 0 false accepts
where nothink emits EOS (natural stop) but the answer is wrong.
VGE catches a fraction of these at negligible cost (~64 tokens).

Usage:
    python scripts/run_vge_iris.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --n_samples 500 \
        --b1 512 --b2_max 2048 --b_answer 256 \
        --b_verify 64 \
        --seed 42 --output_dir results/vge_iris
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

# Reuse all core functions from run_iris.py
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
    run_town_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VGE: Verification stage
# ---------------------------------------------------------------------------
VERIFY_PROMPTS = {
    "gsm8k": (
        "A student solved the following math problem and got the answer below. "
        "Check if the answer is correct. Reply with ONLY 'Yes' or 'No'.\n\n"
        "Problem: {question}\n"
        "Student's answer: {answer}\n\n"
        "Is this correct?"
    ),
    "math500": (
        "A student solved the following math problem and got the answer below. "
        "Check if the answer is correct. Reply with ONLY 'Yes' or 'No'.\n\n"
        "Problem: {question}\n"
        "Student's answer: {answer}\n\n"
        "Is this correct?"
    ),
}


def verify_answer(
    model,
    tokenizer,
    question: str,
    answer: str,
    b_verify: int = 64,
    benchmark: str = "gsm8k",
) -> Tuple[bool, str, int, float]:
    """Run verification on a candidate answer.

    Returns (accept, raw_text, n_tokens, elapsed).
    accept=True means the model thinks the answer is correct.
    """
    template = VERIFY_PROMPTS.get(benchmark, VERIFY_PROMPTS["gsm8k"])
    verify_content = template.format(question=question, answer=answer)

    messages = [{"role": "user", "content": verify_content}]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    chat_kwargs["enable_thinking"] = False
    try:
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

    text, n_tokens, elapsed, _ = generate_simple(
        model, tokenizer, prompt, max_new_tokens=b_verify, temperature=0.0,
    )

    # Parse verification response
    text_lower = text.strip().lower()
    # Accept if clearly "yes"; reject otherwise (conservative)
    if text_lower.startswith("yes"):
        accept = True
    elif text_lower.startswith("no"):
        accept = False
    else:
        # Ambiguous → reject (escalate to be safe)
        accept = False

    return accept, text.strip(), n_tokens, elapsed


# ---------------------------------------------------------------------------
# VGE-IRIS full pipeline for a single sample
# ---------------------------------------------------------------------------
def run_vge_iris_sample(
    model,
    tokenizer,
    question: str,
    b1: int = 256,
    b2_max: int = 512,
    b_answer: int = 128,
    b_verify: int = 64,
    chunk_size: int = 32,
    tau_h: float = 1.5,
    tau_s: float = 50.0,
    min_chunks: int = 2,
    total_budget_cap: Optional[int] = None,
    benchmark: str = "gsm8k",
) -> Dict:
    """Run VGE-IRIS cascade for a single question.

    Stage 0:   nothink@B1 → hit budget → escalate directly
    Stage 0.5: if natural stop → verify with nothink@B_verify
               → accept: return Stage 0 answer
               → reject: escalate to Stage 1
    Stage 1:   adaptive thinking@B2_max
    Stage 2:   decoupled answer@B_answer
    """
    result = {}

    # ======= STAGE 0: Non-thinking probe =======
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

    # If hit budget → escalate directly (no verification needed)
    if hit_budget_s0:
        escalate = True
        result["stage0_5"] = {"skipped": True, "reason": "hit_budget"}
    else:
        # ======= STAGE 0.5: Verification gate =======
        if pred_s0 is not None:
            accept, verify_text, verify_tokens, verify_elapsed = verify_answer(
                model, tokenizer, question, str(pred_s0),
                b_verify=b_verify, benchmark=benchmark,
            )
            result["stage0_5"] = {
                "verify_text": verify_text,
                "verify_tokens": verify_tokens,
                "verify_elapsed": round(verify_elapsed, 4),
                "accept": accept,
            }
            escalate = not accept
        else:
            # No prediction extracted → escalate
            result["stage0_5"] = {"skipped": True, "reason": "no_pred"}
            escalate = True

    # If Stage 0 + verification accepted → return
    if not escalate:
        result["final_stage"] = 0
        result["pred"] = pred_s0
        result["pred_source"] = source_s0
        tokens_total = tokens_s0 + result["stage0_5"].get("verify_tokens", 0)
        elapsed_total = elapsed_s0 + result["stage0_5"].get("verify_elapsed", 0)
        result["tokens_total"] = tokens_total
        result["elapsed_total"] = round(elapsed_total, 4)
        result["stage1"] = None
        result["stage2"] = None
        result["stop_reason"] = "vge_accept"
        return result

    # ======= STAGE 1: Adaptive thinking =======
    tokens_used_so_far = tokens_s0 + result["stage0_5"].get("verify_tokens", 0)
    elapsed_so_far = elapsed_s0 + result["stage0_5"].get("verify_elapsed", 0)

    if total_budget_cap is not None:
        remaining = total_budget_cap - tokens_used_so_far - b_answer
        b2_effective = min(b2_max, max(chunk_size, remaining))
    else:
        b2_effective = b2_max

    prompt_s1 = build_prompt(question, tokenizer, enable_thinking=True, benchmark=benchmark)
    s1_result = generate_adaptive_thinking(
        model, tokenizer, prompt_s1,
        max_think_tokens=b2_effective,
        chunk_size=chunk_size,
        tau_h=tau_h,
        tau_s=tau_s,
        min_chunks=min_chunks,
    )

    result["stage1"] = {
        "tokens_generated": s1_result["n_tokens_generated"],
        "tokens_used": s1_result["n_tokens_used"],
        "tokens_saved": s1_result["tokens_saved"],
        "savings_ratio": s1_result["savings_ratio"],
        "stop_reason": s1_result["stop_reason"],
        "elapsed": s1_result["elapsed_s"],
    }

    # If thinking produced a natural stop with answer, extract directly
    if s1_result["stop_reason"] == "natural_stop":
        if "</think>" in s1_result["full_text"]:
            after_think = s1_result["full_text"].split("</think>", 1)[1]
            pred_s1_direct, source_s1_direct = parse_prediction_dispatch(after_think, benchmark)
            if pred_s1_direct is not None:
                result["final_stage"] = 1
                result["pred"] = pred_s1_direct
                result["pred_source"] = f"s1_{source_s1_direct}"
                result["tokens_total"] = tokens_used_so_far + s1_result["n_tokens_generated"]
                result["elapsed_total"] = round(elapsed_so_far + s1_result["elapsed_s"], 4)
                result["stage2"] = {"skipped": True, "reason": "s1_has_answer"}
                result["stop_reason"] = "stage1_complete"
                return result

    # ======= STAGE 2: Decoupled answer generation =======
    thinking_trace = s1_result["thinking_text"]
    answer_text, tokens_s2, elapsed_s2 = generate_decoupled_answer(
        model, tokenizer, question, thinking_trace, answer_budget=b_answer,
        benchmark=benchmark,
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
    result["tokens_total"] = tokens_used_so_far + s1_result["n_tokens_used"] + tokens_s2
    result["elapsed_total"] = round(elapsed_so_far + s1_result["elapsed_s"] + elapsed_s2, 4)
    result["stop_reason"] = f"stage2_after_{s1_result['stop_reason']}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VGE-IRIS: Verification-Gated Escalation + IRIS cascade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # VGE-IRIS hyperparameters
    parser.add_argument("--b1", type=int, default=256, help="Stage 0 nothink budget")
    parser.add_argument("--b2_max", type=int, default=512, help="Max Stage 1 thinking budget")
    parser.add_argument("--b_answer", type=int, default=128, help="Stage 2 answer budget")
    parser.add_argument("--b_verify", type=int, default=64, help="VGE verification budget")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--tau_h", type=float, default=1.5)
    parser.add_argument("--tau_s", type=float, default=50.0)
    parser.add_argument("--min_chunks", type=int, default=2)
    parser.add_argument("--total_budget_cap", type=int, default=None)

    # Also run baseline IRIS for comparison
    parser.add_argument("--run_baseline_iris", action="store_true", default=True,
                        help="Run baseline IRIS (without VGE) for direct comparison")

    # Also run TOWN for comparison
    parser.add_argument("--run_town", action="store_true", default=False)
    parser.add_argument("--town_b2", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/vge_iris"
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load data
    items = load_benchmark_data(args.benchmark, args.n_samples, args.seed)
    n = len(items)
    log.info(f"Loaded {n} {args.benchmark} samples (seed={args.seed})")

    # ========= RUN VGE-IRIS =========
    log.info(f"VGE-IRIS: B1={args.b1}, B2_max={args.b2_max}, B_answer={args.b_answer}, "
             f"B_verify={args.b_verify}")

    vge_results = []
    vge_correct = 0
    vge_stage_counts = {0: 0, 1: 0, 2: 0}
    vge_total_tokens = 0
    vge_stop_reasons = {}
    vge_caught = 0  # false accepts caught by VGE
    vge_wrong_reject = 0  # correct answers wrongly rejected by VGE

    for i, item in enumerate(items):
        result = run_vge_iris_sample(
            model, tokenizer, item["question"],
            b1=args.b1, b2_max=args.b2_max, b_answer=args.b_answer,
            b_verify=args.b_verify,
            chunk_size=args.chunk_size, tau_h=args.tau_h, tau_s=args.tau_s,
            min_chunks=args.min_chunks, total_budget_cap=args.total_budget_cap,
            benchmark=args.benchmark,
        )

        correct = is_correct_dispatch(result["pred"], item["gold"], args.benchmark)
        if correct:
            vge_correct += 1

        stage = result["final_stage"]
        vge_stage_counts[stage] = vge_stage_counts.get(stage, 0) + 1
        vge_total_tokens += result["tokens_total"]
        sr = result["stop_reason"]
        vge_stop_reasons[sr] = vge_stop_reasons.get(sr, 0) + 1

        # Track VGE effectiveness
        s0_5 = result.get("stage0_5", {})
        if not s0_5.get("skipped", False) and "accept" in s0_5:
            s0_correct = is_correct_dispatch(
                result["stage0"]["pred"], item["gold"], args.benchmark
            )
            if not s0_5["accept"] and s0_correct:
                vge_wrong_reject += 1  # VGE incorrectly rejected a correct answer
            elif not s0_5["accept"] and not s0_correct:
                vge_caught += 1  # VGE correctly caught a wrong answer

        row = {
            "idx": i,
            "gold": item["gold"],
            "correct": int(correct),
            "pred": result["pred"],
            "pred_source": result.get("pred_source", ""),
            "final_stage": result["final_stage"],
            "tokens_total": result["tokens_total"],
            "elapsed_total": result["elapsed_total"],
            "stop_reason": result["stop_reason"],
            "s0_hit_budget": result["stage0"]["hit_budget"],
            "s0_pred": result["stage0"]["pred"],
            "s0_tokens": result["stage0"]["tokens"],
        }
        # VGE details
        if not s0_5.get("skipped", False) and "accept" in s0_5:
            row["vge_accept"] = s0_5["accept"]
            row["vge_tokens"] = s0_5["verify_tokens"]
            row["vge_text"] = s0_5["verify_text"][:100]  # truncate for CSV
        else:
            row["vge_accept"] = None
            row["vge_tokens"] = 0
            row["vge_text"] = ""

        # Stage 1 details
        if result["stage1"] is not None:
            row["s1_tokens_used"] = result["stage1"]["tokens_used"]
            row["s1_stop_reason"] = result["stage1"]["stop_reason"]

        vge_results.append(row)

        if (i + 1) % 20 == 0 or i == n - 1:
            done = i + 1
            acc = vge_correct / done
            avg_tok = vge_total_tokens / done
            log.info(
                f"  VGE-IRIS [{done}/{n}] acc={acc:.3f} avg_tok={avg_tok:.0f} "
                f"stages={vge_stage_counts} caught={vge_caught} wrong_rej={vge_wrong_reject}"
            )

        if (i + 1) % args.checkpoint_every == 0:
            ckpt = {"meta": {"n_done": i + 1}, "vge_results": vge_results}
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_vge_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2, default=str)

    vge_acc = vge_correct / n if n > 0 else 0.0
    vge_avg_tokens = vge_total_tokens / n if n > 0 else 0.0

    # ========= RUN BASELINE IRIS (no VGE) for comparison =========
    from run_iris import run_iris_sample

    iris_results = []
    iris_acc = 0.0
    iris_avg_tokens = 0.0
    if args.run_baseline_iris:
        log.info(f"\nBaseline IRIS (no VGE): B1={args.b1}, B2_max={args.b2_max}, B_answer={args.b_answer}")
        iris_correct = 0
        iris_total_tokens = 0

        for i, item in enumerate(items):
            result = run_iris_sample(
                model, tokenizer, item["question"],
                b1=args.b1, b2_max=args.b2_max, b_answer=args.b_answer,
                chunk_size=args.chunk_size, tau_h=args.tau_h, tau_s=args.tau_s,
                min_chunks=args.min_chunks, total_budget_cap=args.total_budget_cap,
                benchmark=args.benchmark,
            )
            correct = is_correct_dispatch(result["pred"], item["gold"], args.benchmark)
            if correct:
                iris_correct += 1
            iris_total_tokens += result["tokens_total"]

            iris_results.append({
                "idx": i,
                "gold": item["gold"],
                "correct": int(correct),
                "pred": result["pred"],
                "final_stage": result["final_stage"],
                "tokens_total": result["tokens_total"],
                "stop_reason": result["stop_reason"],
            })

            if (i + 1) % 20 == 0 or i == n - 1:
                done = i + 1
                acc = iris_correct / done
                log.info(f"  IRIS [{done}/{n}] acc={acc:.3f}")

        iris_acc = iris_correct / n if n > 0 else 0.0
        iris_avg_tokens = iris_total_tokens / n if n > 0 else 0.0

    # ========= RUN TOWN (optional) =========
    town_results = []
    town_acc = 0.0
    town_avg_tokens = 0.0
    if args.run_town:
        log.info(f"\nTOWN baseline: B1={args.b1}, B2={args.town_b2}")
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
            town_results.append({
                "idx": i, "gold": item["gold"], "correct": int(correct),
                "pred": result["pred"], "stage": result["stage"],
                "tokens_total": result["tokens_total"],
            })

            if (i + 1) % 20 == 0 or i == n - 1:
                done = i + 1
                log.info(f"  TOWN [{done}/{n}] acc={town_correct/done:.3f}")

        town_acc = town_correct / n if n > 0 else 0.0
        town_avg_tokens = town_total_tokens / n if n > 0 else 0.0

    # ========= SUMMARY =========
    log.info("\n" + "=" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("=" * 60)
    log.info(f"  VGE-IRIS:  acc={vge_acc:.4f}  avg_tok={vge_avg_tokens:.0f}")
    log.info(f"    Stage distribution: {vge_stage_counts}")
    log.info(f"    VGE caught (true positive):  {vge_caught}")
    log.info(f"    VGE wrong reject (false neg): {vge_wrong_reject}")
    if args.run_baseline_iris:
        log.info(f"  IRIS:      acc={iris_acc:.4f}  avg_tok={iris_avg_tokens:.0f}")
        log.info(f"  Delta:     {(vge_acc - iris_acc)*100:+.1f}pp")
    if args.run_town:
        log.info(f"  TOWN:      acc={town_acc:.4f}  avg_tok={town_avg_tokens:.0f}")

    # Save full results
    summary = {
        "meta": {
            "script": "run_vge_iris.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "seed": args.seed,
        },
        "vge_config": {
            "b1": args.b1,
            "b2_max": args.b2_max,
            "b_answer": args.b_answer,
            "b_verify": args.b_verify,
            "chunk_size": args.chunk_size,
            "tau_h": args.tau_h,
            "tau_s": args.tau_s,
        },
        "vge_iris_results": {
            "accuracy": round(vge_acc, 4),
            "avg_tokens": round(vge_avg_tokens, 2),
            "stage_distribution": vge_stage_counts,
            "stop_reasons": vge_stop_reasons,
            "vge_caught": vge_caught,
            "vge_wrong_reject": vge_wrong_reject,
        },
        "per_sample_vge": vge_results,
    }

    if args.run_baseline_iris:
        summary["baseline_iris"] = {
            "accuracy": round(iris_acc, 4),
            "avg_tokens": round(iris_avg_tokens, 2),
        }
        summary["per_sample_iris"] = iris_results

    if args.run_town:
        summary["town_results"] = {
            "accuracy": round(town_acc, 4),
            "avg_tokens": round(town_avg_tokens, 2),
        }

    out_path = os.path.join(
        args.output_dir,
        f"vge_iris_{args.benchmark}_b{args.b2_max}_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nSaved to {out_path}")

    # Also save per-sample CSV for analysis
    csv_path = os.path.join(
        args.output_dir,
        f"per_sample_vge_{args.benchmark}_b{args.b2_max}_{timestamp}.csv"
    )
    if vge_results:
        keys = [k for k in vge_results[0].keys() if k != "vge_text"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(vge_results)
        log.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
