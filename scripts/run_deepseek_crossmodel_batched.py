#!/usr/bin/env python3
"""
Cross-model-family validation: DeepSeek-R1-Distill-Llama-8B.
Batched inference version — 3-4x faster than sequential via GPU saturation.

Key differences from run_deepseek_crossmodel.py:
  - Left-padded batch generation (batch_size configurable, default 16)
  - Greedy decode → identical results to sequential version
  - Checkpoint after each (mode, budget) → resume from crash
  - inf/nan safe normalize_number

Usage:
    python3 run_deepseek_crossmodel_batched.py --benchmark gsm8k --budgets 256 512 1024 2048 --batch_size 16
    python3 run_deepseek_crossmodel_batched.py --benchmark math500 --budgets 512 1024 2048 4096 --batch_size 8
"""
import argparse
import csv
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Answer extraction ──
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_last_number(text: str) -> Optional[str]:
    m = NUM_RE.findall(text)
    return m[-1] if m else None


def extract_final_number(text: str) -> Optional[str]:
    m = list(FINAL_ANSWER_RE.finditer(text))
    return m[-1].group(1) if m else None


def extract_boxed_number(text: str) -> Optional[str]:
    for m in reversed(list(BOXED_RE.finditer(text))):
        v = extract_last_number(m.group(1))
        if v is not None:
            return v
    return None


def parse_prediction(text: str) -> Tuple[Optional[str], str]:
    think_end = text.rfind("</think>")
    search = text[think_end:] if think_end >= 0 else text
    f = extract_final_number(search)
    if f:
        return f, "final_marker"
    b = extract_boxed_number(search)
    if b:
        return b, "boxed"
    t = extract_last_number(search)
    if t:
        return t, "fallback_last"
    return None, "none"


def normalize_number(s: str) -> str:
    s = s.replace(",", "").strip()
    try:
        v = float(s)
        if math.isinf(v) or math.isnan(v):
            return s
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        return s


def is_correct(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    return normalize_number(pred) == normalize_number(gold)


def build_prompt_think(question: str, tokenizer, benchmark: str) -> str:
    if benchmark == "math500":
        content = f"Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.\n\n{question}"
    else:
        content = f"Solve this math problem step by step. At the end, write 'Final answer: <number>'.\n\n{question}"
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_nothink(question: str, tokenizer, benchmark: str) -> str:
    if benchmark == "math500":
        content = f"Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.\n\n{question}"
    else:
        content = f"Solve this math problem step by step. At the end, write 'Final answer: <number>'.\n\n{question}"
    msgs = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt = prompt + "\n</think>\n\n"
    return prompt


def load_benchmark(benchmark: str, seed: int, n_samples: int = 0):
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for row in ds:
            q = row["question"]
            a_text = row["answer"]
            gold = a_text.split("####")[-1].strip() if "####" in a_text else extract_last_number(a_text)
            items.append({"question": q, "gold": str(gold)})
    elif benchmark == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = []
        for row in ds:
            q = row["problem"]
            gold = row["answer"]
            items.append({"question": q, "gold": str(gold)})
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if 0 < n_samples < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n_samples)

    return items


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
) -> List[torch.Tensor]:
    """Left-padded batch generation. Returns list of new-token tensors."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Extract new tokens per sample (strip the prompt part)
    new_tokens_list = []
    for i in range(len(prompts)):
        prompt_len = inputs["attention_mask"][i].sum().item()
        new_tokens = outputs[i][prompt_len:]
        # Strip padding from the end
        if tokenizer.pad_token_id is not None:
            mask = new_tokens != tokenizer.pad_token_id
            if mask.any():
                last_real = mask.nonzero()[-1].item() + 1
                new_tokens = new_tokens[:last_real]
            else:
                new_tokens = new_tokens[:0]
        new_tokens_list.append(new_tokens)

    return new_tokens_list


def process_sample(
    tokenizer, new_tokens: torch.Tensor, gold: str, budget: int
) -> dict:
    """Process a single sample's output tokens into metrics."""
    n_tokens = len(new_tokens)
    clean_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    hit_budget = n_tokens >= budget
    natural_stop = not hit_budget

    pred, pred_source = parse_prediction(clean_text)
    correct = is_correct(pred, gold)
    has_final = pred_source == "final_marker"

    return {
        "pred": pred,
        "pred_source": pred_source,
        "correct": correct,
        "n_tokens": n_tokens,
        "natural_stop": natural_stop,
        "has_final": has_final,
    }


def load_checkpoint(ckpt_path: str) -> dict:
    """Load checkpoint if it exists."""
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt_path: str, completed: set, all_results: list, summary: dict):
    """Save checkpoint after each (mode, budget) completes."""
    with open(ckpt_path, "w") as f:
        json.dump({
            "completed": list(completed),
            "all_results": all_results,
            "summary_per_mode": summary.get("per_mode", {}),
        }, f)


def run_experiment(args):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, f"checkpoint_{args.benchmark}.json")

    log.info(f"Model: {args.model}")
    log.info(f"Benchmark: {args.benchmark}, seed: {args.seed}")
    log.info(f"Budgets: {args.budgets}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Output: {out_dir}")

    # Load checkpoint
    ckpt = load_checkpoint(ckpt_path)
    completed = set(ckpt.get("completed", []))
    all_results = ckpt.get("all_results", [])
    if completed:
        log.info(f"Resuming from checkpoint: {len(completed)} configs done")

    # Load model
    cache_dir = args.cache_dir or "/workspace/hf_cache"
    log.info(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    log.info("Model loaded.")

    # Load data
    items = load_benchmark(args.benchmark, args.seed, args.n_samples)
    log.info(f"Loaded {len(items)} samples")

    summary = {
        "meta": {
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": len(items),
            "seed": args.seed,
            "budgets": args.budgets,
            "modes": args.modes,
            "batch_size": args.batch_size,
            "timestamp": ts,
        },
        "per_mode": ckpt.get("summary_per_mode", {}),
    }

    for mode in args.modes:
        enable_thinking = mode == "thinking"
        log.info(f"\n{'='*60}")
        log.info(f"Mode: {mode} (enable_thinking={enable_thinking})")
        log.info(f"{'='*60}")

        for budget in args.budgets:
            config_key = f"{mode}@{budget}"
            if config_key in completed:
                log.info(f"\n--- {config_key} --- SKIPPED (checkpoint)")
                continue

            log.info(f"\n--- {config_key} ---")

            # Build all prompts
            prompts = []
            golds = []
            questions = []
            for item in items:
                q = item["question"]
                if enable_thinking:
                    prompts.append(build_prompt_think(q, tokenizer, args.benchmark))
                else:
                    prompts.append(build_prompt_nothink(q, tokenizer, args.benchmark))
                golds.append(item["gold"])
                questions.append(q)

            # Process in batches
            correct_count = 0
            total_tokens = 0
            natural_stop_count = 0
            natural_stop_correct = 0
            hit_budget_count = 0
            hit_budget_correct = 0
            has_final_count = 0

            n = len(prompts)
            bs = args.batch_size
            t0_total = time.time()

            for batch_start in range(0, n, bs):
                batch_end = min(batch_start + bs, n)
                batch_prompts = prompts[batch_start:batch_end]
                batch_golds = golds[batch_start:batch_end]
                batch_questions = questions[batch_start:batch_end]

                t0 = time.time()
                new_tokens_list = generate_batch(
                    model, tokenizer, batch_prompts, budget
                )
                batch_latency = time.time() - t0

                per_sample_latency = batch_latency / len(batch_prompts)

                for j, (new_tokens, gold, q) in enumerate(
                    zip(new_tokens_list, batch_golds, batch_questions)
                ):
                    idx = batch_start + j
                    result = process_sample(tokenizer, new_tokens, gold, budget)

                    if result["correct"]:
                        correct_count += 1
                    total_tokens += result["n_tokens"]

                    if result["natural_stop"]:
                        natural_stop_count += 1
                        if result["correct"]:
                            natural_stop_correct += 1
                    else:
                        hit_budget_count += 1
                        if result["correct"]:
                            hit_budget_correct += 1

                    if result["has_final"]:
                        has_final_count += 1

                    all_results.append({
                        "idx": idx,
                        "mode": mode,
                        "budget": budget,
                        "question": q[:100],
                        "gold": gold,
                        "pred": result["pred"],
                        "pred_source": result["pred_source"],
                        "correct": result["correct"],
                        "n_tokens": result["n_tokens"],
                        "natural_stop": result["natural_stop"],
                        "has_final": result["has_final"],
                        "latency_s": round(per_sample_latency, 3),
                    })

                # Log progress
                done = batch_end
                acc_so_far = correct_count / done
                tok_per_s = total_tokens / (time.time() - t0_total)
                log.info(
                    f"  [{done}/{n}] acc={acc_so_far:.1%} "
                    f"avg_tok={total_tokens/done:.0f} "
                    f"natural_stop={natural_stop_count}/{done} "
                    f"tok/s={tok_per_s:.0f}"
                )

            total_latency = time.time() - t0_total
            acc = correct_count / n
            avg_tok = total_tokens / n
            avg_lat = total_latency / n
            ns_rate = natural_stop_count / n
            ns_acc = natural_stop_correct / natural_stop_count if natural_stop_count > 0 else 0
            hb_acc = hit_budget_correct / hit_budget_count if hit_budget_count > 0 else 0

            log.info(
                f"\n{config_key} FINAL: acc={acc:.1%} avg_tok={avg_tok:.0f} "
                f"natural_stop={ns_rate:.1%} ns_acc={ns_acc:.1%} hb_acc={hb_acc:.1%} "
                f"wall={total_latency:.0f}s tok/s={total_tokens/total_latency:.0f}"
            )

            summary["per_mode"].setdefault(mode, {})[str(budget)] = {
                "accuracy": round(acc, 4),
                "avg_tokens": round(avg_tok, 1),
                "avg_latency_s": round(avg_lat, 3),
                "wall_time_s": round(total_latency, 1),
                "natural_stop_rate": round(ns_rate, 4),
                "natural_stop_accuracy": round(ns_acc, 4) if natural_stop_count > 0 else None,
                "hit_budget_rate": round(1 - ns_rate, 4),
                "hit_budget_accuracy": round(hb_acc, 4) if hit_budget_count > 0 else None,
                "has_final_rate": round(has_final_count / n, 4),
            }

            # Checkpoint after each config
            completed.add(config_key)
            save_checkpoint(ckpt_path, completed, all_results, summary)
            log.info(f"  Checkpoint saved ({len(completed)} configs done)")

    # Compute thinking tax (only if both modes ran)
    summary["thinking_tax"] = {}
    for budget in args.budgets:
        b = str(budget)
        nt = summary["per_mode"].get("nothink", {}).get(b)
        tk = summary["per_mode"].get("thinking", {}).get(b)
        if nt and tk:
            summary["thinking_tax"][b] = {
                "nothink_accuracy": nt["accuracy"],
                "thinking_accuracy": tk["accuracy"],
                "tax_pp": round((nt["accuracy"] - tk["accuracy"]) * 100, 1),
            }

    # Save final results
    summary_path = os.path.join(out_dir, f"summary_{model_short}_{args.benchmark}_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {summary_path}")

    csv_path = os.path.join(out_dir, f"per_sample_{model_short}_{args.benchmark}_{ts}.csv")
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        log.info(f"Per-sample CSV saved to {csv_path}")

    # Print summary
    if summary["thinking_tax"]:
        log.info("\n" + "=" * 70)
        log.info(f"THINKING TAX SUMMARY: {model_short} on {args.benchmark}")
        log.info("=" * 70)
        log.info(f"{'Budget':>8} {'Nothink':>10} {'Think':>10} {'Tax':>10}")
        for budget in args.budgets:
            b = str(budget)
            if b in summary["thinking_tax"]:
                tax = summary["thinking_tax"][b]
                log.info(f"{budget:>8} {tax['nothink_accuracy']:>9.1%} {tax['thinking_accuracy']:>9.1%} {tax['tax_pp']:>+9.1f}pp")

    # Clean up checkpoint on success
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        log.info("Checkpoint cleaned up (run complete)")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-R1 cross-model thinking tax (batched)")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--output_dir", default="results/deepseek_crossmodel")
    parser.add_argument("--cache_dir", default="/workspace/hf_cache")
    parser.add_argument("--modes", nargs="+", default=["nothink", "thinking"],
                        choices=["nothink", "thinking"])
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_experiment(args)
