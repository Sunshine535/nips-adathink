#!/usr/bin/env python3
"""
Cross-model-family validation: DeepSeek-R1-Distill-Llama-8B on GSM8K.
Runs both thinking and nothink modes to demonstrate the thinking tax
generalizes beyond Qwen family.

DeepSeek-R1 doesn't natively support enable_thinking=False, so we manually
construct the nothink prompt by closing the <think> block immediately.

Usage:
    python3 run_deepseek_crossmodel.py --benchmark gsm8k --budgets 256 512 1024 2048 --seed 42
    python3 run_deepseek_crossmodel.py --benchmark math500 --budgets 512 1024 2048 4096 --seed 42
"""
import argparse
import csv
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

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
    """Extract prediction. Returns (pred, source)."""
    # Strip thinking block if present
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
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def is_correct(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    return normalize_number(pred) == normalize_number(gold)


def build_prompt_think(question: str, tokenizer, benchmark: str) -> str:
    """Standard thinking prompt using chat template."""
    if benchmark == "math500":
        content = f"Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.\n\n{question}"
    else:
        content = f"Solve this math problem step by step. At the end, write 'Final answer: <number>'.\n\n{question}"
    msgs = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_prompt_nothink(question: str, tokenizer, benchmark: str) -> str:
    """Nothink prompt: close the <think> block immediately so the model
    starts generating the answer directly. Budget is used entirely for answer."""
    if benchmark == "math500":
        content = f"Solve the following math problem. Show your work and put your final answer in \\boxed{{}}.\n\n{question}"
    else:
        content = f"Solve this math problem step by step. At the end, write 'Final answer: <number>'.\n\n{question}"
    msgs = [{"role": "user", "content": content}]
    # Build with chat template (includes <think>\n)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Close thinking block immediately
    prompt = prompt + "\n</think>\n\n"
    return prompt


def load_benchmark(benchmark: str, seed: int, n_samples: int = 0):
    """Load GSM8K or MATH-500 dataset."""
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for row in ds:
            q = row["question"]
            # Extract gold from answer field
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

    if n_samples > 0 and n_samples < len(items):
        rng = random.Random(seed)
        items = rng.sample(items, n_samples)

    return items


def run_experiment(args):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_short = args.model.split("/")[-1]
    out_dir = os.path.join(args.output_dir, f"deepseek_crossmodel_{args.benchmark}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    log.info(f"Model: {args.model}")
    log.info(f"Benchmark: {args.benchmark}, seed: {args.seed}")
    log.info(f"Budgets: {args.budgets}")
    log.info(f"Output: {out_dir}")

    # Load model
    cache_dir = args.cache_dir or "/workspace/hf_cache"
    log.info(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    log.info(f"Loading model from {args.model}...")
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

    # Results storage
    all_results = []
    summary = {
        "meta": {
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": len(items),
            "seed": args.seed,
            "budgets": args.budgets,
            "modes": ["nothink", "thinking"],
            "timestamp": ts,
        },
        "per_mode": {},
    }

    for mode in ["nothink", "thinking"]:
        enable_thinking = mode == "thinking"
        log.info(f"\n{'='*60}")
        log.info(f"Mode: {mode} (enable_thinking={enable_thinking})")
        log.info(f"{'='*60}")

        for budget in args.budgets:
            log.info(f"\n--- {mode}@{budget} ---")
            correct_count = 0
            total_tokens = 0
            total_latency = 0.0
            natural_stop_count = 0
            natural_stop_correct = 0
            hit_budget_count = 0
            hit_budget_correct = 0
            has_final_count = 0

            for i, item in enumerate(items):
                q = item["question"]
                gold = item["gold"]

                # Build prompt
                if enable_thinking:
                    prompt = build_prompt_think(q, tokenizer, args.benchmark)
                else:
                    prompt = build_prompt_nothink(q, tokenizer, args.benchmark)

                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

                t0 = time.time()
                with torch.no_grad():
                    out = model.generate(
                        input_ids,
                        max_new_tokens=budget,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                    )
                latency = time.time() - t0

                new_tokens = out[0][input_ids.shape[1]:]
                n_tokens = len(new_tokens)
                raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
                clean_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Determine if natural stop or hit budget
                eos_id = tokenizer.eos_token_id
                hit_budget = n_tokens >= budget
                natural_stop = not hit_budget

                # Parse prediction
                pred, pred_source = parse_prediction(clean_text)
                correct = is_correct(pred, gold)
                has_final = pred_source == "final_marker"

                if correct:
                    correct_count += 1
                total_tokens += n_tokens
                total_latency += latency

                if natural_stop:
                    natural_stop_count += 1
                    if correct:
                        natural_stop_correct += 1
                else:
                    hit_budget_count += 1
                    if correct:
                        hit_budget_correct += 1

                if has_final:
                    has_final_count += 1

                all_results.append({
                    "idx": i,
                    "mode": mode,
                    "budget": budget,
                    "question": q[:100],
                    "gold": gold,
                    "pred": pred,
                    "pred_source": pred_source,
                    "correct": correct,
                    "n_tokens": n_tokens,
                    "natural_stop": natural_stop,
                    "has_final": has_final,
                    "latency_s": round(latency, 3),
                })

                if (i + 1) % 50 == 0 or (i + 1) == len(items):
                    acc_so_far = correct_count / (i + 1)
                    log.info(
                        f"  [{i+1}/{len(items)}] acc={acc_so_far:.1%} "
                        f"avg_tok={total_tokens/(i+1):.0f} "
                        f"natural_stop={natural_stop_count}/{i+1}"
                    )

            n = len(items)
            acc = correct_count / n
            avg_tok = total_tokens / n
            avg_lat = total_latency / n
            ns_rate = natural_stop_count / n
            ns_acc = natural_stop_correct / natural_stop_count if natural_stop_count > 0 else 0
            hb_acc = hit_budget_correct / hit_budget_count if hit_budget_count > 0 else 0

            log.info(f"\n{mode}@{budget} FINAL: acc={acc:.1%} avg_tok={avg_tok:.0f} "
                     f"natural_stop={ns_rate:.1%} ns_acc={ns_acc:.1%} hb_acc={hb_acc:.1%}")

            summary["per_mode"].setdefault(mode, {})[str(budget)] = {
                "accuracy": round(acc, 4),
                "avg_tokens": round(avg_tok, 1),
                "avg_latency_s": round(avg_lat, 3),
                "natural_stop_rate": round(ns_rate, 4),
                "natural_stop_accuracy": round(ns_acc, 4) if natural_stop_count > 0 else None,
                "hit_budget_rate": round(1 - ns_rate, 4),
                "hit_budget_accuracy": round(hb_acc, 4) if hit_budget_count > 0 else None,
                "has_final_rate": round(has_final_count / n, 4),
            }

    # Compute thinking tax
    summary["thinking_tax"] = {}
    for budget in args.budgets:
        b = str(budget)
        nt_acc = summary["per_mode"]["nothink"][b]["accuracy"]
        tk_acc = summary["per_mode"]["thinking"][b]["accuracy"]
        summary["thinking_tax"][b] = {
            "nothink_accuracy": nt_acc,
            "thinking_accuracy": tk_acc,
            "tax_pp": round((nt_acc - tk_acc) * 100, 1),
        }

    # Save results
    summary_path = os.path.join(out_dir, f"summary_{model_short}_{args.benchmark}_{ts}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {summary_path}")

    csv_path = os.path.join(out_dir, f"per_sample_{model_short}_{args.benchmark}_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    log.info(f"Per-sample CSV saved to {csv_path}")

    # Print summary table
    log.info("\n" + "=" * 70)
    log.info(f"THINKING TAX SUMMARY: {model_short} on {args.benchmark}")
    log.info("=" * 70)
    log.info(f"{'Budget':>8} {'Nothink':>10} {'Think':>10} {'Tax':>10}")
    for budget in args.budgets:
        b = str(budget)
        tax = summary["thinking_tax"][b]
        log.info(f"{budget:>8} {tax['nothink_accuracy']:>9.1%} {tax['thinking_accuracy']:>9.1%} {tax['tax_pp']:>+9.1f}pp")

    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek-R1 cross-model thinking tax experiment")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--output_dir", default="results/deepseek_crossmodel")
    parser.add_argument("--cache_dir", default="/workspace/hf_cache")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_experiment(args)
