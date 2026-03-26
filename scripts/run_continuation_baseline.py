#!/usr/bin/env python3
"""
Continuation-from-b1 baseline: Instead of discarding the b1 trace and generating
fresh at b*, continue from the b1 prefix up to b* tokens. This tests whether
the two-pass overhead can be eliminated via prefix reuse.

Usage:
    python run_continuation_baseline.py \
        --model Qwen/Qwen3.5-27B \
        --benchmark gsm8k \
        --budgets 128,256,512 \
        --data_seed 42 \
        --num_samples 40 \
        --output_dir ../results/continuation_baseline
"""
import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_gsm8k_experiment import (
    parse_prediction,
    project_final_answer,
    build_prompt,
    model_input_device,
    GenOutput,
)


def generate_continuation(
    model, tokenizer, prompt: str,
    b1_tokens: int, b_star_tokens: int,
    temperature: float = 0.0,
) -> tuple:
    """Generate at b1, then continue to b_star from the same prefix."""
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=b1_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    start = time.perf_counter()
    with torch.no_grad():
        out_b1 = model.generate(**inputs, **gen_kwargs)
    b1_elapsed = time.perf_counter() - start
    b1_gen_ids = out_b1[0][in_len:]
    b1_text = tokenizer.decode(b1_gen_ids, skip_special_tokens=True)
    b1_actual_tokens = int(b1_gen_ids.shape[0])

    if b_star_tokens <= b1_tokens:
        return b1_text, b1_actual_tokens, b1_elapsed, b1_text, b1_actual_tokens

    remaining_tokens = b_star_tokens - b1_actual_tokens
    if remaining_tokens <= 0:
        return b1_text, b1_actual_tokens, b1_elapsed, b1_text, b1_actual_tokens

    gen_kwargs_cont = dict(
        max_new_tokens=remaining_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs_cont["temperature"] = temperature

    start2 = time.perf_counter()
    with torch.no_grad():
        out_full = model.generate(out_b1, **gen_kwargs_cont)
    cont_elapsed = time.perf_counter() - start2

    full_gen_ids = out_full[0][in_len:]
    full_text = tokenizer.decode(full_gen_ids, skip_special_tokens=True)
    full_actual_tokens = int(full_gen_ids.shape[0])
    total_elapsed = b1_elapsed + cont_elapsed

    return b1_text, b1_actual_tokens, total_elapsed, full_text, full_actual_tokens


def load_benchmark(name, num_samples, data_seed):
    """Load benchmark questions."""
    rng = random.Random(data_seed)
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        questions = [(ds[i]["question"], ds[i]["answer"]) for i in indices]
    elif name == "math500":
        ds = load_dataset("hendrycks/competition_math", split="test")
        indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        questions = [(ds[i]["problem"], ds[i]["solution"]) for i in indices]
    else:
        raise ValueError(f"Unknown benchmark: {name}")
    return questions


def extract_ground_truth(answer_text, benchmark):
    """Extract numeric answer from ground truth."""
    if benchmark == "gsm8k":
        match = re.search(r"####\s*([\d,.-]+)", answer_text)
        if match:
            return match.group(1).replace(",", "")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--budgets", type=str, default="128,256,512",
                        help="Comma-separated budget tiers")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="../results/continuation_baseline")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    b1, b_mid, b_max = budgets[0], budgets[1], budgets[-1]

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    questions = load_benchmark(args.benchmark, args.num_samples, args.data_seed)
    print(f"Loaded {len(questions)} questions from {args.benchmark}")

    results = []
    for i, (question, answer) in enumerate(questions):
        gt = extract_ground_truth(answer, args.benchmark)
        prompt = build_prompt(question, args.enable_thinking)

        b1_text, b1_tokens, b1_time, _, _ = generate_continuation(
            model, tokenizer, prompt, b1, b1, temperature=0.0
        )
        b1_pred, b1_has_final, _ = parse_prediction(b1_text)
        b1_correct = (b1_pred == gt) if (b1_pred and gt) else False

        b1_features = {
            "has_answer": b1_has_final,
            "token_util": b1_tokens / b1,
        }

        selected_budget = b1 if b1_has_final else b_max

        if selected_budget > b1:
            _, _, cont_time, full_text, full_tokens = generate_continuation(
                model, tokenizer, prompt, b1, selected_budget, temperature=0.0
            )
            full_pred, full_has_final, _ = parse_prediction(full_text)
            full_correct = (full_pred == gt) if (full_pred and gt) else False
            total_tokens = full_tokens  # continuation reuses prefix
        else:
            full_text = b1_text
            full_tokens = b1_tokens
            full_pred = b1_pred
            full_correct = b1_correct
            cont_time = b1_time
            total_tokens = b1_tokens

        results.append({
            "idx": i, "question": question[:80], "ground_truth": gt,
            "selected_budget": selected_budget,
            "b1_tokens": b1_tokens, "b1_correct": int(b1_correct),
            "final_tokens": full_tokens, "final_correct": int(full_correct),
            "total_tokens": total_tokens,
            "time_s": cont_time,
        })

        if (i + 1) % 10 == 0:
            acc = sum(r["final_correct"] for r in results) / len(results)
            avg_tok = sum(r["total_tokens"] for r in results) / len(results)
            print(f"  [{i+1}/{len(questions)}] acc={acc:.3f} avg_tok={avg_tok:.1f}")

    n = len(results)
    acc = sum(r["final_correct"] for r in results) / n
    avg_tok = sum(r["total_tokens"] for r in results) / n

    summary = {
        "benchmark": args.benchmark, "model": args.model,
        "budgets": budgets, "data_seed": args.data_seed, "seed": args.seed,
        "n": n, "accuracy": acc, "avg_tokens": avg_tok,
        "method": "continuation_from_b1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.output_dir, f"continuation_{args.benchmark}_{ts}.csv")
    out_json = os.path.join(args.output_dir, f"continuation_{args.benchmark}_{ts}.json")

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Continuation Baseline ===")
    print(f"Accuracy: {acc:.3f}")
    print(f"Avg tokens: {avg_tok:.1f}")
    print(f"Results: {out_csv}")


if __name__ == "__main__":
    main()
