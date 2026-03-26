#!/usr/bin/env python3
"""
Multi-seed evaluation: Run the template controller with multiple algorithm seeds
to assess stochastic decoding variance (not just data-seed variance).

Usage:
    python run_multiseed_eval.py \
        --model Qwen/Qwen3.5-27B \
        --benchmark gsm8k \
        --budgets 128,256,512 \
        --algo_seeds 11,42,123 \
        --data_seed 42 \
        --num_samples 40 \
        --output_dir ../results/multiseed
"""
import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_gsm8k_experiment import (
    parse_prediction,
    project_final_answer,
    build_prompt,
    model_input_device,
    generate_once,
)


def load_benchmark(name, num_samples, data_seed):
    rng = random.Random(data_seed)
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [(ds[i]["question"], ds[i]["answer"]) for i in indices]
    elif name == "math500":
        ds = load_dataset("hendrycks/competition_math", split="test")
        indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))
        return [(ds[i]["problem"], ds[i]["solution"]) for i in indices]
    raise ValueError(f"Unknown benchmark: {name}")


def extract_ground_truth(answer_text, benchmark):
    if benchmark == "gsm8k":
        match = re.search(r"####\s*([\d,.-]+)", answer_text)
        return match.group(1).replace(",", "") if match else None
    return None


def run_fixed_budget(model, tokenizer, questions, budget, seed,
                     benchmark, enable_thinking=True):
    """Run a fixed budget evaluation with a given seed."""
    torch.manual_seed(seed)
    results = []
    for i, (question, answer) in enumerate(questions):
        gt = extract_ground_truth(answer, benchmark)
        prompt = build_prompt(question, enable_thinking)
        out = generate_once(model, tokenizer, prompt, max_new_tokens=budget, temperature=0.0)
        pred, has_final, _ = parse_prediction(out.text)

        if pred is None:
            proj_pred, proj_tokens, _, _, _ = project_final_answer(
                model, tokenizer, question, out.text, max_new_tokens=32,
            )
            pred = proj_pred

        correct = (pred == gt) if (pred and gt) else False
        results.append({
            "idx": i, "correct": int(correct),
            "tokens": out.new_tokens, "pred": pred, "gt": gt,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--budgets", type=str, default="128,256,512")
    parser.add_argument("--algo_seeds", type=str, default="11,42,123",
                        help="Algorithm seeds to evaluate")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default="../results/multiseed")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    algo_seeds = [int(s) for s in args.algo_seeds.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    questions = load_benchmark(args.benchmark, args.num_samples, args.data_seed)
    print(f"Loaded {len(questions)} questions, data_seed={args.data_seed}")

    all_results = {}

    for budget in budgets:
        for seed in algo_seeds:
            key = f"budget_{budget}_seed_{seed}"
            print(f"\n=== Budget={budget}, Seed={seed} ===")

            results = run_fixed_budget(
                model, tokenizer, questions, budget, seed,
                args.benchmark, args.enable_thinking,
            )
            acc = sum(r["correct"] for r in results) / len(results)
            avg_tok = sum(r["tokens"] for r in results) / len(results)

            all_results[key] = {
                "budget": budget, "algo_seed": seed,
                "accuracy": acc, "avg_tokens": avg_tok, "n": len(results),
            }
            print(f"  acc={acc:.3f} avg_tok={avg_tok:.1f}")

    print("\n=== Cross-Seed Variance Analysis ===")
    for budget in budgets:
        accs = [all_results[f"budget_{budget}_seed_{s}"]["accuracy"]
                for s in algo_seeds]
        print(f"Budget {budget}: mean_acc={np.mean(accs):.3f} "
              f"std={np.std(accs):.3f} range=[{min(accs):.3f}, {max(accs):.3f}]")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(
        args.output_dir, f"multiseed_{args.benchmark}_{ts}.json"
    )
    with open(out_json, "w") as f:
        json.dump({
            "benchmark": args.benchmark, "model": args.model,
            "budgets": budgets, "algo_seeds": algo_seeds,
            "data_seed": args.data_seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
