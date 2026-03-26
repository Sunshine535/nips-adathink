#!/usr/bin/env python3
"""
Fair self-consistency baselines with per-sample budgets large enough to produce
parseable CoT traces. Tests SC@2×256 and SC@4×128 at matched total cost (512).

Usage:
    python run_fair_sc_baseline.py \
        --model Qwen/Qwen3.5-27B \
        --benchmark gsm8k \
        --sc_configs "2x256,4x128,1x512" \
        --data_seed 42 \
        --num_samples 40 \
        --output_dir ../results/fair_sc
"""
import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
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
    generate_once,
)


def sc_majority_vote(predictions):
    """Return the majority answer from a list of predictions."""
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None
    counts = Counter(valid)
    return counts.most_common(1)[0][0]


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
        if match:
            return match.group(1).replace(",", "")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--sc_configs", type=str, default="2x256,4x128,8x64,1x512",
                        help="SC configs as KxB pairs, comma-separated")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num_samples", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output_dir", type=str, default="../results/fair_sc")
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    args = parser.parse_args()

    configs = []
    for cfg in args.sc_configs.split(","):
        k, b = cfg.strip().split("x")
        configs.append((int(k), int(b)))

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

    all_results = {}

    for K, B in configs:
        total_budget = K * B
        config_name = f"SC@{K}x{B}"
        print(f"\n=== {config_name} (total budget: {total_budget}) ===")

        results = []
        for i, (question, answer) in enumerate(questions):
            gt = extract_ground_truth(answer, args.benchmark)
            prompt = build_prompt(question, args.enable_thinking)

            predictions = []
            total_tokens = 0
            total_time = 0.0

            for sample_idx in range(K):
                torch.manual_seed(args.seed * 1000 + i * 100 + sample_idx)
                out = generate_once(model, tokenizer, prompt, max_new_tokens=B,
                                    temperature=args.temperature)
                pred, has_final, _ = parse_prediction(out.text)

                if pred is None and not has_final:
                    proj_pred, proj_tokens, proj_time, _, _ = project_final_answer(
                        model, tokenizer, question, out.text, max_new_tokens=32,
                    )
                    pred = proj_pred
                    total_tokens += proj_tokens
                    total_time += proj_time

                predictions.append(pred)
                total_tokens += out.new_tokens
                total_time += out.elapsed_s

            majority = sc_majority_vote(predictions)
            correct = (majority == gt) if (majority and gt) else False
            parseable = sum(1 for p in predictions if p is not None)

            results.append({
                "idx": i, "question": question[:80], "ground_truth": gt,
                "K": K, "B": B, "total_budget": total_budget,
                "predictions": str(predictions), "majority": majority,
                "correct": int(correct), "parseable_count": parseable,
                "total_tokens": total_tokens, "time_s": total_time,
            })

            if (i + 1) % 10 == 0:
                acc = sum(r["correct"] for r in results) / len(results)
                parse_rate = sum(r["parseable_count"] for r in results) / (len(results) * K)
                print(f"  [{i+1}/{len(questions)}] acc={acc:.3f} parse_rate={parse_rate:.2f}")

        n = len(results)
        acc = sum(r["correct"] for r in results) / n
        avg_tok = sum(r["total_tokens"] for r in results) / n
        parse_rate = sum(r["parseable_count"] for r in results) / (n * K)

        all_results[config_name] = {
            "K": K, "B": B, "total_budget": K * B,
            "n": n, "accuracy": acc, "avg_tokens": avg_tok,
            "parse_rate": parse_rate,
        }

        print(f"  {config_name}: acc={acc:.3f} avg_tok={avg_tok:.1f} parse_rate={parse_rate:.2f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.output_dir, f"fair_sc_{args.benchmark}_{ts}.json")
    with open(out_json, "w") as f:
        json.dump({
            "benchmark": args.benchmark, "model": args.model,
            "data_seed": args.data_seed, "seed": args.seed,
            "temperature": args.temperature,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": all_results,
        }, f, indent=2)

    print(f"\n=== Summary ===")
    for name, r in all_results.items():
        print(f"{name}: acc={r['accuracy']:.3f} avg_tok={r['avg_tokens']:.1f} "
              f"parse={r['parse_rate']:.2f}")
    print(f"\nResults: {out_json}")


if __name__ == "__main__":
    main()
