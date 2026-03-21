#!/usr/bin/env python3
"""Self-Consistency@k baseline using vLLM for MATH-500 and BBH benchmarks.

For each sample, generates k responses at a given per-sample budget, then
takes the majority vote. Compares against greedy at a matched total budget.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# benchmarks module in same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import (
    load_benchmark,
    parse_prediction,
    math_answers_equiv,
    default_budgets,
)


def answers_equiv(pred, gold, benchmark):
    """Compare prediction to gold. pred may be str or tuple (from parse_prediction)."""
    pred_str = pred[0] if isinstance(pred, tuple) else pred
    if not pred_str or not gold:
        return False
    if benchmark == "math500":
        return math_answers_equiv(pred_str, gold)
    return pred_str.strip().lower() == gold.strip().lower()

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def majority_vote(preds: List[str], benchmark: str) -> str:
    """Return majority answer, preferring most common exact string."""
    if not preds:
        return ""
    filtered = [p for p in preds if p.strip()]
    if not filtered:
        return ""
    counts = Counter(filtered)
    return counts.most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser(description="SC@k baseline with vLLM")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--benchmark", type=str, default="math500", choices=["math500", "bbh", "gsm8k"])
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--sc_k", type=int, default=4, help="Number of SC samples")
    ap.add_argument("--sc_budget", type=int, default=512, help="Per-sample token budget for SC")
    ap.add_argument("--greedy_budget", type=int, default=2048, help="Greedy baseline budget (should be ~= sc_k * sc_budget)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_seed", type=int, default=101)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--gpu_utilization", type=float, default=0.85)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading benchmark: {args.benchmark}, n={args.n_samples}, seed={args.data_seed}")
    dataset = load_benchmark(args.benchmark, n_samples=args.n_samples, seed=args.data_seed)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed")
        sys.exit(1)

    model_name = args.model.split("/")[-1].replace("-", "_").replace(".", "_")
    is_large = any(k in args.model.lower() for k in ["27b", "70b", "72b"])
    tp = 2 if is_large else 1

    print(f"Loading model: {args.model} (tp={tp})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_utilization,
        max_model_len=max(args.greedy_budget, args.sc_budget) + 2048,
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()

    # Build prompts
    prompts = []
    for item in dataset:
        q = item.question
        messages = [
            {"role": "system", "content": "Solve the problem step by step. End with 'Final answer: <answer>'"},
            {"role": "user", "content": q},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # 1. Greedy decode at greedy_budget
    print(f"\n=== Greedy@{args.greedy_budget} ({len(prompts)} samples) ===")
    greedy_params = SamplingParams(
        max_tokens=args.greedy_budget,
        temperature=0.0,
        top_p=1.0,
        seed=args.seed,
    )
    t0 = time.time()
    greedy_outputs = llm.generate(prompts, greedy_params)
    greedy_time = time.time() - t0
    print(f"  Done in {greedy_time:.1f}s")

    # 2. SC@k at sc_budget each
    print(f"\n=== SC@{args.sc_k} (budget={args.sc_budget} each, {len(prompts)*args.sc_k} total generations) ===")
    sc_params = SamplingParams(
        max_tokens=args.sc_budget,
        temperature=args.temperature,
        top_p=0.95,
        n=args.sc_k,
        seed=args.seed,
    )
    t0 = time.time()
    sc_outputs = llm.generate(prompts, sc_params)
    sc_time = time.time() - t0
    print(f"  Done in {sc_time:.1f}s")

    # Evaluate
    rows = []
    greedy_correct = 0
    sc_correct = 0

    for i, item in enumerate(dataset):
        gold = item.gold

        # Greedy
        greedy_text = greedy_outputs[i].outputs[0].text
        greedy_tokens = len(greedy_outputs[i].outputs[0].token_ids)
        greedy_pred_raw = parse_prediction(greedy_text, args.benchmark)
        greedy_pred = greedy_pred_raw[0] if isinstance(greedy_pred_raw, tuple) else greedy_pred_raw
        greedy_c = 1 if answers_equiv(greedy_pred, gold, args.benchmark) else 0
        greedy_correct += greedy_c

        # SC@k
        sc_preds = []
        sc_total_tokens = 0
        for j in range(min(args.sc_k, len(sc_outputs[i].outputs))):
            out = sc_outputs[i].outputs[j]
            pred_raw = parse_prediction(out.text, args.benchmark)
            pred = pred_raw[0] if isinstance(pred_raw, tuple) else pred_raw
            sc_preds.append(pred if pred else "")
            sc_total_tokens += len(out.token_ids)

        sc_final = majority_vote(sc_preds, args.benchmark)
        sc_c = 1 if answers_equiv(sc_final, gold, args.benchmark) else 0
        sc_correct += sc_c

        rows.append({
            "idx": i,
            "question": item.question[:100],
            "gold": gold,
            "greedy_pred": greedy_pred,
            "greedy_correct": greedy_c,
            "greedy_tokens": greedy_tokens,
            "sc_preds": "|".join(sc_preds),
            "sc_final": sc_final,
            "sc_correct": sc_c,
            "sc_total_tokens": sc_total_tokens,
        })

    n = len(dataset)
    greedy_acc = greedy_correct / n
    sc_acc = sc_correct / n
    avg_greedy_tok = sum(r["greedy_tokens"] for r in rows) / n
    avg_sc_tok = sum(r["sc_total_tokens"] for r in rows) / n

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "meta": {
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "sc_k": args.sc_k,
            "sc_budget": args.sc_budget,
            "greedy_budget": args.greedy_budget,
            "temperature": args.temperature,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "timestamp": ts,
        },
        "greedy": {
            "accuracy": greedy_acc,
            "avg_tokens": avg_greedy_tok,
            "total_time_s": greedy_time,
        },
        "sc": {
            "accuracy": sc_acc,
            "avg_total_tokens": avg_sc_tok,
            "total_time_s": sc_time,
        },
        "delta": {
            "accuracy": sc_acc - greedy_acc,
            "tokens": avg_sc_tok - avg_greedy_tok,
        },
    }

    out_json = os.path.join(args.results_dir, f"sc_baseline_{args.benchmark}_{model_name}_{ts}.json")
    out_csv = os.path.join(args.results_dir, f"sc_baseline_per_sample_{args.benchmark}_{model_name}_{ts}.csv")

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n=== RESULTS ===")
    print(f"  Greedy@{args.greedy_budget}: acc={greedy_acc:.4f}, avg_tokens={avg_greedy_tok:.0f}")
    print(f"  SC@{args.sc_k}(budget={args.sc_budget}): acc={sc_acc:.4f}, avg_tokens={avg_sc_tok:.0f}")
    print(f"  Delta: acc={sc_acc-greedy_acc:+.4f}, tokens={avg_sc_tok-avg_greedy_tok:+.0f}")
    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
