#!/usr/bin/env python3
"""
Standalone TOWN baseline runner — no IRIS dependency.
Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_town_only.py \
        --model Qwen/Qwen3-8B --benchmark math500 --n_samples 500 \
        --b1 512 --b2 4096 --seed 42 --output_dir results/town_b4096_a100
"""
import json, os, argparse, time, logging, datetime
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Import shared utilities
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import (
    load_math500, load_gsm8k, load_benchmark,
    parse_prediction_math, parse_prediction_gsm8k,
    is_correct_math, is_correct_gsm8k, is_correct,
    BenchmarkSample,
)


def build_prompt(question, tokenizer, enable_thinking, benchmark="gsm8k"):
    """Matches run_iris.py build_prompt exactly."""
    SYSTEM_TEXTS = {
        "gsm8k": (
            "You are a careful math solver. Solve the problem step by step briefly. "
            "End with a single line: Final answer: <number>."
        ),
        "math500": (
            "You are an expert mathematician. Solve the following problem step by step. "
            "Put your final answer inside \\boxed{}."
        ),
    }
    system_text = SYSTEM_TEXTS.get(benchmark, SYSTEM_TEXTS["gsm8k"])
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    chat_kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_simple(model, tokenizer, prompt, max_new_tokens, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=(temperature > 0),
        )
    elapsed = time.time() - t0
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    n_tokens = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    hit_budget = (n_tokens >= max_new_tokens)
    return text, n_tokens, elapsed, hit_budget


def parse_pred(text, benchmark):
    if benchmark == "math500":
        pred, _, _ = parse_prediction_math(text)
    else:
        pred, _, _ = parse_prediction_gsm8k(text)
    return pred


def check_correct(pred, gold, benchmark):
    return is_correct(pred, gold, benchmark=benchmark)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", type=str, default="math500", choices=["gsm8k", "math500"])
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--b1", type=int, default=512, help="Stage 1 nothink budget")
    parser.add_argument("--b2", type=int, default=4096, help="Stage 2 thinking budget")
    parser.add_argument("--output_dir", type=str, default="results/town_standalone")
    parser.add_argument("--checkpoint_every", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data — matches run_iris.py's load_math500_for_iris / load_gsm8k
    if args.benchmark == "math500":
        raw = load_math500(split="test")
        items = [{"question": s.question, "gold": s.gold} for s in raw[:args.n_samples]]
    else:
        raw = load_gsm8k(split="test")
        items = [{"question": s.question, "gold": s.gold} for s in raw[:args.n_samples]]
    n = len(items)
    log.info(f"Loaded {n} samples from {args.benchmark}")

    # Load model
    log.info(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded.")

    # Run TOWN
    results = []
    correct_count = 0
    total_tokens = 0
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    for i, item in enumerate(items):
        question = item["question"]
        gold = item["gold"]

        # Stage 1: nothink
        prompt_s1 = build_prompt(question, tokenizer, enable_thinking=False, benchmark=args.benchmark)
        text_s1, tokens_s1, elapsed_s1, hit_budget_s1 = generate_simple(
            model, tokenizer, prompt_s1, max_new_tokens=args.b1, temperature=0.0,
        )
        pred_s1 = parse_pred(text_s1, args.benchmark)

        if not hit_budget_s1:
            stage = 1
            pred = pred_s1
            tok_total = tokens_s1
            elapsed_total = elapsed_s1
            stop_reason = "stage1_natural_stop"
            text_s2 = ""
            tokens_s2 = 0
        else:
            # Stage 2: think
            prompt_s2 = build_prompt(question, tokenizer, enable_thinking=True, benchmark=args.benchmark)
            text_s2, tokens_s2, elapsed_s2, _ = generate_simple(
                model, tokenizer, prompt_s2, max_new_tokens=args.b2, temperature=0.0,
            )
            pred_s2 = parse_pred(text_s2, args.benchmark)
            stage = 2
            pred = pred_s2
            tok_total = tokens_s1 + tokens_s2
            elapsed_total = elapsed_s1 + elapsed_s2
            stop_reason = "stage2_town"

        c = check_correct(pred, gold, args.benchmark)
        if c:
            correct_count += 1
        total_tokens += tok_total

        results.append({
            "idx": i,
            "gold": gold,
            "correct": int(c),
            "pred": pred,
            "stage": stage,
            "tokens_total": tok_total,
            "tokens_s1": tokens_s1,
            "tokens_s2": tokens_s2,
            "stop_reason": stop_reason,
        })

        if (i + 1) % 20 == 0 or i == n - 1:
            acc = correct_count / (i + 1)
            avg = total_tokens / (i + 1)
            log.info(f"  TOWN [{i+1}/{n}] acc={acc:.3f} avg_tokens={avg:.0f}")

        # Checkpoint
        if (i + 1) % args.checkpoint_every == 0:
            ckpt = {
                "n_done": i + 1,
                "accuracy": round(correct_count / (i + 1), 4),
                "avg_tokens": round(total_tokens / (i + 1), 2),
                "results": results[:],
            }
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_town_{i+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(ckpt, f, indent=2)
            log.info(f"  Checkpoint saved: {ckpt_path}")

    # Final summary
    acc = correct_count / n
    avg_tok = total_tokens / n
    summary = {
        "meta": {
            "script": "run_town_only.py",
            "timestamp_utc": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": n,
            "seed": args.seed,
        },
        "config": {"b1": args.b1, "b2": args.b2},
        "results": {
            "accuracy": round(acc, 4),
            "avg_tokens": round(avg_tok, 2),
        },
        "per_sample": results,
    }
    out_path = os.path.join(
        args.output_dir,
        f"town_{args.benchmark}_b1{args.b1}_b2{args.b2}_{timestamp}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nFinal: accuracy={acc:.3f}, avg_tokens={avg_tok:.0f}")
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
