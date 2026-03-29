#!/usr/bin/env python3
"""Confidence-Gated Early Exit (CGEE) for Thinking Models.

Core insight: The biggest waste in fixed-budget thinking is NOT on easy questions
(which finish early anyway) but on IMPOSSIBLE questions that burn full budget
and still produce wrong answers.

At Fixed-512:
  - 62% samples stop early → 97% accuracy (already efficient!)
  - 38% samples hit budget → only 13% accuracy (massive waste!)

CGEE method:
  1. Generate with max_budget tokens
  2. At each checkpoint (128, 256, 384, ...), compute a confidence signal
  3. If confidence is low AND we haven't produced a "Final answer:" yet → STOP EARLY
  4. Redirect saved tokens to other questions or skip entirely

Confidence signals (computed during generation, zero extra cost):
  - Repetition rate: high repetition → model is stuck
  - Token probability entropy: high entropy → model is uncertain
  - Progress indicator: has the model produced any numbers/equations?
  - Partial answer presence: any "Final answer:" or boxed{} started?

This paper studies which signals work and quantifies savings.

Usage:
  python scripts/run_confidence_gated_exit.py \
    --model Qwen/Qwen3-8B \
    --n_samples 200 \
    --max_budget 512 \
    --checkpoints 128 256 384 \
    --seed 42
"""

import argparse
import json
import logging
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone

import torch
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_last_number(text):
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def to_float(s):
    if s is None:
        return None
    s = s.replace(",", "").strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                d = float(parts[1])
                return float(parts[0]) / d if d != 0 else None
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred, gold, tol=1e-6):
    p, g = to_float(pred), to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def get_gold_from_gsm8k(answer_field):
    if "####" in answer_field:
        after = answer_field.split("####")[-1]
        match = NUM_RE.search(after)
        if match:
            return match.group(0)
    return extract_last_number(answer_field)


def parse_prediction(text):
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", ""), True, "final_answer"
    m = BOXED_RE.search(text)
    if m:
        inner = m.group(1).replace(",", "")
        num = NUM_RE.search(inner)
        if num:
            return num.group(0), True, "boxed"
    last = extract_last_number(text)
    if last:
        return last.replace(",", ""), False, "last_number"
    return None, False, "none"


def compute_confidence_signals(text, generated_tokens=None):
    """Compute confidence signals from generated text (zero extra cost).

    Returns dict with multiple signals that can be combined for gating.
    """
    signals = {}

    # 1. Repetition rate: count repeated n-grams
    words = text.split()
    if len(words) >= 4:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        repeated = sum(c - 1 for c in bigram_counts.values() if c > 1)
        signals["repetition_rate"] = repeated / max(total_bigrams, 1)
    else:
        signals["repetition_rate"] = 0.0

    # 2. Has mathematical content (numbers, operators, equations)
    math_patterns = re.findall(r'[\d+\-*/=]', text)
    signals["math_density"] = len(math_patterns) / max(len(text), 1)

    # 3. Progress: has any intermediate numbers appeared?
    numbers = NUM_RE.findall(text)
    signals["number_count"] = len(numbers)

    # 4. Has started forming an answer?
    signals["has_partial_answer"] = bool(
        FINAL_ANSWER_RE.search(text) or BOXED_RE.search(text) or
        re.search(r"(?:therefore|thus|so|hence|=)\s*\d", text, re.IGNORECASE)
    )

    # 5. Text length relative to tokens (short text per token = padding/repetition)
    signals["text_density"] = len(text.strip()) / max(len(words), 1)

    # 6. Is the model stuck in a loop? (last N lines same)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if len(lines) >= 4:
        last_4 = lines[-4:]
        unique = len(set(last_4))
        signals["line_diversity"] = unique / 4.0
    else:
        signals["line_diversity"] = 1.0

    return signals


def should_early_exit(signals, strategy="conservative"):
    """Decide if generation should be stopped early based on confidence signals.

    Strategies:
    - conservative: only stop on very clear stuck/loop patterns
    - moderate: stop on low confidence + no progress
    - aggressive: stop early on any sign of difficulty
    """
    if strategy == "conservative":
        # Only stop if model is clearly stuck (high repetition + no answer forming)
        return (signals["repetition_rate"] > 0.3 and
                not signals["has_partial_answer"] and
                signals["line_diversity"] < 0.5)

    elif strategy == "moderate":
        # Stop if repetitive OR no math progress
        stuck = signals["repetition_rate"] > 0.2 and signals["line_diversity"] < 0.75
        no_progress = signals["number_count"] < 2 and not signals["has_partial_answer"]
        return stuck or (no_progress and signals["repetition_rate"] > 0.15)

    elif strategy == "aggressive":
        # More aggressive pruning
        return (signals["repetition_rate"] > 0.15 or
                (signals["number_count"] < 3 and signals["math_density"] < 0.02) or
                signals["line_diversity"] < 0.5)

    return False


def load_model_and_tokenizer(model_id, device_map="auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device_map, trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def model_input_device(model):
    if hasattr(model, "device"):
        dev = model.device
        if isinstance(dev, torch.device):
            return dev
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for _, device in hf_map.items():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    return next(model.parameters()).device


def generate_with_checkpoints(model, tokenizer, prompt, max_budget, checkpoints, strategy,
                              enable_thinking=True):
    """Generate tokens with intermediate checkpoints for confidence gating.

    At each checkpoint, evaluate confidence signals. If low confidence detected,
    can exit early to save tokens.

    Returns: (text, tokens_used, exit_point, signals_history, hit_budget)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    # Sort checkpoints and add max_budget as final
    sorted_checkpoints = sorted(set(list(checkpoints) + [max_budget]))

    gen_kwargs = dict(
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_ids = inputs["input_ids"]
    total_generated = 0
    exit_point = "max_budget"
    signals_history = []
    early_exited = False

    start = time.perf_counter()

    prev_tokens = 0
    for cp in sorted_checkpoints:
        tokens_to_gen = cp - prev_tokens
        if tokens_to_gen <= 0:
            continue

        with torch.no_grad():
            out = model.generate(
                input_ids=all_ids,
                attention_mask=torch.ones_like(all_ids),
                max_new_tokens=tokens_to_gen,
                **gen_kwargs,
            )

        new_tokens = out[0][all_ids.shape[1]:]
        total_generated += int(new_tokens.shape[0])
        all_ids = out

        # Check if model naturally stopped (generated fewer tokens than requested)
        actually_generated = int(new_tokens.shape[0])
        naturally_stopped = actually_generated < int(tokens_to_gen * 0.95)

        if naturally_stopped:
            exit_point = f"natural_stop_{total_generated}"
            break

        # At checkpoint: compute confidence signals
        if cp < max_budget:
            gen_text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
            signals = compute_confidence_signals(gen_text)
            signals["checkpoint"] = cp
            signals["tokens_so_far"] = total_generated
            signals_history.append(signals)

            if should_early_exit(signals, strategy):
                exit_point = f"confidence_exit_{cp}"
                early_exited = True
                break

        prev_tokens = cp

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = out[0][in_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit_budget = total_generated >= int(max_budget * 0.95) and not early_exited

    return text, total_generated, exit_point, signals_history, hit_budget, elapsed, early_exited


def build_prompt(question, tokenizer, enable_thinking=True):
    system_text = (
        "You are a careful math solver. Solve the problem step by step briefly. "
        "End with a single line: Final answer: <number>."
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        chat_kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **chat_kwargs)
    except Exception:
        return f"{system_text}\n\nQuestion: {question}\nSolution:\n"


def load_gsm8k(n_samples, seed):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(idxs)
    selected = [ds[i] for i in idxs[:n_samples]]
    items = []
    for raw in selected:
        gold = get_gold_from_gsm8k(raw["answer"])
        items.append({"question": raw["question"], "gold": gold})
    return items


def main():
    parser = argparse.ArgumentParser(description="Confidence-Gated Early Exit")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--max_budget", type=int, default=512)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[128, 256, 384])
    parser.add_argument("--strategies", nargs="+", default=["conservative", "moderate", "aggressive"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/confidence_exit")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device_map)

    log.info(f"Loading {args.benchmark} (n={args.n_samples})...")
    items = load_gsm8k(args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    all_results = {}

    # First: run baseline (no early exit) to get reference
    log.info("Running baseline (no confidence gating)...")
    baseline_results = []
    correct_bl = 0
    total_tok_bl = 0

    for i, item in enumerate(items):
        prompt = build_prompt(item["question"], tokenizer, enable_thinking=True)
        text, tokens, exit_pt, signals_hist, hit_budget, latency, early_exited = \
            generate_with_checkpoints(model, tokenizer, prompt, args.max_budget,
                                     [], "conservative")  # empty checkpoints = no gating

        pred, has_final, pred_source = parse_prediction(text)
        c = is_correct(pred, item["gold"])
        if c:
            correct_bl += 1
        total_tok_bl += tokens

        baseline_results.append({
            "idx": i, "pred": pred, "correct": c, "tokens": tokens,
            "hit_budget": hit_budget, "exit_point": exit_pt,
            "has_final": has_final, "pred_source": pred_source,
        })

        if (i + 1) % 20 == 0 or i == len(items) - 1:
            log.info(f"  Baseline [{i+1}/{len(items)}] acc={correct_bl/(i+1):.3f}, "
                     f"avg_tok={total_tok_bl/(i+1):.0f}")

    n = len(items)
    all_results["baseline"] = {
        "accuracy": correct_bl / n,
        "avg_tokens": total_tok_bl / n,
        "per_sample": baseline_results,
    }
    log.info(f"  Baseline FINAL: acc={correct_bl/n:.3f}, avg_tok={total_tok_bl/n:.0f}")

    # Now run each strategy
    for strategy in args.strategies:
        log.info(f"Running strategy: {strategy}...")
        results = []
        correct = 0
        total_tok = 0
        early_exits = 0

        for i, item in enumerate(items):
            prompt = build_prompt(item["question"], tokenizer, enable_thinking=True)
            text, tokens, exit_pt, signals_hist, hit_budget, latency, early_exited = \
                generate_with_checkpoints(model, tokenizer, prompt, args.max_budget,
                                         args.checkpoints, strategy)

            pred, has_final, pred_source = parse_prediction(text)
            c = is_correct(pred, item["gold"])
            if c:
                correct += 1
            total_tok += tokens
            if early_exited:
                early_exits += 1

            # Compare with baseline
            bl_correct = baseline_results[i]["correct"]

            results.append({
                "idx": i, "pred": pred, "correct": c, "tokens": tokens,
                "hit_budget": hit_budget, "exit_point": exit_pt,
                "early_exited": early_exited, "has_final": has_final,
                "pred_source": pred_source,
                "baseline_correct": bl_correct,
                "signals": signals_hist,
            })

            if (i + 1) % 20 == 0 or i == len(items) - 1:
                log.info(f"  {strategy} [{i+1}/{len(items)}] acc={correct/(i+1):.3f}, "
                         f"avg_tok={total_tok/(i+1):.0f}, early_exit={early_exits/(i+1):.1%}")

        all_results[strategy] = {
            "accuracy": correct / n,
            "avg_tokens": total_tok / n,
            "early_exit_rate": early_exits / n,
            "per_sample": results,
        }

        # Analyze: how many samples saved by early exit were actually wrong in baseline?
        saved_correct = sum(1 for r in results if r["early_exited"] and r["baseline_correct"])
        saved_wrong = sum(1 for r in results if r["early_exited"] and not r["baseline_correct"])
        log.info(f"  {strategy} FINAL: acc={correct/n:.3f}, avg_tok={total_tok/n:.0f}, "
                 f"early_exit={early_exits/n:.1%}")
        log.info(f"  Early-exited samples: {early_exits} total, "
                 f"{saved_wrong} were wrong in baseline (good saves), "
                 f"{saved_correct} were correct (bad saves)")

    # Summary
    log.info("")
    log.info("=" * 80)
    log.info(f"{'Strategy':<15s} {'Accuracy':>10s} {'Avg Tok':>10s} {'Early Exit':>12s} {'Token Save':>12s}")
    log.info("-" * 80)
    bl_tok = all_results["baseline"]["avg_tokens"]
    for label in ["baseline"] + args.strategies:
        r = all_results[label]
        ee = r.get("early_exit_rate", 0)
        save = 1 - r["avg_tokens"] / bl_tok if bl_tok > 0 else 0
        log.info(f"{label:<15s} {r['accuracy']:>9.1%} {r['avg_tokens']:>10.0f} "
                 f"{ee:>11.1%} {save:>11.1%}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "meta": {
            "method": "confidence_gated_early_exit",
            "timestamp": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": len(items),
            "max_budget": args.max_budget,
            "checkpoints": args.checkpoints,
            "strategies": args.strategies,
            "seed": args.seed,
        },
        "results": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
            for k, v in all_results.items()
        },
        "per_sample": {k: v["per_sample"] for k, v in all_results.items()},
    }
    fname = f"cgee_{model_tag}_{args.benchmark}_{timestamp}.json"
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"Saved to {fpath}")


if __name__ == "__main__":
    main()
