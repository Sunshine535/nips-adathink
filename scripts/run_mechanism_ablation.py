#!/usr/bin/env python3
"""Mechanism ablation: free-continuation vs mode-switch extraction.

For samples that exhaust Stage-2 thinking budget, compare:
  (A) Free continuation: continue generating in THINKING mode for b_answer more tokens
  (B) Decoupled extraction: switch to NOTHINK mode with thinking prefix (our Stage-3)

Both use the same thinking prefix and same answer budget. Only the mode differs.
This directly tests whether mode-switching is the key mechanism.

Usage:
    python scripts/run_mechanism_ablation.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --n_samples 200 \
        --b1 512 --b2_max 4096 --b_answer 512 \
        --seed 42
"""
import argparse, json, logging, os, random, re, sys, time
from datetime import datetime, timezone
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import parse_prediction_math, is_correct_math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_benchmark(benchmark, n, seed):
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed)
        random.shuffle(idxs)
        return [{"q": ds[i]["question"],
                 "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")}
                for i in idxs[:n]]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = list(ds)
        random.seed(seed)
        random.shuffle(items)
        return [{"q": s["problem"], "gold": s["answer"]} for s in items[:n]]


def generate_nothink(model, tokenizer, question, budget, benchmark):
    messages = [
        {"role": "system", "content": "You are a careful math solver."},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)

    dev = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][in_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text, len(gen), len(gen) < budget


def generate_thinking(model, tokenizer, question, budget):
    messages = [
        {"role": "system", "content": "You are a careful math solver. Think step by step."},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=True)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)

    dev = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][in_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    natural_stop = len(gen) < budget
    return text, len(gen), natural_stop


def extract_free_continuation(model, tokenizer, question, thinking_prefix, answer_budget):
    """Variant A: continue generating in THINKING mode from truncated prefix."""
    messages = [
        {"role": "system", "content": "You are a careful math solver. Think step by step."},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True,
                                                enable_thinking=True)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)

    prompt = prompt + thinking_prefix

    dev = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=answer_budget, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][in_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text, len(gen)


def extract_mode_switch(model, tokenizer, question, thinking_prefix, answer_budget, benchmark):
    """Variant B: switch to NOTHINK mode with thinking prefix (our Stage-3)."""
    clean_trace = thinking_prefix
    for tag in ["<think>", "</think>", "<|think|>", "<|/think|>"]:
        clean_trace = clean_trace.replace(tag, "")
    clean_trace = clean_trace.strip()

    if benchmark == "math500":
        system_text = (
            "You are an expert mathematician. I have done most of the reasoning already. "
            "Your job is ONLY to extract or compute the final answer. "
            "Do not re-solve or re-explain. Output ONLY \\boxed{your_answer} and nothing else. "
            "Examples: \\boxed{42}, \\boxed{\\frac{3}{4}}, \\boxed{x^2+1}."
        )
    else:
        system_text = (
            "You are a careful math solver. Solve the problem step by step briefly. "
            "End with a single line: Final answer: <number>."
        )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question},
    ]
    try:
        base_prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=False)
    except TypeError:
        base_prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)

    if clean_trace:
        if benchmark == "math500":
            prompt = base_prompt + f"Based on my reasoning: {clean_trace}\n\nThe final answer is "
        else:
            prompt = base_prompt + f"Based on my reasoning: {clean_trace}\n\nFinal answer: "
    else:
        prompt = base_prompt

    dev = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=answer_budget, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    gen = out[0][in_len:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    return text, len(gen)


def parse_answer(text, benchmark):
    if benchmark == "math500":
        pred, _, src = parse_prediction_math(text)
        return pred, src
    else:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        return (nums[-1].replace(",", "") if nums else None), "regex"


def check_correct(pred, gold, benchmark):
    if benchmark == "math500":
        return is_correct_math(pred, gold)
    else:
        try:
            return pred is not None and abs(float(pred) - float(gold)) < 1e-4
        except (ValueError, TypeError):
            return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--b1", type=int, default=512)
    p.add_argument("--b2_max", type=int, default=4096)
    p.add_argument("--b_answer", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/mechanism_ablation")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    items = load_benchmark(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = []
    counts = {"stage1_pass": 0, "stage2_natural": 0, "stage2_budget": 0}
    correct_counts = {"town": 0, "free_cont": 0, "mode_switch": 0}

    for i, item in enumerate(items):
        r = {"idx": i, "gold": item["gold"]}

        # Stage 1: nothink triage
        s1_text, s1_tok, s1_natural = generate_nothink(
            model, tok, item["q"], args.b1, args.benchmark)

        if s1_natural:
            pred, src = parse_answer(s1_text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            r.update({"stage": 1, "pred_town": pred, "pred_fc": pred, "pred_ms": pred,
                      "correct_town": c, "correct_fc": c, "correct_ms": c,
                      "tokens_total": s1_tok})
            counts["stage1_pass"] += 1
            if c:
                correct_counts["town"] += 1
                correct_counts["free_cont"] += 1
                correct_counts["mode_switch"] += 1
            results.append(r)
            log.info(f"[{i+1}/{len(items)}] Stage 1 pass: correct={c}")
            continue

        # Stage 2: thinking
        s2_text, s2_tok, s2_natural = generate_thinking(
            model, tok, item["q"], args.b2_max)

        if s2_natural:
            pred, src = parse_answer(s2_text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            r.update({"stage": 2, "s2_natural": True,
                      "pred_town": pred, "pred_fc": pred, "pred_ms": pred,
                      "correct_town": c, "correct_fc": c, "correct_ms": c,
                      "tokens_total": args.b1 + s2_tok})
            counts["stage2_natural"] += 1
            if c:
                correct_counts["town"] += 1
                correct_counts["free_cont"] += 1
                correct_counts["mode_switch"] += 1
            results.append(r)
            log.info(f"[{i+1}/{len(items)}] Stage 2 natural stop: correct={c}")
            continue

        # Stage 2 budget exhausted — HERE IS THE ABLATION
        counts["stage2_budget"] += 1

        # TOWN baseline: just parse answer from truncated thinking
        pred_town, src_town = parse_answer(s2_text, args.benchmark)
        c_town = check_correct(pred_town, item["gold"], args.benchmark)
        if c_town:
            correct_counts["town"] += 1

        # Variant A: Free continuation (continue in thinking mode)
        fc_text, fc_tok = extract_free_continuation(
            model, tok, item["q"], s2_text, args.b_answer)
        full_fc = s2_text + fc_text
        pred_fc, src_fc = parse_answer(full_fc, args.benchmark)
        c_fc = check_correct(pred_fc, item["gold"], args.benchmark)
        if c_fc:
            correct_counts["free_cont"] += 1

        # Variant B: Mode-switch extraction (our Stage-3)
        ms_text, ms_tok = extract_mode_switch(
            model, tok, item["q"], s2_text, args.b_answer, args.benchmark)
        pred_ms, src_ms = parse_answer(ms_text, args.benchmark)
        c_ms = check_correct(pred_ms, item["gold"], args.benchmark)
        if c_ms:
            correct_counts["mode_switch"] += 1

        r.update({
            "stage": 3,
            "s2_natural": False,
            "s2_tokens": s2_tok,
            "pred_town": pred_town, "correct_town": c_town,
            "pred_fc": pred_fc, "correct_fc": c_fc, "fc_tokens": fc_tok,
            "pred_ms": pred_ms, "correct_ms": c_ms, "ms_tokens": ms_tok,
            "tokens_total_town": args.b1 + s2_tok,
            "tokens_total_fc": args.b1 + s2_tok + fc_tok,
            "tokens_total_ms": args.b1 + s2_tok + ms_tok,
        })
        results.append(r)
        log.info(f"[{i+1}/{len(items)}] Stage 2 budget: "
                 f"TOWN={c_town} FC={c_fc} MS={c_ms}")

        # Checkpoint every 25 samples
        if (i + 1) % 25 == 0:
            n = i + 1
            log.info(f"--- Checkpoint {n}/{len(items)} ---")
            log.info(f"  TOWN:        {correct_counts['town']}/{n} = {correct_counts['town']/n*100:.1f}%")
            log.info(f"  Free-cont:   {correct_counts['free_cont']}/{n} = {correct_counts['free_cont']/n*100:.1f}%")
            log.info(f"  Mode-switch: {correct_counts['mode_switch']}/{n} = {correct_counts['mode_switch']/n*100:.1f}%")

    # Final summary
    n = len(results)
    summary = {
        "meta": {
            "model": args.model, "benchmark": args.benchmark,
            "n_samples": n, "seed": args.seed,
            "b1": args.b1, "b2_max": args.b2_max, "b_answer": args.b_answer,
            "timestamp": ts,
        },
        "stage_distribution": counts,
        "accuracy": {
            "town": correct_counts["town"] / n,
            "free_continuation": correct_counts["free_cont"] / n,
            "mode_switch": correct_counts["mode_switch"] / n,
        },
        "per_sample": results,
    }

    # McNemar: free_cont vs mode_switch on stage-3 samples
    stage3 = [r for r in results if r["stage"] == 3]
    if stage3:
        both = sum(1 for r in stage3 if r["correct_fc"] and r["correct_ms"])
        fc_only = sum(1 for r in stage3 if r["correct_fc"] and not r["correct_ms"])
        ms_only = sum(1 for r in stage3 if not r["correct_fc"] and r["correct_ms"])
        neither = sum(1 for r in stage3 if not r["correct_fc"] and not r["correct_ms"])
        summary["mcnemar_fc_vs_ms"] = {
            "n_stage3": len(stage3),
            "both_correct": both, "fc_only": fc_only,
            "ms_only": ms_only, "neither": neither,
            "discordant": fc_only + ms_only,
        }

    fname = f"ablation_{args.benchmark}_{ts}.json"
    outpath = os.path.join(args.output_dir, fname)
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved to {outpath}")

    log.info(f"\n=== FINAL RESULTS (n={n}) ===")
    log.info(f"Stage distribution: {counts}")
    log.info(f"TOWN:          {correct_counts['town']}/{n} = {correct_counts['town']/n*100:.1f}%")
    log.info(f"Free-cont:     {correct_counts['free_cont']}/{n} = {correct_counts['free_cont']/n*100:.1f}%")
    log.info(f"Mode-switch:   {correct_counts['mode_switch']}/{n} = {correct_counts['mode_switch']/n*100:.1f}%")
    if stage3:
        mc = summary["mcnemar_fc_vs_ms"]
        log.info(f"McNemar (FC vs MS on stage-3): disc={mc['discordant']}, "
                 f"FC-only={mc['fc_only']}, MS-only={mc['ms_only']}")


if __name__ == "__main__":
    main()
