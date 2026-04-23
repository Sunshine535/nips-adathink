#!/usr/bin/env python3
"""Pure mode-switch ablation: SAME prompt, ONLY enable_thinking differs.

For Stage-2-exhausted samples, both variants get:
  - Same system prompt
  - Same question
  - Same thinking prefix as context
  - Same answer scaffold ("Based on my reasoning, the answer is")
  - Same answer budget

Only difference: enable_thinking=True vs False in chat template.
This isolates the mode-switch effect from prompt engineering.

Usage:
    python scripts/run_pure_mode_ablation.py \
        --model Qwen/Qwen3-8B --benchmark gsm8k \
        --n_samples 200 --b1 256 --b2_max 512 --b_answer 128 --seed 42
"""
import argparse, json, logging, os, random, re, sys, time
from datetime import datetime, timezone

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


def generate_nothink(model, tok, question, budget):
    messages = [
        {"role": "system", "content": "You are a careful math solver."},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    dev = model_input_device(model)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True), len(gen), len(gen) < budget


def generate_thinking(model, tok, question, budget):
    messages = [
        {"role": "system", "content": "You are a careful math solver."},
        {"role": "user", "content": question},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=True)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    dev = model_input_device(model)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True), len(gen), len(gen) < budget


def extract_with_mode(model, tok, question, thinking_prefix, answer_budget,
                      benchmark, enable_thinking):
    """Extract answer with IDENTICAL prompt, only enable_thinking differs."""
    clean_trace = thinking_prefix
    for tag in ["<think>", "</think>", "<|think|>", "<|/think|>"]:
        clean_trace = clean_trace.replace(tag, "")
    clean_trace = clean_trace.strip()

    system_text = "You are a careful math solver."
    if benchmark == "math500":
        user_text = question + "\n\nHere is my partial reasoning:\n" + clean_trace + "\n\nPlease give the final answer."
    else:
        user_text = question + "\n\nHere is my partial reasoning:\n" + clean_trace + "\n\nWhat is the final numerical answer?"

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=enable_thinking)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)

    dev = model_input_device(model)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=answer_budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    text = tok.decode(gen, skip_special_tokens=True)
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
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="gsm8k", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--b1", type=int, default=256)
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/pure_mode_ablation")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    items = load_benchmark(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = []
    counts = {"stage1_pass": 0, "stage2_natural": 0, "stage2_budget": 0}
    correct = {"think_extract": 0, "nothink_extract": 0, "town": 0}

    for i, item in enumerate(items):
        r = {"idx": i, "gold": item["gold"]}

        s1_text, s1_tok, s1_natural = generate_nothink(model, tok, item["q"], args.b1)
        if s1_natural:
            pred, _ = parse_answer(s1_text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            r.update({"stage": 1, "correct_think": c, "correct_nothink": c,
                      "correct_town": c})
            counts["stage1_pass"] += 1
            if c:
                correct["think_extract"] += 1
                correct["nothink_extract"] += 1
                correct["town"] += 1
            results.append(r)
            log.info(f"[{i+1}/{len(items)}] S1 pass: {c}")
            continue

        s2_text, s2_tok, s2_natural = generate_thinking(model, tok, item["q"], args.b2_max)
        if s2_natural:
            pred, _ = parse_answer(s2_text, args.benchmark)
            c = check_correct(pred, item["gold"], args.benchmark)
            r.update({"stage": 2, "correct_think": c, "correct_nothink": c,
                      "correct_town": c})
            counts["stage2_natural"] += 1
            if c:
                correct["think_extract"] += 1
                correct["nothink_extract"] += 1
                correct["town"] += 1
            results.append(r)
            log.info(f"[{i+1}/{len(items)}] S2 natural: {c}")
            continue

        counts["stage2_budget"] += 1

        pred_town, _ = parse_answer(s2_text, args.benchmark)
        c_town = check_correct(pred_town, item["gold"], args.benchmark)
        if c_town: correct["town"] += 1

        # PURE ABLATION: same prompt, only enable_thinking differs
        think_text, think_tok = extract_with_mode(
            model, tok, item["q"], s2_text, args.b_answer,
            args.benchmark, enable_thinking=True)
        pred_think, _ = parse_answer(think_text, args.benchmark)
        c_think = check_correct(pred_think, item["gold"], args.benchmark)
        if c_think: correct["think_extract"] += 1

        nothink_text, nothink_tok = extract_with_mode(
            model, tok, item["q"], s2_text, args.b_answer,
            args.benchmark, enable_thinking=False)
        pred_nothink, _ = parse_answer(nothink_text, args.benchmark)
        c_nothink = check_correct(pred_nothink, item["gold"], args.benchmark)
        if c_nothink: correct["nothink_extract"] += 1

        r.update({"stage": 3, "correct_town": c_town,
                  "correct_think": c_think, "correct_nothink": c_nothink,
                  "think_tokens": think_tok, "nothink_tokens": nothink_tok})
        results.append(r)
        log.info(f"[{i+1}/{len(items)}] S3: town={c_town} think={c_think} nothink={c_nothink}")

        if (i + 1) % 25 == 0:
            n = i + 1
            log.info(f"--- Checkpoint {n} ---")
            for k, v in correct.items():
                log.info(f"  {k}: {v}/{n} = {v/n*100:.1f}%")

    n = len(results)
    stage3 = [r for r in results if r["stage"] == 3]
    mcnemar = {}
    if stage3:
        both = sum(1 for r in stage3 if r["correct_think"] and r["correct_nothink"])
        think_only = sum(1 for r in stage3 if r["correct_think"] and not r["correct_nothink"])
        nothink_only = sum(1 for r in stage3 if not r["correct_think"] and r["correct_nothink"])
        neither = sum(1 for r in stage3 if not r["correct_think"] and not r["correct_nothink"])
        mcnemar = {"n_stage3": len(stage3), "both": both, "think_only": think_only,
                   "nothink_only": nothink_only, "neither": neither,
                   "discordant": think_only + nothink_only}

    summary = {
        "meta": {"model": args.model, "benchmark": args.benchmark,
                 "n_samples": n, "seed": args.seed, "timestamp": ts,
                 "b1": args.b1, "b2_max": args.b2_max, "b_answer": args.b_answer,
                 "ablation_type": "pure_mode_only"},
        "stages": counts,
        "accuracy": {
            "town": correct["town"] / n,
            "think_extract": correct["think_extract"] / n,
            "nothink_extract": correct["nothink_extract"] / n,
        },
        "mcnemar_think_vs_nothink": mcnemar,
        "per_sample": results,
    }

    fname = f"pure_ablation_{args.benchmark}_{ts}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== PURE MODE ABLATION (n={n}) ===")
    log.info(f"Stages: {counts}")
    log.info(f"TOWN:            {correct['town']}/{n} = {correct['town']/n*100:.1f}%")
    log.info(f"Think-extract:   {correct['think_extract']}/{n} = {correct['think_extract']/n*100:.1f}%")
    log.info(f"Nothink-extract: {correct['nothink_extract']}/{n} = {correct['nothink_extract']/n*100:.1f}%")
    if mcnemar:
        log.info(f"McNemar (think vs nothink on S3): disc={mcnemar['discordant']}, "
                 f"think_only={mcnemar['think_only']}, nothink_only={mcnemar['nothink_only']}")


if __name__ == "__main__":
    main()
