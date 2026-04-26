#!/usr/bin/env python3
"""Evaluate CART transducer — V3 with frozen-prefix and arm_name.

Fixes from GPT-5.5:
- arm_name in output meta (not just mode)
- prefix_generator: base model generates prefix, LoRA only for answer (default)
- checkpoint hard-fail (no silent fallback)
- checkpoint_loaded logged per sample

Usage:
    python3 scripts/eval_cart_transducer.py \
        --model Qwen/Qwen3-8B --checkpoint checkpoints/cart/dev \
        --benchmark math500 --n_samples 50 --seed 42 \
        --trace_conditioned --b2_max 512 --arm_name cart_prefix_conditioned
"""
import argparse, hashlib, json, logging, os, random, re, sys, time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import parse_prediction_math, is_correct_math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_eval_data(benchmark, n, seed):
    from datasets import load_dataset
    if benchmark == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        return [{"q": ds[i]["question"],
                 "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", ""),
                 "idx": i} for i in idxs[:n]]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        idxs = list(range(len(ds)))
        random.seed(seed); random.shuffle(idxs)
        return [{"q": ds[i]["problem"], "gold": str(ds[i]["answer"]),
                 "idx": i} for i in idxs[:n]]


def generate_thinking_trace(model, tok, question, max_tokens):
    """Generate thinking trace using the given model (should be BASE for clean ablation)."""
    messages = [{"role": "system", "content": "You are a careful math solver. Think step by step."},
                {"role": "user", "content": question}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True), len(gen)


def transducer_generate(model, tok, question, prefix, max_answer_tokens, trace_conditioned):
    if trace_conditioned and prefix:
        clean = prefix
        for tag in ["<think>", "</think>"]:
            clean = clean.replace(tag, "")
        system = ("You are an expert mathematician. Given the partial reasoning below, "
                  "extract or compute the final answer.")
        user_text = f"Question: {question}\n\nPartial reasoning: {clean}\n\nFinal answer:"
    else:
        system = "You are an expert mathematician. Solve this problem."
        user_text = f"Question: {question}\n\nFinal answer:"

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_text}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)
    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_answer_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    return tok.decode(gen, skip_special_tokens=True), len(gen)


def parse_answer(text, benchmark):
    if benchmark == "math500":
        pred, _, src = parse_prediction_math(text)
        return pred, src
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
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trace_conditioned", action="store_true")
    p.add_argument("--b2_max", type=int, default=512)
    p.add_argument("--b_answer", type=int, default=128)
    p.add_argument("--arm_name", default=None,
                   help="V3: explicit arm name for ablation report")
    p.add_argument("--prefix_generator", default="base",
                   choices=["base", "answer_model"],
                   help="V3: who generates prefix. 'base'=frozen base model (clean)")
    p.add_argument("--output_dir", default="results/cart/eval")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    # Determine arm_name
    if args.arm_name:
        arm_name = args.arm_name
    elif args.checkpoint and args.trace_conditioned:
        arm_name = "cart_prefix_conditioned"
    elif args.checkpoint:
        arm_name = "cart_question_only"
    else:
        arm_name = "existing_fragment"

    # Determine mode for backward compat
    mode = "prefix_conditioned" if args.trace_conditioned else "question_only"

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # V3: Load base model first (always needed for prefix generation)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    base_model.eval()

    # V3: Load answer model (base + LoRA if checkpoint provided)
    if args.checkpoint:
        from peft import PeftModel
        log.info(f"Loading LoRA from {args.checkpoint}")
        answer_model = PeftModel.from_pretrained(base_model, args.checkpoint)
        trainable = sum(p.numel() for p in answer_model.parameters() if p.requires_grad)
        log.info(f"LoRA loaded: {trainable:,} trainable params")
        checkpoint_loaded = True
        answer_model.eval()
    else:
        answer_model = base_model
        checkpoint_loaded = False
        log.info("No checkpoint — using base model for answers")

    # V3: Decide prefix generator
    if args.prefix_generator == "base":
        prefix_model = base_model
        log.info("Prefix generator: BASE model (frozen, clean ablation)")
    else:
        prefix_model = answer_model
        log.info("Prefix generator: ANSWER model (LoRA-loaded, confounded)")

    items = load_eval_data(args.benchmark, args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} eval items, arm={arm_name}, prefix_gen={args.prefix_generator}")
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    results = []
    n_correct = 0
    total_tok = 0

    for i, item in enumerate(items):
        prefix = ""
        prefix_tok = 0
        if args.trace_conditioned:
            prefix, prefix_tok = generate_thinking_trace(
                prefix_model, tok, item["q"], args.b2_max)

        answer_text, answer_tok = transducer_generate(
            answer_model, tok, item["q"], prefix, args.b_answer, args.trace_conditioned)
        pred, src = parse_answer(answer_text, args.benchmark)
        correct = check_correct(pred, item["gold"], args.benchmark)
        if correct: n_correct += 1
        sample_tok = prefix_tok + answer_tok
        total_tok += sample_tok

        q_hash = hashlib.sha256(item["q"].encode()).hexdigest()[:16]
        results.append({
            "idx": item["idx"], "question_hash": q_hash,
            "gold": item["gold"], "pred": pred, "pred_source": src,
            "correct": correct,
            "prefix_tokens": prefix_tok, "answer_tokens": answer_tok,
            "total_tokens": sample_tok,
            "arm_name": arm_name,
            "trace_conditioned": args.trace_conditioned,
            "prefix_generator": args.prefix_generator,
            "checkpoint_requested": args.checkpoint,
            "checkpoint_loaded": checkpoint_loaded,
        })

        if (i + 1) % 10 == 0:
            acc = n_correct / (i + 1)
            log.info(f"[{i+1}/{len(items)}] acc={acc*100:.1f}% avg_tok={total_tok/(i+1):.0f}")

    n = len(results)
    summary = {
        "meta": {"model": args.model, "checkpoint": args.checkpoint,
                 "benchmark": args.benchmark, "n": n, "seed": args.seed,
                 "mode": mode, "arm_name": arm_name,
                 "prefix_generator": args.prefix_generator,
                 "checkpoint_loaded": checkpoint_loaded,
                 "b2_max": args.b2_max, "b_answer": args.b_answer,
                 "timestamp": ts, "schema_version": 3},
        "accuracy": n_correct / n,
        "avg_tokens": total_tok / n,
        "n_correct": n_correct,
        "per_sample": results,
    }

    fname = f"cart_{arm_name}_{args.benchmark}_{ts}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== CART Eval ({arm_name}, prefix_gen={args.prefix_generator}) ===")
    log.info(f"Accuracy: {n_correct}/{n} = {n_correct/n*100:.1f}%")
    log.info(f"Avg tokens: {total_tok/n:.0f}")


if __name__ == "__main__":
    main()
