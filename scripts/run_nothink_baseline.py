#!/usr/bin/env python3
"""Quick baseline: Non-thinking mode accuracy at different budgets.

Critical pilot experiment to determine if non-thinking probes can
serve as difficulty oracle for thinking-mode routing.

Usage:
  python scripts/run_nothink_baseline.py \
    --model Qwen/Qwen3-8B \
    --n_samples 200 \
    --budgets 32 64 128 256 512 \
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


def parse_prediction(text, strict_final_only=False):
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


def generate_once(model, tokenizer, prompt, max_new_tokens, temperature=0.0, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens, do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start
    gen_ids = out[0][in_len:]
    n_tokens = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit_budget = n_tokens >= int(max_new_tokens * 0.95)
    return text, n_tokens, elapsed, hit_budget


def build_prompt(question, tokenizer, enable_thinking):
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
    parser = argparse.ArgumentParser(description="Non-thinking mode baseline")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", default="gsm8k")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/speculation")
    parser.add_argument("--device_map", default="auto")
    # Also run thinking mode for comparison
    parser.add_argument("--also_thinking", action="store_true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.split("/")[-1].replace(".", "_")

    model, tokenizer = load_model_and_tokenizer(args.model, args.device_map)

    log.info(f"Loading {args.benchmark} (n={args.n_samples})...")
    items = load_gsm8k(args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    results = {}

    for enable_thinking in ([False, True] if args.also_thinking else [False]):
        mode = "thinking" if enable_thinking else "nothink"

        for budget in args.budgets:
            label = f"{mode}_{budget}"
            log.info(f"Running {label}...")

            correct = 0
            total_tokens = 0
            early_stops = 0
            has_finals = 0
            per_sample = []

            for i, item in enumerate(items):
                prompt = build_prompt(item["question"], tokenizer, enable_thinking=enable_thinking)
                text, tokens, latency, hit_budget = generate_once(
                    model, tokenizer, prompt, max_new_tokens=budget, temperature=0.0,
                )
                total_tokens += tokens
                if not hit_budget:
                    early_stops += 1

                pred, has_final, pred_source = parse_prediction(text, strict_final_only=False)
                if has_final:
                    has_finals += 1

                c = is_correct(pred, item["gold"])
                if c:
                    correct += 1

                per_sample.append({
                    "idx": i,
                    "pred": pred,
                    "correct": c,
                    "tokens": tokens,
                    "hit_budget": hit_budget,
                    "has_final": has_final,
                    "pred_source": pred_source,
                })

                if (i + 1) % 20 == 0 or i == len(items) - 1:
                    log.info(f"  [{i+1}/{len(items)}] {label}: acc={correct/(i+1):.3f}, "
                             f"avg_tok={total_tokens/(i+1):.0f}, early_stop={early_stops/(i+1):.1%}, "
                             f"has_final={has_finals/(i+1):.1%}")

            n = len(items)
            results[label] = {
                "accuracy": correct / n,
                "avg_tokens": total_tokens / n,
                "early_stop_rate": early_stops / n,
                "has_final_rate": has_finals / n,
                "per_sample": per_sample,
            }

            log.info(f"  {label} FINAL: acc={correct/n:.3f}, avg_tok={total_tokens/n:.0f}, "
                     f"early_stop={early_stops/n:.1%}, has_final={has_finals/n:.1%}")

    # Summary comparison
    log.info("")
    log.info("=" * 70)
    log.info(f"{'Config':<20s} {'Accuracy':>10s} {'Avg Tok':>10s} {'EarlyStop':>10s} {'HasFinal':>10s}")
    log.info("-" * 70)
    for label in sorted(results.keys()):
        r = results[label]
        log.info(f"{label:<20s} {r['accuracy']:>9.1%} {r['avg_tokens']:>10.0f} "
                 f"{r['early_stop_rate']:>9.1%} {r['has_final_rate']:>9.1%}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out = {
        "meta": {
            "method": "nothink_baseline",
            "timestamp": timestamp,
            "model": args.model,
            "benchmark": args.benchmark,
            "n_samples": len(items),
            "budgets": args.budgets,
            "seed": args.seed,
        },
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"} for k, v in results.items()},
        "per_sample": {k: v["per_sample"] for k, v in results.items()},
    }
    fname = f"nothink_baseline_{model_tag}_{args.benchmark}_{timestamp}.json"
    fpath = os.path.join(args.output_dir, fname)
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"Saved to {fpath}")


if __name__ == "__main__":
    main()
