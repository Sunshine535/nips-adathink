#!/usr/bin/env python3
"""Full-scale DeepSeek-R1-Distill-Llama-8B thinking-mode experiment on GSM8K.
Demonstrates the same truncation-driven collapse as Qwen family.
"""
import csv
import json
import logging
import os
import random
import re
import time
import torch
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[::]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_last_number(text):
    m = NUM_RE.findall(text)
    return m[-1] if m else None


def extract_final_number(text):
    m = list(FINAL_ANSWER_RE.finditer(text))
    return m[-1].group(1) if m else None


def extract_boxed_number(text):
    for m in reversed(list(BOXED_RE.finditer(text))):
        v = extract_last_number(m.group(1))
        if v is not None:
            return v
    return None


def parse_prediction(text):
    think_end = text.rfind("</think>")
    search = text[think_end:] if think_end >= 0 else text
    f = extract_final_number(search)
    if f:
        return f, "final_marker"
    b = extract_boxed_number(search)
    if b:
        return b, "boxed"
    t = extract_last_number(search)
    if t:
        return t, "fallback_last"
    return None, "none"


def normalize_number(s):
    s = s.replace(",", "").strip()
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def is_correct(pred, gold):
    if pred is None:
        return False
    return normalize_number(pred) == normalize_number(gold)


def main():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    cache_dir = "/workspace/hf_cache"
    budgets = [256, 512, 1024, 2048]
    seed = 42

    random.seed(seed)
    torch.manual_seed(seed)

    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds:
        q = row["question"]
        a = row["answer"]
        gold = a.split("####")[-1].strip() if "####" in a else extract_last_number(a)
        items.append({"question": q, "gold": str(gold)})
    log.info("Loaded %d GSM8K samples", len(items))

    # Load model
    log.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    log.info("Model loaded.")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = "results/deepseek_crossmodel/deepseek_gsm8k_full_%s" % ts
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    summary = {
        "meta": {
            "model": model_name,
            "benchmark": "gsm8k",
            "n": len(items),
            "seed": seed,
            "budgets": budgets,
            "timestamp": ts,
        },
        "per_budget": {},
    }

    for budget in budgets:
        log.info("\n=== think@%d ===", budget)
        correct_count = 0
        total_tokens = 0
        natural_stop_count = 0
        natural_stop_correct = 0
        hit_budget_count = 0
        hit_budget_correct = 0
        has_final = 0

        for i, item in enumerate(items):
            q = item["question"]
            prompt_text = (
                "Solve this math problem step by step. "
                "At the end, write 'Final answer: <number>'.\n\n" + q
            )
            msgs = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
                model.device
            )

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=budget,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            latency = time.time() - t0

            new_tokens = out[0][input_ids.shape[1] :]
            n_tokens = len(new_tokens)
            clean = tokenizer.decode(new_tokens, skip_special_tokens=True)

            hit_b = n_tokens >= budget
            nat_stop = not hit_b

            pred, src = parse_prediction(clean)
            correct = is_correct(pred, item["gold"])

            if correct:
                correct_count += 1
            total_tokens += n_tokens
            if nat_stop:
                natural_stop_count += 1
                if correct:
                    natural_stop_correct += 1
            else:
                hit_budget_count += 1
                if correct:
                    hit_budget_correct += 1
            if src == "final_marker":
                has_final += 1

            all_results.append(
                {
                    "idx": i,
                    "budget": budget,
                    "gold": item["gold"],
                    "pred": pred,
                    "src": src,
                    "correct": correct,
                    "n_tokens": n_tokens,
                    "natural_stop": nat_stop,
                    "latency": round(latency, 3),
                }
            )

            if (i + 1) % 100 == 0 or (i + 1) == len(items):
                acc_so_far = correct_count / (i + 1)
                avg_tok = total_tokens / (i + 1)
                log.info(
                    "  [%d/%d] acc=%.1f%% avg_tok=%.0f nat_stop=%d/%d",
                    i + 1,
                    len(items),
                    acc_so_far * 100,
                    avg_tok,
                    natural_stop_count,
                    i + 1,
                )

        n = len(items)
        acc = correct_count / n
        ns_rate = natural_stop_count / n
        ns_acc = (
            natural_stop_correct / natural_stop_count
            if natural_stop_count > 0
            else 0
        )
        hb_acc = (
            hit_budget_correct / hit_budget_count if hit_budget_count > 0 else 0
        )

        summary["per_budget"][str(budget)] = {
            "accuracy": round(acc, 4),
            "avg_tokens": round(total_tokens / n, 1),
            "natural_stop_rate": round(ns_rate, 4),
            "natural_stop_accuracy": (
                round(ns_acc, 4) if natural_stop_count > 0 else None
            ),
            "hit_budget_accuracy": (
                round(hb_acc, 4) if hit_budget_count > 0 else None
            ),
            "has_final_rate": round(has_final / n, 4),
        }
        log.info(
            "think@%d FINAL: acc=%.1f%% avg_tok=%.0f nat_stop=%.1f%% ns_acc=%.1f%%",
            budget,
            acc * 100,
            total_tokens / n,
            ns_rate * 100,
            ns_acc * 100,
        )

    # Save
    with open("%s/summary.json" % out_dir, "w") as f:
        json.dump(summary, f, indent=2)
    with open("%s/per_sample.csv" % out_dir, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    log.info("\nResults saved to %s", out_dir)
    log.info("\n=== SUMMARY ===")
    for b in budgets:
        s = summary["per_budget"][str(b)]
        log.info(
            "think@%d: acc=%.1f%% nat_stop=%.1f%%",
            b,
            s["accuracy"] * 100,
            s["natural_stop_rate"] * 100,
        )


if __name__ == "__main__":
    main()
