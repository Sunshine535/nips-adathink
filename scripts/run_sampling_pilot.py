#!/usr/bin/env python3
"""Sampling robustness pilot: Does the thinking tax survive non-greedy decoding?

Run think@256 and nothink@256 on GSM8K (n=200 subset) at temperature 0.7.
This tests whether the tax persists under stochastic decoding, not just greedy.
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
FINAL_RE = re.compile(
    r"(?:final answer\s*[::]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_last_number(text):
    m = NUM_RE.findall(text)
    return m[-1] if m else None


def extract_final_number(text):
    m = list(FINAL_RE.finditer(text))
    return m[-1].group(1) if m else None


def extract_boxed_number(text):
    for m in reversed(list(BOXED_RE.finditer(text))):
        v = extract_last_number(m.group(1))
        if v is not None:
            return v
    return None


def parse_prediction(text, mode):
    if mode == "think":
        think_end = text.rfind("</think>")
        search = text[think_end:] if think_end >= 0 else text
    else:
        search = text
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


def parse_gold(text):
    m = re.search(r"####\s*(.+)$", text.strip(), re.MULTILINE)
    if m:
        return m.group(1).strip().replace(",", "")
    return extract_last_number(text) or ""


def main():
    model_name = "Qwen/Qwen3-8B"
    cache_dir = "/workspace/hf_cache"
    budget = 256
    n_samples = 200
    temperature = 0.7
    seed = 42

    random.seed(seed)
    torch.manual_seed(seed)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = "/workspace/nips-adathink/results/sampling_pilot_%s" % ts
    os.makedirs(out_dir, exist_ok=True)

    log.info("Loading model: %s" % model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    log.info("Loading GSM8K test set...")
    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=cache_dir)
    items = []
    for row in ds:
        items.append({"question": row["question"], "gold": parse_gold(row["answer"])})

    random.shuffle(items)
    items = items[:n_samples]
    log.info("Using %d samples (shuffled, seed=%d)" % (len(items), seed))

    results = {}

    for mode in ["think", "nothink"]:
        log.info("\n=== %s@%d (temperature=%.1f) ===" % (mode, budget, temperature))
        correct_count = 0
        nat_stop_count = 0
        total_tokens = 0
        mode_results = []

        for i, item in enumerate(items):
            messages = [{"role": "user", "content": item["question"]}]

            try:
                if mode == "nothink":
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                else:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=True,
                    )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )

            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=budget,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )

            gen_ids = out[0][input_ids.shape[1]:]
            n_tokens = len(gen_ids)
            output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            nat_stop = n_tokens < budget
            if nat_stop:
                nat_stop_count += 1
            total_tokens += n_tokens

            pred, src = parse_prediction(output_text, mode)
            correct = is_correct(pred, item["gold"])
            if correct:
                correct_count += 1

            mode_results.append({
                "idx": i, "mode": mode, "budget": budget,
                "gold": item["gold"], "pred": pred, "src": src,
                "correct": correct, "n_tokens": n_tokens,
                "natural_stop": nat_stop,
            })

            if (i + 1) % 50 == 0 or (i + 1) == len(items):
                acc_so_far = correct_count / (i + 1)
                ns_rate = nat_stop_count / (i + 1)
                log.info(
                    "  [%d/%d] acc=%.1f%% nat_stop=%.1f%% avg_tok=%.0f",
                    i + 1, len(items), acc_so_far * 100, ns_rate * 100,
                    total_tokens / (i + 1),
                )

        n = len(items)
        acc = correct_count / n
        ns_rate = nat_stop_count / n
        log.info(
            "%s@%d FINAL: acc=%.1f%% nat_stop=%.1f%% avg_tok=%.0f",
            mode, budget, acc * 100, ns_rate * 100, total_tokens / n,
        )
        results[mode] = {
            "accuracy": round(acc, 4),
            "natural_stop_rate": round(ns_rate, 4),
            "avg_tokens": round(total_tokens / n, 1),
            "n": n,
        }

        # Save per-sample
        csv_path = "%s/per_sample_%s.csv" % (out_dir, mode)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mode_results[0].keys())
            writer.writeheader()
            writer.writerows(mode_results)

    # Summary
    summary = {
        "model": model_name,
        "benchmark": "gsm8k",
        "n_samples": n_samples,
        "budget": budget,
        "temperature": temperature,
        "seed": seed,
        "results": results,
        "tax_pp": round((results["nothink"]["accuracy"] - results["think"]["accuracy"]) * 100, 1),
    }

    with open("%s/summary.json" % out_dir, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n=== SUMMARY ===")
    log.info("think@%d (tau=%.1f): acc=%.1f%%" % (budget, temperature, results["think"]["accuracy"] * 100))
    log.info("nothink@%d (tau=%.1f): acc=%.1f%%" % (budget, temperature, results["nothink"]["accuracy"] * 100))
    log.info("Tax: %.1f pp" % summary["tax_pp"])
    log.info("Results saved to %s" % out_dir)


if __name__ == "__main__":
    main()
