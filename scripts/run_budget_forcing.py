#!/usr/bin/env python3
"""Budget forcing baseline (Muennighoff et al. 2025, s1-style).

Two variants:
    - "wait_extend": if reasoning naturally stops before budget, inject
      "Wait," to force continued thinking (extend chain).
    - "early_stop": if budget reached before natural stop, inject
      "Final answer:" to force immediate answer emission (truncate chain).

This competes with TOWN/IRIS by directly controlling chain length
without the triage + decoupled-extraction architecture.

Usage:
    python scripts/run_budget_forcing.py \\
        --model Qwen/Qwen3-8B --benchmark math500 --n_samples 200 \\
        --budget 2048 --variant early_stop --seed 42
"""
import argparse, json, logging, os, random, sys, time
from datetime import datetime, timezone
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import parse_prediction_math, is_correct_math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

WAIT_TOKEN = "\n\nWait,"
STOP_TOKEN = "\n\nFinal answer:"


def load_math500(n, seed):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    items = list(ds); random.seed(seed); random.shuffle(items)
    return [{"q": s["problem"], "gold": s["answer"]} for s in items[:n]]


def load_gsm8k(n, seed):
    import re
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(ds))); random.seed(seed); random.shuffle(idxs)
    items = []
    for i in idxs[:n]:
        raw = ds[i]
        ans = raw["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"q": raw["question"], "gold": ans})
    return items


def generate_with_forcing(model, tok, prompt, budget, variant):
    """Generate with budget forcing.

    early_stop: at budget, append STOP_TOKEN and generate ≤32 more tokens.
    wait_extend: if EOS before 0.5*budget, inject WAIT_TOKEN and continue.
    """
    dev = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt").to(dev)
    in_len = inputs["input_ids"].shape[1]

    # Stage 1: normal generation with full budget
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
    gen = out[0][in_len:]
    gen_len = len(gen)
    eos = gen_len < budget  # hit EOS before budget

    text = tok.decode(gen, skip_special_tokens=True)

    if variant == "early_stop" and not eos:
        # Budget exhausted, inject "Final answer:" and generate up to 32 more
        forced_prompt = prompt + text + STOP_TOKEN
        inputs2 = tok(forced_prompt, return_tensors="pt").to(dev)
        in_len2 = inputs2["input_ids"].shape[1]
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=32, do_sample=False,
                                  pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        text += STOP_TOKEN + tok.decode(out2[0][in_len2:], skip_special_tokens=True)

    elif variant == "wait_extend" and eos and gen_len < budget // 2:
        # Natural stop but short — try to extend thinking
        forced_prompt = prompt + text + WAIT_TOKEN
        inputs2 = tok(forced_prompt, return_tensors="pt").to(dev)
        in_len2 = inputs2["input_ids"].shape[1]
        remaining = budget - gen_len - 3
        with torch.no_grad():
            out2 = model.generate(**inputs2, max_new_tokens=remaining, do_sample=False,
                                  pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        text += WAIT_TOKEN + tok.decode(out2[0][in_len2:], skip_special_tokens=True)

    # Count ALL generated tokens including forced/injected
    total_generated = gen_len
    if variant == "early_stop" and not eos:
        forced_tokens = len(out2[0][in_len2:])
        total_generated = gen_len + forced_tokens
    elif variant == "wait_extend" and eos and gen_len < budget // 2:
        extended_tokens = len(out2[0][in_len2:])
        total_generated = gen_len + extended_tokens

    return text, total_generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--budget", type=int, default=2048)
    p.add_argument("--variant", default="early_stop",
                   choices=["early_stop", "wait_extend"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/budget_forcing")
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    log.info(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if args.benchmark == "math500":
        items = load_math500(args.n_samples, args.seed)
    else:
        items = load_gsm8k(args.n_samples, args.seed)
    log.info(f"Loaded {len(items)} items")

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    correct_count = 0
    total_tokens = 0
    t_start = time.time()

    for i, item in enumerate(items):
        # Build chat prompt with thinking mode enabled
        messages = [
            {"role": "system", "content": "You are a careful math solver."},
            {"role": "user", "content": item["q"]},
        ]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=True)
        except TypeError:
            prompt = tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)

        text, n_tok = generate_with_forcing(model, tok, prompt,
                                            args.budget, args.variant)

        if args.benchmark == "math500":
            pred, has_final, src = parse_prediction_math(text)
            c = 1 if is_correct_math(pred, item["gold"]) else 0
        else:
            # simple GSM8K parsing
            import re
            nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text.split("Final answer:")[-1] if "Final answer:" in text else text)
            pred = nums[-1].replace(",", "") if nums else None
            try:
                c = 1 if pred and abs(float(pred) - float(item["gold"])) < 1e-4 else 0
            except (ValueError, TypeError):
                c = 0
            src = "gsm8k_last"

        correct_count += c
        total_tokens += n_tok
        results.append({"idx": i, "gold": item["gold"], "pred": str(pred),
                        "correct": c, "tokens": n_tok, "pred_source": src})

        if (i+1) % 20 == 0:
            log.info(f"  [{i+1}/{len(items)}] acc={correct_count/(i+1):.3f} avg_tok={total_tokens/(i+1):.0f}")

    elapsed = time.time() - t_start
    accuracy = correct_count / len(items)
    avg_tokens = total_tokens / len(items)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = args.model.split("/")[-1].replace(".", "_")
    fname = f"bforce_{args.variant}_{tag}_{args.benchmark}_b{args.budget}_{ts}.json"
    out_path = os.path.join(args.output_dir, fname)

    summary = {
        "meta": {"model": args.model, "benchmark": args.benchmark,
                 "n_samples": len(items), "budget": args.budget,
                 "variant": args.variant, "seed": args.seed,
                 "elapsed_s": elapsed},
        "summary": {"accuracy": accuracy, "avg_tokens": avg_tokens,
                    "n_correct": correct_count},
        "per_sample": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info(f"Done. Acc={accuracy:.3f} avg_tok={avg_tokens:.0f} elapsed={elapsed:.0f}s")
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
