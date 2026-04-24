#!/usr/bin/env python3
"""Evaluate Elastic Reasoning (E1) model as trained baseline.

E1 protocol: generate thinking with budget → force </think> → generate solution.
Uses HuggingFace transformers (no vLLM needed).

Usage:
    python scripts/run_elastic_reasoning_eval.py \
        --model Salesforce/E1-Math-7B \
        --benchmark gsm8k --n_samples 200 \
        --thinking_budget 512 --solution_budget 512 --seed 42
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
        random.seed(seed); random.shuffle(idxs)
        return [{"q": ds[i]["question"],
                 "gold": ds[i]["answer"].split("####")[-1].strip().replace(",", "")}
                for i in idxs[:n]]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = list(ds)
        random.seed(seed); random.shuffle(items)
        return [{"q": s["problem"], "gold": s["answer"]} for s in items[:n]]


def e1_generate(model, tok, question, thinking_budget, solution_budget):
    """E1 two-phase generation: thinking (budgeted) → forced </think> → solution."""
    messages = [{"role": "user", "content": question}]
    try:
        prompt = tok.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    except Exception:
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    dev = model_input_device(model)

    # Phase 1: generate thinking with budget
    # E1 models start with <think> token automatically
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    # Find </think> token id
    think_end_candidates = ["</think>", "<|/think|>", "</think>\n"]
    think_end_id = None
    for candidate in think_end_candidates:
        ids = tok.encode(candidate, add_special_tokens=False)
        if len(ids) == 1:
            think_end_id = ids[0]
            break
    if think_end_id is None:
        think_end_id = tok.encode("</think>", add_special_tokens=False)[-1]

    start = time.perf_counter()

    # Phase 1: thinking generation (stop at </think> or budget)
    with torch.no_grad():
        out1 = model.generate(
            **inputs, max_new_tokens=thinking_budget, do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=[tok.eos_token_id, think_end_id],
        )

    thinking_ids = out1[0][in_len:]
    thinking_text = tok.decode(thinking_ids, skip_special_tokens=False)
    thinking_tokens = len(thinking_ids)
    natural_think_stop = thinking_tokens < thinking_budget

    # Phase 2: force </think> and generate solution
    if not natural_think_stop:
        # Append </think>\n\n to force solution phase
        think_end_tokens = tok.encode("</think>\n\n", add_special_tokens=False)
        phase2_input = torch.cat([out1[0], torch.tensor(think_end_tokens, device=dev)])
    else:
        # Thinking completed naturally, might already have </think>
        if "</think>" not in thinking_text:
            think_end_tokens = tok.encode("</think>\n\n", add_special_tokens=False)
            phase2_input = torch.cat([out1[0], torch.tensor(think_end_tokens, device=dev)])
        else:
            phase2_input = out1[0]

    phase2_input = phase2_input.unsqueeze(0)
    p2_len = phase2_input.shape[1]

    with torch.no_grad():
        out2 = model.generate(
            input_ids=phase2_input, max_new_tokens=solution_budget,
            do_sample=False, pad_token_id=tok.eos_token_id,
        )

    solution_ids = out2[0][p2_len:]
    solution_text = tok.decode(solution_ids, skip_special_tokens=True)
    solution_tokens = len(solution_ids)

    elapsed = time.perf_counter() - start
    total_tokens = thinking_tokens + solution_tokens

    return {
        "thinking_text": thinking_text,
        "solution_text": solution_text,
        "thinking_tokens": thinking_tokens,
        "solution_tokens": solution_tokens,
        "total_tokens": total_tokens,
        "natural_think_stop": natural_think_stop,
        "elapsed": round(elapsed, 2),
    }


def parse_answer(text, benchmark):
    if benchmark == "math500":
        pred, _, _ = parse_prediction_math(text)
        return pred
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def check_correct(pred, gold, benchmark):
    if benchmark == "math500":
        return is_correct_math(pred, gold)
    try:
        return pred is not None and abs(float(pred) - float(gold)) < 1e-4
    except (ValueError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Salesforce/E1-Math-7B")
    p.add_argument("--benchmark", default="gsm8k", choices=["math500", "gsm8k"])
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--thinking_budget", type=int, default=512)
    p.add_argument("--solution_budget", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="results/elastic_reasoning")
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
    n_correct = 0
    total_tokens = 0

    for i, item in enumerate(items):
        gen = e1_generate(model, tok, item["q"], args.thinking_budget, args.solution_budget)
        pred = parse_answer(gen["solution_text"], args.benchmark)
        c = check_correct(pred, item["gold"], args.benchmark)
        if c: n_correct += 1
        total_tokens += gen["total_tokens"]

        results.append({
            "idx": i, "gold": item["gold"], "pred": pred, "correct": c,
            "thinking_tokens": gen["thinking_tokens"],
            "solution_tokens": gen["solution_tokens"],
            "total_tokens": gen["total_tokens"],
            "natural_think_stop": gen["natural_think_stop"],
            "elapsed": gen["elapsed"],
        })

        if (i + 1) % 20 == 0:
            acc = n_correct / (i + 1)
            avg_tok = total_tokens / (i + 1)
            log.info(f"[{i+1}/{len(items)}] acc={acc*100:.1f}% avg_tok={avg_tok:.0f}")

    n = len(results)
    summary = {
        "meta": {"model": args.model, "benchmark": args.benchmark,
                 "n": n, "seed": args.seed, "ts": ts,
                 "thinking_budget": args.thinking_budget,
                 "solution_budget": args.solution_budget},
        "accuracy": n_correct / n,
        "avg_tokens": total_tokens / n,
        "n_correct": n_correct,
        "per_sample": results,
    }

    fname = f"e1_{args.benchmark}_{ts}.json"
    with open(os.path.join(args.output_dir, fname), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n=== E1 Evaluation ({args.model}) ===")
    log.info(f"Benchmark: {args.benchmark}, n={n}")
    log.info(f"Accuracy: {n_correct}/{n} = {n_correct/n*100:.1f}%")
    log.info(f"Avg tokens: {total_tokens/n:.0f} (think={args.thinking_budget}, sol={args.solution_budget})")


if __name__ == "__main__":
    main()
