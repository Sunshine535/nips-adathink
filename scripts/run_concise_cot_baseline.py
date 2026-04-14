#!/usr/bin/env python3
"""Concise-CoT baseline: thinking mode with concise instruction.
Tests whether prompting alone can reduce truncation waste.
Comparison: standard think@B vs concise-think@B vs nothink@B.
"""
import json, logging, os, random, re, time, torch
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

def extract_last_number(text):
    m = NUM_RE.findall(text)
    return m[-1] if m else None

def extract_boxed(text):
    for m in reversed(list(BOXED_RE.finditer(text))):
        v = extract_last_number(m.group(1))
        if v: return v
    return None

def parse_prediction(text, benchmark):
    think_end = text.rfind("</think>")
    search = text[think_end:] if think_end >= 0 else text
    if benchmark == "math500":
        b = extract_boxed(search)
        if b: return b, "boxed"
    fa = re.findall(r"(?:final answer\s*[::]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?)", search, re.I)
    if fa: return fa[-1], "final_marker"
    n = extract_last_number(search)
    if n: return n, "fallback"
    return None, "none"

def normalize_number(s):
    s = s.replace(",", "").strip()
    try:
        v = float(s)
        if not (v == v and abs(v) != float('inf')):
            return s
        return str(int(v)) if v == int(v) else str(v)
    except (ValueError, OverflowError):
        return s

def is_correct(pred, gold):
    if pred is None: return False
    return normalize_number(pred) == normalize_number(gold)

CONCISE_PROMPT_GSM8K = "Solve this math problem. Be very concise - skip unnecessary steps, show only key calculations. Write 'Final answer: <number>' at the end.\n\n"
CONCISE_PROMPT_MATH = "Solve this problem concisely. Skip verbose explanations, show only essential steps. Put your final answer in \\boxed{}.\n\n"
STANDARD_PROMPT_GSM8K = "Solve this math problem step by step. At the end, write 'Final answer: <number>'.\n\n"
STANDARD_PROMPT_MATH = "Solve the following math problem. Show your work and put your final answer in \\boxed{}.\n\n"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=0)
    parser.add_argument("--output_dir", default="results/concise_cot")
    parser.add_argument("--cache_dir", default="/root/.cache/huggingface/hub")
    args = parser.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    if args.benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"question": r["question"], "gold": r["answer"].split("####")[-1].strip()} for r in ds]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": r["problem"], "gold": str(r["answer"])} for r in ds]

    if args.n_samples > 0:
        items = random.Random(args.seed).sample(items, min(args.n_samples, len(items)))
    log.info("Loaded %d samples", len(items))

    log.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"{args.benchmark}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    modes = [
        ("concise_think", True),
        ("standard_think", True),
        ("nothink", False),
    ]

    summary = {"meta": {"model": args.model, "benchmark": args.benchmark, "n": len(items), "seed": args.seed, "budgets": args.budgets, "timestamp": ts}, "results": {}}

    for mode_name, thinking in modes:
        summary["results"][mode_name] = {}
        for budget in args.budgets:
            log.info("\n=== %s@%d ===", mode_name, budget)
            correct_count = 0; nat_stop = 0; total_tok = 0

            for i, item in enumerate(items):
                q = item["question"]
                if mode_name == "concise_think":
                    prefix = CONCISE_PROMPT_MATH if args.benchmark == "math500" else CONCISE_PROMPT_GSM8K
                else:
                    prefix = STANDARD_PROMPT_MATH if args.benchmark == "math500" else STANDARD_PROMPT_GSM8K

                msgs = [{"role": "user", "content": prefix + q}]
                prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

                with torch.no_grad():
                    out = model.generate(input_ids, max_new_tokens=budget, do_sample=False, temperature=None, top_p=None)
                new_tokens = out[0][input_ids.shape[1]:]
                n_tok = len(new_tokens)
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                pred, src = parse_prediction(text, args.benchmark)
                correct = is_correct(pred, item["gold"])
                if correct: correct_count += 1
                if n_tok < budget: nat_stop += 1
                total_tok += n_tok

                if (i+1) % 100 == 0 or (i+1) == len(items):
                    log.info("  [%d/%d] acc=%.1f%% avg_tok=%.0f nat_stop=%d", i+1, len(items), 100*correct_count/(i+1), total_tok/(i+1), nat_stop)

            acc = correct_count / len(items)
            summary["results"][mode_name][str(budget)] = {
                "accuracy": round(acc, 4),
                "n_correct": correct_count,
                "n_total": len(items),
                "avg_tokens": round(total_tok / len(items), 1),
                "natural_stop_rate": round(nat_stop / len(items), 4),
            }

    # Summary table
    log.info("\n=== CONCISE-CoT BASELINE SUMMARY ===")
    log.info("%8s %15s %15s %15s", "Budget", "Concise Think", "Standard Think", "Nothink")
    for b in args.budgets:
        bs = str(b)
        c = summary["results"]["concise_think"].get(bs, {}).get("accuracy", 0)
        s = summary["results"]["standard_think"].get(bs, {}).get("accuracy", 0)
        n = summary["results"]["nothink"].get(bs, {}).get("accuracy", 0)
        log.info("%8d %14.1f%% %14.1f%% %14.1f%%", b, 100*c, 100*s, 100*n)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved to %s", out_dir)

if __name__ == "__main__":
    main()
