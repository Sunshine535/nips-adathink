#!/usr/bin/env python3
"""Multi-seed robustness check for headline comparisons.
Runs think and nothink at key budgets across 3 seeds to report variance.
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

def run_one(model, tokenizer, items, budget, thinking, benchmark, seed):
    """Run one configuration, return accuracy and stats."""
    torch.manual_seed(seed)
    correct = 0; nat_stop = 0; total_tok = 0

    prompt_prefix = ("Solve the following math problem. Show your work and put your final answer in \\boxed{}.\n\n"
                     if benchmark == "math500" else
                     "Solve this math problem step by step. At the end, write 'Final answer: <number>'.\n\n")

    for i, item in enumerate(items):
        msgs = [{"role": "user", "content": prompt_prefix + item["question"]}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=budget, do_sample=False, temperature=None, top_p=None)
        new_tokens = out[0][input_ids.shape[1]:]
        n_tok = len(new_tokens)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred, _ = parse_prediction(text, benchmark)
        if is_correct(pred, item["gold"]): correct += 1
        if n_tok < budget: nat_stop += 1
        total_tok += n_tok

        if (i+1) % 200 == 0 or (i+1) == len(items):
            log.info("  [%d/%d] acc=%.1f%%", i+1, len(items), 100*correct/(i+1))

    n = len(items)
    return {"accuracy": round(correct/n, 4), "n_correct": correct, "n_total": n,
            "avg_tokens": round(total_tok/n, 1), "natural_stop_rate": round(nat_stop/n, 4)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--budgets", type=int, nargs="+", default=[512])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--output_dir", default="results/multiseed")
    parser.add_argument("--cache_dir", default="/root/.cache/huggingface/hub")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.benchmark == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = [{"question": r["question"], "gold": r["answer"].split("####")[-1].strip()} for r in ds]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": r["problem"], "gold": str(r["answer"])} for r in ds]
    log.info("Loaded %d samples", len(items))

    log.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"{args.benchmark}_gpu{args.gpu}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for budget in args.budgets:
        for mode_name, thinking in [("think", True), ("nothink", False)]:
            for seed in args.seeds:
                key = f"{mode_name}@{budget}_seed{seed}"
                log.info("\n=== %s ===", key)
                r = run_one(model, tokenizer, items, budget, thinking, args.benchmark, seed)
                results[key] = r
                log.info("  Result: acc=%.1f%% avg_tok=%.0f nat_stop=%.1f%%",
                         100*r["accuracy"], r["avg_tokens"], 100*r["natural_stop_rate"])

    # Summary
    log.info("\n=== MULTI-SEED ROBUSTNESS SUMMARY ===")
    for budget in args.budgets:
        for mode in ["think", "nothink"]:
            accs = [results[f"{mode}@{budget}_seed{s}"]["accuracy"] for s in args.seeds]
            mean_acc = sum(accs)/len(accs)
            std_acc = (sum((a-mean_acc)**2 for a in accs)/len(accs))**0.5
            log.info("%s@%d: %.1f%% +/- %.1f%% (seeds: %s)",
                     mode, budget, 100*mean_acc, 100*std_acc,
                     ", ".join(f"{100*a:.1f}%" for a in accs))

    summary = {"meta": {"model": args.model, "benchmark": args.benchmark, "seeds": args.seeds,
                        "budgets": args.budgets, "gpu": args.gpu, "timestamp": ts},
               "results": results}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved to %s", out_dir)

if __name__ == "__main__":
    main()
