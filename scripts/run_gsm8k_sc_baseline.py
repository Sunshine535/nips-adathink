#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def extract_number(text: str) -> Optional[str]:
    if not text:
        return None
    for marker in ["Final answer:", "final answer:", "The answer is", "the answer is"]:
        idx = text.rfind(marker)
        if idx != -1:
            tail = text[idx + len(marker) :]
            match = NUM_RE.search(tail)
            if match:
                return match.group(0)
    matches = NUM_RE.findall(text)
    if matches:
        return matches[-1]
    return None


def to_float(num_str: Optional[str]) -> Optional[float]:
    if num_str is None:
        return None
    s = num_str.replace(",", "").strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                denom = float(parts[1])
                if denom == 0:
                    return None
                return float(parts[0]) / denom
            except ValueError:
                return None
    try:
        return float(s)
    except ValueError:
        return None


def get_gold_from_gsm8k(answer_field: str) -> Optional[str]:
    if "####" in answer_field:
        after = answer_field.split("####")[-1]
        match = NUM_RE.search(after)
        if match:
            return match.group(0)
    return extract_number(answer_field)


def is_correct(pred: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    p = to_float(pred)
    g = to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


def build_prompt(
    question: str,
    tokenizer=None,
    prompt_format: str = "chat",
    direct_answer: bool = False,
    enable_thinking: Optional[bool] = False,
) -> str:
    if direct_answer:
        system_text = (
            "You are a careful math solver. "
            "Return only one line in this exact format: Final answer: <number>."
        )
    else:
        system_text = (
            "You are a careful math solver. Solve the problem step by step briefly. "
            "End with a single line: Final answer: <number>."
        )

    if prompt_format == "chat" and tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
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
            pass
    return f"{system_text}\n\nQuestion: {question}\nSolution:\n"


def get_rank_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def maybe_init_distributed(world_size: int, local_rank: int) -> bool:
    if world_size <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("WORLD_SIZE>1 detected but CUDA is unavailable.")
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=60),
            device_id=local_rank,
        )
    except TypeError:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    return True


def maybe_cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def model_input_device(model) -> torch.device:
    if hasattr(model, "device"):
        dev = model.device
        if isinstance(dev, torch.device):
            return dev
    return next(model.parameters()).device


def check_local_model_snapshot(model_id: str) -> Tuple[str, List[str]]:
    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(repo_id=model_id, local_files_only=True)
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    single_path = os.path.join(snapshot_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        required = sorted(set(index_data.get("weight_map", {}).values()))
        missing = [name for name in required if not os.path.exists(os.path.join(snapshot_dir, name))]
        return snapshot_dir, missing
    if os.path.exists(single_path):
        return snapshot_dir, []
    return snapshot_dir, ["model.safetensors.index.json (or model.safetensors)"]


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, int, float]:
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = out[0][in_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, int(gen_ids.shape[0]), elapsed


def majority_vote(preds: List[Optional[str]]) -> Optional[str]:
    vals = [p for p in preds if p is not None]
    if not vals:
        return None
    c = Counter(vals)
    return c.most_common(1)[0][0]


def summarize(records: List[Dict], key: str) -> Dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_latency_s": 0.0}
    n = len(records)
    return {
        "accuracy": sum(r[f"{key}_correct"] for r in records) / n,
        "avg_tokens": sum(r[f"{key}_tokens"] for r in records) / n,
        "avg_latency_s": sum(r[f"{key}_latency_s"] for r in records) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-consistency baseline on GSM8K.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--data_seed", type=int, default=303)
    parser.add_argument("--greedy_budget", type=int, default=256)
    parser.add_argument("--sc_budget", type=int, default=64)
    parser.add_argument("--sc_n", type=int, default=4)
    parser.add_argument("--sc_temperature", type=float, default=0.7)
    parser.add_argument("--prompt_format", choices=["plain", "chat"], default="chat")
    parser.add_argument("--direct_answer", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--results_dir", type=str, default="methods/01_adathink/results")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--skip_local_model_check", action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size = get_rank_info()
    distributed = False
    try:
        distributed = maybe_init_distributed(world_size, local_rank)

        def print0(msg: str) -> None:
            if rank == 0:
                print(msg, flush=True)

        use_cuda = torch.cuda.is_available()
        if not use_cuda and not args.allow_cpu:
            raise RuntimeError("CUDA required but unavailable. Pass --allow_cpu to override.")

        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed + rank)

        if not args.skip_local_model_check:
            snapshot_dir, missing = check_local_model_snapshot(args.model)
            if missing:
                raise RuntimeError(
                    f"Local model snapshot incomplete. model={args.model}, snapshot={snapshot_dir}, missing={missing}"
                )

        os.makedirs(args.results_dir, exist_ok=True)

        print0(f"Loading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        dtype = torch.bfloat16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=dtype)
        if use_cuda and distributed:
            model.to(torch.device(f"cuda:{local_rank}"))
        elif use_cuda:
            model.to(torch.device("cuda"))
        else:
            model.to(torch.device("cpu"))
        model.eval()

        ds = load_dataset("gsm8k", "main", split=args.split)
        ds = ds.shuffle(seed=args.data_seed)
        n = min(args.n_samples, len(ds))
        ds = ds.select(range(n))

        local_target = sum(1 for i in range(n) if (i % world_size) == rank)
        done = 0
        records = []
        for i, ex in enumerate(ds):
            if (i % world_size) != rank:
                continue
            q = ex["question"]
            gold = get_gold_from_gsm8k(ex["answer"])
            prompt = build_prompt(
                q,
                tokenizer=tokenizer,
                prompt_format=args.prompt_format,
                direct_answer=args.direct_answer,
                enable_thinking=True if args.enable_thinking else False,
            )

            greedy_text, greedy_tok, greedy_lat = generate_once(
                model, tokenizer, prompt, args.greedy_budget, temperature=0.0
            )
            greedy_pred = extract_number(greedy_text)

            sc_preds: List[Optional[str]] = []
            sc_tokens = 0
            sc_latency = 0.0
            for _ in range(args.sc_n):
                text, tok, lat = generate_once(
                    model,
                    tokenizer,
                    prompt,
                    args.sc_budget,
                    temperature=args.sc_temperature,
                )
                sc_preds.append(extract_number(text))
                sc_tokens += tok
                sc_latency += lat
            sc_pred = majority_vote(sc_preds)

            row = {
                "idx": i,
                "gold": gold,
                "greedy_pred": greedy_pred,
                "greedy_correct": int(is_correct(greedy_pred, gold)),
                "greedy_tokens": greedy_tok,
                "greedy_latency_s": greedy_lat,
                "sc_pred": sc_pred,
                "sc_correct": int(is_correct(sc_pred, gold)),
                "sc_tokens": sc_tokens,
                "sc_latency_s": sc_latency,
            }
            records.append(row)
            done += 1
            if done % 5 == 0 or done == local_target:
                print(f"[rank {rank}] Processed {done}/{local_target}", flush=True)

        if distributed:
            gathered = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(records, gathered, dst=0)
            if rank != 0:
                return
            records = []
            for shard in gathered:
                if shard:
                    records.extend(shard)
        records.sort(key=lambda x: x["idx"])

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_tag = args.model.split("/")[-1].replace("-", "_")
        summary = {
            "meta": {
                "timestamp_utc": ts,
                "model": args.model,
                "split": args.split,
                "n_samples": n,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "world_size": world_size,
                "greedy_budget": args.greedy_budget,
                "sc_budget": args.sc_budget,
                "sc_n": args.sc_n,
                "sc_temperature": args.sc_temperature,
                "prompt_format": args.prompt_format,
                "direct_answer": bool(args.direct_answer),
                "enable_thinking": bool(args.enable_thinking),
            },
            "greedy": summarize(records, "greedy"),
            "self_consistency": summarize(records, "sc"),
        }
        summary["delta_sc_minus_greedy"] = {
            "accuracy": summary["self_consistency"]["accuracy"] - summary["greedy"]["accuracy"],
            "avg_tokens": summary["self_consistency"]["avg_tokens"] - summary["greedy"]["avg_tokens"],
            "avg_latency_s": summary["self_consistency"]["avg_latency_s"] - summary["greedy"]["avg_latency_s"],
        }

        json_path = os.path.join(args.results_dir, f"sc_baseline_{model_tag}_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(args.results_dir, f"sc_baseline_per_sample_{model_tag}_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=sorted(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        print0("=== SC Baseline Summary ===")
        print0(json.dumps(summary, indent=2, ensure_ascii=False))
        print0(f"Saved: {json_path}")
        print0(f"Saved: {csv_path}")
    finally:
        maybe_cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
