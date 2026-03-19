#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import product
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

    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        for _, device in hf_map.items():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
        for _, device in hf_map.items():
            if isinstance(device, str) and device == "cpu":
                return torch.device("cpu")

    return next(model.parameters()).device


def check_local_model_snapshot(model_id: str) -> Tuple[str, List[str]]:
    from huggingface_hub import snapshot_download

    snapshot_dir = snapshot_download(repo_id=model_id, local_files_only=True)
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    single_path = os.path.join(snapshot_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        required = sorted(set(weight_map.values()))
        missing = [name for name in required if not os.path.exists(os.path.join(snapshot_dir, name))]
        return snapshot_dir, missing

    if os.path.exists(single_path):
        return snapshot_dir, []

    return snapshot_dir, ["model.safetensors.index.json (or model.safetensors)"]


@dataclass
class GenOutput:
    text: str
    new_tokens: int
    elapsed_s: float


def generate_once(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> GenOutput:
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    start = time.perf_counter()
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = out[0][in_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return GenOutput(text=text, new_tokens=int(gen_ids.shape[0]), elapsed_s=elapsed)


def run_fixed_budget(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> Tuple[Optional[str], int, float]:
    out = generate_once(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    pred = extract_number(out.text)
    return pred, out.new_tokens, out.elapsed_s


def collect_chunk_trace(
    model,
    tokenizer,
    prompt: str,
    chunk_plan: List[int],
    max_total_tokens: int,
) -> List[Dict]:
    cumulative_text = ""
    total_tokens = 0
    total_time = 0.0
    previous_pred = None
    stable_count = 0
    steps = []

    for i, chunk in enumerate(chunk_plan):
        if total_tokens >= max_total_tokens:
            break
        allowed = min(chunk, max_total_tokens - total_tokens)
        out = generate_once(
            model,
            tokenizer,
            prompt + cumulative_text,
            max_new_tokens=allowed,
            temperature=0.0,
        )
        cumulative_text += out.text
        total_tokens += out.new_tokens
        total_time += out.elapsed_s

        pred = extract_number(cumulative_text)
        if pred is not None:
            if pred == previous_pred:
                stable_count += 1
            else:
                stable_count = 0
            previous_pred = pred

        steps.append(
            {
                "step_idx": i + 1,
                "pred": pred,
                "cum_tokens": total_tokens,
                "cum_latency_s": total_time,
                "has_final": int("final answer" in cumulative_text.lower()),
                "stable_count": stable_count,
            }
        )
    return steps


def pick_step_by_policy(
    steps: List[Dict],
    min_stable: int,
    require_final: int,
    min_tokens_before_stop: int,
    max_steps: int,
) -> Tuple[Dict, int]:
    chosen = steps[min(max_steps, len(steps)) - 1]
    stopped_early = 0
    for step in steps:
        if step["step_idx"] > max_steps:
            break
        pred_ok = step["pred"] is not None
        stable_ok = step["stable_count"] >= min_stable
        final_ok = (step["has_final"] == 1) if require_final else True
        tokens_ok = step["cum_tokens"] >= min_tokens_before_stop
        if pred_ok and stable_ok and final_ok and tokens_ok:
            chosen = step
            stopped_early = 1
            break
    return chosen, stopped_early


def eval_policy(records: List[Dict], policy: Dict) -> Dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_latency_s": 0.0, "early_stop_rate": 0.0}
    corr = 0
    tok = 0.0
    lat = 0.0
    early = 0
    for r in records:
        chosen, stopped_early = pick_step_by_policy(
            r["steps"],
            min_stable=policy["min_stable"],
            require_final=policy["require_final"],
            min_tokens_before_stop=policy["min_tokens_before_stop"],
            max_steps=policy["max_steps"],
        )
        corr += int(is_correct(chosen["pred"], r["gold"]))
        tok += float(chosen["cum_tokens"])
        lat += float(chosen["cum_latency_s"])
        early += int(stopped_early)
    n = len(records)
    return {
        "accuracy": corr / n,
        "avg_tokens": tok / n,
        "avg_latency_s": lat / n,
        "early_stop_rate": early / n,
    }


def summarize_fixed(records: List[Dict], budget: int) -> Dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_latency_s": 0.0}
    n = len(records)
    corr = sum(int(r[f"fixed_{budget}_correct"]) for r in records) / n
    tok = sum(float(r[f"fixed_{budget}_tokens"]) for r in records) / n
    lat = sum(float(r[f"fixed_{budget}_latency_s"]) for r in records) / n
    return {"accuracy": corr, "avg_tokens": tok, "avg_latency_s": lat}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a data-driven adaptive stopping policy on GSM8K.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--n_train", type=int, default=120)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train_data_seed", type=int, default=101)
    parser.add_argument("--test_data_seed", type=int, default=303)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--chunk_plan", type=int, nargs="+", default=[64, 64, 128])
    parser.add_argument("--max_total_tokens", type=int, default=256)
    parser.add_argument("--lambda_cost", type=float, default=0.05)
    parser.add_argument(
        "--min_stable_grid",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--require_final_grid",
        type=int,
        nargs="+",
        default=[0, 1],
    )
    parser.add_argument(
        "--min_tokens_grid",
        type=int,
        nargs="+",
        default=[0, 16, 32, 48, 64],
    )
    parser.add_argument(
        "--max_steps_grid",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument("--results_dir", type=str, default="methods/01_adathink/results")
    parser.add_argument(
        "--prompt_format",
        type=str,
        choices=["plain", "chat"],
        default="chat",
    )
    parser.add_argument("--direct_answer", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
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
            raise RuntimeError("CUDA is required but unavailable. Pass --allow_cpu to override.")

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
        model_dtype = torch.bfloat16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            dtype=model_dtype,
        )
        if use_cuda and distributed:
            model.to(torch.device(f"cuda:{local_rank}"))
        elif use_cuda:
            model.to(torch.device("cuda"))
        else:
            model.to(torch.device("cpu"))

        if hasattr(model, "generation_config"):
            model.generation_config.do_sample = False
            for k in ("top_p", "top_k", "temperature"):
                if hasattr(model.generation_config, k):
                    setattr(model.generation_config, k, None)
        model.eval()

        def load_subset(split: str, n_samples: int, data_seed: int):
            ds = load_dataset("gsm8k", "main", split=split)
            ds = ds.shuffle(seed=data_seed)
            n = min(n_samples, len(ds))
            ds = ds.select(range(n))
            return ds, n

        train_ds, n_train = load_subset(args.train_split, args.n_train, args.train_data_seed)
        test_ds, n_test = load_subset(args.test_split, args.n_test, args.test_data_seed)

        print0(f"Collecting train traces: n={n_train}, world_size={world_size}")
        train_records_local: List[Dict] = []
        train_target = sum(1 for i in range(n_train) if (i % world_size) == rank)
        train_done = 0
        for i, ex in enumerate(train_ds):
            if (i % world_size) != rank:
                continue
            question = ex["question"]
            gold = get_gold_from_gsm8k(ex["answer"])
            prompt = build_prompt(
                question,
                tokenizer=tokenizer,
                prompt_format=args.prompt_format,
                direct_answer=args.direct_answer,
                enable_thinking=True if args.enable_thinking else False,
            )
            row = {"idx": i, "gold": gold}
            for b in args.budgets:
                pred, toks, lat = run_fixed_budget(model, tokenizer, prompt, b)
                row[f"fixed_{b}_pred"] = pred
                row[f"fixed_{b}_tokens"] = toks
                row[f"fixed_{b}_latency_s"] = lat
                row[f"fixed_{b}_correct"] = int(is_correct(pred, gold))
            row["steps"] = collect_chunk_trace(
                model,
                tokenizer,
                prompt,
                chunk_plan=args.chunk_plan,
                max_total_tokens=args.max_total_tokens,
            )
            train_records_local.append(row)
            train_done += 1
            if train_done % 5 == 0 or train_done == train_target:
                print(f"[rank {rank}] Train processed {train_done}/{train_target}", flush=True)

        print0(f"Collecting test traces: n={n_test}, world_size={world_size}")
        test_records_local: List[Dict] = []
        test_target = sum(1 for i in range(n_test) if (i % world_size) == rank)
        test_done = 0
        for i, ex in enumerate(test_ds):
            if (i % world_size) != rank:
                continue
            question = ex["question"]
            gold = get_gold_from_gsm8k(ex["answer"])
            prompt = build_prompt(
                question,
                tokenizer=tokenizer,
                prompt_format=args.prompt_format,
                direct_answer=args.direct_answer,
                enable_thinking=True if args.enable_thinking else False,
            )
            row = {"idx": i, "gold": gold}
            for b in args.budgets:
                pred, toks, lat = run_fixed_budget(model, tokenizer, prompt, b)
                row[f"fixed_{b}_pred"] = pred
                row[f"fixed_{b}_tokens"] = toks
                row[f"fixed_{b}_latency_s"] = lat
                row[f"fixed_{b}_correct"] = int(is_correct(pred, gold))
            row["steps"] = collect_chunk_trace(
                model,
                tokenizer,
                prompt,
                chunk_plan=args.chunk_plan,
                max_total_tokens=args.max_total_tokens,
            )
            test_records_local.append(row)
            test_done += 1
            if test_done % 5 == 0 or test_done == test_target:
                print(f"[rank {rank}] Test processed {test_done}/{test_target}", flush=True)

        if distributed:
            gathered_train = [None for _ in range(world_size)] if rank == 0 else None
            gathered_test = [None for _ in range(world_size)] if rank == 0 else None
            dist.gather_object(train_records_local, gathered_train, dst=0)
            dist.gather_object(test_records_local, gathered_test, dst=0)
            if rank != 0:
                return
            train_records = []
            test_records = []
            for shard in gathered_train:
                if shard:
                    train_records.extend(shard)
            for shard in gathered_test:
                if shard:
                    test_records.extend(shard)
        else:
            train_records = train_records_local
            test_records = test_records_local

        train_records.sort(key=lambda x: x["idx"])
        test_records.sort(key=lambda x: x["idx"])

        # Grid-search a data-driven stop policy on train split.
        best_policy = None
        best_score = -1e9
        policy_scores = []
        for min_stable, require_final, min_tokens, max_steps in product(
            args.min_stable_grid,
            args.require_final_grid,
            args.min_tokens_grid,
            args.max_steps_grid,
        ):
            policy = {
                "min_stable": int(min_stable),
                "require_final": int(require_final),
                "min_tokens_before_stop": int(min_tokens),
                "max_steps": int(max_steps),
            }
            metrics = eval_policy(train_records, policy)
            score = metrics["accuracy"] - args.lambda_cost * (metrics["avg_tokens"] / max(1, args.max_total_tokens))
            policy_scores.append({"policy": policy, "metrics": metrics, "score": score})
            if score > best_score:
                best_score = score
                best_policy = policy

        assert best_policy is not None

        # Evaluate best policy on test.
        best_test = eval_policy(test_records, best_policy)

        # Heuristic baseline equivalent to previous stop rule.
        heuristic_policy = {
            "min_stable": 1,
            "require_final": 1,
            "min_tokens_before_stop": 0,
            "max_steps": len(args.chunk_plan),
        }
        heuristic_test = eval_policy(test_records, heuristic_policy)

        fixed_summary = {str(b): summarize_fixed(test_records, b) for b in args.budgets}

        # Save per-sample for best policy.
        per_sample = []
        for r in test_records:
            chosen, stopped = pick_step_by_policy(
                r["steps"],
                min_stable=best_policy["min_stable"],
                require_final=best_policy["require_final"],
                min_tokens_before_stop=best_policy["min_tokens_before_stop"],
                max_steps=best_policy["max_steps"],
            )
            row = {
                "idx": r["idx"],
                "gold": r["gold"],
                "best_pred": chosen["pred"],
                "best_correct": int(is_correct(chosen["pred"], r["gold"])),
                "best_tokens": chosen["cum_tokens"],
                "best_latency_s": chosen["cum_latency_s"],
                "best_stopped_early": int(stopped),
            }
            for b in args.budgets:
                row[f"fixed_{b}_pred"] = r[f"fixed_{b}_pred"]
                row[f"fixed_{b}_correct"] = r[f"fixed_{b}_correct"]
                row[f"fixed_{b}_tokens"] = r[f"fixed_{b}_tokens"]
                row[f"fixed_{b}_latency_s"] = r[f"fixed_{b}_latency_s"]
            per_sample.append(row)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_tag = args.model.split("/")[-1].replace("-", "_")
        summary = {
            "meta": {
                "timestamp_utc": ts,
                "model": args.model,
                "seed": args.seed,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "n_train": n_train,
                "n_test": n_test,
                "train_data_seed": args.train_data_seed,
                "test_data_seed": args.test_data_seed,
                "world_size": world_size,
                "budgets": args.budgets,
                "chunk_plan": args.chunk_plan,
                "max_total_tokens": args.max_total_tokens,
                "lambda_cost": args.lambda_cost,
                "prompt_format": args.prompt_format,
                "direct_answer": bool(args.direct_answer),
                "enable_thinking": bool(args.enable_thinking),
            },
            "best_policy": best_policy,
            "best_policy_test": best_test,
            "heuristic_policy": heuristic_policy,
            "heuristic_policy_test": heuristic_test,
            "fixed_test": fixed_summary,
            "top5_train_policies": sorted(policy_scores, key=lambda x: x["score"], reverse=True)[:5],
        }

        json_path = os.path.join(args.results_dir, f"policy_search_{model_tag}_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(args.results_dir, f"policy_search_per_sample_{model_tag}_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(per_sample[0].keys()) if per_sample else [])
            if per_sample:
                writer.writeheader()
                writer.writerows(per_sample)

        print0("=== Policy Search Summary ===")
        print0(json.dumps(summary, indent=2, ensure_ascii=False))
        print0(f"Saved: {json_path}")
        print0(f"Saved: {csv_path}")
    finally:
        maybe_cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
