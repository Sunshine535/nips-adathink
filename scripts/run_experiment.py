#!/usr/bin/env python3
"""Generalized fixed-budget vs adaptive-budget experiment for multiple benchmarks.

Supports GSM8K, MATH-500, and BBH. Produces per_sample CSV files in the same
column format as the original run_gsm8k_experiment.py so that all downstream
controller and significance scripts work without modification.

Usage:
    python run_experiment.py --benchmark gsm8k --model Qwen/Qwen3.5-27B ...
    python run_experiment.py --benchmark math500 --model Qwen/Qwen3.5-27B ...
    python run_experiment.py --benchmark bbh --bbh_task all --model Qwen/Qwen3.5-27B ...
"""

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks import (
    BenchmarkSample,
    build_prompt,
    default_budgets,
    is_correct,
    load_benchmark,
    parse_prediction,
)

DEFAULT_LOW_COST_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_MAIN_MODEL = "Qwen/Qwen3.5-27B"


# ---------------------------------------------------------------------------
# Distributed helpers (unchanged from run_gsm8k_experiment.py)
# ---------------------------------------------------------------------------

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
            backend="nccl", timeout=timedelta(minutes=60), device_id=local_rank,
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
        required = sorted(set(index_data.get("weight_map", {}).values()))
        missing = [n for n in required if not os.path.exists(os.path.join(snapshot_dir, n))]
        return snapshot_dir, missing
    if os.path.exists(single_path):
        return snapshot_dir, []
    return snapshot_dir, ["model.safetensors.index.json (or model.safetensors)"]


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

class GenOutput:
    __slots__ = ("text", "new_tokens", "elapsed_s")

    def __init__(self, text: str, new_tokens: int, elapsed_s: float):
        self.text = text
        self.new_tokens = new_tokens
        self.elapsed_s = elapsed_s


def generate_once(
    model, tokenizer, prompt: str, max_new_tokens: int, temperature: float = 0.0,
) -> GenOutput:
    inputs = tokenizer(prompt, return_tensors="pt")
    target_device = model_input_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    elapsed = time.perf_counter() - start

    gen_ids = out[0][in_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return GenOutput(text=text, new_tokens=int(gen_ids.shape[0]), elapsed_s=elapsed)


def verify_answer(model, tokenizer, question: str, candidate: str) -> bool:
    prompt = (
        "You are verifying a math answer. "
        "Given the question and candidate final answer, reply with only 'yes' or 'no'.\n\n"
        f"Question: {question}\n"
        f"Candidate answer: {candidate}\n"
        "Is the candidate correct?"
    )
    out = generate_once(model, tokenizer, prompt, max_new_tokens=4, temperature=0.0)
    return out.text.strip().lower().startswith("yes")


def project_final_answer(
    model, tokenizer, question: str, draft: str, benchmark: str, max_new_tokens: int = 32,
) -> Tuple[Optional[str], int, float, str, bool]:
    if benchmark in ("math500", "math"):
        instr = "Output exactly the final answer inside \\boxed{}."
    elif benchmark == "bbh":
        instr = "Output exactly one line: The answer is <answer>."
    else:
        instr = "Output exactly one line: Final answer: <number>"

    projection_prompt = (
        f"Read the question and draft solution. {instr}\n\n"
        f"Question: {question}\n\n"
        f"Draft solution:\n{draft}\n\n"
    )
    out = generate_once(model, tokenizer, projection_prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    pred, has_final, _ = parse_prediction(out.text, benchmark=benchmark, strict_final_only=False)
    return pred, out.new_tokens, out.elapsed_s, out.text, has_final


# ---------------------------------------------------------------------------
# Core experiment routines
# ---------------------------------------------------------------------------

def run_fixed_budget(
    model, tokenizer, prompt: str, question: str, gold: str, max_new_tokens: int,
    benchmark: str, is_mc: bool = True,
    strict_final_only: bool = False,
    projection_on_missing_final: bool = False,
    projection_max_tokens: int = 32,
) -> Tuple[Optional[str], int, float, str, int, str, int, int, float]:
    out = generate_once(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    pred, has_final, pred_source = parse_prediction(
        out.text, benchmark=benchmark, strict_final_only=strict_final_only, is_mc=is_mc,
    )
    projection_used = 0
    projection_tokens = 0
    projection_latency_s = 0.0
    raw = out.text

    if pred is None and projection_on_missing_final:
        p_pred, p_toks, p_lat, p_text, p_has_final = project_final_answer(
            model, tokenizer, question, raw, benchmark=benchmark, max_new_tokens=projection_max_tokens,
        )
        projection_used = 1
        projection_tokens = p_toks
        projection_latency_s = p_lat
        raw = raw + "\n\n[projection]\n" + p_text
        out.new_tokens += p_toks
        out.elapsed_s += p_lat
        if p_pred is not None:
            pred = p_pred
            has_final = p_has_final
            pred_source = "projection"

    return (pred, out.new_tokens, out.elapsed_s, raw, int(has_final),
            pred_source, projection_used, projection_tokens, projection_latency_s)


def run_adaptive(
    model, tokenizer, prompt: str, question: str, gold: str,
    chunk_plan: List[int], max_total_tokens: int, use_verifier: bool,
    benchmark: str, is_mc: bool = True,
    strict_final_only: bool = False,
    projection_on_missing_final: bool = False,
    projection_max_tokens: int = 32,
) -> Tuple[Optional[str], int, float, str, int, bool, int, str, int, int, float]:
    cumulative_text = ""
    total_tokens = 0
    total_time = 0.0
    verification_calls = 0
    stopped_early = False
    previous_pred = None
    stable_count = 0

    for chunk in chunk_plan:
        if total_tokens >= max_total_tokens:
            break
        allowed = min(chunk, max_total_tokens - total_tokens)
        out = generate_once(
            model, tokenizer, prompt + cumulative_text, max_new_tokens=allowed, temperature=0.0,
        )
        cumulative_text += out.text
        total_tokens += out.new_tokens
        total_time += out.elapsed_s

        pred, pred_has_final, _ = parse_prediction(
            cumulative_text, benchmark=benchmark, strict_final_only=strict_final_only, is_mc=is_mc,
        )
        if pred is None:
            continue

        if pred == previous_pred:
            stable_count += 1
        else:
            stable_count = 0
        previous_pred = pred

        verifier_ok = True
        if use_verifier:
            verification_calls += 1
            verifier_ok = verify_answer(model, tokenizer, question, pred)

        if stable_count >= 1 and pred_has_final and verifier_ok:
            stopped_early = True
            break

    final_pred, final_has_final, final_source = parse_prediction(
        cumulative_text, benchmark=benchmark, strict_final_only=strict_final_only, is_mc=is_mc,
    )

    projection_used = 0
    projection_tokens = 0
    projection_latency_s = 0.0
    if final_pred is None and projection_on_missing_final:
        p_pred, p_toks, p_lat, p_text, p_has_final = project_final_answer(
            model, tokenizer, question, cumulative_text, benchmark=benchmark,
            max_new_tokens=projection_max_tokens,
        )
        projection_used = 1
        projection_tokens = p_toks
        projection_latency_s = p_lat
        total_tokens += p_toks
        total_time += p_lat
        cumulative_text += "\n\n[projection]\n" + p_text
        if p_pred is not None:
            final_pred = p_pred
            final_has_final = p_has_final
            final_source = "projection"

    return (final_pred, total_tokens, total_time, cumulative_text,
            verification_calls, stopped_early, int(final_has_final), final_source,
            projection_used, projection_tokens, projection_latency_s)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize(records: List[Dict], key_prefix: str) -> Dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_latency_s": 0.0}
    n = len(records)
    return {
        "accuracy": sum(r[f"{key_prefix}_correct"] for r in records) / n,
        "avg_tokens": sum(r[f"{key_prefix}_tokens"] for r in records) / n,
        "avg_latency_s": sum(r[f"{key_prefix}_latency_s"] for r in records) / n,
    }


def compute_oer(records: List[Dict], short_key: str, long_key: str) -> float:
    if not records:
        return 0.0
    bad = sum(1 for r in records if r[f"{short_key}_correct"] and not r[f"{long_key}_correct"])
    return bad / len(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-budget vs adaptive-budget experiment on GSM8K/MATH-500/BBH"
    )
    parser.add_argument(
        "--benchmark", type=str, default="gsm8k",
        choices=["gsm8k", "math500", "bbh"],
        help="Benchmark to evaluate on.",
    )
    parser.add_argument(
        "--bbh_task", type=str, default="all",
        help="BBH subtask name (or 'all'). Only used when --benchmark=bbh.",
    )
    parser.add_argument(
        "--model_tier", type=str, choices=["low_cost", "main"], default="low_cost",
        help="Preset model tier used when --model is not provided.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"HF model id. Defaults: low_cost={DEFAULT_LOW_COST_MODEL}, main={DEFAULT_MAIN_MODEL}",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Token budgets. Defaults vary by benchmark.")
    parser.add_argument("--adaptive_chunks", type=int, nargs="+", default=None)
    parser.add_argument("--adaptive_max_total", type=int, default=None)
    parser.add_argument("--no_verifier", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--single_process_device_map_auto", action="store_true")
    parser.add_argument("--skip_local_model_check", action="store_true")
    parser.add_argument("--prompt_format", type=str, choices=["plain", "chat"], default="chat")
    parser.add_argument("--direct_answer", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--strict_final_only", action="store_true")
    parser.add_argument("--projection_on_missing_final", action="store_true")
    parser.add_argument("--projection_max_tokens", type=int, default=32)
    parser.add_argument("--results_dir", type=str, default="results")
    # Ablation flags
    parser.add_argument("--ablation_halting_only", action="store_true",
                        help="Ablation: only allow early-stop, no continue/branch.")
    parser.add_argument("--ablation_no_branch", action="store_true",
                        help="Ablation: disable branching in adaptive.")
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_LOW_COST_MODEL if args.model_tier == "low_cost" else DEFAULT_MAIN_MODEL
    if args.data_seed is None:
        args.data_seed = args.seed
    if args.budgets is None:
        args.budgets = default_budgets(args.benchmark, enable_thinking=args.enable_thinking)
    if args.adaptive_chunks is None:
        b = args.budgets
        if len(b) >= 3:
            args.adaptive_chunks = [b[0], b[0], b[1] - b[0] if b[1] > b[0] else b[0]]
        else:
            args.adaptive_chunks = [b[0]] * 3
    if args.adaptive_max_total is None:
        args.adaptive_max_total = max(args.budgets)

    rank, local_rank, world_size = get_rank_info()
    distributed = False

    try:
        distributed = maybe_init_distributed(world_size, local_rank)

        if not args.skip_local_model_check:
            check_payload = {"ok": True, "snapshot_dir": "", "missing": [], "error": ""}
            if rank == 0:
                try:
                    snapshot_dir, missing = check_local_model_snapshot(args.model)
                    check_payload["snapshot_dir"] = snapshot_dir
                    check_payload["missing"] = missing
                    if missing:
                        check_payload["ok"] = False
                except Exception as e:
                    check_payload["ok"] = False
                    check_payload["error"] = str(e)
            if distributed:
                obj_list = [check_payload]
                dist.broadcast_object_list(obj_list, src=0)
                check_payload = obj_list[0]
            if not check_payload["ok"]:
                err = check_payload.get("error", "")
                raise RuntimeError(
                    f"Local model snapshot issue. model={args.model}, "
                    f"missing={check_payload['missing']}, error={err}"
                )

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

        os.makedirs(args.results_dir, exist_ok=True)

        # --- Load model ---
        print0(f"Loading model: {args.model}")
        print0(f"Runtime: cuda={use_cuda}, world_size={world_size}, rank={rank}")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model_dtype = torch.bfloat16 if use_cuda else torch.float32
        model_kwargs = {"trust_remote_code": True, "dtype": model_dtype}
        if use_cuda and not distributed and args.single_process_device_map_auto and torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if use_cuda and distributed:
            model.to(torch.device(f"cuda:{local_rank}"))
        elif use_cuda and "device_map" not in model_kwargs:
            model.to(torch.device("cuda"))
        elif not use_cuda:
            model.to(torch.device("cpu"))
        if hasattr(model, "generation_config"):
            model.generation_config.do_sample = False
            for k in ("top_p", "top_k", "temperature"):
                if hasattr(model.generation_config, k):
                    setattr(model.generation_config, k, None)
        model.eval()

        # --- Load dataset ---
        print0(f"Loading benchmark: {args.benchmark}" + (f" (task={args.bbh_task})" if args.benchmark == "bbh" else ""))
        load_kwargs = {"split": args.split}
        if args.benchmark == "bbh":
            load_kwargs["task"] = args.bbh_task
        samples = load_benchmark(args.benchmark, **load_kwargs)

        rng = random.Random(args.data_seed)
        rng.shuffle(samples)
        n = min(args.n_samples, len(samples))
        samples = samples[:n]
        print0(f"Running {n} samples with budgets={args.budgets}")

        local_target = sum(1 for i in range(n) if (i % world_size) == rank)
        records: List[Dict] = []
        local_done = 0

        for i, sample in enumerate(samples):
            if (i % world_size) != rank:
                continue

            question = sample.question
            gold = sample.gold
            is_mc = sample.meta.get("is_mc", True)

            row: Dict = {
                "idx": i,
                "question": question,
                "gold": gold,
                "benchmark": args.benchmark,
            }
            # Preserve BBH task info
            if "task" in sample.meta:
                row["bbh_task"] = sample.meta["task"]
            if "subject" in sample.meta:
                row["math_subject"] = sample.meta["subject"]
            if "level" in sample.meta:
                row["math_level"] = sample.meta["level"]

            prompt = build_prompt(
                question,
                benchmark=args.benchmark,
                tokenizer=tokenizer,
                prompt_format=args.prompt_format,
                direct_answer=args.direct_answer,
                enable_thinking=True if args.enable_thinking else False,
            )

            # --- Fixed budgets ---
            for b in args.budgets:
                (pred, toks, lat, raw, has_final, pred_source,
                 proj_used, proj_toks, proj_lat) = run_fixed_budget(
                    model, tokenizer, prompt, question, gold, b,
                    benchmark=args.benchmark, is_mc=is_mc,
                    strict_final_only=args.strict_final_only,
                    projection_on_missing_final=args.projection_on_missing_final,
                    projection_max_tokens=args.projection_max_tokens,
                )
                row[f"fixed_{b}_pred"] = pred
                row[f"fixed_{b}_tokens"] = toks
                row[f"fixed_{b}_latency_s"] = lat
                row[f"fixed_{b}_correct"] = int(is_correct(pred, gold, args.benchmark, is_mc=is_mc))
                row[f"fixed_{b}_has_final"] = has_final
                row[f"fixed_{b}_pred_source"] = pred_source
                row[f"fixed_{b}_projection_used"] = proj_used
                row[f"fixed_{b}_projection_tokens"] = proj_toks
                row[f"fixed_{b}_projection_latency_s"] = proj_lat
                row[f"fixed_{b}_raw"] = raw

            # --- Adaptive ---
            adaptive_chunks = args.adaptive_chunks
            if args.ablation_halting_only:
                adaptive_chunks = [args.adaptive_max_total]

            (adap_pred, adap_toks, adap_lat, adap_raw, ver_calls, stopped_early,
             adap_has_final, adap_source, adap_proj_used, adap_proj_toks, adap_proj_lat) = run_adaptive(
                model, tokenizer, prompt, question, gold,
                chunk_plan=adaptive_chunks,
                max_total_tokens=args.adaptive_max_total,
                use_verifier=not args.no_verifier,
                benchmark=args.benchmark, is_mc=is_mc,
                strict_final_only=args.strict_final_only,
                projection_on_missing_final=args.projection_on_missing_final,
                projection_max_tokens=args.projection_max_tokens,
            )

            row["adaptive_pred"] = adap_pred
            row["adaptive_tokens"] = adap_toks
            row["adaptive_latency_s"] = adap_lat
            row["adaptive_correct"] = int(is_correct(adap_pred, gold, args.benchmark, is_mc=is_mc))
            row["adaptive_verifier_calls"] = ver_calls
            row["adaptive_stopped_early"] = int(stopped_early)
            row["adaptive_has_final"] = adap_has_final
            row["adaptive_pred_source"] = adap_source
            row["adaptive_projection_used"] = adap_proj_used
            row["adaptive_projection_tokens"] = adap_proj_toks
            row["adaptive_projection_latency_s"] = adap_proj_lat
            row["adaptive_raw"] = adap_raw

            records.append(row)
            local_done += 1

            if (local_done % 5 == 0) or (local_done == local_target):
                print(f"[rank {rank}] Processed {local_done}/{local_target}", flush=True)

        # --- Gather distributed results ---
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
        bench_tag = args.benchmark
        if args.benchmark == "bbh" and args.bbh_task != "all":
            bench_tag = f"bbh_{args.bbh_task}"

        # --- Build summary ---
        summary: Dict = {
            "meta": {
                "timestamp_utc": ts,
                "benchmark": args.benchmark,
                "bbh_task": args.bbh_task if args.benchmark == "bbh" else None,
                "model": args.model,
                "n_samples": n,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "world_size": world_size,
                "distributed": bool(distributed),
                "budgets": args.budgets,
                "adaptive_chunks": args.adaptive_chunks,
                "adaptive_max_total": args.adaptive_max_total,
                "use_verifier": not args.no_verifier,
                "prompt_format": args.prompt_format,
                "direct_answer": bool(args.direct_answer),
                "enable_thinking": bool(args.enable_thinking),
                "strict_final_only": bool(args.strict_final_only),
                "projection_on_missing_final": bool(args.projection_on_missing_final),
                "projection_max_tokens": int(args.projection_max_tokens),
                "ablation_halting_only": bool(args.ablation_halting_only),
                "ablation_no_branch": bool(args.ablation_no_branch),
            },
            "fixed": {},
            "adaptive": {},
            "overthinking": {},
        }

        for b in args.budgets:
            summary["fixed"][str(b)] = summarize(records, f"fixed_{b}")
            summary["fixed"][str(b)]["final_rate"] = sum(
                r.get(f"fixed_{b}_has_final", 0) for r in records
            ) / max(1, len(records))
            summary["fixed"][str(b)]["projection_rate"] = sum(
                r.get(f"fixed_{b}_projection_used", 0) for r in records
            ) / max(1, len(records))

        summary["adaptive"] = summarize(records, "adaptive")
        summary["adaptive"]["avg_verifier_calls"] = sum(r["adaptive_verifier_calls"] for r in records) / max(1, len(records))
        summary["adaptive"]["early_stop_rate"] = sum(r["adaptive_stopped_early"] for r in records) / max(1, len(records))
        summary["adaptive"]["final_rate"] = sum(r.get("adaptive_has_final", 0) for r in records) / max(1, len(records))

        if len(args.budgets) >= 2:
            short_b = min(args.budgets)
            long_b = max(args.budgets)
            summary["overthinking"][f"fixed_{short_b}_vs_fixed_{long_b}"] = compute_oer(
                records, f"fixed_{short_b}", f"fixed_{long_b}"
            )

        # Per-task breakdown for BBH
        if args.benchmark == "bbh":
            task_records: Dict[str, List[Dict]] = {}
            for r in records:
                t = r.get("bbh_task", "unknown")
                task_records.setdefault(t, []).append(r)
            summary["per_task"] = {}
            for t, t_recs in sorted(task_records.items()):
                summary["per_task"][t] = {
                    "n": len(t_recs),
                    "fixed": {str(b): summarize(t_recs, f"fixed_{b}") for b in args.budgets},
                    "adaptive": summarize(t_recs, "adaptive"),
                }

        # Per-subject/level breakdown for MATH-500
        if args.benchmark == "math500":
            for group_key in ("math_subject", "math_level"):
                group_records: Dict[str, List[Dict]] = {}
                for r in records:
                    g = r.get(group_key, "unknown")
                    group_records.setdefault(g, []).append(r)
                summary[f"per_{group_key}"] = {}
                for g, g_recs in sorted(group_records.items()):
                    summary[f"per_{group_key}"][g] = {
                        "n": len(g_recs),
                        "fixed": {str(b): summarize(g_recs, f"fixed_{b}") for b in args.budgets},
                        "adaptive": summarize(g_recs, "adaptive"),
                    }

        # --- Save outputs ---
        json_path = os.path.join(args.results_dir, f"summary_{bench_tag}_{model_tag}_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(args.results_dir, f"per_sample_{bench_tag}_{model_tag}_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=sorted(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)

        print0("=== Summary ===")
        print0(json.dumps(summary, indent=2, ensure_ascii=False))
        print0(f"Saved: {json_path}")
        print0(f"Saved: {csv_path}")
    finally:
        maybe_cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
