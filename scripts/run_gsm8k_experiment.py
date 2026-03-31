#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
DEFAULT_LOW_COST_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_MAIN_MODEL = "Qwen/Qwen3.5-27B"


def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None

    matches = NUM_RE.findall(text)
    if matches:
        return matches[-1]
    return None


def extract_final_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(FINAL_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1)
    return None


def extract_boxed_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(BOXED_RE.finditer(text))
    for m in reversed(matches):
        inner = m.group(1)
        val = extract_last_number(inner)
        if val is not None:
            return val
    return None


def has_explicit_final(text: str) -> bool:
    return extract_final_number(text) is not None


def parse_prediction(text: str, strict_final_only: bool = False) -> Tuple[Optional[str], bool, str]:
    final = extract_final_number(text)
    if final is not None:
        return final, True, "final_marker"

    boxed = extract_boxed_number(text)
    if boxed is not None:
        return boxed, False, "boxed"

    if strict_final_only:
        return None, False, "none"

    tail = extract_last_number(text)
    if tail is not None:
        return tail, False, "fallback_last"
    return None, False, "none"


def extract_number(text: str) -> Optional[str]:
    pred, _, _ = parse_prediction(text, strict_final_only=False)
    return pred


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
    # GSM8K gold answer format usually includes "#### <number>".
    if "####" in answer_field:
        after = answer_field.split("####")[-1]
        match = NUM_RE.search(after)
        if match:
            return match.group(0)
    return extract_last_number(answer_field)


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


@dataclass
class GenOutput:
    text: str
    new_tokens: int
    elapsed_s: float


def get_rank_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def maybe_init_distributed(world_size: int, local_rank: int) -> bool:
    if world_size <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError(
            "WORLD_SIZE>1 detected but CUDA is unavailable. "
            "Install CUDA-enabled PyTorch before using torchrun multi-GPU."
        )
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
    """Validate that all required safetensor shards exist in local HF cache."""
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


def verify_answer(model, tokenizer, question: str, candidate: str) -> bool:
    prompt = (
        "You are verifying a math answer. "
        "Given the question and candidate final answer, reply with only 'yes' or 'no'.\n\n"
        f"Question: {question}\n"
        f"Candidate answer: {candidate}\n"
        "Is the candidate correct?"
    )
    out = generate_once(model, tokenizer, prompt, max_new_tokens=4, temperature=0.0)
    decision = out.text.strip().lower()
    return decision.startswith("yes")


def project_final_answer(
    model,
    tokenizer,
    question: str,
    draft: str,
    max_new_tokens: int = 16,
) -> Tuple[Optional[str], int, float, str, bool]:
    projection_prompt = (
        "Read the question and draft solution. "
        "Output exactly one line in this format: Final answer: <number>\n\n"
        f"Question: {question}\n\n"
        f"Draft solution:\n{draft}\n\n"
        "Final answer:"
    )
    out = generate_once(
        model,
        tokenizer,
        projection_prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    pred, has_final, _ = parse_prediction(out.text, strict_final_only=False)
    return pred, out.new_tokens, out.elapsed_s, out.text, has_final


def run_fixed_budget(
    model,
    tokenizer,
    prompt: str,
    question: str,
    max_new_tokens: int,
    strict_final_only: bool = False,
    projection_on_missing_final: bool = False,
    projection_max_tokens: int = 16,
) -> Tuple[Optional[str], int, float, str, int, str, int, int, float]:
    out = generate_once(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    pred, has_final, pred_source = parse_prediction(out.text, strict_final_only=strict_final_only)
    projection_used = 0
    projection_tokens = 0
    projection_latency_s = 0.0
    raw = out.text

    if pred is None and projection_on_missing_final:
        p_pred, p_toks, p_lat, p_text, p_has_final = project_final_answer(
            model, tokenizer, question, raw, max_new_tokens=projection_max_tokens
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

    return (
        pred,
        out.new_tokens,
        out.elapsed_s,
        raw,
        int(has_final),
        pred_source,
        projection_used,
        projection_tokens,
        projection_latency_s,
    )


def run_adaptive(
    model,
    tokenizer,
    prompt: str,
    question: str,
    chunk_plan: List[int],
    max_total_tokens: int,
    use_verifier: bool,
    strict_final_only: bool = False,
    projection_on_missing_final: bool = False,
    projection_max_tokens: int = 16,
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
            model,
            tokenizer,
            prompt + cumulative_text,
            max_new_tokens=allowed,
            temperature=0.0,
        )
        cumulative_text += out.text
        total_tokens += out.new_tokens
        total_time += out.elapsed_s

        pred, pred_has_final, _ = parse_prediction(
            cumulative_text, strict_final_only=strict_final_only
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
        cumulative_text, strict_final_only=strict_final_only
    )

    projection_used = 0
    projection_tokens = 0
    projection_latency_s = 0.0
    if final_pred is None and projection_on_missing_final:
        p_pred, p_toks, p_lat, p_text, p_has_final = project_final_answer(
            model, tokenizer, question, cumulative_text, max_new_tokens=projection_max_tokens
        )
        projection_used = 1
        projection_tokens = p_toks
        projection_latency_s = p_lat
        total_tokens += p_toks
        total_time += p_lat
        cumulative_text = cumulative_text + "\n\n[projection]\n" + p_text
        if p_pred is not None:
            final_pred = p_pred
            final_has_final = p_has_final
            final_source = "projection"

    return (
        final_pred,
        total_tokens,
        total_time,
        cumulative_text,
        verification_calls,
        stopped_early,
        int(final_has_final),
        final_source,
        projection_used,
        projection_tokens,
        projection_latency_s,
    )


def summarize(records: List[Dict], key_prefix: str) -> Dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "avg_tokens": 0.0, "avg_latency_s": 0.0}
    acc = sum(r[f"{key_prefix}_correct"] for r in records) / len(records)
    avg_tokens = sum(r[f"{key_prefix}_tokens"] for r in records) / len(records)
    avg_latency = sum(r[f"{key_prefix}_latency_s"] for r in records) / len(records)
    return {
        "accuracy": acc,
        "avg_tokens": avg_tokens,
        "avg_latency_s": avg_latency,
    }


def compute_oer(records: List[Dict], short_key: str, long_key: str) -> float:
    if not records:
        return 0.0
    bad = 0
    for r in records:
        if r[f"{short_key}_correct"] and not r[f"{long_key}_correct"]:
            bad += 1
    return bad / len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSM8K fixed-budget vs adaptive budget experiment")
    parser.add_argument(
        "--model_tier",
        type=str,
        choices=["low_cost", "main"],
        default="low_cost",
        help="Preset model tier used when --model is not provided",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Optional explicit HF model id override. "
            f"Defaults: low_cost={DEFAULT_LOW_COST_MODEL}, main={DEFAULT_MAIN_MODEL}"
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help="Dataset shuffle seed. Defaults to --seed when not provided.",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--adaptive_chunks", type=int, nargs="+", default=[64, 64, 128])
    parser.add_argument("--adaptive_max_total", type=int, default=256)
    parser.add_argument("--no_verifier", action="store_true")
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU fallback. By default this script requires CUDA.",
    )
    parser.add_argument(
        "--single_process_device_map_auto",
        action="store_true",
        help=(
            "When running without torchrun and with multiple GPUs visible, "
            "load with device_map='auto'. Preferred multi-GPU mode is torchrun."
        ),
    )
    parser.add_argument(
        "--skip_local_model_check",
        action="store_true",
        help="Skip local snapshot completeness check before loading model.",
    )
    parser.add_argument(
        "--prompt_format",
        type=str,
        choices=["plain", "chat"],
        default="chat",
        help="Prompt format. 'chat' is recommended for Qwen3 series.",
    )
    parser.add_argument(
        "--direct_answer",
        action="store_true",
        help="Ask model to output only 'Final answer: <number>' for tighter token budgets.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode when prompt_format=chat (Qwen3 supports this switch).",
    )
    parser.add_argument(
        "--strict_final_only",
        action="store_true",
        help="Only accept explicit final-answer markers (or projection output) as valid prediction.",
    )
    parser.add_argument(
        "--projection_on_missing_final",
        action="store_true",
        help="Run short projection pass when final-answer marker is missing.",
    )
    parser.add_argument(
        "--projection_max_tokens",
        type=int,
        default=16,
        help="Max new tokens for projection pass.",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    if args.model is None:
        args.model = DEFAULT_LOW_COST_MODEL if args.model_tier == "low_cost" else DEFAULT_MAIN_MODEL
    if args.data_seed is None:
        args.data_seed = args.seed

    rank, local_rank, world_size = get_rank_info()
    distributed = False
    try:
        distributed = maybe_init_distributed(world_size, local_rank)

        if not args.skip_local_model_check:
            check_payload = {
                "ok": True,
                "snapshot_dir": "",
                "missing": [],
                "error": "",
            }
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
                if check_payload["missing"]:
                    raise RuntimeError(
                        "Local model snapshot is incomplete. "
                        f"model={args.model}, snapshot={check_payload['snapshot_dir']}, "
                        f"missing={check_payload['missing']}. "
                        "Please finish manual download, then rerun."
                    )
                err_detail = check_payload["error"]
                err_suffix = f" detail={err_detail}" if err_detail else ""
                raise RuntimeError(
                    "Local model snapshot not available or unreadable. "
                    f"model={args.model}. Please complete manual download first.{err_suffix}"
                )

        def print0(msg: str) -> None:
            if rank == 0:
                print(msg, flush=True)

        use_cuda = torch.cuda.is_available()
        if not use_cuda and not args.allow_cpu:
            raise RuntimeError(
                "CUDA is required but not available in this environment. "
                "Install CUDA-enabled PyTorch, or pass --allow_cpu to override."
            )

        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        if use_cuda:
            torch.cuda.manual_seed_all(args.seed + rank)

        os.makedirs(args.results_dir, exist_ok=True)

        print0(f"Loading model: {args.model}")
        print0(
            f"Runtime: cuda={use_cuda}, world_size={world_size}, rank={rank}, local_rank={local_rank}"
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model_dtype = torch.bfloat16 if use_cuda else torch.float32
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": model_dtype,
        }
        if use_cuda and not distributed and args.single_process_device_map_auto and torch.cuda.device_count() > 1:
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if use_cuda and distributed:
            model.to(torch.device(f"cuda:{local_rank}"))
        elif use_cuda and "device_map" not in model_kwargs:
            model.to(torch.device("cuda"))
        elif not use_cuda:
            model.to(torch.device("cpu"))
        # Keep deterministic greedy defaults and avoid noisy invalid generation-arg warnings.
        if hasattr(model, "generation_config"):
            model.generation_config.do_sample = False
            for k in ("top_p", "top_k", "temperature"):
                if hasattr(model.generation_config, k):
                    setattr(model.generation_config, k, None)
        model.eval()

        print0("Loading GSM8K...")
        ds = load_dataset("gsm8k", "main", split=args.split)
        ds = ds.shuffle(seed=args.data_seed)
        n = min(args.n_samples, len(ds))
        ds = ds.select(range(n))

        model_tag_inc = args.model.split("/")[-1].replace("-", "_")
        incremental_path = os.path.join(
            args.results_dir,
            f"incremental_{model_tag_inc}_{args.seed}_rank{rank}.jsonl",
        )
        done_indices: set = set()
        records: List[Dict] = []
        if os.path.exists(incremental_path):
            with open(incremental_path, "r", encoding="utf-8") as _fp:
                for _line in _fp:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = json.loads(_line)
                        done_indices.add(_rec["idx"])
                        records.append(_rec)
                    except (json.JSONDecodeError, KeyError):
                        continue
            if done_indices:
                print0(f"[resume] Loaded {len(done_indices)} existing results from {incremental_path}")

        local_target = sum(1 for i in range(n) if (i % world_size) == rank and i not in done_indices)
        local_done = 0

        inc_fp = open(incremental_path, "a", encoding="utf-8")

        for i, ex in enumerate(ds):
            if (i % world_size) != rank:
                continue
            if i in done_indices:
                continue

            question = ex["question"]
            gold = get_gold_from_gsm8k(ex["answer"])

            row: Dict = {
                "idx": i,
                "question": question,
                "gold": gold,
            }
            prompt = build_prompt(
                question,
                tokenizer=tokenizer,
                prompt_format=args.prompt_format,
                direct_answer=args.direct_answer,
                enable_thinking=True if args.enable_thinking else False,
            )

            for b in args.budgets:
                (
                    pred,
                    toks,
                    lat,
                    raw,
                    has_final,
                    pred_source,
                    projection_used,
                    projection_tokens,
                    projection_latency_s,
                ) = run_fixed_budget(
                    model,
                    tokenizer,
                    prompt,
                    question,
                    b,
                    strict_final_only=args.strict_final_only,
                    projection_on_missing_final=args.projection_on_missing_final,
                    projection_max_tokens=args.projection_max_tokens,
                )
                row[f"fixed_{b}_pred"] = pred
                row[f"fixed_{b}_tokens"] = toks
                row[f"fixed_{b}_latency_s"] = lat
                row[f"fixed_{b}_correct"] = int(is_correct(pred, gold))
                row[f"fixed_{b}_has_final"] = has_final
                row[f"fixed_{b}_pred_source"] = pred_source
                row[f"fixed_{b}_projection_used"] = projection_used
                row[f"fixed_{b}_projection_tokens"] = projection_tokens
                row[f"fixed_{b}_projection_latency_s"] = projection_latency_s
                row[f"fixed_{b}_raw"] = raw

            (
                adap_pred,
                adap_toks,
                adap_lat,
                adap_raw,
                ver_calls,
                stopped_early,
                adap_has_final,
                adap_pred_source,
                adap_projection_used,
                adap_projection_tokens,
                adap_projection_latency_s,
            ) = run_adaptive(
                model,
                tokenizer,
                prompt,
                question,
                chunk_plan=args.adaptive_chunks,
                max_total_tokens=args.adaptive_max_total,
                use_verifier=not args.no_verifier,
                strict_final_only=args.strict_final_only,
                projection_on_missing_final=args.projection_on_missing_final,
                projection_max_tokens=args.projection_max_tokens,
            )

            row["adaptive_pred"] = adap_pred
            row["adaptive_tokens"] = adap_toks
            row["adaptive_latency_s"] = adap_lat
            row["adaptive_correct"] = int(is_correct(adap_pred, gold))
            row["adaptive_verifier_calls"] = ver_calls
            row["adaptive_stopped_early"] = int(stopped_early)
            row["adaptive_has_final"] = adap_has_final
            row["adaptive_pred_source"] = adap_pred_source
            row["adaptive_projection_used"] = adap_projection_used
            row["adaptive_projection_tokens"] = adap_projection_tokens
            row["adaptive_projection_latency_s"] = adap_projection_latency_s
            row["adaptive_raw"] = adap_raw

            records.append(row)
            inc_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            inc_fp.flush()
            local_done += 1

            if (local_done % 5 == 0) or (local_done == local_target):
                print(f"[rank {rank}] Processed {local_done}/{local_target}", flush=True)

        inc_fp.close()

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

        summary: Dict[str, Dict] = {
            "meta": {
                "timestamp_utc": ts,
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
            summary["fixed"][str(b)]["avg_projection_tokens"] = sum(
                r.get(f"fixed_{b}_projection_tokens", 0) for r in records
            ) / max(1, len(records))

        summary["adaptive"] = summarize(records, "adaptive")
        summary["adaptive"]["avg_verifier_calls"] = sum(r["adaptive_verifier_calls"] for r in records) / max(1, len(records))
        summary["adaptive"]["early_stop_rate"] = sum(r["adaptive_stopped_early"] for r in records) / max(1, len(records))
        summary["adaptive"]["final_rate"] = sum(r.get("adaptive_has_final", 0) for r in records) / max(
            1, len(records)
        )
        summary["adaptive"]["projection_rate"] = sum(
            r.get("adaptive_projection_used", 0) for r in records
        ) / max(1, len(records))
        summary["adaptive"]["avg_projection_tokens"] = sum(
            r.get("adaptive_projection_tokens", 0) for r in records
        ) / max(1, len(records))

        if len(args.budgets) >= 2:
            short_b = min(args.budgets)
            long_b = max(args.budgets)
            summary["overthinking"][f"fixed_{short_b}_vs_fixed_{long_b}"] = compute_oer(
                records, f"fixed_{short_b}", f"fixed_{long_b}"
            )
            summary["overthinking"][f"fixed_{short_b}_vs_adaptive"] = compute_oer(
                records, f"fixed_{short_b}", "adaptive"
            )

        json_path = os.path.join(args.results_dir, f"summary_{model_tag}_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(args.results_dir, f"per_sample_{model_tag}_{ts}.csv")
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
