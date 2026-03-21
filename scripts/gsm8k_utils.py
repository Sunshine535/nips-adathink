#!/usr/bin/env python3
"""Shared utilities for all AdaThink GSM8K experiment scripts.

Centralises answer extraction, numeric comparison, prompt construction,
distributed helpers, model utilities, and compute-cost accounting so that
every script uses identical logic.
"""
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer\s*[:：]|the answer is\s*)([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


# ---------------------------------------------------------------------------
# Answer extraction (unified across all scripts)
# ---------------------------------------------------------------------------

def extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = NUM_RE.findall(text)
    return matches[-1] if matches else None


def extract_final_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = list(FINAL_ANSWER_RE.finditer(text))
    return matches[-1].group(1) if matches else None


def extract_boxed_number(text: str) -> Optional[str]:
    if not text:
        return None
    for m in reversed(list(BOXED_RE.finditer(text))):
        val = extract_last_number(m.group(1))
        if val is not None:
            return val
    return None


def has_explicit_final(text: str) -> bool:
    return extract_final_number(text) is not None


def parse_prediction(
    text: str, strict_final_only: bool = False
) -> Tuple[Optional[str], bool, str]:
    """Multi-strategy answer parser used by ALL experiment scripts."""
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
    """Convenience wrapper returning just the extracted number string."""
    pred, _, _ = parse_prediction(text, strict_final_only=False)
    return pred


# ---------------------------------------------------------------------------
# Numeric comparison
# ---------------------------------------------------------------------------

def to_float(num_str) -> Optional[float]:
    if num_str is None:
        return None
    s = str(num_str).replace(",", "").strip()
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
    return extract_last_number(answer_field)


def is_correct(pred, gold, tol: float = 1e-6) -> bool:
    p = to_float(pred)
    g = to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) <= tol * max(1.0, abs(g))


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

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

    if (
        prompt_format == "chat"
        and tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
    ):
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


# ---------------------------------------------------------------------------
# Distributed helpers
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


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

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
        missing = [
            name
            for name in required
            if not os.path.exists(os.path.join(snapshot_dir, name))
        ]
        return snapshot_dir, missing

    if os.path.exists(single_path):
        return snapshot_dir, []

    return snapshot_dir, ["model.safetensors.index.json (or model.safetensors)"]


def prepare_model_for_greedy(model) -> None:
    """Set generation_config to deterministic greedy defaults."""
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for k in ("top_p", "top_k", "temperature"):
            if hasattr(model.generation_config, k):
                delattr(model.generation_config, k)


# ---------------------------------------------------------------------------
# Generation with compute-cost accounting
# ---------------------------------------------------------------------------

@dataclass
class GenOutput:
    text: str
    new_tokens: int
    elapsed_s: float
    prefill_tokens: int = 0


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
    return GenOutput(
        text=text,
        new_tokens=int(gen_ids.shape[0]),
        elapsed_s=elapsed,
        prefill_tokens=in_len,
    )


# ---------------------------------------------------------------------------
# Compute-cost helpers
# ---------------------------------------------------------------------------

def estimate_total_flops_ratio(
    prefill_tokens_per_call: List[int],
    new_tokens_per_call: List[int],
    baseline_prefill: int,
    baseline_new: int,
) -> float:
    """Estimate FLOPs ratio of a multi-call strategy vs a single-call baseline.

    Assumes FLOPs ~ 2 * n_params * seq_len for each forward pass.
    Prefill is a single forward over the full input; decoding is one
    forward per generated token.

    Returns ratio > 1 means the strategy costs MORE than the baseline.
    """
    strategy_cost = 0
    for pf, nt in zip(prefill_tokens_per_call, new_tokens_per_call):
        strategy_cost += pf + nt
    baseline_cost = baseline_prefill + baseline_new
    if baseline_cost == 0:
        return float("inf")
    return strategy_cost / baseline_cost
