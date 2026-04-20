#!/usr/bin/env python3
"""Online chunk-by-chunk Stage 2 generation for IRIS — deployment-faithful accounting.

The original `generate_adaptive_thinking` in `run_iris.py` generates the full
`max_think_tokens` sequence once and then post-hoc locates an optimal stopping
point. That is *analysis-faithful* (the accuracy curves are valid) but not
*deployment-faithful*: the reported savings are truncation accounting of an
already-generated trace, not compute actually avoided.

This module reimplements Stage 2 as a true online chunk-by-chunk loop using a
KV cache. At each chunk boundary we evaluate the IRIS stopping criteria and
halt generation immediately if they are met. As a result,
`n_tokens_generated == n_tokens_used` by construction, and wall-clock / FLOPs
savings are real.

Used by `run_iris.py` when `--online_stage2` is passed.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Set

import numpy as np
import torch


def _resolve_think_end_ids(tokenizer) -> Set[int]:
    """Token IDs that indicate natural end of thinking."""
    ids = set()
    for candidate in ("</think>", "<|/think|>", "</think>\n"):
        toks = tokenizer.encode(candidate, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return ids


def generate_adaptive_thinking_online(
    model,
    tokenizer,
    prompt: str,
    max_think_tokens: int,
    chunk_size: int = 32,
    tau_h: float = 1.5,
    tau_s: float = 50.0,
    min_chunks: int = 2,
) -> Dict:
    """Online chunk-by-chunk generation with adaptive early stopping.

    Stops early (saving both tokens *and* compute) when:
      - model emits `</think>` (natural stop), OR
      - mean per-token entropy within a chunk < tau_h AND hidden-state L2 drift
        between the current and previous chunk < tau_s, after min_chunks chunks.

    Otherwise runs to `max_think_tokens` (budget exhausted).

    Returns the same dict shape as `generate_adaptive_thinking` but with
    `n_tokens_generated == n_tokens_used` (we never generate past the stop
    point) and an `online=True` marker.
    """
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    think_end_ids = _resolve_think_end_ids(tokenizer)

    gen_ids: list[int] = []
    chunk_entropies: list[float] = []
    chunk_stabilities: list[float] = []
    per_token_entropy: list[float] = []
    prev_chunk_h: Optional[torch.Tensor] = None

    actual_stop_token: Optional[int] = None
    stop_reason = "budget_exhausted"
    natural_stop_token: Optional[int] = None
    iris_stop_chunk: Optional[int] = None

    t_start = time.perf_counter()

    # Initial forward — get KV cache and first next-token logits
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    past = out.past_key_values
    last_logits = out.logits[0, -1]

    n_chunks_cap = (max_think_tokens + chunk_size - 1) // chunk_size
    for chunk_idx in range(n_chunks_cap):
        this_chunk_entropies: list[float] = []
        chunk_end_h: Optional[torch.Tensor] = None

        # Generate up to chunk_size tokens in this chunk (greedy)
        for _ in range(chunk_size):
            if len(gen_ids) >= max_think_tokens:
                break

            probs = torch.softmax(last_logits.float(), dim=-1)
            H_step = -(probs * torch.log(probs + 1e-10)).sum().item()
            per_token_entropy.append(H_step)
            this_chunk_entropies.append(H_step)

            next_id = int(last_logits.argmax())
            gen_ids.append(next_id)

            if next_id in think_end_ids and natural_stop_token is None:
                natural_stop_token = len(gen_ids) - 1
                stop_reason = "natural_stop"
                actual_stop_token = len(gen_ids)
                break

            # Forward the single new token through the KV cache
            next_input = torch.tensor([[next_id]], device=device)
            with torch.no_grad():
                out = model(
                    input_ids=next_input,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            past = out.past_key_values
            last_logits = out.logits[0, -1]
            chunk_end_h = out.hidden_states[-1][0, -1].float().detach().cpu()

        if stop_reason == "natural_stop":
            break

        # Chunk-level aggregates
        H_chunk = float(np.mean(this_chunk_entropies)) if this_chunk_entropies else float("inf")
        chunk_entropies.append(H_chunk)

        if prev_chunk_h is not None and chunk_end_h is not None:
            S_chunk = torch.norm(chunk_end_h - prev_chunk_h, p=2).item()
        else:
            S_chunk = float("inf")
        chunk_stabilities.append(S_chunk)
        prev_chunk_h = chunk_end_h

        # IRIS criterion
        if (
            (chunk_idx + 1) >= min_chunks
            and H_chunk < tau_h
            and S_chunk < tau_s
        ):
            iris_stop_chunk = chunk_idx
            stop_reason = "iris_entropy"
            actual_stop_token = len(gen_ids)
            break

        if len(gen_ids) >= max_think_tokens:
            break

    if actual_stop_token is None:
        actual_stop_token = len(gen_ids)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t_start

    # Free KV cache
    del past

    thinking_ids = gen_ids[:actual_stop_token]
    thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    n_tokens_generated = len(gen_ids)  # by construction equals n_tokens_used
    hit_budget = n_tokens_generated >= int(max_think_tokens * 0.95) and actual_stop_token >= n_tokens_generated

    return {
        "thinking_text": thinking_text,
        "full_text": full_text,
        "n_tokens_generated": n_tokens_generated,
        "n_tokens_used": actual_stop_token,
        "tokens_saved": max(0, max_think_tokens - n_tokens_generated),
        "savings_ratio": round(
            max(0, max_think_tokens - n_tokens_generated) / max(1, max_think_tokens),
            4,
        ),
        "stop_reason": stop_reason,
        "iris_stop_chunk": iris_stop_chunk,
        "natural_stop_token": natural_stop_token,
        "hit_budget": hit_budget,
        "elapsed_s": round(elapsed, 3),
        "per_token_entropy": per_token_entropy,
        "chunk_entropies": chunk_entropies,
        "chunk_stabilities": chunk_stabilities,
        "online": True,
    }
