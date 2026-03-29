#!/usr/bin/env python3
"""
Modified inference script: Save model internals (logits, hidden states)
This is the ONLY way to get strong enough signals for SOTA performance
"""
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_with_internals(model, tokenizer, question, max_tokens=256):
    """Generate with full internal state capture."""
    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        output_hidden_states=True,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=False
    )

    # Extract internals
    logits = torch.stack(outputs.scores)  # [seq_len, vocab_size]
    hidden = outputs.hidden_states  # Tuple of hidden states

    # Compute uncertainty from logits
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

    # Compute hidden state variance (instability indicator)
    last_hidden = torch.stack([h[-1] for h in hidden[-5:]])  # Last 5 layers
    hidden_var = last_hidden.var(dim=0).mean().item()

    return {
        'output': tokenizer.decode(outputs.sequences[0]),
        'entropy': entropy,
        'hidden_var': hidden_var,
        'max_prob': probs.max(dim=-1)[0].mean().item()
    }

# This script needs to be run on GPU servers to regenerate all data
print("Script ready. Deploy to GPU servers to collect model internals.")
