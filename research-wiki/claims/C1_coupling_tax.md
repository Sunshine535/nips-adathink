---
type: claim
node_id: claim:C1
title: "Non-thinking mode outperforms thinking mode at all budgets ≤2048"
status: supported
strength: strong
created_at: 2026-04-10
updated_at: 2026-04-10
---

# The Coupling Tax: nothink > think at all tested budgets

## Claim
Under fixed output-token budgets ≤2048, non-thinking mode outperforms thinking mode on GSM8K, MATH-500, and BBH with Qwen-family models.

## Evidence
- **GSM8K (n=1,319)**: +36.2pp at budget 512 (8B) — exp:coupling_tax_gsm8k
- **MATH-500 (n=500)**: +41.8pp at budget 1024 — exp:coupling_tax_math500
- **BBH (n=1,187)**: +33.3pp at budget 256 — exp:bbh
- Cross-scale: confirmed at 8B, 9B, 27B — exp:cross_scale
- Theory: truncation-waste decomposition (Proposition 1)

## Scope
Qwen-family models with explicit `<think>` mode, fixed output-token budgets, structured reasoning tasks.

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
