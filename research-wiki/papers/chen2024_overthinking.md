---
type: paper
node_id: paper:chen2024_overthinking
title: "Do Not Think That Much for 2+3=? On the Overthinking of o1-Like LLMs"
authors: ["Xingyu Chen", "Jiahao Xu", "Tian Tian", "et al."]
year: 2024
venue: arXiv
external_ids:
  arxiv: "2412.21187"
tags: [overthinking, compute-waste, reasoning-efficiency, cot]
relevance: core
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Identifies compute waste in reasoning LLMs on easy problems ("overthinking").

## Problem / Gap
o1-like models spend excessive tokens reasoning about trivially easy problems.

## Method
Analyzes token usage vs difficulty across reasoning benchmarks.

## Key Results
- Easy problems receive disproportionately many reasoning tokens
- Significant compute wasted on problems that don't need reasoning

## Limitations / Failure Modes
- Focuses on "too many tokens" (efficiency), not "truncation when budget-limited" (our failure mode)
- Does not study what happens under fixed token budgets

## Reusable Ingredients
- Framing of compute waste in reasoning
- Easy/hard distinction

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
Complementary work. They study overthinking (too many tokens on easy problems); we study the coupling tax (truncation waste on hard problems under budget constraints). Different failure modes of the same paradigm.
