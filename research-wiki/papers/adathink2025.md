---
type: paper
node_id: paper:adathink2025
title: "AdaThink: Adaptive Thinking Makes LLM Reasoning More Efficient"
authors: ["Jiahao Ji", "Jiaxin Luo", "Yuze Chen", "Sungmin Choi", "Timo Pfister"]
year: 2025
venue: arXiv
external_ids:
  arxiv: "2505.05345"
tags: [adaptive-reasoning, budget-allocation, efficiency, thinking-mode]
relevance: related
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Adaptive allocation of thinking budget based on problem difficulty.

## Problem / Gap
Fixed thinking budgets are wasteful — easy problems need fewer tokens than hard ones.

## Method
Adaptive budget allocation based on estimated difficulty.

## Key Results
- More efficient reasoning through difficulty-adaptive budgets
- Shares our insight that fixed budgets are suboptimal

## Limitations / Failure Modes
- Does not address the coupling tax (reasoning and answering still share budget)
- Does not propose decoupling reasoning from answering

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
Related concurrent work. They optimize budget allocation; we show that even with optimal allocation, coupling reasoning and answering in one stream is fundamentally inefficient.
