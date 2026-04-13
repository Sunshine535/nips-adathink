---
type: paper
node_id: paper:deepseekr1_2025
title: "DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning"
authors: ["Daya Guo", "Dejian Yang", "He Zhang", "et al."]
year: 2025
venue: arXiv
external_ids:
  arxiv: "2501.12948"
tags: [reasoning-llm, rl, thinking-mode, distillation]
relevance: core
created_at: 2026-04-10
updated_at: 2026-04-10
---

# RL-trained reasoning model with explicit thinking traces; distilled variants show similar coupling tax.

## Problem / Gap
How to train LLMs that reason like o1 in an open-source setting.

## Method
RL-based training to produce structured reasoning traces. Distilled into smaller models (8B, 14B, 70B).

## Key Results
- Competitive with o1 on math/code reasoning
- Distilled models retain reasoning capability
- DeepSeek-R1-Distill-Llama-8B used in our cross-architecture validation

## Limitations / Failure Modes
- Verbose reasoning chains — vulnerable to the coupling tax we identify
- No analysis of budget-constrained performance

## Reusable Ingredients
- DeepSeek-R1-Distill-Llama-8B as cross-architecture test model

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
Cross-architecture validation target. Our appendix shows DeepSeek-R1 exhibits consistent utilization collapse and natural-stop patterns, though we lack matched nothink baselines.
