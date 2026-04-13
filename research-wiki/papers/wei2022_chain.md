---
type: paper
node_id: paper:wei2022_chain
title: "Chain-of-thought prompting elicits reasoning in large language models"
authors: ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter", "Fei Xia", "Ed Chi", "Quoc Le", "Denny Zhou"]
year: 2022
venue: NeurIPS
external_ids:
  arxiv: null
  doi: null
tags: [cot, reasoning, prompting, foundational]
relevance: core
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Foundational paper establishing CoT prompting as a reasoning paradigm for LLMs.

## Problem / Gap
LLMs struggle with multi-step reasoning tasks; few-shot prompting alone is insufficient.

## Method
Add intermediate reasoning steps ("chain of thought") to few-shot exemplars. The model then generates step-by-step reasoning before the final answer.

## Key Results
- Dramatic accuracy improvements on arithmetic, commonsense, and symbolic reasoning benchmarks
- Emergent capability: CoT only helps at sufficient model scale (≥100B parameters)
- GSM8K: 8-shot CoT achieves 57% with PaLM 540B (vs 18% standard prompting)

## Assumptions
- Longer reasoning traces = better answers (the "think longer" hypothesis)
- No analysis of what happens when reasoning is truncated or budget-limited

## Limitations / Failure Modes
- No analysis of token efficiency or cost-accuracy tradeoffs
- Assumes unbounded generation length — does not consider truncation
- The "think longer = better" assumption breaks under budget constraints (our paper's core finding)

## Reusable Ingredients
- CoT prompting paradigm
- GSM8K as reasoning benchmark

## Open Questions
- What happens when CoT chains are truncated? (→ our coupling tax finding)
- Is the reasoning trace always necessary, or can the model produce answers directly?

## Claims
- claim:C1 (CoT improves accuracy — challenged by our budget-constrained results)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
This is the foundational paper we challenge. Our coupling tax shows that CoT's "think longer" promise breaks down under budget constraints — the very regime most production deployments operate in.
