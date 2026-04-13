---
type: paper
node_id: paper:snell2024_scaling
title: "Scaling LLM test-time compute optimally can be more effective than scaling model parameters"
authors: ["Charlie Snell", "Jaehoon Lee", "Kelvin Xu", "Aviral Kumar"]
year: 2024
venue: arXiv
external_ids:
  arxiv: "2408.03314"
tags: [test-time-compute, scaling, budget-allocation, reasoning]
relevance: core
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Test-time compute scaling (thinking longer) can substitute for model parameter scaling.

## Problem / Gap
How to optimally allocate test-time compute across different strategies (revision, search, verification).

## Method
Systematic comparison of test-time compute strategies: best-of-N, step-by-step verification, sequential revision. Analyze compute-optimal allocation.

## Key Results
- Test-time compute scaling can outperform 14× larger models on some tasks
- Optimal strategy depends on problem difficulty
- Easy problems: direct answer is compute-optimal; hard problems: benefit from more compute

## Assumptions
- Focus on compute allocation, not on what happens when compute is artificially limited
- Assumes compute can be freely allocated

## Limitations / Failure Modes
- Does not analyze the truncation failure mode
- Does not study what happens with fixed output-token budgets (our regime)

## Reusable Ingredients
- Compute-optimal allocation framework
- Easy/hard problem distinction (related to our triage concept)

## Open Questions
- What is the optimal test-time strategy under *fixed* token budgets? (→ our paper)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
Core related work. We study the complementary regime: what happens when test-time compute is *constrained* rather than freely allocated. Our coupling tax shows that naively allocating more compute to thinking can be strictly worse than direct answering.
