---
type: paper
node_id: paper:nvidia2025_pareto
title: "Reasoning Strategy Optimization: A Pareto Framework for Test-Time Compute"
authors: ["Amrith Setlur", "Chirag Nagpal", "Adam Fisch", "et al."]
year: 2025
venue: arXiv
external_ids:
  arxiv: "2503.04474"
tags: [pareto, test-time-compute, strategy-selection, reasoning]
relevance: related
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Pareto-optimal strategy selection for test-time compute allocation.

## Problem / Gap
Given a compute budget, which reasoning strategy is Pareto-optimal?

## Method
Formal Pareto framework for comparing reasoning strategies at different compute levels.

## Key Results
- Different strategies are optimal at different compute budgets
- Cheap strategies (direct answering) dominate at low budgets

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
Our coupling tax provides the mechanistic explanation for why direct answering dominates at low budgets — truncation waste renders thinking mode strictly inferior below the crossover.
