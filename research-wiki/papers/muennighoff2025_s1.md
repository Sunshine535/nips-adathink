---
type: paper
node_id: paper:muennighoff2025_s1
title: "s1: Simple test-time scaling"
authors: ["Niklas Muennighoff", "Zitong Yang", "Weijia Shi", "et al."]
year: 2025
venue: arXiv
external_ids:
  arxiv: "2501.19393"
tags: [test-time-compute, budget-control, reasoning, simple-baseline]
relevance: core
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Simple budget-forcing via token truncation + "Wait" injection for test-time scaling.

## Problem / Gap
How to control reasoning length without retraining.

## Method
Budget forcing: truncate thinking tokens at a fixed budget, optionally inject "Wait" to extend. Simple but effective.

## Key Results
- Budget forcing works as a simple baseline for controlling reasoning length
- Demonstrates that truncation is a real failure mode (supports our finding)

## Limitations / Failure Modes
- Does not analyze the coupling between reasoning and answer under truncation
- Truncation is treated as a lever, not as a pathology

## Reusable Ingredients
- Budget forcing methodology (we use similar approach)
- The "Wait" injection idea

## Connections
[AUTO-GENERATED from graph/edges.jsonl]

## Relevance to This Project
Directly related. They control reasoning budget; we show what goes wrong when reasoning and answering share that budget. Our coupling tax analysis provides the theoretical explanation for why their budget-forcing approach loses accuracy.
