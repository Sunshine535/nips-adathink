---
type: claim
node_id: claim:C2
title: "Coupling tax worsens with model size (inverse scaling)"
status: supported
strength: moderate
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Inverse scaling: larger models have larger coupling tax

## Claim
The coupling tax amplifies ~2.1× from 8B to 9B/27B, driven by longer reasoning chains.

## Evidence
- GSM8K@512: 36.2pp (8B) → 77.7pp (9B) → 77.2pp (27B) — exp:cross_scale
- Chain completion rate: 37.4% (8B) → 0.7% (27B) at budget 512
- Theory: Proposition 2 (stochastic dominance)

## Caveats
- Jump is 8B→9B, not gradual — 9B and 27B are nearly identical
- All within Qwen family (architecture confound possible)
- "Associated with" longer chains, not definitively "caused by"

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
