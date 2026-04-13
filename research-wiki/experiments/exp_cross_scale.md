---
type: experiment
node_id: exp:cross_scale
title: "Cross-scale coupling tax (8B, 9B, 27B on GSM8K)"
model: Qwen3-8B, Qwen3.5-9B, Qwen3.5-27B
benchmark: GSM8K
n_samples: 1319
status: completed
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Inverse scaling: coupling tax worsens with model size

## Key Results (GSM8K, budget=512)
| Model | Nothink | Think | Tax |
|-------|---------|-------|-----|
| 8B | 93.1% | 56.9% | 36.2pp |
| 9B | 93.2% | 15.5% | 77.7pp |
| 27B | 95.5% | 18.3% | 77.2pp |

- 2.1× amplification from 8B to 9B/27B
- 8B→9B is the big jump; 9B→27B is flat
- Mechanism: larger models generate longer chains that overflow budget more severely
- 0.7% of 27B traces complete within budget 512 (vs 37.4% for 8B)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
