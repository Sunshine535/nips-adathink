---
type: experiment
node_id: exp:town_math500_fullscale
title: "TOWN full-scale on MATH-500 (n=500)"
status: completed
model: Qwen3-8B
benchmark: MATH-500
hardware: H800
seed: 42
created_at: 2026-04-12
updated_at: 2026-04-12
---

# TOWN full-scale on MATH-500

## Setup
- Model: Qwen/Qwen3-8B
- Benchmark: MATH-500 (n=500)
- Configurations: TOWN@2048, TOWN@4096
- B1=512 (nothink triage), thinking fallback at B2
- Hardware: H800
- Seed: 42

## Results
| Config | Acc | 95% CI | Avg Tokens |
|--------|-----|--------|------------|
| TOWN@2048 | 55.0% | [50.6, 59.3] | 1590 |
| TOWN@4096 | 71.8% | [67.7, 75.6] | 2565 |

### IRIS vs TOWN comparison
| Budget | IRIS | TOWN | Gap | p-value |
|--------|------|------|-----|---------|
| 2048 | 67.2% | 55.0% | +12.2pp | <10^-6 |
| 4096 | 74.0% | 71.8% | +2.2pp | 0.28 |

Gap narrowing at B4096 is predicted by coupling-tax theory: as truncation decreases (44.7% natural stop at B4096), decoupled answering provides less additional value.

## Source files
- `results/town/town_math500_B2048_n500_*.json`
- `results/town/town_math500_B4096_n500_*.json`

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
