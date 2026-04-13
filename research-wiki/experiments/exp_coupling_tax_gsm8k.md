---
type: experiment
node_id: exp:coupling_tax_gsm8k
title: "Coupling tax measurement on GSM8K (8B, full-scale)"
model: Qwen/Qwen3-8B
benchmark: GSM8K
n_samples: 1319
seed: 42
engine: HF
status: completed
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Full-scale coupling tax measurement on GSM8K

## Setup
- Model: Qwen3-8B, greedy decoding
- Budgets: 128, 256, 512 (both thinking and non-thinking modes)
- n=1,319 (full GSM8K test set)

## Key Results
| Budget | Nothink Acc | Think Acc | Gap (Tax) |
|--------|-------------|-----------|-----------|
| 128 | 50.8% | 3.0% | +47.8pp |
| 256 | 87.5% | 18.0% | +69.5pp |
| 512 | 93.1% | 56.9% | +36.2pp |

- Natural-stop oracle: 99.0% accuracy when chains complete within budget
- Crossover not reached at budget ≤512

## Data Sources
- `results_kun/nothink_fullset/nothink_baseline_*` (nothink@128/256)
- `results_kun/fulltest/summary_gsm8k_Qwen3_8B_20260324_120316.json` (think@*)
- `results/gap_fill/8b_highbudget/` (nothink@512, think@512)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
