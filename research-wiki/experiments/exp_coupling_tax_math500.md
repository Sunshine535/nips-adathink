---
type: experiment
node_id: exp:coupling_tax_math500
title: "Coupling tax measurement on MATH-500 (8B, full-scale)"
model: Qwen/Qwen3-8B
benchmark: MATH-500
n_samples: 500
seed: 42
engine: HF
status: completed
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Full-scale coupling tax on MATH-500

## Key Results
| Budget | Nothink Acc | Think Acc | Gap (Tax) |
|--------|-------------|-----------|-----------|
| 256 | 16.6% | 4.2% | +12.4pp |
| 512 | 40.6% | 6.2% | +34.4pp |
| 1024 | 59.8% | 18.0% | +41.8pp |
| 2048 | 64.4% | 44.0% | +20.4pp |

- Crossover not reached at budget ≤2048
- Tax is LARGER on harder benchmark (41.8pp vs 36.2pp at comparable budgets)

## Data Sources
- Appendix Table (n=500, full MATH-500)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
