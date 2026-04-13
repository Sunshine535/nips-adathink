---
type: experiment
node_id: exp:bbh
title: "Coupling tax on BIG-Bench Hard (5 subtasks)"
model: Qwen/Qwen3-8B
benchmark: BBH
n_samples: 1187
status: completed
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Coupling tax generalizes beyond math to logical/temporal/spatial reasoning

## Aggregate Results
| Budget | Nothink | Think | Gap |
|--------|---------|-------|-----|
| 256 | 49.9% | 16.6% | +33.3pp |
| 512 | 66.4% | 45.8% | +20.6pp |
| 1024 | 75.1% | 73.6% | +1.4pp |
| 2048 | 75.1% | 86.0% | -11.0pp |

Crossover: between 1024–2048.

## Per-Task Analysis
| Task | n | Tax@256 |
|------|---|---------|
| tracking_shuffled_objects | 250 | +88.0pp |
| boolean_expressions | 250 | +36.8pp |
| date_understanding | 250 | +32.4pp |
| logical_deduction | 250 | ~20pp |
| web_of_lies | 187 | ~15pp |

- Consistent across all 5 tasks — not driven by outliers
- Object tracking has extreme tax (88pp) because traces are invariably truncated

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
