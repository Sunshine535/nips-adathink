---
type: claim
node_id: claim:C8
title: "Coupling tax generalizes to non-mathematical reasoning (BBH)"
status: supported
strength: strong — full-scale n=1187 across 5 diverse tasks
created_at: 2026-04-12
updated_at: 2026-04-12
---

# Coupling tax generalizes beyond mathematics

## Claim
The thinking tax (nothink > think at matched budgets) persists on non-mathematical reasoning tasks, with task-dependent crossover budgets ranging from ~512 to >2048.

## Evidence (Qwen3-8B, n=1187, 5 BBH tasks)

### Aggregate
- think@256=15.8% vs nothink@256=49.1% → **tax=33.3pp**
- think@2048=86.0% — crossover between 1024-2048

### Per-task variation
| Task | Tax@256 | Crossover |
|------|---------|-----------|
| boolean_expressions | +36.8pp | ~512 |
| object_tracking | +88.0pp | >2048 |
| causal_judgement | persists to 2048 | >2048 |
| date_understanding | ~40pp@256 | ~1024 |
| logical_deduction | ~30pp@256 | ~1024 |

### Key insight
Crossover budget varies with task difficulty — matches theory prediction that b* depends on chain-length distribution F_L.

## Source
- exp:bbh (full-scale n=1187)
- results_kun/bbh_full/summary_bbh_Qwen3_8B_20260403_064520.json

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
