---
type: experiment
node_id: exp:mrsd_gsm8k_pilot
title: "MRSD pilot on GSM8K (8B, n=200)"
model: Qwen/Qwen3-8B
benchmark: GSM8K
n_samples: 200
seed: 42
engine: HF
status: completed
created_at: 2026-04-10
updated_at: 2026-04-10
---

# MRSD pilot on GSM8K

## Setup
- B1=256 (triage), B_think=512, B_answer=128, max_rounds=3
- Baselines: nothink@256, think@512, TOWN

## Key Results
| Method | Accuracy | Avg Tokens |
|--------|----------|------------|
| Nothink@256 | 89.0% | 140 |
| Think@512 | 56.9% | 477 |
| TOWN | 89.0% | 180 |
| 1-round MRSD | 93.5% | 181 |
| **MRSD (3-round)** | **94.0%** | 235 |

- +5.0pp over nothink@256 (significant, 95% CI: [2.0, 8.0])
- 10 unique wins, 0 unique losses vs nothink
- 89% of samples resolved at Stage 0 (triage)
- Convergence rate: 98% within 2 rounds

## Verdict
**Positive** — MRSD works on GSM8K where nothink has saturated at ~89%.

## Data Sources
- `results/mrsd_pilot/mrsd_Qwen3_8B_gsm8k_*.json`

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
