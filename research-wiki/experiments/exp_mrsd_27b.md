---
type: experiment
node_id: exp:mrsd_27b
title: "MRSD on 27B — cascade failure"
model: Qwen3.5-27B
benchmark: GSM8K + MATH-500
n_samples: 200
status: partial
created_at: 2026-04-10
updated_at: 2026-04-10
---

# 27B MRSD — cascade hurts on both benchmarks

## GSM8K (completed, n=200)
| Method | Accuracy |
|--------|----------|
| Nothink@256 | 67.5% |
| TOWN | 62.5% |
| MRSD | 60.0% |

- MRSD **-7.5pp below nothink** — cascade hurts
- 80/200 escalated to thinking; only 3 (3.8%) answered correctly
- 97.5% accuracy on Stage 0 resolved samples

## MATH-500 (preliminary, 150/200)
| Method | Accuracy |
|--------|----------|
| Nothink@512 | 24.7% |
| MRSD | 20.7% |

- MRSD **-4.0pp below nothink** — cascade hurts again

## Root Cause
At B_think=512 (GSM8K) and B_think=1024 (MATH-500), 27B thinking traces are too heavily truncated. Partial reasoning is too shallow for decoupled extraction to recover signal.

## Implication
Net recovery condition fails at 27B for these budgets. Need much larger B_think (≥2048?) for 27B cascade to help.

## Data Sources
- `results/mrsd_27b/mrsd_Qwen3_5_27B_gsm8k_*.json`
- `results/mrsd_27b/checkpoint_150.json` (MATH-500 preliminary)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
