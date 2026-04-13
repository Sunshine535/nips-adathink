---
type: experiment
node_id: exp:mrsd_math500_pilot
title: "MRSD pilot on MATH-500 (8B, n=200)"
model: Qwen/Qwen3-8B
benchmark: MATH-500
n_samples: 200
seed: 42
engine: HF
status: completed
created_at: 2026-04-10
updated_at: 2026-04-10
---

# MRSD pilot on MATH-500 — mixed results

## Setup
- B1=512 (triage), B_think=1024, B_answer=256, max_rounds=3

## Key Results (rescored with current parser)
| Method | Accuracy | Avg Tokens |
|--------|----------|------------|
| Nothink@512 | 42.0% (MRSD run) / 47.5% (split-budget run) | 418 |
| **Nothink@1024** | **69.5%** | 600 |
| Think@1024 | 19.5% | 1024 |
| TOWN@1024 | 69.5% | 877 |
| Best split@1024 | 55.0% | 816 |
| 1-round MRSD | 55.5% | 995 |
| MRSD (3-round) | 61.0% | 1823 |

### Critical finding
**nothink@1024 = 69.5% >> MRSD = 61.0%** at 3× fewer tokens.

### Parser inconsistency
- MRSD pilot nothink@512 = 42.0% (rescored from 40.5%)
- Split-budget nothink@512 = 47.5% (different inference run)
- 13 mismatches between experiments on same samples

### Sample-size caveat
- Full-scale nothink@1024 (n=500) = 59.8% — closer to MRSD's 61.0%
- Pilot n=200 overestimates nothink@1024 (69.5% vs 59.8%)

## Verdict
**Negative for MRSD vs compute-matched baseline on pilot.** Potentially neutral on full-scale (59.8% vs ~61%). Budget-scaling ceiling hypothesis: MRSD is ineffective when nothink accuracy is still improving with budget.

## Root Cause Analysis (PENDING)
- B_think=1024 too small? 100% of escalated samples hit ceiling
- Need B_think=2048 experiment
- Need full-scale MRSD (n=500) for definitive comparison

## Data Sources
- `results/mrsd_pilot/mrsd_Qwen3_8B_math500_b1512_bt1024_ba256_r3_20260409_043123.json`
- `results/split_budget/split_budget_Qwen3_8B_math500_n200_B512_1024_20260409_084433.json`
- `results/mrsd_pilot/mrsd_math500_rescored.json`

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
