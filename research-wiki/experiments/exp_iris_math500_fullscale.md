---
type: experiment
node_id: exp:iris_math500_fullscale
title: "IRIS full-scale on MATH-500 (n=500)"
status: completed
model: Qwen3-8B
benchmark: MATH-500
hardware: H800
seed: 42
created_at: 2026-04-12
updated_at: 2026-04-12
---

# IRIS full-scale on MATH-500

## Setup
- Model: Qwen/Qwen3-8B
- Benchmark: MATH-500 (n=500)
- Configurations: IRIS@2048, IRIS@4096
- B1=512 (nothink triage), B_answer=256
- Hardware: H800 (same as nothink@1024 baseline)
- Seed: 42

## Results
| Config | Acc | 95% CI | Avg Tokens | Natural Stop Rate |
|--------|-----|--------|------------|-------------------|
| IRIS@2048 | 67.2% | [63.0, 71.2] | 1573 | 8.8% (of escalated) |
| IRIS@4096 | 74.0% | [70.0, 77.7] | 2401 | 44.7% (of escalated) |

### Stage breakdown (IRIS@4096)
- Stage 0 triage: 216/500 resolved at 91.7%
- Stage 1 escalated: 284 samples
- Natural completions: 127/284 (44.7%) at 71.7% accuracy
- Truncated + decoupled: 157/284 at 51.6% accuracy

### Cross-budget gain
IRIS@2048→4096: +6.8pp (McNemar p=0.0004)

### Pilot reproduction
First 200 samples match pilot exactly: 73.0%@2048, 78.5%@4096.

## Source files
- `results/iris/iris_math500_B2048_n500_*.json`
- `results/iris/iris_math500_B4096_n500_*.json`

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
