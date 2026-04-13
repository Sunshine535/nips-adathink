---
type: experiment
node_id: exp:deepseek_replication
title: "DeepSeek-R1-Distill-Llama-8B full replication"
status: running
model: DeepSeek-R1-Distill-Llama-8B
benchmark: GSM8K + MATH-500
hardware: A100 (GSM8K) + H800 (MATH-500)
seed: 42
created_at: 2026-04-12
updated_at: 2026-04-12
---

# DeepSeek cross-family replication

## Objective
Test whether coupling tax and split-budget recovery generalize beyond Qwen to DeepSeek-R1 model family. This is the CRITICAL gap to best paper (reviewer W1).

## Setup
### Phase 1: Think/Nothink baselines
- **A100 server** (216.81.245.138:17490): GSM8K n=1319, budgets {256,512,1024,2048}, think+nothink
- **H800 server** (222.223.106.147:30022): MATH-500 n=500, budgets {512,1024,2048,4096}, think+nothink
- Nothink simulation: inject `</think>` into prompt (DeepSeek-R1 lacks native enable_thinking=False)

### Phase 2: IRIS on DeepSeek (depends on Phase 1)
- GSM8K IRIS@1024, IRIS@2048
- MATH-500 IRIS@2048
- Script: `scripts/deploy_deepseek_iris.sh`

## Expected Results
If coupling tax generalizes:
1. Nothink should dominate think at budgets ≤2048
2. F_L(b) decomposition should predict accuracy within ~3pp
3. IRIS should recover accuracy over standard think mode

## Status (2026-04-12 18:50 UTC)
- A100: Running GSM8K nothink@256, ~86/1319 samples processed
- H800: Model loaded, starting MATH-500 inference
- Estimated total: 30-40h (A100), 15-20h (H800)

## Source files
- `scripts/run_deepseek_crossmodel.py`
- `scripts/deploy_deepseek_replication.sh`
- `scripts/deploy_deepseek_iris.sh`
- Output: `results/deepseek_crossmodel/`

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
