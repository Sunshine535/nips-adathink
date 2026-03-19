# AdaThink: Adaptive Test-Time Compute Control with Matched-Cost Guarantees

## Overview

This project investigates **adaptive test-time compute allocation** for large language model reasoning. The core insight is that fixed-length chain-of-thought (CoT) wastes compute on easy problems (overthinking) and starves hard problems (underthinking). AdaThink learns a controller that dynamically decides when to stop, continue, verify, or branch during inference—achieving higher accuracy at equal or lower token cost.

**Target venue:** NeurIPS 2026

**Status:** ~88% complete (most mature of 6 research tracks)

## Research Questions

1. At matched average cost, does adaptive control improve accuracy over the best fixed budget?
2. Can adaptive control reduce overthinking error without harming average accuracy?
3. Does external verification improve net quality after counting verifier cost?

## Key Results (Current)

### Primary Scale: Qwen3.5-27B (GSM8K, 23 seeds, n=920)

| Controller | Acc | Tokens | Utility | vs Fixed256 |
|---|---|---|---|---|
| Fixed128 | 0.3370 | 128 | — | baseline |
| Fixed256 | 0.4620 | 286.3 | — | reference |
| Fixed512 | 0.4870 | 512 | — | +0.0250 acc |
| Template Controller | 0.6044 | 269.5 | 0.5254 | **+0.1424 acc, -16.8 tok** |
| Parametric Controller | 0.5696 | 242.7 | 0.4985 | **+0.1076 acc, -43.6 tok** |

Template controller 95% CI vs Fixed256: DeltaAcc [+0.120, +0.165], DeltaTokens [-23.4, -10.2]

### Second Scale: Qwen3-8B Think Mode (7 seeds, n=280)

| Setting | DeltaAcc | DeltaTokens | DeltaUtility | CI covers 0? |
|---|---|---|---|---|
| Quality-first (pen=0.0) | +0.136 | +86.7 | +0.110 | No (significant) |
| Near-cost (pen=0.8) | +0.046 | +11.7 | +0.043 | No (significant) |

## Method Architecture

```
Input Problem
    │
    ├── Budget Chunk 1 (128 tokens) → Extract answer, compute confidence
    │       │
    │       ├── Controller Decision: STOP → Output answer
    │       ├── Controller Decision: CONTINUE → Allocate next chunk
    │       └── Controller Decision: VERIFY → Run verifier, then decide
    │
    ├── Budget Chunk 2 (256 tokens) → Extract answer, compute confidence
    │       │
    │       └── ... (repeat until budget exhausted or STOP)
    │
    └── Final Answer + Cost Accounting
```

### Controller Types
- **Template Controller:** Leave-one-out cross-validation, per-budget accuracy lookup
- **Parametric Controller:** Linear hashed policy with cost penalty
- **Value-Based Controller:** Predicts per-budget correctness probability, optimizes expected utility

## Repository Structure

```
nips-adathink/
├── README.md                 # This file
├── PROPOSAL.md               # Falsifiable thesis and success criteria
├── PLAN.md                   # Stage-gate execution plan
├── EXPERIMENTS.md             # Full evaluation protocol and results log
├── PAPERS.md                 # Core references
├── README_RUN.md             # Runbook for reproducing experiments
├── environment.yml           # Conda environment spec
├── scripts/                  # All experiment and analysis scripts
│   ├── run_gsm8k_experiment.py                  # Main multi-GPU inference
│   ├── run_gsm8k_torchrun_4gpu.sh               # 4-GPU launcher
│   ├── run_gsm8k_policy_search.py               # Policy search experiments
│   ├── run_gsm8k_sc_baseline.py                 # Self-consistency baseline
│   ├── run_learned_budget_controller.py          # Linear controller
│   ├── run_template_budget_controller.py         # Template controller
│   ├── run_value_budget_controller.py            # Value-based controller
│   ├── run_parametric_budget_controller.py       # Parametric controller
│   ├── run_parametric_sweep.py                   # Hyperparameter sweep
│   ├── run_template_controller_significance.py   # Statistical significance
│   └── run_overthinking_aggregate.py             # Multi-seed aggregation
├── results/                  # All experiment outputs (JSON/CSV/logs)
├── shared_scripts/           # HF download and env check utilities
└── docs/                     # Reports, audit logs, status tracking
```

## Quick Start

### Environment Setup

```bash
conda env create -f environment.yml
conda activate nips_adathink
```

### Run Low-Cost Smoke Test (Qwen3-0.6B)

```bash
python scripts/run_gsm8k_experiment.py \
  --model Qwen/Qwen3-0.6B \
  --n_samples 20 --budgets 64 128 \
  --prompt_format chat --direct_answer \
  --enable_thinking --strict_final_only \
  --projection_on_missing_final --seed 42 --data_seed 101
```

### Run Full 4-GPU Experiment (Qwen3.5-27B)

```bash
bash scripts/run_gsm8k_torchrun_4gpu.sh \
  Qwen/Qwen3.5-27B 40 "64 128 256" 42 101
```

### Run Controller Analysis

```bash
python scripts/run_template_controller_significance.py \
  --manifest results/manifest_qwen35_27b_strict_23seed_20260228.json \
  --lambda_cost 0.15
```

## Hardware Requirements

- **Low-cost experiments:** 1x GPU with >=8GB VRAM (Qwen3-0.6B/1.7B)
- **Main experiments:** 4x A100 80GB (Qwen3.5-27B)
- **Storage:** ~50GB for model weights + ~2GB for results

## HuggingFace Mirror Configuration

```bash
export HF_ENDPOINT='https://hf-mirror.com'
export XDG_CACHE_HOME='/path/to/your/cache'
export HF_HOME="$XDG_CACHE_HOME/huggingface"
```

## Remaining Work for Submission

1. Ablation package: halting-only, no-branch, no-verifier
2. Out-of-domain validation (MATH or BBH)
3. Latency wall-clock analysis
4. Camera-ready figures and tables

## References

See [PAPERS.md](PAPERS.md) for full citation list with direct URLs.

## License

Research code for academic use.
