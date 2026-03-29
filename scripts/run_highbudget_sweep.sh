#!/bin/bash
# High-Budget Thinking Token Sweep
# Tests budgets 1024/2048/4096 to understand scaling behavior
# KEY EXPERIMENT: proves adaptive routing value increases at high budgets
set -euo pipefail
cd /workspace/nips-adathink
source .venv/bin/activate 2>/dev/null || true
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

OUTDIR="results/highbudget"
LOGDIR="results/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

echo "=== HIGH-BUDGET THINKING TOKEN SWEEP ==="
echo "Start: $(date)"
echo "Goal: Characterize accuracy/token scaling at high budgets (1024/2048/4096)"

# ============================================================
# Phase 1: High-budget sweep on 200-sample subset (fast)
# Estimate: 1024→~30min, 2048→~60min, 4096→~120min = ~3.5h total
# ============================================================
echo "[$(date)] Phase 1: 200-sample high-budget sweep"
python3 -u scripts/run_gsm8k_experiment.py \
    --benchmark gsm8k \
    --model Qwen/Qwen3-8B \
    --n_samples 200 \
    --seed 42 \
    --data_seed 42 \
    --budgets 1024 2048 4096 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$OUTDIR" \
    --skip_local_model_check
echo "[$(date)] Phase 1 DONE"

# ============================================================
# Phase 2: Full GSM8K (1319) at 1024 tokens
# This is the most important — fills the gap between 512 and 2048
# Estimate: ~3h (1319 samples × 1024 tokens)
# ============================================================
echo "[$(date)] Phase 2: Full GSM8K at 1024 tokens"
python3 -u scripts/run_gsm8k_experiment.py \
    --benchmark gsm8k \
    --model Qwen/Qwen3-8B \
    --n_samples 99999 \
    --seed 42 \
    --data_seed 42 \
    --budgets 1024 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$OUTDIR" \
    --skip_local_model_check
echo "[$(date)] Phase 2 DONE"

# ============================================================
# Phase 3: Non-thinking at high budgets (1024/2048) on 200 samples
# Comparison: how does non-thinking scale with budget?
# ============================================================
echo "[$(date)] Phase 3: Non-thinking high-budget comparison"
python3 -u scripts/run_nothink_baseline.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --budgets 1024 2048 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] Phase 3 DONE"

echo "=== ALL HIGH-BUDGET EXPERIMENTS COMPLETE: $(date) ==="
echo "Results in: $OUTDIR"
ls -la "$OUTDIR"/*.json 2>/dev/null
