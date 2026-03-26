#!/bin/bash
# Deploy Qwen3.5-27B full-dataset experiments
# Second priority: validate 27B results at full scale
set -e

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME"

MODEL="Qwen/Qwen3.5-27B"
RESULTS_DIR="/workspace/nips-adathink/results/fulltest_27b"
mkdir -p "$RESULTS_DIR"
cd /workspace/nips-adathink

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== Qwen3.5-27B Full-Dataset Pipeline ======"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 1: GSM8K Full Dataset (n=1319, 27B) ==="
python3 -u scripts/run_experiment.py \
    --benchmark gsm8k \
    --model "$MODEL" \
    --n_samples 99999 \
    --seed 11 \
    --data_seed 42 \
    --budgets 128 256 512 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$RESULTS_DIR" \
    --skip_local_model_check 2>&1 | tee "$RESULTS_DIR/gsm8k_27b_full.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GSM8K-27B full complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 2: MATH500 Full Dataset (n=500, 27B) ==="
python3 -u scripts/run_experiment.py \
    --benchmark math500 \
    --model "$MODEL" \
    --n_samples 99999 \
    --seed 11 \
    --data_seed 42 \
    --budgets 2048 4096 8192 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$RESULTS_DIR" \
    --skip_local_model_check 2>&1 | tee "$RESULTS_DIR/math500_27b_full.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] MATH500-27B full complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== ALL 27B FULL-DATASET EXPERIMENTS COMPLETE ======"
echo "27B_FULLTEST_PIPELINE_COMPLETE $(date)" > "$RESULTS_DIR/.pipeline_done"
