#!/bin/bash
# Run DeepSeek-R1-Distill-Llama-8B experiment on GSM8K and MATH500
# Usage: CUDA_VISIBLE_DEVICES=X bash scripts/run_deepseek_experiment.sh

set -e

export HF_ENDPOINT=https://hf-mirror.com
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
RESULTS_DIR="/workspace/nips-adathink/results/deepseek"
mkdir -p "$RESULTS_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting DeepSeek-R1-Distill-Llama-8B experiments"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running GSM8K with budgets 256 512 1024..."
python3 -u scripts/run_experiment.py \
    --benchmark gsm8k \
    --model "$MODEL" \
    --n_samples 99999 \
    --seed 11 \
    --data_seed 42 \
    --budgets 256 512 1024 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$RESULTS_DIR" \
    --skip_local_model_check

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GSM8K complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running MATH500 with budgets 1024 2048 4096..."
python3 -u scripts/run_experiment.py \
    --benchmark math500 \
    --model "$MODEL" \
    --n_samples 99999 \
    --seed 11 \
    --data_seed 42 \
    --budgets 1024 2048 4096 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$RESULTS_DIR" \
    --skip_local_model_check

echo "[$(date '+%Y-%m-%d %H:%M:%S')] MATH500 complete."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All DeepSeek experiments complete."
