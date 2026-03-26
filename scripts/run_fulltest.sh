#!/usr/bin/env bash
set -euo pipefail

# Full-benchmark evaluation for AdaThink
# Uses Qwen3-8B with thinking mode on GSM8K (1319 questions) and MATH500 (500 questions)
# Runs on single GPU with device_map auto for multi-GPU setups

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export CUDA_VISIBLE_DEVICES=${1:-0}

MODEL="Qwen/Qwen3-8B"
RESULTS_DIR="/workspace/nips-adathink/results/fulltest"
mkdir -p "$RESULTS_DIR"

echo "=== Full-benchmark evaluation ==="
echo "Model: $MODEL"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Results: $RESULTS_DIR"

cd /workspace/nips-adathink/scripts

echo "--- GSM8K full test (1319 questions) ---"
python3 -u run_experiment.py \
    --benchmark gsm8k \
    --model "$MODEL" \
    --n_samples 2000 \
    --seed 11 \
    --data_seed 0 \
    --budgets 128 256 512 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --skip_local_model_check \
    --single_process_device_map_auto \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/gsm8k_full.log"

echo "--- MATH500 full test (500 questions) ---"
python3 -u run_experiment.py \
    --benchmark math500 \
    --model "$MODEL" \
    --n_samples 500 \
    --seed 11 \
    --data_seed 0 \
    --budgets 2048 4096 8192 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --skip_local_model_check \
    --single_process_device_map_auto \
    --results_dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/math500_full.log"

echo "=== Full-test evaluation complete ==="
