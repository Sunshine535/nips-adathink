#!/bin/bash
# Deploy DeepSeek-R1-Distill-Llama-8B full experiments
# Most critical experiment for paper: non-Qwen model validation
set -e

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
RESULTS_DIR="/workspace/nips-adathink/results/deepseek"
mkdir -p "$RESULTS_DIR"
cd /workspace/nips-adathink

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== DeepSeek-R1 Full Experiment Pipeline ======"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 1: GSM8K Full Dataset (n=1319), all budgets ==="
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
    --skip_local_model_check 2>&1 | tee "$RESULTS_DIR/gsm8k_full.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] GSM8K full complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 2: MATH500 Full Dataset (n=500), all budgets ==="
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
    --skip_local_model_check 2>&1 | tee "$RESULTS_DIR/math500_full.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] MATH500 full complete."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 3: Multi-seed subsets for template controller ==="
for DSEED in 101 202 303 404 505 606 707; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GSM8K subset seed=$DSEED (n=40)..."
    python3 -u scripts/run_experiment.py \
        --benchmark gsm8k \
        --model "$MODEL" \
        --n_samples 40 \
        --seed 11 \
        --data_seed $DSEED \
        --budgets 256 512 1024 \
        --enable_thinking \
        --strict_final_only \
        --projection_on_missing_final \
        --projection_max_tokens 16 \
        --prompt_format chat \
        --results_dir "$RESULTS_DIR" \
        --skip_local_model_check 2>&1 | tail -5
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Phase 4: MATH500 Multi-seed subsets ==="
for DSEED in 101 202 303 404 505 606 707; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] MATH500 subset seed=$DSEED (n=40)..."
    python3 -u scripts/run_experiment.py \
        --benchmark math500 \
        --model "$MODEL" \
        --n_samples 40 \
        --seed 11 \
        --data_seed $DSEED \
        --budgets 1024 2048 4096 \
        --enable_thinking \
        --strict_final_only \
        --projection_on_missing_final \
        --projection_max_tokens 16 \
        --prompt_format chat \
        --results_dir "$RESULTS_DIR" \
        --skip_local_model_check 2>&1 | tail -5
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ====== ALL DeepSeek EXPERIMENTS COMPLETE ======"
echo "DEEPSEEK_PIPELINE_COMPLETE $(date)" > "$RESULTS_DIR/.pipeline_done"
