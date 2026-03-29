#!/bin/bash
# Server B: Qwen3.5-27B full-dataset validation
# Priority: Honest feature controller on 27B full datasets

set -e

export HF_ENDPOINT=https://hf-mirror.com
cd /workspace/nips-adathink
source .venv/bin/activate

MODEL="Qwen/Qwen3.5-27B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Server B: Qwen3.5-27B Full Dataset Validation ==="
echo "Model: $MODEL"
echo "Start: $(date)"

# GSM8K full (n=1319)
echo "Running GSM8K full..."
python scripts/run_gsm8k_experiment.py \
  --model_name_or_path "$MODEL" \
  --benchmark gsm8k \
  --n_samples 1319 \
  --seed 11 \
  --data_seed 0 \
  --budgets 128 256 512 \
  --adaptive_chunks 128 128 128 \
  --adaptive_max_total 512 \
  --enable_thinking \
  --prompt_format chat \
  --direct_answer \
  --use_verifier \
  --strict_final_only \
  --projection_on_missing_final \
  --output_dir results/fulltest_27b \
  2>&1 | tee results/fulltest_27b/gsm8k_full_${TIMESTAMP}.log

# MATH500 full
echo "Running MATH500 full..."
python scripts/run_gsm8k_experiment.py \
  --model_name_or_path "$MODEL" \
  --benchmark math500 \
  --n_samples 500 \
  --seed 11 \
  --data_seed 0 \
  --budgets 512 1024 2048 \
  --adaptive_chunks 512 512 512 \
  --adaptive_max_total 2048 \
  --enable_thinking \
  --prompt_format chat \
  --direct_answer \
  --use_verifier \
  --strict_final_only \
  --projection_on_missing_final \
  --output_dir results/fulltest_27b \
  2>&1 | tee results/fulltest_27b/math500_full_${TIMESTAMP}.log

echo "=== Server B Complete: $(date) ==="
echo "Results in: results/fulltest_27b/"
