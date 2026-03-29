#!/bin/bash
# Server A: DeepSeek-R1-Distill-Llama-8B
# Priority: Cross-family validation (生死线实验)

set -e

export HF_ENDPOINT=https://hf-mirror.com
cd /workspace/nips-adathink
source .venv/bin/activate

MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Server A: DeepSeek Cross-Family Validation ==="
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
  --output_dir results/deepseek \
  2>&1 | tee results/deepseek/gsm8k_full_${TIMESTAMP}.log

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
  --output_dir results/deepseek \
  2>&1 | tee results/deepseek/math500_full_${TIMESTAMP}.log

echo "=== Server A Complete: $(date) ==="
echo "Results in: results/deepseek/"
