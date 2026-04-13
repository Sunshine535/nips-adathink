#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B IRIS experiments
# PURPOSE: Test split-budget generation on non-Qwen model family
# PREREQUISITE: deploy_deepseek_replication.sh completed (provides think/nothink baselines)
#
# GPU requirement: 1x A100/H800, ~4-6 hours total
# Output: results/deepseek_iris/
#
# Usage:
#   screen -dmS deepseek_iris bash scripts/deploy_deepseek_iris.sh

set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

mkdir -p results/deepseek_iris

echo "========================================"
echo "IRIS on DeepSeek-R1-Distill-Llama-8B"
echo "========================================"

echo ""
echo "Phase 1: GSM8K IRIS@1024 (n=1319)"
echo "========================================"

python3 scripts/run_iris.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --benchmark gsm8k \
    --b1 256 \
    --b_think 1024 \
    --b_answer 256 \
    --n_samples 0 \
    --seed 42 \
    --output_dir results/deepseek_iris \
    2>&1 | tee results/deepseek_iris/gsm8k_iris_1024.log

echo ""
echo "Phase 2: GSM8K IRIS@2048 (n=1319)"
echo "========================================"

python3 scripts/run_iris.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --benchmark gsm8k \
    --b1 256 \
    --b_think 2048 \
    --b_answer 256 \
    --n_samples 0 \
    --seed 42 \
    --output_dir results/deepseek_iris \
    2>&1 | tee results/deepseek_iris/gsm8k_iris_2048.log

echo ""
echo "Phase 3: MATH-500 IRIS@2048 (n=500)"
echo "========================================"

python3 scripts/run_iris.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --benchmark math500 \
    --b1 512 \
    --b_think 2048 \
    --b_answer 256 \
    --n_samples 0 \
    --seed 42 \
    --output_dir results/deepseek_iris \
    2>&1 | tee results/deepseek_iris/math500_iris_2048.log

echo ""
echo "========================================"
echo "DeepSeek IRIS experiments complete!"
echo "Results: results/deepseek_iris/"
echo "========================================"
