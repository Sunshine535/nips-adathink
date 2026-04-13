#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B full replication for cross-family validation
# PURPOSE: Generate matched think/nothink budget sweep on GSM8K (n=1319)
#          This is THE critical experiment for best-paper territory.
#
# GPU requirement: 1x A100/H800, ~8-12 hours total
# Output: results/deepseek_crossmodel/
#
# Usage:
#   screen -dmS deepseek_repl bash scripts/deploy_deepseek_replication.sh
#   # or manually:
#   bash scripts/deploy_deepseek_replication.sh

set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate

echo "========================================"
echo "Phase 1: GSM8K full dataset (n=1319)"
echo "Budgets: 256, 512, 1024, 2048"
echo "Both nothink + thinking modes"
echo "========================================"

python3 scripts/run_deepseek_crossmodel.py \
    --benchmark gsm8k \
    --budgets 256 512 1024 2048 \
    --seed 42 \
    --n_samples 0 \
    --output_dir results/deepseek_crossmodel \
    --cache_dir /workspace/hf_cache \
    2>&1 | tee results/deepseek_crossmodel/gsm8k_full_replication.log

echo ""
echo "========================================"
echo "Phase 2: MATH-500 full dataset (n=500)"
echo "Budgets: 512, 1024, 2048, 4096"
echo "Both nothink + thinking modes"
echo "========================================"

python3 scripts/run_deepseek_crossmodel.py \
    --benchmark math500 \
    --budgets 512 1024 2048 4096 \
    --seed 42 \
    --n_samples 0 \
    --output_dir results/deepseek_crossmodel \
    --cache_dir /workspace/hf_cache \
    2>&1 | tee results/deepseek_crossmodel/math500_full_replication.log

echo ""
echo "========================================"
echo "DeepSeek replication complete!"
echo "Results: results/deepseek_crossmodel/"
echo "========================================"
echo ""
echo "Next step: adapt run_iris.py for DeepSeek and run IRIS@{1024,2048} on GSM8K"
