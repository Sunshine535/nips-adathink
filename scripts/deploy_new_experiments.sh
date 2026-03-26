#!/bin/bash
# Deploy all new experiments requested by reviewer.
# Run on remote server with GPU access.
#
# Usage: bash deploy_new_experiments.sh
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
cd "$(dirname "$0")/.."

RESULTS_BASE="results"
LOG_DIR="logs/new_experiments"
mkdir -p "$LOG_DIR"

echo "=== Deploying New Experiments ==="
echo "Time: $(date)"

# 1. Fair SC baselines (SC@2x256, SC@4x128 at matched total cost)
echo "[1/3] Fair SC baselines..."
screen -dmS fair_sc bash -c "
  source .venv/bin/activate 2>/dev/null || true
  python scripts/run_fair_sc_baseline.py \
    --model Qwen/Qwen3.5-27B \
    --benchmark gsm8k \
    --sc_configs '2x256,4x128,1x512' \
    --data_seed 42 --seed 11 \
    --num_samples 40 \
    --output_dir $RESULTS_BASE/fair_sc \
    2>&1 | tee $LOG_DIR/fair_sc_gsm8k.log
  echo 'DONE: Fair SC GSM8K' >> $LOG_DIR/status.txt
"

# 2. Continuation-from-b1 baseline (prefix reuse)
echo "[2/3] Continuation baseline..."
screen -dmS continuation bash -c "
  source .venv/bin/activate 2>/dev/null || true
  python scripts/run_continuation_baseline.py \
    --model Qwen/Qwen3.5-27B \
    --benchmark gsm8k \
    --budgets 128,256,512 \
    --data_seed 42 --seed 11 \
    --num_samples 40 \
    --output_dir $RESULTS_BASE/continuation_baseline \
    2>&1 | tee $LOG_DIR/continuation_gsm8k.log
  echo 'DONE: Continuation GSM8K' >> $LOG_DIR/status.txt
"

# 3. Multi-seed evaluation
echo "[3/3] Multi-seed evaluation..."
screen -dmS multiseed bash -c "
  source .venv/bin/activate 2>/dev/null || true
  python scripts/run_multiseed_eval.py \
    --model Qwen/Qwen3.5-27B \
    --benchmark gsm8k \
    --budgets 128,256,512 \
    --algo_seeds '11,42,123,7,2024' \
    --data_seed 42 \
    --num_samples 40 \
    --output_dir $RESULTS_BASE/multiseed \
    2>&1 | tee $LOG_DIR/multiseed_gsm8k.log
  echo 'DONE: Multi-seed GSM8K' >> $LOG_DIR/status.txt
"

echo ""
echo "All experiments launched in screen sessions:"
echo "  - fair_sc: SC@2x256, SC@4x128, SC@1x512"
echo "  - continuation: Continuation-from-b1 baseline"
echo "  - multiseed: 5 algorithm seeds × 3 budgets"
echo ""
echo "Monitor: screen -r <name>"
echo "Status: cat $LOG_DIR/status.txt"
