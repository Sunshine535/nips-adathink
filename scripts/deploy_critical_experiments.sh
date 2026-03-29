#!/bin/bash
# deploy_critical_experiments.sh
# Runs critical experiments for paper revision in priority order
# Deploy on Server2 after thinking@256 fullset completes

set -e
cd /workspace/nips-adathink
source .venv/bin/activate
export HF_HOME=/workspace/.cache/huggingface

RESULTS_DIR="results/critical_revision"
mkdir -p "$RESULTS_DIR" results/logs

echo "$(date): Starting critical experiments for paper revision"

# ==========================================
# PRIORITY 1: TOWN end-to-end (8B GSM8K)
# ==========================================
echo "$(date): [P1] Running TOWN end-to-end on full GSM8K (n=1319)"
python3 -u scripts/run_town.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 99999 \
    --b1 256 --b2 512 \
    --seed 42 \
    --output_dir "$RESULTS_DIR/town_8b" \
    2>&1 | tee results/logs/town_8b_gsm8k.log
echo "$(date): [P1] TOWN 8B complete"

# ==========================================
# PRIORITY 2: High-budget thinking sweep (8B)
# ==========================================
echo "$(date): [P2] Running high-budget thinking sweep (1024/2048/4096)"
python3 -u scripts/run_gsm8k_experiment.py \
    --model Qwen/Qwen3-8B \
    --n_samples 99999 \
    --seed 42 --data_seed 42 \
    --budgets 1024 2048 4096 \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --projection_max_tokens 16 \
    --prompt_format chat \
    --results_dir "$RESULTS_DIR/highbudget_8b" \
    --skip_local_model_check \
    2>&1 | tee results/logs/highbudget_8b_gsm8k.log
echo "$(date): [P2] High-budget 8B complete"

# ==========================================
# PRIORITY 3: Nothink on GSM8K full set (explicit 128/512)
# ==========================================
echo "$(date): [P3] Running nothink@128 and nothink@512 on full GSM8K"
for B in 128 512; do
    echo "$(date): [P3] Running nothink@$B"
    python3 -u scripts/run_nothink_baseline.py \
        --model Qwen/Qwen3-8B \
        --benchmark gsm8k \
        --n_samples 99999 \
        --budgets $B \
        --seed 42 \
        --output_dir "$RESULTS_DIR/nothink_8b" \
        2>&1 | tee results/logs/nothink_8b_b${B}.log
done
echo "$(date): [P3] Nothink baselines complete"

# ==========================================
# PRIORITY 4: TOWN with different B1 values (ablation)
# ==========================================
echo "$(date): [P4] Running TOWN ablation: B1 sweep"
for B1 in 128 512; do
    echo "$(date): [P4] TOWN with B1=$B1, B2=512"
    python3 -u scripts/run_town.py \
        --model Qwen/Qwen3-8B \
        --benchmark gsm8k \
        --n_samples 99999 \
        --b1 $B1 --b2 512 \
        --seed 42 \
        --output_dir "$RESULTS_DIR/town_ablation" \
        2>&1 | tee results/logs/town_8b_b1${B1}.log
done
echo "$(date): [P4] TOWN ablation complete"

echo "$(date): ALL CRITICAL EXPERIMENTS COMPLETE"
echo "Results saved to: $RESULTS_DIR/"
