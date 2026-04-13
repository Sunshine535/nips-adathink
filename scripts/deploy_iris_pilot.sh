#!/bin/bash
# Deploy IRIS entropy dynamics pilot experiment
# Waits for current gap-fill experiments to finish, then runs the pilot
# Usage: bash scripts/deploy_iris_pilot.sh

set -e

cd /workspace/nips-adathink
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/.cache/huggingface

echo "============================================"
echo "[$(date)] IRIS Entropy Dynamics Pilot"
echo "============================================"

# Phase 1: Wait for gap-fill experiments to finish
echo "[$(date)] Waiting for gap-fill experiments to finish..."
while true; do
    N_RUNNING=$(ps aux | grep "run_nothink_baseline" | grep -v grep | wc -l)
    if [ "$N_RUNNING" -eq 0 ]; then
        echo "[$(date)] Gap-fill experiments finished!"
        break
    fi
    echo "[$(date)] Still waiting... $N_RUNNING processes running"
    sleep 300  # Check every 5 minutes
done

# Phase 2: Run entropy pilot on GPU 0 (Qwen3-8B, 200 samples, think@512)
echo "[$(date)] Starting entropy pilot: Qwen3-8B, 200 samples, budget=512"
export CUDA_VISIBLE_DEVICES=0

python3 -u scripts/collect_entropy_dynamics.py \
    --model Qwen/Qwen3-8B \
    --budget 512 \
    --n_samples 200 \
    --chunk_size 32 \
    --seed 42 \
    --output_dir results/entropy_dynamics \
    --checkpoint_every 50 \
    2>&1

echo "[$(date)] Pilot complete on budget=512!"

# Phase 3: Run entropy pilot at budget=256 and budget=1024 for comparison
echo "[$(date)] Starting entropy pilot: budget=256"
python3 -u scripts/collect_entropy_dynamics.py \
    --model Qwen/Qwen3-8B \
    --budget 256 \
    --n_samples 200 \
    --chunk_size 32 \
    --seed 42 \
    --output_dir results/entropy_dynamics \
    2>&1

echo "[$(date)] Starting entropy pilot: budget=1024"
python3 -u scripts/collect_entropy_dynamics.py \
    --model Qwen/Qwen3-8B \
    --budget 1024 \
    --n_samples 200 \
    --chunk_size 32 \
    --seed 42 \
    --output_dir results/entropy_dynamics \
    2>&1

# Phase 4: Quick IRIS pilot (200 samples) vs TOWN comparison
echo "[$(date)] Starting IRIS vs TOWN comparison (200 samples)"
python3 -u scripts/run_iris.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --b1 256 --b2_max 512 --b_answer 128 \
    --chunk_size 32 \
    --tau_h 1.5 --tau_s 50.0 \
    --run_town --town_b2 512 \
    --seed 42 \
    --output_dir results/iris \
    2>&1

echo "============================================"
echo "[$(date)] ALL IRIS PILOT EXPERIMENTS COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Check GO/NO-GO: cat results/entropy_dynamics/go_no_go_summary_*.json"
echo "  2. If GO: run full IRIS eval with --n_samples 99999"
echo "  3. Generate figures: python scripts/plot_entropy_dynamics.py --input 'results/entropy_dynamics/entropy_dynamics_*.json'"
