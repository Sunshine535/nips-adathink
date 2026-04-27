#!/bin/bash
# Bug-fix rerun: Stage 3 \boxed{} parser fix + online accounting
# Generated 2026-04-27
# Push this commit to servers first, then run.

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== Bug-Fix Rerun Suite ==="
echo "Commit: $(git rev-parse --short HEAD)"
echo "CUDA: $CUDA_VISIBLE_DEVICES"
echo "HF: $HF_ENDPOINT"
date

# ---- PRIORITY 1: 8B MATH-500 @b4096 (online, n=500) ----
echo "[P1] 8B MATH-500 @b4096 online (n=500, seed=42)"
python scripts/run_iris.py \
    --model Qwen/Qwen3-8B \
    --benchmark math500 \
    --n_samples 500 --seed 42 \
    --b1 512 --b2_max 4096 --b_answer 512 \
    --online_stage2 \
    --run_town --town_b2 4096 \
    --output_dir results/bugfix_rerun_8b_math500_b4096 \
    --checkpoint_every 25

# ---- PRIORITY 3: 2x2 Factorial on MATH-500 ----
echo "[P3] 2x2 Factorial MATH-500 (n=200, seed=42)"
python scripts/run_factorial_ablation.py \
    --model Qwen/Qwen3-8B \
    --benchmark math500 \
    --n_samples 200 --seed 42 \
    --b1 512 --b2_max 1024 --b_answer 256 \
    --output_dir results/bugfix_factorial_math500

# ---- PRIORITY 4: 8B MATH-500 @b2048 ----
echo "[P4] 8B MATH-500 @b2048 online (n=500, seed=42)"
python scripts/run_iris.py \
    --model Qwen/Qwen3-8B \
    --benchmark math500 \
    --n_samples 500 --seed 42 \
    --b1 512 --b2_max 2048 --b_answer 512 \
    --online_stage2 \
    --run_town --town_b2 2048 \
    --output_dir results/bugfix_rerun_8b_math500_b2048 \
    --checkpoint_every 25

# ---- PRIORITY 5: Multiseed for best setting ----
for SEED in 123 456; do
    echo "[P5] 8B MATH-500 @b4096 online (n=500, seed=$SEED)"
    python scripts/run_iris.py \
        --model Qwen/Qwen3-8B \
        --benchmark math500 \
        --n_samples 500 --seed $SEED \
        --b1 512 --b2_max 4096 --b_answer 512 \
        --online_stage2 \
        --run_town --town_b2 4096 \
        --output_dir results/bugfix_rerun_8b_math500_b4096_s${SEED} \
        --checkpoint_every 25
done

echo "=== All 8B experiments done ==="
date
