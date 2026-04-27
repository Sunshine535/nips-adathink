#!/bin/bash
# Bug-fix rerun: 27B experiments — run on H800
# Generated 2026-04-27

set -euo pipefail
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== 27B Bug-Fix Rerun (H800) ==="
echo "Commit: $(git rev-parse --short HEAD)"
date

# ---- PRIORITY 2: 27B MATH-500 @b4096 (online, n=200) ----
echo "[P2] 27B MATH-500 @b4096 online (n=200, seed=42)"
python scripts/run_iris.py \
    --model Qwen/Qwen3.5-27B \
    --benchmark math500 \
    --n_samples 200 --seed 42 \
    --b1 512 --b2_max 4096 --b_answer 512 \
    --online_stage2 \
    --run_town --town_b2 4096 \
    --output_dir results/bugfix_rerun_27b_math500_b4096 \
    --checkpoint_every 25

# ---- 27B MATH-500 @b2048 ----
echo "[P2b] 27B MATH-500 @b2048 online (n=200, seed=42)"
python scripts/run_iris.py \
    --model Qwen/Qwen3.5-27B \
    --benchmark math500 \
    --n_samples 200 --seed 42 \
    --b1 512 --b2_max 2048 --b_answer 512 \
    --online_stage2 \
    --run_town --town_b2 2048 \
    --output_dir results/bugfix_rerun_27b_math500_b2048 \
    --checkpoint_every 25

# ---- 27B GSM8K @b4096 (verify existing clean result) ----
echo "[P2c] 27B GSM8K @b4096 online (n=200, seed=42)"
python scripts/run_iris.py \
    --model Qwen/Qwen3.5-27B \
    --benchmark gsm8k \
    --n_samples 200 --seed 42 \
    --b1 256 --b2_max 4096 --b_answer 512 \
    --online_stage2 \
    --run_town --town_b2 4096 \
    --output_dir results/bugfix_rerun_27b_gsm8k_b4096 \
    --checkpoint_every 25

echo "=== All 27B experiments done ==="
date
