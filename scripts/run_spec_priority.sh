#!/bin/bash
# Priority experiments: V2 configs with higher probe budgets
# Run these BEFORE the low-priority V1 ablations (K=8, seed=123)
set -euo pipefail
cd /workspace/nips-adathink
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

OUTDIR="results/speculation"
mkdir -p "$OUTDIR"

echo "=== PRIORITY V2 Pipeline ==="
echo "Start: $(date)"

# V2-A: K=2, probe=256 → total explore=512 (budget-matched with Fixed-512)
# THIS IS THE MOST IMPORTANT EXPERIMENT
echo "[$(date)] V2-A: K=2, probe=256, med=512, hard=1024"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 256 \
    --medium_budget 512 \
    --hard_budget 1024 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] V2-A DONE"

# V2-E: K=2, probe=512 → each path gets full 512 budget
echo "[$(date)] V2-E: K=2, probe=512, med=512, hard=1024"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 512 \
    --medium_budget 512 \
    --hard_budget 1024 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] V2-E DONE"

# V2-B: K=4, probe=128, LOWER thresholds (enable hard route)
echo "[$(date)] V2-B: K=4, probe=128, easy=0.90, med=0.60"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --easy_threshold 0.90 \
    --medium_threshold 0.60 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] V2-B DONE"

# V2-C: K=2, probe=256, tight thresholds
echo "[$(date)] V2-C: K=2, probe=256, easy=0.95, med=0.70"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 256 \
    --medium_budget 512 \
    --hard_budget 1024 \
    --temperature 0.7 \
    --easy_threshold 0.95 \
    --medium_threshold 0.70 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] V2-C DONE"

# V2-D: K=3, probe=170
echo "[$(date)] V2-D: K=3, probe=170"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 3 \
    --probe_budget 170 \
    --medium_budget 340 \
    --hard_budget 680 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] V2-D DONE"

# LOW-PRIORITY V1 ablations
echo "[$(date)] V1-P4: K=8 ablation (low priority)"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 8 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] V1-P4 DONE"

echo "[$(date)] V1-P5: Seed=123 (low priority)"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 123 \
    --output_dir "$OUTDIR"
echo "[$(date)] V1-P5 DONE"

echo "=== ALL PRIORITY EXPERIMENTS COMPLETE: $(date) ==="
echo "Results in: $OUTDIR"
ls -la "$OUTDIR"/*.json 2>/dev/null | wc -l
echo "JSON files generated"
