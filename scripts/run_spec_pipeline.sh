#!/bin/bash
# Reasoning Speculation full pipeline - runs sequentially in background
set -euo pipefail
cd /workspace/nips-adathink
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

LOGDIR="results/logs"
OUTDIR="results/speculation"
mkdir -p "$LOGDIR" "$OUTDIR"

echo "=== Reasoning Speculation Pipeline ==="
echo "Start: $(date)"
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.free --format=csv
echo ""

# Phase 1: Validation (n=50)
echo "[$(date)] Phase 1: Validation run (n=50, K=4)"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 50 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] Phase 1 DONE"
echo ""

# Phase 2: Full run (n=200)
echo "[$(date)] Phase 2: Full run (n=200, K=4)"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] Phase 2 DONE"
echo ""

# Phase 3: K=2 ablation
echo "[$(date)] Phase 3: K=2 ablation (n=200)"
python3 -u scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] Phase 3 DONE"
echo ""

# Phase 4: K=8 ablation
echo "[$(date)] Phase 4: K=8 ablation (n=200)"
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
echo "[$(date)] Phase 4 DONE"
echo ""

# Phase 5: Different seed (n=200, K=4)
echo "[$(date)] Phase 5: Seed=123 (n=200, K=4)"
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
echo "[$(date)] Phase 5 DONE"
echo ""

echo "=== ALL PHASES COMPLETE: $(date) ==="
echo "Results in: $OUTDIR"
ls -la "$OUTDIR"/*.json 2>/dev/null | wc -l
echo "JSON files generated"
