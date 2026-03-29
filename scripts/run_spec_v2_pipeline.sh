#!/bin/bash
# Reasoning Speculation V2 pipeline — fixed exploration overhead
# Key change: higher probe budget, varied K configurations, budget-matched comparisons
set -euo pipefail
cd /workspace/nips-adathink
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

LOGDIR="results/logs"
OUTDIR="results/speculation"
mkdir -p "$LOGDIR" "$OUTDIR"

echo "=== Reasoning Speculation V2 Pipeline ==="
echo "Start: $(date)"
echo ""

# ============================================================
# Configuration: Budget-matched experiments
# Total exploration budget ≈ 512 tokens (same as Fixed-512)
# ============================================================

# V2-A: K=2, probe=256 → explore=512, medium=512, hard=1024
echo "[$(date)] V2-A: K=2, probe=256 (total explore≈512)"
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
echo ""

# V2-B: K=4, probe=128 (original) but LOWER thresholds to enable hard route
echo "[$(date)] V2-B: K=4, probe=128, lower thresholds (easy=0.9, medium=0.6)"
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
echo ""

# V2-C: K=2, probe=256, tighter thresholds
echo "[$(date)] V2-C: K=2, probe=256, tight thresholds (easy=0.95, medium=0.7)"
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
echo ""

# V2-D: K=3, probe=170 ≈ 510 total explore
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
echo ""

# V2-E: K=2, probe=512 → each path gets full 512 budget (explore=1024 total, but strong signal)
echo "[$(date)] V2-E: K=2, probe=512, medium=512, hard=1024"
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
echo ""

echo "=== ALL V2 PHASES COMPLETE: $(date) ==="
echo "Results in: $OUTDIR"
ls -la "$OUTDIR"/*.json 2>/dev/null | wc -l
echo "JSON files generated"
