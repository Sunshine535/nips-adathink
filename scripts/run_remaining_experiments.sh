#!/bin/bash
# Run all remaining experiments after K=8 completes
# This includes: Phase 5 (seed=123), V2 priority experiments, V3 experiments
set -euo pipefail
cd /workspace/nips-adathink
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

OUTDIR="results/speculation"
mkdir -p "$OUTDIR"

echo "=== REMAINING EXPERIMENTS ==="
echo "Start: $(date)"

# ============================================================
# PHASE 5: Seed=123 ablation (V1, low priority)
# ============================================================
echo "[$(date)] Phase 5: K=4, seed=123"
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

# ============================================================
# V2-A: K=2, probe=256 (MOST IMPORTANT — budget-matched with Fixed-512)
# ============================================================
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

# ============================================================
# V2-E: K=2, probe=512 (SECOND MOST IMPORTANT)
# ============================================================
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

# ============================================================
# V2-B: K=4, probe=128, LOWER thresholds
# ============================================================
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

# ============================================================
# V2-C: K=2, probe=256, tight thresholds
# ============================================================
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

# ============================================================
# V2-D: K=3, probe=170
# ============================================================
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

# ============================================================
# V3 EXPERIMENTS (with no-think probes + quality-aware routing)
# ============================================================
echo "[$(date)] V3-A: K=2, probe=256, V3 optimizations"
python3 -u scripts/run_reasoning_speculation_v3.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 256 \
    --medium_budget 512 \
    --hard_budget 1024 \
    --temperature 0.7 \
    --seed 42 \
    --v3 \
    --output_dir "$OUTDIR"
echo "[$(date)] V3-A DONE"

echo "[$(date)] V3-B: K=2, probe=256, V3 WITH think probes (ablation)"
python3 -u scripts/run_reasoning_speculation_v3.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 2 \
    --probe_budget 256 \
    --medium_budget 512 \
    --hard_budget 1024 \
    --temperature 0.7 \
    --seed 42 \
    --v3 \
    --think_probes \
    --output_dir "$OUTDIR"
echo "[$(date)] V3-B DONE"

echo "=== ALL REMAINING EXPERIMENTS COMPLETE: $(date) ==="
echo "Results in: $OUTDIR"
ls -la "$OUTDIR"/*.json 2>/dev/null | wc -l
echo "JSON files generated"
