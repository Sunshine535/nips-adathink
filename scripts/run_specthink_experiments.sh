#!/bin/bash
# Speculative Thinking (SpecThink) experiments
# Tests adaptive token allocation via natural-stop detection
set -euo pipefail
cd /workspace/nips-adathink
export HF_HOME=/workspace/.cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

OUTDIR="results/speculation"
mkdir -p "$OUTDIR"

echo "=== SPECULATIVE THINKING EXPERIMENTS ==="
echo "Start: $(date)"

# ============================================================
# Experiment 1: Main configuration (probe=256, extend=512, max=1024)
# Budget-matched comparison with Fixed-512
# ============================================================
echo "[$(date)] SpecThink-1: probe=256, extend=512, max=1024 (with baselines)"
python3 -u scripts/run_speculative_thinking.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --probe_budget 256 \
    --extend_budget 512 \
    --max_budget 1024 \
    --temperature 0.0 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] SpecThink-1 DONE"

# ============================================================
# Experiment 2: Conservative (probe=128, extend=256, max=512)
# Comparable budget to Fixed-512
# ============================================================
echo "[$(date)] SpecThink-2: probe=128, extend=256, max=512"
python3 -u scripts/run_speculative_thinking.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --probe_budget 128 \
    --extend_budget 256 \
    --max_budget 512 \
    --temperature 0.0 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] SpecThink-2 DONE"

# ============================================================
# Experiment 3: Aggressive (probe=128, extend=512, max=1024)
# Skip the 256 level entirely
# ============================================================
echo "[$(date)] SpecThink-3: probe=128, extend=512, max=1024 (no cascade)"
python3 -u scripts/run_speculative_thinking.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --probe_budget 128 \
    --extend_budget 512 \
    --max_budget 1024 \
    --no_cascade \
    --temperature 0.0 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] SpecThink-3 DONE"

# ============================================================
# Experiment 4: Large probe (probe=512, extend=1024, max=2048)
# Test with generous budgets
# ============================================================
echo "[$(date)] SpecThink-4: probe=512, extend=1024, max=2048"
python3 -u scripts/run_speculative_thinking.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --probe_budget 512 \
    --extend_budget 1024 \
    --max_budget 2048 \
    --temperature 0.0 \
    --seed 42 \
    --output_dir "$OUTDIR"
echo "[$(date)] SpecThink-4 DONE"

# ============================================================
# Experiment 5: Full GSM8K test (n=1319) with best config
# Will run after we identify best config from above
# ============================================================
echo "[$(date)] SpecThink-Full: probe=256, extend=512, max=1024, n=1319"
python3 -u scripts/run_speculative_thinking.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 99999 \
    --probe_budget 256 \
    --extend_budget 512 \
    --max_budget 1024 \
    --temperature 0.0 \
    --seed 42 \
    --output_dir "$OUTDIR" \
    --run_baselines
echo "[$(date)] SpecThink-Full DONE"

# ============================================================
# Experiment 6: Seed ablation
# ============================================================
echo "[$(date)] SpecThink-Seed: probe=256, extend=512, max=1024, seed=123"
python3 -u scripts/run_speculative_thinking.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --probe_budget 256 \
    --extend_budget 512 \
    --max_budget 1024 \
    --temperature 0.0 \
    --seed 123 \
    --output_dir "$OUTDIR"
echo "[$(date)] SpecThink-Seed DONE"

echo "=== ALL SPECTHINK EXPERIMENTS COMPLETE: $(date) ==="
echo "Results in: $OUTDIR"
ls -la "$OUTDIR"/specthink*.json 2>/dev/null | wc -l
echo "SpecThink JSON files generated"
