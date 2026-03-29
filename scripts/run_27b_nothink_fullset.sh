#!/usr/bin/env bash
# Run nothink@128/256/512 for Qwen3.5-27B on full GSM8K (n=1319)
# Critical experiment: validates that nothink >> thinking for 27B too
set -euo pipefail
export PYTHONUNBUFFERED=1

MODEL="Qwen/Qwen3.5-27B"
OUTDIR="results/nothink_fullset_27b"
LOGDIR="results/logs"
SEED=42

mkdir -p "$OUTDIR" "$LOGDIR"

LOGFILE="$LOGDIR/nothink_fullset_27b.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting 27B nothink fullset experiments" | tee -a "$LOGFILE"

# Phase 1: nothink@256 (most critical — direct comparison to thinking@256=7.9%)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running 27B nothink@256..." | tee -a "$LOGFILE"
python3 -u scripts/run_nothink_baseline.py \
    --model "$MODEL" \
    --benchmark gsm8k \
    --n_samples 99999 \
    --budgets 256 \
    --seed "$SEED" \
    --output_dir "$OUTDIR" \
    2>&1 | tee -a "$LOGFILE"

# Phase 2: nothink@128
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running 27B nothink@128..." | tee -a "$LOGFILE"
python3 -u scripts/run_nothink_baseline.py \
    --model "$MODEL" \
    --benchmark gsm8k \
    --n_samples 99999 \
    --budgets 128 \
    --seed "$SEED" \
    --output_dir "$OUTDIR" \
    2>&1 | tee -a "$LOGFILE"

# Phase 3: nothink@512
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Running 27B nothink@512..." | tee -a "$LOGFILE"
python3 -u scripts/run_nothink_baseline.py \
    --model "$MODEL" \
    --benchmark gsm8k \
    --n_samples 99999 \
    --budgets 512 \
    --seed "$SEED" \
    --output_dir "$OUTDIR" \
    2>&1 | tee -a "$LOGFILE"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] All 27B nothink experiments COMPLETE" | tee -a "$LOGFILE"
