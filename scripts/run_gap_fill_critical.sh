#!/usr/bin/env bash
# ============================================================================
# run_gap_fill_critical.sh — Gap-fill experiments for paper audit
# ============================================================================
#
# Purpose:
#   Fill MISSING data points identified during the paper audit (Phase A3).
#   These experiments are required before the paper can finalize Table 1
#   and the model-size-scaling analysis.
#
# Missing experiments:
#   1. 9B nothink@256/512/1024 (+ thinking by default)
#   2. 8B think@1024/2048 and nothink@512/1024/2048 (HF engine)
#   3. 8B nothink fullset @512 (full n=1319)
#
# GPU-hour estimates (1x A100-80G):
#   Section 1 (9B, 6 runs):  ~8 GPU-h
#   Section 2 (8B, 5 runs):  ~6 GPU-h
#   Section 3 (8B, 1 run):   ~1 GPU-h
#   Total:                    ~15 GPU-h
#
# Remote servers:
#   Server A: ssh -p 11839 -i ~/.ssh/kun_ed25519 root@216.81.151.3
#   Server B: ssh root@216.81.245.127 -p 15276 -i ~/.ssh/kun_ed25519
#
# Usage:
#   # Full run (all 3 sections):
#   bash scripts/run_gap_fill_critical.sh
#
#   # Run specific section only:
#   bash scripts/run_gap_fill_critical.sh --only 1   # 9B only
#   bash scripts/run_gap_fill_critical.sh --only 2   # 8B high-budget
#   bash scripts/run_gap_fill_critical.sh --only 3   # 8B nothink@512 fullset
#
#   # Background via screen (recommended):
#   screen -dmS gap_fill bash -c 'cd /workspace/nips-adathink && source .venv/bin/activate && bash scripts/run_gap_fill_critical.sh 2>&1 | tee results/logs/gap_fill_critical.log'
#   screen -r gap_fill
# ============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

# ---- Configuration ----
cd /workspace/nips-adathink
source .venv/bin/activate 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

SEED=42
N_SAMPLES=99999  # effectively "all" — GSM8K test has 1319 items
LOGDIR="results/logs"
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

mkdir -p results/gap_fill results/logs

# ---- Parse args ----
ONLY_SECTION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only) ONLY_SECTION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- GPU check ----
echo "============================================"
echo " Gap-Fill Critical Experiments"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"

if ! command -v nvidia-smi &>/dev/null; then
    echo "[FATAL] nvidia-smi not found. No GPU driver?"
    exit 1
fi

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')
echo "  GPUs: $NUM_GPUS x ${GPU_MEM}MiB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -4
echo "============================================"

if [ "${GPU_MEM:-0}" -lt 35000 ]; then
    echo "[WARN] GPU memory < 35GB. 9B model may require >40GB. Proceeding anyway..."
fi

# ---- Helper function ----
run_nothink() {
    local model="$1"
    local budgets="$2"
    local output_dir="$3"
    local logfile="$4"
    local tag="$5"

    mkdir -p "$output_dir"
    echo "[$(date -u +%H:%M:%S)] START  $tag  model=$model budgets=$budgets"
    python3 -u scripts/run_nothink_baseline.py \
        --model "$model" \
        --benchmark gsm8k \
        --n_samples "$N_SAMPLES" \
        --budgets $budgets \
        --seed "$SEED" \
        --output_dir "$output_dir" \
        2>&1 | tee -a "$logfile"
    echo "[$(date -u +%H:%M:%S)] DONE   $tag"
    echo ""
}

# ============================================================================
# Section 1: 9B nothink@256/512/1024 (also runs thinking mode automatically)
# Model: Qwen/Qwen3.5-9B
# Note: run_nothink_baseline.py has --also_thinking=True by default,
#       so each budget runs BOTH nothink and thinking modes.
# Estimated: ~8 GPU-h (6 runs: 3 budgets x 2 modes x 1319 samples)
# ============================================================================
if [ -z "$ONLY_SECTION" ] || [ "$ONLY_SECTION" = "1" ]; then
    echo "============================================"
    echo " Section 1/3: 9B nothink+thinking @256/512/1024"
    echo " Model:     Qwen/Qwen3.5-9B"
    echo " Budgets:   256 512 1024"
    echo " Estimated: ~8 GPU-hours"
    echo "============================================"

    OUTDIR_9B="results/gap_fill/9b_nothink"
    LOGFILE_9B="$LOGDIR/gap_fill_9b_nothink_${TIMESTAMP}.log"

    # Run budgets one-by-one to allow partial recovery on failure
    for B in 256 512 1024; do
        run_nothink \
            "Qwen/Qwen3.5-9B" \
            "$B" \
            "$OUTDIR_9B" \
            "$LOGFILE_9B" \
            "9B@$B"
    done

    echo "[Section 1 COMPLETE] Results in: $OUTDIR_9B/"
    ls -lh "$OUTDIR_9B"/*.json 2>/dev/null || echo "(no json files yet)"
    echo ""
fi

# ============================================================================
# Section 2: 8B high-budget think@1024/2048 + nothink@512/1024/2048
# Model: Qwen/Qwen3-8B
# Note: Running budgets 512 1024 2048 with --also_thinking gives us:
#       nothink@512, nothink@1024, nothink@2048,
#       thinking@512, thinking@1024, thinking@2048
#       (thinking@512 is a bonus / validation point)
# Estimated: ~6 GPU-hours (6 runs: 3 budgets x 2 modes x 1319 samples)
# ============================================================================
if [ -z "$ONLY_SECTION" ] || [ "$ONLY_SECTION" = "2" ]; then
    echo "============================================"
    echo " Section 2/3: 8B nothink+thinking @512/1024/2048"
    echo " Model:     Qwen/Qwen3-8B"
    echo " Budgets:   512 1024 2048"
    echo " Estimated: ~6 GPU-hours"
    echo "============================================"

    OUTDIR_8B="results/gap_fill/8b_highbudget"
    LOGFILE_8B="$LOGDIR/gap_fill_8b_highbudget_${TIMESTAMP}.log"

    for B in 512 1024 2048; do
        run_nothink \
            "Qwen/Qwen3-8B" \
            "$B" \
            "$OUTDIR_8B" \
            "$LOGFILE_8B" \
            "8B@$B"
    done

    echo "[Section 2 COMPLETE] Results in: $OUTDIR_8B/"
    ls -lh "$OUTDIR_8B"/*.json 2>/dev/null || echo "(no json files yet)"
    echo ""
fi

# ============================================================================
# Section 3: 8B nothink fullset @512
# Model: Qwen/Qwen3-8B
# Purpose: Get the full n=1319 nothink@512 number for Table 1.
#          Section 2 already runs nothink@512, but this is a dedicated run
#          with its own output dir for clarity if Section 2 was skipped.
# Estimated: ~1 GPU-hour (2 runs: 1 budget x 2 modes x 1319 samples)
# ============================================================================
if [ -z "$ONLY_SECTION" ] || [ "$ONLY_SECTION" = "3" ]; then
    echo "============================================"
    echo " Section 3/3: 8B nothink fullset @512"
    echo " Model:     Qwen/Qwen3-8B"
    echo " Budgets:   512"
    echo " Estimated: ~1 GPU-hour"
    echo "============================================"

    OUTDIR_8B512="results/gap_fill/8b_nothink_512"
    LOGFILE_8B512="$LOGDIR/gap_fill_8b_nothink_512_${TIMESTAMP}.log"

    # Check if Section 2 already produced this result
    EXISTING=$(find results/gap_fill/8b_highbudget -name '*nothink*512*' -o -name '*Qwen3_8B*nothink*512*' 2>/dev/null | head -1)
    if [ -n "$EXISTING" ]; then
        echo "[INFO] Section 2 may already contain nothink@512 result: $EXISTING"
        echo "[INFO] Running anyway for a dedicated output in $OUTDIR_8B512"
    fi

    run_nothink \
        "Qwen/Qwen3-8B" \
        "512" \
        "$OUTDIR_8B512" \
        "$LOGFILE_8B512" \
        "8B-fullset@512"

    echo "[Section 3 COMPLETE] Results in: $OUTDIR_8B512/"
    ls -lh "$OUTDIR_8B512"/*.json 2>/dev/null || echo "(no json files yet)"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "============================================"
echo " ALL GAP-FILL EXPERIMENTS COMPLETE"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================"
echo ""
echo "Result directories:"
for d in results/gap_fill/9b_nothink results/gap_fill/8b_highbudget results/gap_fill/8b_nothink_512; do
    if [ -d "$d" ]; then
        count=$(find "$d" -name '*.json' 2>/dev/null | wc -l)
        echo "  $d/ ($count json files)"
    fi
done
echo ""
echo "Next steps:"
echo "  1. Verify accuracy numbers against paper claims"
echo "  2. Sync results to local: scp -r -P <port> root@<host>:/workspace/nips-adathink/results/gap_fill/ results_kun/gap_fill/"
echo "  3. Update Table 1 and model-size-scaling analysis"
