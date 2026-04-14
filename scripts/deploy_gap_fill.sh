#!/bin/bash
# =============================================================================
# Gap-Fill Experiment Deployment — 2026-04-14
# Fills critical claim-evidence gaps for NeurIPS submission
# =============================================================================
#
# SERVER ASSIGNMENT:
#   Server 1 (2×A100): 8B GSM8K full-scale IRIS (n=1319) — validates Theorem 1
#   Server 2 (1×A100): 8B MATH-500 IRIS multi-budget sweep — cross-benchmark
#   Server 3 (H800):   27B IRIS high-budget (b2=2048/4096) — 27B diagnosis
#
# PRIORITY ORDER (by paper impact):
#   P0: 27B IRIS with scaled budgets (validates "budget insufficiency" claim)
#   P1: 8B GSM8K full-scale IRIS (n=1319, provides α_extract^hard for Thm 1)
#   P2: 8B MATH-500 IRIS multi-budget (cross-benchmark IRIS validation)
#
# ESTIMATED TIME:
#   Server 1: ~3-4h (1319 samples, 8B model)
#   Server 2: ~2-3h (500 samples × 2 budgets)
#   Server 3: ~4-6h (200 samples, 27B model, high budgets)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_DIR}/results/gap_fill_$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$HF_HOME"

echo "=================================================="
echo "Gap-Fill Experiments — $(date)"
echo "Results: $RESULTS_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=================================================="

# ---- Detect which server we're on ----
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Detected: ${NUM_GPUS}x ${GPU_NAME}"

# ---- Auto-select experiment based on flag ----
EXPERIMENT="${1:-auto}"

case "$EXPERIMENT" in
  # ===== SERVER 1: 8B GSM8K Full-Scale IRIS =====
  server1|gsm8k_fullscale)
    echo ""
    echo "[P1] 8B GSM8K Full-Scale IRIS (n=1319)"
    echo "  Purpose: Provides real α_extract^hard for Theorem 1"
    echo "  Current: Only TOWN cascade data (62.8%), need actual IRIS"
    echo ""

    cd "$PROJECT_DIR"
    source .venv/bin/activate 2>/dev/null || true

    python scripts/run_iris.py \
      --model Qwen/Qwen3-8B \
      --benchmark gsm8k \
      --n_samples 1319 \
      --b1 256 --b2_max 512 --b_answer 128 \
      --run_town --town_b2 512 \
      --seed 42 \
      --output_dir "$RESULTS_DIR/iris_gsm8k_8b_fullscale" \
      --checkpoint_every 100 \
      2>&1 | tee "$RESULTS_DIR/log_iris_gsm8k_8b_fullscale.txt"

    echo "[P1] DONE: 8B GSM8K Full-Scale IRIS"

    # Follow up: seed 123 for robustness
    echo ""
    echo "[P4] 8B GSM8K IRIS seed=123 (robustness check)"
    python scripts/run_iris.py \
      --model Qwen/Qwen3-8B \
      --benchmark gsm8k \
      --n_samples 200 \
      --b1 256 --b2_max 512 --b_answer 128 \
      --seed 123 \
      --output_dir "$RESULTS_DIR/iris_gsm8k_8b_seed123" \
      --checkpoint_every 50 \
      2>&1 | tee "$RESULTS_DIR/log_iris_gsm8k_8b_seed123.txt"

    echo "[P4] DONE: Seed 123 robustness"
    ;;

  # ===== SERVER 2: 8B MATH-500 IRIS Multi-Budget =====
  server2|math500_sweep)
    echo ""
    echo "[P2] 8B MATH-500 IRIS Budget Sweep (n=500, b2=1024/2048)"
    echo "  Purpose: Cross-benchmark IRIS validation at multiple budgets"
    echo ""

    cd "$PROJECT_DIR"
    source .venv/bin/activate 2>/dev/null || true

    # b2_max=1024
    echo "--- IRIS MATH-500 b2=1024 ---"
    python scripts/run_iris.py \
      --model Qwen/Qwen3-8B \
      --benchmark math500 \
      --n_samples 500 \
      --b1 512 --b2_max 1024 --b_answer 256 \
      --run_town --town_b2 1024 \
      --seed 42 \
      --output_dir "$RESULTS_DIR/iris_math500_8b_b1024" \
      --checkpoint_every 50 \
      2>&1 | tee "$RESULTS_DIR/log_iris_math500_8b_b1024.txt"

    # b2_max=2048
    echo "--- IRIS MATH-500 b2=2048 ---"
    python scripts/run_iris.py \
      --model Qwen/Qwen3-8B \
      --benchmark math500 \
      --n_samples 500 \
      --b1 512 --b2_max 2048 --b_answer 256 \
      --run_town --town_b2 2048 \
      --seed 42 \
      --output_dir "$RESULTS_DIR/iris_math500_8b_b2048" \
      --checkpoint_every 50 \
      2>&1 | tee "$RESULTS_DIR/log_iris_math500_8b_b2048.txt"

    echo "[P2] DONE: MATH-500 IRIS sweep"
    ;;

  # ===== SERVER 3 (H800): 27B IRIS High-Budget =====
  server3|27b_highbudget)
    echo ""
    echo "[P0] 27B IRIS High-Budget (GSM8K b2=2048, MATH-500 b2=4096)"
    echo "  Purpose: Diagnose whether 27B underperformance is budget-limited"
    echo "  Current: 27B MRSD@512=60% < nothink@67.5% (budget insufficient)"
    echo "  Hypothesis: With b2=2048+, split-budget should recover gains"
    echo ""

    cd "$PROJECT_DIR"
    source .venv/bin/activate 2>/dev/null || true

    # 27B GSM8K with b2=2048
    echo "--- 27B IRIS GSM8K b2=2048 (n=200) ---"
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_iris.py \
      --model Qwen/Qwen3.5-27B \
      --benchmark gsm8k \
      --n_samples 200 \
      --b1 256 --b2_max 2048 --b_answer 256 \
      --run_town --town_b2 2048 \
      --seed 42 \
      --output_dir "$RESULTS_DIR/iris_gsm8k_27b_b2048" \
      --checkpoint_every 25 \
      2>&1 | tee "$RESULTS_DIR/log_iris_gsm8k_27b_b2048.txt"

    # 27B MATH-500 with b2=4096
    echo "--- 27B IRIS MATH-500 b2=4096 (n=200) ---"
    CUDA_VISIBLE_DEVICES=0,1 python scripts/run_iris.py \
      --model Qwen/Qwen3.5-27B \
      --benchmark math500 \
      --n_samples 200 \
      --b1 512 --b2_max 4096 --b_answer 256 \
      --run_town --town_b2 4096 \
      --seed 42 \
      --output_dir "$RESULTS_DIR/iris_math500_27b_b4096" \
      --checkpoint_every 25 \
      2>&1 | tee "$RESULTS_DIR/log_iris_math500_27b_b4096.txt"

    echo "[P0] DONE: 27B high-budget IRIS"
    ;;

  *)
    echo "Usage: $0 {server1|server2|server3}"
    echo ""
    echo "  server1 — 8B GSM8K full-scale IRIS (n=1319) + seed=123"
    echo "  server2 — 8B MATH-500 IRIS b1024/b2048 sweep (n=500)"
    echo "  server3 — 27B IRIS high-budget GSM8K+MATH-500 (n=200)"
    exit 1
    ;;
esac

echo ""
echo "=================================================="
echo "Experiment complete — $(date)"
echo "Results in: $RESULTS_DIR"
echo "=================================================="
