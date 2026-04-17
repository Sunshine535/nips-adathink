#!/bin/bash
# =============================================================================
# Baseline Verification — Independent reproduction of paper's core claims
# =============================================================================
#
# PURPOSE: Before trusting ANY method results, independently verify that
#   the baseline observations (thinking tax) are real and match the paper.
#
# WHAT WE VERIFY:
#   1. Qwen3-8B think@256 << nothink@256 on GSM8K (paper: 18.0% vs 87.5%)
#   2. Qwen3-8B think@512 << nothink@512 on GSM8K (paper: 56.9% vs 93.1%)
#   3. Truncation rate at b=256 ≈ 98% (paper: 1.4% natural stop)
#   4. Truncation rate at b=512 ≈ 63% (paper: 37.4% natural stop)
#   5. α_c(512) ≈ 99% (completed chains highly accurate)
#
# SAMPLE SIZE: n=200 (quick verification, ~1h)
#   If consistent with paper, confidence is high since paper uses n=1319
#   with same seed=42 subset matching n=200 pilot data
#
# EXPECTED RUN TIME: ~45 min on 1×A100
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${PROJECT_DIR}/results/baseline_verification_$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME" 2>/dev/null || true

echo "=================================================="
echo "Baseline Verification — $(date)"
echo "Purpose: Independently verify paper's core claims"
echo "=================================================="

cd "$PROJECT_DIR"

# ---- Experiment 1: GSM8K budget sweep (think + nothink) ----
echo ""
echo "[VERIFY-1] GSM8K Think vs Nothink Budget Sweep (n=200)"
echo "  Paper claims:"
echo "    think@128  = 3.0%   nothink@128  = 50.8%"
echo "    think@256  = 18.0%  nothink@256  = 87.5%"
echo "    think@512  = 56.9%  nothink@512  = 93.1%"
echo ""

python3 scripts/run_nothink_baseline.py \
  --model Qwen/Qwen3-8B \
  --benchmark gsm8k \
  --n_samples 200 \
  --budgets 128 256 512 1024 \
  --also_thinking \
  --seed 42 \
  --output_dir "$RESULTS_DIR/gsm8k_8b_sweep" \
  2>&1 | tee "$RESULTS_DIR/log_gsm8k_verify.txt"

echo ""
echo "[VERIFY-1] DONE"

# ---- Experiment 2: MATH-500 budget sweep ----
echo ""
echo "[VERIFY-2] MATH-500 Think vs Nothink Budget Sweep (n=200)"
echo "  Paper claims:"
echo "    think@512  = 6.2%   nothink@512  = 40.6%"
echo "    think@1024 = 18.0%  nothink@1024 = 59.8%"
echo ""

python3 scripts/run_nothink_baseline.py \
  --model Qwen/Qwen3-8B \
  --benchmark math500 \
  --n_samples 200 \
  --budgets 512 1024 \
  --also_thinking \
  --seed 42 \
  --output_dir "$RESULTS_DIR/math500_8b_sweep" \
  2>&1 | tee "$RESULTS_DIR/log_math500_verify.txt"

echo ""
echo "[VERIFY-2] DONE"

echo ""
echo "=================================================="
echo "Baseline Verification Complete — $(date)"
echo "Results in: $RESULTS_DIR"
echo ""
echo "NEXT STEP: Compare results with paper claims."
echo "  If consistent → baseline confirmed → proceed with method verification"
echo "  If inconsistent → investigate discrepancy before ANY method claims"
echo "=================================================="
