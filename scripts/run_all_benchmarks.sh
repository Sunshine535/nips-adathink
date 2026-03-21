#!/usr/bin/env bash
set -euo pipefail

# Run full experiment suite across MATH-500 and BBH benchmarks.
# GSM8K already done (23 seeds). This covers the new benchmarks.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh Qwen/Qwen3.5-27B
#   NGPU=8 bash scripts/run_all_benchmarks.sh Qwen/Qwen3.5-27B

MODEL="${1:?Provide model id, e.g. Qwen/Qwen3.5-27B}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_SAMPLES="${N_SAMPLES:-40}"
SEEDS=(101 202 303 404 505 606 707 808 909 1001 1103 1205 1307 1409 1511 1613 1717)

echo "============================================="
echo "  AdaThink Full Benchmark Suite"
echo "  Model:   $MODEL"
echo "  Seeds:   ${#SEEDS[@]}"
echo "  Samples: $N_SAMPLES per seed"
echo "============================================="

echo ""
echo ">>> Phase 1: MATH-500 (${#SEEDS[@]} seeds)"
for s in "${SEEDS[@]}"; do
    echo "--- MATH-500, data_seed=$s ---"
    bash "$SCRIPT_DIR/run_experiment_torchrun_4gpu.sh" math500 "$MODEL" "$N_SAMPLES" "$s"
done

echo ""
echo ">>> Phase 2: BBH (${#SEEDS[@]} seeds)"
for s in "${SEEDS[@]}"; do
    echo "--- BBH, data_seed=$s ---"
    bash "$SCRIPT_DIR/run_experiment_torchrun_4gpu.sh" bbh "$MODEL" "$N_SAMPLES" "$s"
done

echo ""
echo "============================================="
echo "  All benchmark experiments completed."
echo "============================================="
