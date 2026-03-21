#!/usr/bin/env bash
set -euo pipefail

# Run ablation experiments for AdaThink.
# Three ablations: halting-only, no-verifier, no-branch (full fixed-budget only).
#
# Usage:
#   bash scripts/run_ablation.sh Qwen/Qwen3.5-27B gsm8k
#   bash scripts/run_ablation.sh Qwen/Qwen3.5-27B math500

MODEL="${1:?Provide model id}"
BENCHMARK="${2:-gsm8k}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
N_SAMPLES="${N_SAMPLES:-40}"
SEEDS=(101 202 303 404 505)

case "$BENCHMARK" in
    gsm8k)
        BUDGETS="128 256 512"
        ADAPTIVE_CHUNKS="128 128 256"
        ADAPTIVE_MAX_TOTAL=512
        ;;
    math500)
        BUDGETS="256 512 1024"
        ADAPTIVE_CHUNKS="256 256 512"
        ADAPTIVE_MAX_TOTAL=1024
        ;;
    bbh)
        BUDGETS="128 256 512"
        ADAPTIVE_CHUNKS="128 128 256"
        ADAPTIVE_MAX_TOTAL=512
        ;;
esac

COMMON_ARGS="--benchmark $BENCHMARK --model $MODEL --n_samples $N_SAMPLES \
    --budgets $BUDGETS --adaptive_chunks $ADAPTIVE_CHUNKS \
    --adaptive_max_total $ADAPTIVE_MAX_TOTAL \
    --prompt_format chat --enable_thinking \
    --strict_final_only --projection_on_missing_final \
    --results_dir $RESULTS_DIR"

NGPU="${NGPU:-4}"

echo "============================================="
echo "  AdaThink Ablation Suite"
echo "  Model:     $MODEL"
echo "  Benchmark: $BENCHMARK"
echo "============================================="

# 1. No-verifier ablation
echo ""
echo ">>> Ablation 1: No Verifier"
for s in "${SEEDS[@]}"; do
    echo "--- no_verifier, data_seed=$s ---"
    torchrun --nproc_per_node="$NGPU" --master_port=29501 \
        "$SCRIPT_DIR/run_experiment.py" \
        $COMMON_ARGS --data_seed "$s" --no_verifier
done

# 2. Halting-only ablation (single-chunk adaptive: can only stop early, no continue/branch)
echo ""
echo ">>> Ablation 2: Halting Only"
for s in "${SEEDS[@]}"; do
    echo "--- halting_only, data_seed=$s ---"
    torchrun --nproc_per_node="$NGPU" --master_port=29502 \
        "$SCRIPT_DIR/run_experiment.py" \
        $COMMON_ARGS --data_seed "$s" --ablation_halting_only
done

# 3. No-branch ablation (adaptive with 2 chunks only, no third branching chunk)
echo ""
echo ">>> Ablation 3: No Branch (2-chunk adaptive)"
for s in "${SEEDS[@]}"; do
    echo "--- no_branch, data_seed=$s ---"
    ABLATION_CHUNKS="${ADAPTIVE_CHUNKS%% *} ${ADAPTIVE_CHUNKS%% *}"
    ABLATION_MAX=$(( ${ADAPTIVE_CHUNKS%% *} * 2 ))
    torchrun --nproc_per_node="$NGPU" --master_port=29503 \
        "$SCRIPT_DIR/run_experiment.py" \
        --benchmark "$BENCHMARK" --model "$MODEL" --n_samples "$N_SAMPLES" \
        --budgets $BUDGETS \
        --adaptive_chunks $ABLATION_CHUNKS \
        --adaptive_max_total $ABLATION_MAX \
        --prompt_format chat --enable_thinking \
        --strict_final_only --projection_on_missing_final \
        --results_dir "$RESULTS_DIR" \
        --data_seed "$s" --ablation_no_branch
done

echo ""
echo "============================================="
echo "  Ablation experiments completed."
echo "============================================="
