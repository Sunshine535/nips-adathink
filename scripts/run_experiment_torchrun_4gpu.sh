#!/usr/bin/env bash
set -euo pipefail

# Generalized multi-GPU launcher for AdaThink experiments.
#
# Usage:
#   bash scripts/run_experiment_torchrun_4gpu.sh gsm8k  Qwen/Qwen3.5-27B 40 101
#   bash scripts/run_experiment_torchrun_4gpu.sh math500 Qwen/Qwen3.5-27B 40 101
#   bash scripts/run_experiment_torchrun_4gpu.sh bbh     Qwen/Qwen3.5-27B 40 101 [bbh_task]

BENCHMARK="${1:?Usage: $0 <benchmark> <model> <n_samples> <data_seed> [bbh_task] [extra_args...]}"
MODEL="${2:?Provide model id, e.g. Qwen/Qwen3.5-27B}"
N="${3:-40}"
DATA_SEED="${4:-101}"
BBH_TASK="${5:-all}"

shift 4 2>/dev/null || true
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
mkdir -p "$RESULTS_DIR"

NGPU="${NGPU:-4}"

BENCH_ARGS=""
if [ "$BENCHMARK" = "bbh" ]; then
    BENCH_ARGS="--bbh_task $BBH_TASK"
fi

# Budget configuration per benchmark
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
    *)
        echo "Unknown benchmark: $BENCHMARK"
        exit 1
        ;;
esac

echo "=== AdaThink Experiment ==="
echo "Benchmark:  $BENCHMARK"
echo "Model:      $MODEL"
echo "N samples:  $N"
echo "Data seed:  $DATA_SEED"
echo "Budgets:    $BUDGETS"
echo "GPUs:       $NGPU"
echo "=========================="

torchrun --nproc_per_node="$NGPU" --master_port="${MASTER_PORT:-29500}" \
    "$SCRIPT_DIR/run_experiment.py" \
    --benchmark "$BENCHMARK" \
    $BENCH_ARGS \
    --model "$MODEL" \
    --n_samples "$N" \
    --data_seed "$DATA_SEED" \
    --budgets $BUDGETS \
    --adaptive_chunks $ADAPTIVE_CHUNKS \
    --adaptive_max_total $ADAPTIVE_MAX_TOTAL \
    --prompt_format chat \
    --enable_thinking \
    --strict_final_only \
    --projection_on_missing_final \
    --results_dir "$RESULTS_DIR" \
    $EXTRA_ARGS
