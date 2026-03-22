#!/usr/bin/env bash
# ============================================================================
# AdaThink — Master Experiment Orchestration
#
# Runs all experimental phases:
#   Phase 0: Download models
#   Phase 1: Budget sweep (64/128/256 tokens) on GSM8K
#   Phase 2: Self-consistency baseline (SC@8, SC@16)
#   Phase 3: Learned budget controller
#   Phase 4: Value-based budget controller
#   Phase 5: Policy search
#   Phase 6: 8B dual-scale validation
#   Phase 7: Significance tests + final figures
#
# Usage:
#   bash scripts/run_all_experiments.sh
#   bash scripts/run_all_experiments.sh --from-phase 3
#   bash scripts/run_all_experiments.sh --only-phase 2
# ============================================================================

set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_utils.sh"
auto_setup

# --- Activate project venv (created by setup.sh) ---
PROJ_DIR_ROOT="$(dirname "$SCRIPT_DIR")"
if [ -f "$PROJ_DIR_ROOT/.venv/bin/activate" ]; then
    source "$PROJ_DIR_ROOT/.venv/bin/activate"
fi
export PATH="$HOME/.local/bin:$PATH"

PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RESULTS_DIR="results"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "$LOG_DIR"

# Parse arguments
FROM_PHASE=0
ONLY_PHASE=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --from-phase) FROM_PHASE="$2"; shift 2 ;;
        --only-phase) ONLY_PHASE="$2"; shift 2 ;;
        *) echo "WARNING: Unknown arg: $1 (ignored)"; shift ;;
    esac
done

should_run() {
    local phase=$1
    if [[ $ONLY_PHASE -ge 0 ]]; then [[ $phase -eq $ONLY_PHASE ]]; else [[ $phase -ge $FROM_PHASE ]]; fi
}

log_phase() { echo ""; echo "=== Phase $1: $2 === [$(date '+%Y-%m-%d %H:%M:%S')]"; echo ""; }

SEEDS=(42 123 456)
MODEL_27B="Qwen/Qwen3.5-27B"
MODEL_8B="Qwen/Qwen3.5-8B"

echo "============================================================"
echo " AdaThink — Full Experiment Pipeline"
echo " GPUs: $NUM_GPUS × $GPU_CLASS"
echo " Started: $(date)"
echo "============================================================"

# Phase 0: Download models
if should_run 0; then
    log_phase 0 "Download models"
    python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, gc
for m in ['$MODEL_27B', '$MODEL_8B']:
    print(f'Downloading {m}...')
    tok = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True)
    print(f'  OK: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params')
    del model; gc.collect()
" 2>&1 | tee "$LOG_DIR/phase0_download.log"
fi

# Phase 1: Budget sweep
if should_run 1; then
    log_phase 1 "Budget sweep (64/128/256)"
    for BUDGET in 64 128 256; do
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 1] Budget=$BUDGET Seed=$SEED"
            $(get_torchrun_cmd) scripts/run_gsm8k_experiment.py \
                --max_new_tokens "$BUDGET" \
                --seed "$SEED" \
                --output_dir "${RESULTS_DIR}/budget_sweep/budget${BUDGET}/seed${SEED}" \
                2>&1 | tee "$LOG_DIR/phase1_b${BUDGET}_s${SEED}.log" || true
        done
    done
fi

# Phase 2: Self-consistency baseline
if should_run 2; then
    log_phase 2 "Self-consistency baseline"
    for SC in 8 16; do
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 2] SC@${SC} Seed=$SEED"
            $(get_torchrun_cmd) scripts/run_gsm8k_sc_baseline.py \
                --num_samples "$SC" \
                --seed "$SEED" \
                --output_dir "${RESULTS_DIR}/sc_baseline/sc${SC}/seed${SEED}" \
                2>&1 | tee "$LOG_DIR/phase2_sc${SC}_s${SEED}.log" || true
        done
    done
fi

# Phase 3: Learned controller
if should_run 3; then
    log_phase 3 "Learned budget controller"
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 3] Learned controller Seed=$SEED"
        $(get_torchrun_cmd) scripts/run_learned_budget_controller.py \
            --seed "$SEED" \
            --output_dir "${RESULTS_DIR}/learned_controller/seed${SEED}" \
            2>&1 | tee "$LOG_DIR/phase3_learned_s${SEED}.log" || true
    done
fi

# Phase 4: Value controller
if should_run 4; then
    log_phase 4 "Value-based budget controller"
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 4] Value controller Seed=$SEED"
        $(get_torchrun_cmd) scripts/run_value_budget_controller.py \
            --seed "$SEED" \
            --output_dir "${RESULTS_DIR}/value_controller/seed${SEED}" \
            2>&1 | tee "$LOG_DIR/phase4_value_s${SEED}.log" || true
    done
fi

# Phase 5: Policy search
if should_run 5; then
    log_phase 5 "Policy search"
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 5] Policy search Seed=$SEED"
        $(get_torchrun_cmd) scripts/run_gsm8k_policy_search.py \
            --seed "$SEED" \
            --output_dir "${RESULTS_DIR}/policy_search/seed${SEED}" \
            2>&1 | tee "$LOG_DIR/phase5_policy_s${SEED}.log" || true
    done
fi

# Phase 6: 8B dual-scale
if should_run 6; then
    log_phase 6 "8B dual-scale validation"
    python scripts/run_8b_think_postprocess_after_seeds.py \
        --output_dir "${RESULTS_DIR}/dual_scale_8b" \
        2>&1 | tee "$LOG_DIR/phase6_8b.log" || true
fi

# Phase 7: Significance tests + final figures
if should_run 7; then
    log_phase 7 "Significance tests + final figures"
    $(get_torchrun_cmd) scripts/run_template_controller_significance.py \
        --results_dir "${RESULTS_DIR}" \
        --output_dir "${RESULTS_DIR}/figures" \
        2>&1 | tee "$LOG_DIR/phase7_significance.log" || true
fi

echo ""
echo "============================================================"
echo " AdaThink — Pipeline Complete [$(date)]"
echo " Results: ${RESULTS_DIR}/"
echo "============================================================"

# --- Pipeline completion marker ---
DONE_FILE="$(dirname "$(dirname "${BASH_SOURCE[0]}")")/results/.pipeline_done"
mkdir -p "$(dirname "$DONE_FILE")"
cat > "$DONE_FILE" << DONEEOF
{
  "project": "$(basename "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS:-unknown}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo ""
echo "[PIPELINE_COMPLETE] All experiments finished successfully."
echo "  Marker: $DONE_FILE"
echo "  Run 'bash collect_results.sh' to package results."
