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

PHASE_MARKERS="${RESULTS_DIR}/.phase_markers"
mkdir -p "$PHASE_MARKERS"

phase_done() {
    local phase=$1
    [ -f "$PHASE_MARKERS/phase${phase}.done" ] && [ "${FORCE_RERUN:-0}" != "1" ]
}

mark_phase_done() {
    local phase=$1
    echo "{\"phase\":$phase,\"completed\":\"$(date -u '+%Y-%m-%dT%H:%M:%SZ')\",\"hostname\":\"$(hostname)\"}" \
        > "$PHASE_MARKERS/phase${phase}.done"
}

should_run() {
    local phase=$1
    if phase_done "$phase"; then
        echo "[SKIP] Phase $phase already completed (marker: $PHASE_MARKERS/phase${phase}.done)"
        return 1
    fi
    if [[ $ONLY_PHASE -ge 0 ]]; then [[ $phase -eq $ONLY_PHASE ]]; else [[ $phase -ge $FROM_PHASE ]]; fi
}

log_phase() { echo ""; echo "=== Phase $1: $2 === [$(date '+%Y-%m-%d %H:%M:%S')]"; echo ""; }

SEEDS=(42 123 456)
MODEL_27B="Qwen/Qwen3.5-27B"
MODEL_8B="Qwen/Qwen3-8B"

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
    mark_phase_done 0
fi

# Phase 1: Budget sweep on GSM8K (27B model, multiple budgets per seed)
if should_run 1; then
    log_phase 1 "Budget sweep (64/128/256) on GSM8K"
    PHASE_FAILURES=0
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 1] Budgets=64,128,256 Seed=$SEED Model=$MODEL_27B"
        $(get_torchrun_cmd) scripts/run_gsm8k_experiment.py \
            --model "$MODEL_27B" \
            --budgets 64 128 256 \
            --seed "$SEED" \
            --results_dir "${RESULTS_DIR}" \
            --enable_thinking \
            2>&1 | tee "$LOG_DIR/phase1_s${SEED}.log" || PHASE_FAILURES=$((PHASE_FAILURES + 1))
    done
    [ "$PHASE_FAILURES" -eq 0 ] && mark_phase_done 1 || echo "[WARN] Phase 1 had $PHASE_FAILURES failure(s); not marking as done"
fi

# Phase 2: Self-consistency baseline
if should_run 2; then
    log_phase 2 "Self-consistency baseline (SC@8, SC@16)"
    PHASE_FAILURES=0
    for SC in 8 16; do
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 2] SC@${SC} Seed=$SEED Model=$MODEL_27B"
            $(get_torchrun_cmd) scripts/run_gsm8k_sc_baseline.py \
                --model "$MODEL_27B" \
                --sc_n "$SC" \
                --seed "$SEED" \
                --results_dir "${RESULTS_DIR}" \
                --enable_thinking \
                2>&1 | tee "$LOG_DIR/phase2_sc${SC}_s${SEED}.log" || PHASE_FAILURES=$((PHASE_FAILURES + 1))
        done
    done
    [ "$PHASE_FAILURES" -eq 0 ] && mark_phase_done 2 || echo "[WARN] Phase 2 had $PHASE_FAILURES failure(s); not marking as done"
fi

# Phase 3: Learned budget controller (post-processing, no GPU needed)
if should_run 3; then
    log_phase 3 "Learned budget controller"
    INPUT_CSVS=( $(ls ${RESULTS_DIR}/per_sample_*.csv 2>/dev/null) )
    if [ ${#INPUT_CSVS[@]} -gt 0 ]; then
        PHASE_FAILURES=0
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 3] Learned controller Seed=$SEED (${#INPUT_CSVS[@]} input CSVs)"
            python scripts/run_learned_budget_controller.py \
                --input_csvs "${INPUT_CSVS[@]}" \
                --seed "$SEED" \
                --output_dir "${RESULTS_DIR}" \
                2>&1 | tee "$LOG_DIR/phase3_learned_s${SEED}.log" || PHASE_FAILURES=$((PHASE_FAILURES + 1))
        done
        [ "$PHASE_FAILURES" -eq 0 ] && mark_phase_done 3 || echo "[WARN] Phase 3 had $PHASE_FAILURES failure(s); not marking as done"
    else
        echo "[Phase 3] SKIPPED: no per_sample CSVs found in ${RESULTS_DIR}/"
    fi
fi

# Phase 4: Value-based budget controller (post-processing, no GPU needed)
if should_run 4; then
    log_phase 4 "Value-based budget controller"
    INPUT_CSVS=( $(ls ${RESULTS_DIR}/per_sample_*.csv 2>/dev/null) )
    if [ ${#INPUT_CSVS[@]} -gt 0 ]; then
        PHASE_FAILURES=0
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 4] Value controller Seed=$SEED (${#INPUT_CSVS[@]} input CSVs)"
            python scripts/run_value_budget_controller.py \
                --input_csvs "${INPUT_CSVS[@]}" \
                --seed "$SEED" \
                2>&1 | tee "$LOG_DIR/phase4_value_s${SEED}.log" || PHASE_FAILURES=$((PHASE_FAILURES + 1))
        done
        [ "$PHASE_FAILURES" -eq 0 ] && mark_phase_done 4 || echo "[WARN] Phase 4 had $PHASE_FAILURES failure(s); not marking as done"
    else
        echo "[Phase 4] SKIPPED: no per_sample CSVs found in ${RESULTS_DIR}/"
    fi
fi

# Phase 5: Policy search on GSM8K
if should_run 5; then
    log_phase 5 "Policy search"
    PHASE_FAILURES=0
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 5] Policy search Seed=$SEED Model=$MODEL_27B"
        $(get_torchrun_cmd) scripts/run_gsm8k_policy_search.py \
            --model "$MODEL_27B" \
            --seed "$SEED" \
            --results_dir "${RESULTS_DIR}" \
            --enable_thinking \
            2>&1 | tee "$LOG_DIR/phase5_policy_s${SEED}.log" || PHASE_FAILURES=$((PHASE_FAILURES + 1))
    done
    [ "$PHASE_FAILURES" -eq 0 ] && mark_phase_done 5 || echo "[WARN] Phase 5 had $PHASE_FAILURES failure(s); not marking as done"
fi

# Phase 6: 8B dual-scale validation
if should_run 6; then
    log_phase 6 "8B dual-scale validation"
    python scripts/run_8b_think_postprocess_after_seeds.py \
        --results_dir "${RESULTS_DIR}" \
        2>&1 | tee "$LOG_DIR/phase6_8b.log" && mark_phase_done 6 || echo "[WARN] Phase 6 failed; not marking as done"
fi

# Phase 7: Significance tests
if should_run 7; then
    log_phase 7 "Significance tests"
    PHASE_FAILURES=0
    for f in ${RESULTS_DIR}/template_controller_rows_*.csv; do
        [ -f "$f" ] || continue
        echo "[Phase 7] Significance: $f"
        python scripts/run_template_controller_significance.py \
            --rows_csv "$f" \
            2>&1 | tee -a "$LOG_DIR/phase7_significance.log" || PHASE_FAILURES=$((PHASE_FAILURES + 1))
    done
    [ "$PHASE_FAILURES" -eq 0 ] && mark_phase_done 7 || echo "[WARN] Phase 7 had $PHASE_FAILURES failure(s); not marking as done"
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
