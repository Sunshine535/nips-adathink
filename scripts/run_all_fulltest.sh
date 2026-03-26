#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5
export TRANSFORMERS_VERBOSITY=error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJ_DIR/results/fulltest"
mkdir -p "$RESULTS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

MODEL_8B="Qwen/Qwen3-8B"
MODEL_27B="Qwen/Qwen3.5-27B"

# ── Phase 1: Download & cache 8B model ──────────────────────────────────────
log "Phase 1: Downloading $MODEL_8B ..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_id = '$MODEL_8B'
print(f'Downloading tokenizer for {model_id}...')
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'Downloading model for {model_id}...')
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cpu')
print('8B model cached successfully.')
"
log "Phase 1 complete."

# ── Phase 2: 8B Full-tests ──────────────────────────────────────────────────
run_8b_fulltest() {
    local bench="$1"
    local budgets="$2"
    local marker="$RESULTS_DIR/.done_8b_${bench}"

    if [ -f "$marker" ]; then
        log "SKIP: 8B ${bench} already done."
        return 0
    fi

    log "Running 8B ${bench} full-test with budgets=${budgets} ..."
    cd "$PROJ_DIR"
    python3 -u scripts/run_experiment.py \
        --benchmark "$bench" \
        --model "$MODEL_8B" \
        --n_samples 99999 \
        --seed 11 \
        --data_seed 42 \
        --budgets $budgets \
        --enable_thinking \
        --strict_final_only \
        --projection_on_missing_final \
        --projection_max_tokens 16 \
        --prompt_format chat \
        --results_dir "$RESULTS_DIR" \
        --skip_local_model_check \
        $([ "$bench" = "bbh" ] && echo "--bbh_task all")

    touch "$marker"
    log "DONE: 8B ${bench} full-test."
}

run_8b_fulltest gsm8k  "128 256 512"
run_8b_fulltest math500 "512 1024 2048"
run_8b_fulltest bbh     "256 512 1024"

log "All 8B full-tests complete."

# ── Phase 3: Download & cache 27B model ─────────────────────────────────────
log "Phase 3: Downloading $MODEL_27B ..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_id = '$MODEL_27B'
print(f'Downloading tokenizer for {model_id}...')
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'Downloading model for {model_id}...')
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cpu')
print('27B model cached successfully.')
"
log "Phase 3 complete."

# ── Phase 4: 27B Full-tests ─────────────────────────────────────────────────
run_27b_fulltest() {
    local bench="$1"
    local budgets="$2"
    local marker="$RESULTS_DIR/.done_27b_${bench}"

    if [ -f "$marker" ]; then
        log "SKIP: 27B ${bench} already done."
        return 0
    fi

    log "Running 27B ${bench} full-test with budgets=${budgets} ..."
    cd "$PROJ_DIR"
    python3 -u scripts/run_experiment.py \
        --benchmark "$bench" \
        --model "$MODEL_27B" \
        --n_samples 99999 \
        --seed 11 \
        --data_seed 42 \
        --budgets $budgets \
        --enable_thinking \
        --strict_final_only \
        --projection_on_missing_final \
        --projection_max_tokens 16 \
        --prompt_format chat \
        --results_dir "$RESULTS_DIR" \
        --skip_local_model_check \
        $([ "$bench" = "bbh" ] && echo "--bbh_task all")

    touch "$marker"
    log "DONE: 27B ${bench} full-test."
}

run_27b_fulltest gsm8k  "128 256 512"
run_27b_fulltest math500 "2048 4096 8192"
run_27b_fulltest bbh     "1024 2048 4096"

log "ALL FULL-TEST EXPERIMENTS COMPLETE."
date > "$RESULTS_DIR/.all_done"
