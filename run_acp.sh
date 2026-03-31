#!/bin/bash
# =============================================================================
# SenseCore ACP launch script for nips-adathink
#
# Expects Docker image with: PyTorch 2.10, Transformers 5.3.0,
#   TRL, PEFT, DeepSpeed, Accelerate pre-installed.
#
# Usage:
#   bash run_acp.sh                   # full pipeline from Phase 1
#   bash run_acp.sh --from-phase 3    # resume from Phase 3
#   bash run_acp.sh --only-phase 1    # run single phase
#   FORCE_RERUN=1 bash run_acp.sh     # ignore phase markers
# =============================================================================
set -euo pipefail

# === Paths ===
PROJECT_DIR="/data/szs/250010072/nwh/nips-adathink"
DATA_DIR="/data/szs/share/adathink"

MODEL_27B="/data/szs/share/Qwen3.5-27B"
MODEL_8B="/data/szs/share/Qwen3.5-9B"

# === Environment ===
export HF_HOME="${DATA_DIR}/hf_cache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8

mkdir -p "$HF_HOME"

# === GPU detection ===
if ! command -v nvidia-smi &>/dev/null; then
    echo "[ERROR] nvidia-smi not found"
    exit 1
fi
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[ERROR] No GPUs detected"
    exit 1
fi

GPU_IDS=""
for ((i=0; i<NUM_GPUS; i++)); do
    [ -n "$GPU_IDS" ] && GPU_IDS="${GPU_IDS},"
    GPU_IDS="${GPU_IDS}${i}"
done
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | tr -d ' ')

# Validate via torch (use total_memory, not total_mem)
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPUs:    {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    mem_gib = props.total_memory / (1024**3)
    print(f'  GPU {i}: {props.name}  {mem_gib:.1f} GiB')
"

echo "============================================================"
echo " SenseCore ACP — nips-adathink"
echo " GPUs: ${NUM_GPUS} × ${GPU_MEM_MIB} MiB"
echo " PROJECT_DIR: ${PROJECT_DIR}"
echo " DATA_DIR:    ${DATA_DIR}"
echo " MODEL_27B:   ${MODEL_27B}"
echo " MODEL_8B:    ${MODEL_8B}"
echo " HF_HOME:     ${HF_HOME}"
echo "============================================================"

# === Validate local models ===
for M in "$MODEL_27B" "$MODEL_8B"; do
    if [ ! -d "$M" ]; then
        echo "[ERROR] Model dir not found: $M"
        exit 1
    fi
    echo "[OK] Model found: $M"
done

# === Project setup ===
cd "$PROJECT_DIR"

# Symlink results/ and logs/ to shared storage for persistence
SHARED_RESULTS="${DATA_DIR}/results"
SHARED_LOGS="${DATA_DIR}/logs"
mkdir -p "$SHARED_RESULTS" "$SHARED_LOGS"

if [ ! -L "$PROJECT_DIR/results" ]; then
    if [ -d "$PROJECT_DIR/results" ]; then
        # Merge existing results into shared, then replace with symlink
        cp -rn "$PROJECT_DIR/results/"* "$SHARED_RESULTS/" 2>/dev/null || true
        rm -rf "$PROJECT_DIR/results"
    fi
    ln -sf "$SHARED_RESULTS" "$PROJECT_DIR/results"
    echo "[SYMLINK] results/ -> $SHARED_RESULTS"
fi

LOG_DIR="${PROJECT_DIR}/results/logs"
if [ ! -L "$LOG_DIR" ] && [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$SHARED_LOGS"
    ln -sf "$SHARED_LOGS" "$LOG_DIR"
    echo "[SYMLINK] results/logs/ -> $SHARED_LOGS"
fi
mkdir -p "$PROJECT_DIR/results/logs"

# === Install missing deps (Docker should have most) ===
pip install --quiet datasets scipy matplotlib pandas huggingface_hub tqdm pyyaml 2>/dev/null || true

# === Phase markers for resume ===
RESULTS_DIR="results"
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

get_torchrun_cmd() {
    local nproc="${1:-$NUM_GPUS}"
    echo "torchrun --nproc_per_node=$nproc --master_port=$(( RANDOM % 10000 + 20000 ))"
}

# === Parse arguments ===
FROM_PHASE=1
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
    if phase_done "$phase"; then
        echo "[SKIP] Phase $phase already completed (marker: $PHASE_MARKERS/phase${phase}.done)"
        return 1
    fi
    if [[ $ONLY_PHASE -ge 0 ]]; then [[ $phase -eq $ONLY_PHASE ]]; else [[ $phase -ge $FROM_PHASE ]]; fi
}

log_phase() { echo ""; echo "=== Phase $1: $2 === [$(date '+%Y-%m-%d %H:%M:%S')]"; echo ""; }

SEEDS=(42 123 456)

echo ""
echo "============================================================"
echo " Pipeline starting: $(date)"
echo " FROM_PHASE=$FROM_PHASE  ONLY_PHASE=$ONLY_PHASE"
echo " FORCE_RERUN=${FORCE_RERUN:-0}"
echo "============================================================"

# Phase 0: SKIP — models are pre-installed at local paths
echo "[SKIP] Phase 0: Models pre-installed at $MODEL_27B, $MODEL_8B"
mark_phase_done 0

# Phase 1: Budget sweep on GSM8K (27B)
if should_run 1; then
    log_phase 1 "Budget sweep (64/128/256) on GSM8K"
    P1_FAIL=0
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 1] Budgets=64,128,256 Seed=$SEED Model=$MODEL_27B"
        $(get_torchrun_cmd) scripts/run_gsm8k_experiment.py \
            --model "$MODEL_27B" \
            --budgets 64 128 256 \
            --seed "$SEED" \
            --results_dir "${RESULTS_DIR}" \
            --enable_thinking \
            --skip_local_model_check \
            2>&1 | tee "results/logs/phase1_s${SEED}.log" || { echo "[ERROR] Phase 1 seed=$SEED failed"; P1_FAIL=1; }
    done
    if [ $P1_FAIL -eq 0 ]; then mark_phase_done 1; else echo "[ERROR] Phase 1 had failures"; exit 1; fi
fi

# Phase 2: Self-consistency baseline (SC@8, SC@16)
if should_run 2; then
    log_phase 2 "Self-consistency baseline (SC@8, SC@16)"
    P2_FAIL=0
    for SC in 8 16; do
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 2] SC@${SC} Seed=$SEED Model=$MODEL_27B"
            $(get_torchrun_cmd) scripts/run_gsm8k_sc_baseline.py \
                --model "$MODEL_27B" \
                --sc_n "$SC" \
                --seed "$SEED" \
                --results_dir "${RESULTS_DIR}" \
                --enable_thinking \
                --skip_local_model_check \
                2>&1 | tee "results/logs/phase2_sc${SC}_s${SEED}.log" || { echo "[ERROR] Phase 2 SC@${SC} seed=$SEED failed"; P2_FAIL=1; }
        done
    done
    if [ $P2_FAIL -eq 0 ]; then mark_phase_done 2; else echo "[ERROR] Phase 2 had failures"; exit 1; fi
fi

# Phase 3: Learned budget controller (CPU post-processing)
if should_run 3; then
    log_phase 3 "Learned budget controller"
    INPUT_CSVS=( $(ls ${RESULTS_DIR}/per_sample_*.csv 2>/dev/null) )
    if [ ${#INPUT_CSVS[@]} -gt 0 ]; then
        P3_FAIL=0
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 3] Learned controller Seed=$SEED (${#INPUT_CSVS[@]} input CSVs)"
            python scripts/run_learned_budget_controller.py \
                --input_csvs "${INPUT_CSVS[@]}" \
                --seed "$SEED" \
                --output_dir "${RESULTS_DIR}" \
                2>&1 | tee "results/logs/phase3_learned_s${SEED}.log" || P3_FAIL=$((P3_FAIL + 1))
        done
        [ "$P3_FAIL" -eq 0 ] && mark_phase_done 3 || echo "[WARN] Phase 3 had $P3_FAIL failure(s); not marking as done"
    else
        echo "[Phase 3] SKIPPED: no per_sample CSVs found in ${RESULTS_DIR}/"
    fi
fi

# Phase 4: Value-based budget controller (CPU post-processing)
if should_run 4; then
    log_phase 4 "Value-based budget controller"
    INPUT_CSVS=( $(ls ${RESULTS_DIR}/per_sample_*.csv 2>/dev/null) )
    if [ ${#INPUT_CSVS[@]} -gt 0 ]; then
        P4_FAIL=0
        for SEED in "${SEEDS[@]}"; do
            echo "[Phase 4] Value controller Seed=$SEED (${#INPUT_CSVS[@]} input CSVs)"
            python scripts/run_value_budget_controller.py \
                --input_csvs "${INPUT_CSVS[@]}" \
                --seed "$SEED" \
                2>&1 | tee "results/logs/phase4_value_s${SEED}.log" || P4_FAIL=$((P4_FAIL + 1))
        done
        [ "$P4_FAIL" -eq 0 ] && mark_phase_done 4 || echo "[WARN] Phase 4 had $P4_FAIL failure(s); not marking as done"
    else
        echo "[Phase 4] SKIPPED: no per_sample CSVs found in ${RESULTS_DIR}/"
    fi
fi

# Phase 5: Policy search on GSM8K (27B)
if should_run 5; then
    log_phase 5 "Policy search"
    P5_FAIL=0
    for SEED in "${SEEDS[@]}"; do
        echo "[Phase 5] Policy search Seed=$SEED Model=$MODEL_27B"
        $(get_torchrun_cmd) scripts/run_gsm8k_policy_search.py \
            --model "$MODEL_27B" \
            --seed "$SEED" \
            --results_dir "${RESULTS_DIR}" \
            --enable_thinking \
            --skip_local_model_check \
            2>&1 | tee "results/logs/phase5_policy_s${SEED}.log" || { echo "[ERROR] Phase 5 seed=$SEED failed"; P5_FAIL=1; }
    done
    if [ $P5_FAIL -eq 0 ]; then mark_phase_done 5; else echo "[ERROR] Phase 5 had failures"; exit 1; fi
fi

# Phase 6: 8B dual-scale validation
if should_run 6; then
    log_phase 6 "8B dual-scale validation"
    python scripts/run_8b_think_postprocess_after_seeds.py \
        --results_dir "${RESULTS_DIR}" \
        2>&1 | tee "results/logs/phase6_8b.log" && mark_phase_done 6 || echo "[WARN] Phase 6 failed; not marking as done"
fi

# Phase 7: Significance tests
if should_run 7; then
    log_phase 7 "Significance tests"
    P7_FAIL=0
    for f in ${RESULTS_DIR}/template_controller_rows_*.csv; do
        [ -f "$f" ] || continue
        echo "[Phase 7] Significance: $f"
        python scripts/run_template_controller_significance.py \
            --rows_csv "$f" \
            2>&1 | tee -a "results/logs/phase7_significance.log" || P7_FAIL=$((P7_FAIL + 1))
    done
    [ "$P7_FAIL" -eq 0 ] && mark_phase_done 7 || echo "[WARN] Phase 7 had $P7_FAIL failure(s); not marking as done"
fi

# === Done ===
echo ""
echo "============================================================"
echo " AdaThink — Pipeline Complete [$(date)]"
echo " Results: ${RESULTS_DIR}/ -> ${SHARED_RESULTS}"
echo " Logs:    results/logs/"
echo "============================================================"

cat > "${RESULTS_DIR}/.pipeline_done" << DONEEOF
{
  "project": "nips-adathink",
  "completed_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "gpus": "${NUM_GPUS}",
  "model_27b": "${MODEL_27B}",
  "model_8b": "${MODEL_8B}",
  "status": "PIPELINE_COMPLETE"
}
DONEEOF
echo "[PIPELINE_COMPLETE] All experiments finished."
