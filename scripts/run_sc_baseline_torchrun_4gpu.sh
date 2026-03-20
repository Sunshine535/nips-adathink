#!/usr/bin/env bash
set -euo pipefail

if [ -z "${TORCHRUN_BIN:-}" ]; then
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
if [ -z "${TORCHRUN_BIN}" ]; then
  echo "torchrun not found in PATH. Activate a CUDA conda env first." >&2
  exit 1
fi

: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${XDG_CACHE_HOME:=$HOME/.cache}"
: "${HF_HOME:=${XDG_CACHE_HOME}/huggingface}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

export HF_ENDPOINT XDG_CACHE_HOME HF_HOME CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

"${TORCHRUN_BIN}" --standalone --nproc_per_node=8 \
  "${SCRIPT_DIR}/run_gsm8k_sc_baseline.py" "$@"
