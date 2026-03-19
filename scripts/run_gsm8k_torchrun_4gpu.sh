#!/usr/bin/env bash
set -euo pipefail

# Defaults can be overridden by exporting env vars before execution.
if [ -z "${TORCHRUN_BIN:-}" ]; then
  TORCHRUN_BIN="$(command -v torchrun || true)"
fi
if [ -z "${TORCHRUN_BIN}" ]; then
  echo "torchrun not found in PATH. Activate a CUDA conda env first." >&2
  exit 1
fi
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${XDG_CACHE_HOME:=/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh}"
: "${HF_HOME:=${XDG_CACHE_HOME}/huggingface}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"

export HF_ENDPOINT XDG_CACHE_HOME HF_HOME CUDA_VISIBLE_DEVICES

"${TORCHRUN_BIN}" --standalone --nproc_per_node=4 \
  methods/01_adathink/scripts/run_gsm8k_experiment.py "$@"
