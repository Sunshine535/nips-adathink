#!/usr/bin/env bash
set -euo pipefail

# You can override these before running.
: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${XDG_CACHE_HOME:=/mnt/ae22ef98-e02a-4dfa-acf4-a307e8941cd9/nwh}"
: "${HF_HOME:=${XDG_CACHE_HOME}/huggingface}"

export HF_ENDPOINT XDG_CACHE_HOME HF_HOME

# Prevent accidental multi-starts that fight over HF lock files.
LOCK_FILE="${XDG_CACHE_HOME}/.download_hf_models.lock"
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another download job is already running (lock: $LOCK_FILE)." >&2
  echo "Stop the existing job first, or wait for it to finish." >&2
  exit 1
fi

# Default to downloading all groups. You can pass custom args, e.g.:
#   bash scripts/download_hf_models.sh --group qwen3_low_cost qwen35_main
#   bash scripts/download_hf_models.sh --group qwen35_large --dry-run
#   bash scripts/download_hf_models.sh --group embedding reranker --dry-run

if [ "${CONDA_DEFAULT_ENV:-}" = "nips_adathink" ]; then
  python -u scripts/download_hf_models.py "$@"
elif command -v conda >/dev/null 2>&1 && conda env list | rg -q '^nips_adathink\s'; then
  # Keep stdout/stderr streaming in real-time; default conda-run captures output.
  conda run --no-capture-output -n nips_adathink python -u scripts/download_hf_models.py "$@"
else
  python -u scripts/download_hf_models.py "$@"
fi
