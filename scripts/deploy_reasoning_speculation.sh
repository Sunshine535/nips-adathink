#!/bin/bash
# Deploy and run Reasoning Speculation experiments on remote GPU servers.
#
# Usage:
#   bash scripts/deploy_reasoning_speculation.sh [server]
#
# Servers:
#   server1 = 216.81.151.3:11839  (A100)
#   server2 = 216.81.245.127:15276 (A100-SXM4-80GB)

set -euo pipefail

SERVER="${1:-server1}"
SSH_KEY="$HOME/.ssh/kun_ed25519"

if [ "$SERVER" = "server1" ]; then
    SSH_CMD="ssh -p 11839 -i $SSH_KEY root@216.81.151.3"
    SCP_CMD="scp -P 11839 -i $SSH_KEY"
elif [ "$SERVER" = "server2" ]; then
    SSH_CMD="ssh -p 15276 -i $SSH_KEY root@216.81.245.127"
    SCP_CMD="scp -P 15276 -i $SSH_KEY"
else
    echo "Unknown server: $SERVER"
    echo "Usage: $0 [server1|server2]"
    exit 1
fi

REMOTE_DIR="/workspace/nips-adathink"

echo "=== Deploying to $SERVER ==="

# 1. Sync the new script
echo "Syncing scripts..."
$SCP_CMD scripts/run_reasoning_speculation.py ${SSH_CMD##* }:$REMOTE_DIR/scripts/

# 2. Run experiments in screen sessions
echo "Launching experiments..."

$SSH_CMD << 'REMOTE_SCRIPT'
set -euo pipefail
cd /workspace/nips-adathink
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Phase 1: Small validation run (8B, 50 samples)
echo "=== Phase 1: Validation (8B, n=50) ==="
screen -dmS spec_val bash -c '
cd /workspace/nips-adathink
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

python scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 50 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir results/speculation \
    --run_baselines \
    2>&1 | tee results/logs/speculation_val_8b.log

echo "Validation done! Check results/speculation/"

# If validation passes (nonzero accuracy), run full 200 samples
echo "=== Phase 2: Full run (8B, n=200) ==="
python scripts/run_reasoning_speculation.py \
    --model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --n_samples 200 \
    --k_paths 4 \
    --probe_budget 128 \
    --medium_budget 256 \
    --hard_budget 512 \
    --temperature 0.7 \
    --seed 42 \
    --output_dir results/speculation \
    --run_baselines \
    2>&1 | tee results/logs/speculation_full_8b.log

# Phase 3: K ablation
for K in 2 8; do
    echo "=== K=$K ablation ==="
    python scripts/run_reasoning_speculation.py \
        --model Qwen/Qwen3-8B \
        --benchmark gsm8k \
        --n_samples 200 \
        --k_paths $K \
        --probe_budget 128 \
        --medium_budget 256 \
        --hard_budget 512 \
        --temperature 0.7 \
        --seed 42 \
        --output_dir results/speculation \
        2>&1 | tee results/logs/speculation_k${K}_8b.log
done

echo "All Reasoning Speculation experiments done!"
'

echo "Screen session 'spec_val' launched."
echo "Monitor with: screen -r spec_val"
echo "Logs: results/logs/speculation_*.log"
REMOTE_SCRIPT

echo "=== Deployment complete ==="
echo ""
echo "Monitor progress:"
echo "  $SSH_CMD 'tail -f $REMOTE_DIR/results/logs/speculation_val_8b.log'"
echo ""
echo "Collect results:"
echo "  $SCP_CMD ${SSH_CMD##* }:$REMOTE_DIR/results/speculation/ results_kun/speculation/"
