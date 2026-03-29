#!/bin/bash
# Deploy to GPU servers and rerun experiments with model internals
set -e

echo "=== Deploying Model Internals Experiment ==="

# Server configuration
SERVER_A="216.81.151.3"
PORT_A="11839"
SERVER_B="216.81.245.127"
PORT_B="15276"
KEY="~/.ssh/kun_ed25519"

# Sync code
echo "[1/3] Syncing code to servers..."
rsync -avz -e "ssh -p $PORT_A -i $KEY" \
    --exclude='results/' --exclude='.git/' \
    /home/nwh/nips_15/github_repos/nips-adathink/ \
    root@$SERVER_A:/workspace/nips-adathink-internals/

# Deploy experiment
echo "[2/3] Starting GPU experiments..."
ssh -p $PORT_A -i $KEY root@$SERVER_A << 'ENDSSH'
cd /workspace/nips-adathink-internals
source .venv/bin/activate

# Run with model internals capture
python3 scripts/run_with_model_internals.py \
    --model Qwen/Qwen3.5-27B \
    --dataset gsm8k \
    --budgets 128 256 512 \
    --save_internals \
    --output results/internals/
ENDSSH

echo "[3/3] Experiments launched on GPU"
echo "Monitor: ssh -p $PORT_A -i $KEY root@$SERVER_A 'tail -f /workspace/nips-adathink-internals/run.log'"
