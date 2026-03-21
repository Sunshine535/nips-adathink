#!/bin/bash
set -e
cd /workspace/nips-adathink

echo "=== SC@k Baselines for All New Benchmarks ==="
echo ""

# MATH-500 8B: greedy@2048, SC@4 with 512 each (matched cost: 4x512=2048)
echo "[$(date +%H:%M:%S)] MATH-500 8B..."
for seed in 101 202 303; do
  echo "  data_seed=$seed"
  python3 scripts/run_sc_baseline_vllm.py \
    --model Qwen/Qwen3-8B --benchmark math500 \
    --n_samples 40 --sc_k 4 --sc_budget 512 --greedy_budget 2048 \
    --enable_thinking --data_seed $seed --results_dir results 2>&1 | grep -E "RESULTS|Greedy|SC@|Delta|Saved"
done

# BBH 8B: greedy@1024, SC@4 with 256 each
echo ""
echo "[$(date +%H:%M:%S)] BBH 8B..."
for seed in 101 202 303; do
  echo "  data_seed=$seed"
  python3 scripts/run_sc_baseline_vllm.py \
    --model Qwen/Qwen3-8B --benchmark bbh \
    --n_samples 40 --sc_k 4 --sc_budget 256 --greedy_budget 1024 \
    --enable_thinking --data_seed $seed --results_dir results 2>&1 | grep -E "RESULTS|Greedy|SC@|Delta|Saved"
done

# Kill any leftover vLLM processes before switching models
pkill -f "vllm" 2>/dev/null || true
sleep 5

# MATH-500 27B: greedy@8192, SC@4 with 2048 each
echo ""
echo "[$(date +%H:%M:%S)] MATH-500 27B..."
for seed in 101 202 303; do
  echo "  data_seed=$seed"
  python3 scripts/run_sc_baseline_vllm.py \
    --model Qwen/Qwen3.5-27B --benchmark math500 \
    --n_samples 40 --sc_k 4 --sc_budget 2048 --greedy_budget 8192 \
    --data_seed $seed --results_dir results 2>&1 | grep -E "RESULTS|Greedy|SC@|Delta|Saved"
done

# BBH 27B: greedy@4096, SC@4 with 1024 each
echo ""
echo "[$(date +%H:%M:%S)] BBH 27B..."
for seed in 101 202 303; do
  echo "  data_seed=$seed"
  python3 scripts/run_sc_baseline_vllm.py \
    --model Qwen/Qwen3.5-27B --benchmark bbh \
    --n_samples 40 --sc_k 4 --sc_budget 1024 --greedy_budget 4096 \
    --data_seed $seed --results_dir results 2>&1 | grep -E "RESULTS|Greedy|SC@|Delta|Saved"
done

echo ""
echo "=== ALL SC BASELINES DONE ==="
