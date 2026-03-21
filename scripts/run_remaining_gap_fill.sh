#!/bin/bash
set -e
cd /workspace/nips-adathink

echo "=== Phase 3: Parametric Controller (remaining) ==="

# bbh_27b
echo "[$(date +%H:%M:%S)] Parametric Controller bbh_27b..."
python3 scripts/run_parametric_budget_controller.py \
  --input_csvs results/per_sample_bbh_Qwen3.5_27B_20260320_1*.csv \
  --lambda_cost 0.15 --norm_tokens 4096.0 \
  --output_json results/param_controller_bbh_27b_20260320_160051.json \
  --output_csv results/param_controller_rows_bbh_27b_20260320_160051.csv 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

# math500_8b
echo "[$(date +%H:%M:%S)] Parametric Controller math500_8b..."
python3 scripts/run_parametric_budget_controller.py \
  --input_csvs results/per_sample_math500_Qwen3_8B_20260320_0*.csv \
  --lambda_cost 0.15 --norm_tokens 2048.0 \
  --output_json results/param_controller_math500_8b_20260320_160051.json \
  --output_csv results/param_controller_rows_math500_8b_20260320_160051.csv 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

# bbh_8b
echo "[$(date +%H:%M:%S)] Parametric Controller bbh_8b..."
python3 scripts/run_parametric_budget_controller.py \
  --input_csvs results/per_sample_bbh_Qwen3_8B_20260320_0*.csv \
  --lambda_cost 0.15 --norm_tokens 1024.0 \
  --output_json results/param_controller_bbh_8b_20260320_160051.json \
  --output_csv results/param_controller_rows_bbh_8b_20260320_160051.csv 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

echo ""
echo "=== Phase 4: Overthinking Aggregate ==="

# math500_27b
echo "[$(date +%H:%M:%S)] Overthinking math500_27b..."
python3 scripts/run_overthinking_aggregate.py \
  --input_csvs results/per_sample_math500_Qwen3.5_27B_20260320_0*.csv \
  --output_json results/overthinking_math500_27b_17seed_20260320_160051.json 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

# bbh_27b
echo "[$(date +%H:%M:%S)] Overthinking bbh_27b..."
python3 scripts/run_overthinking_aggregate.py \
  --input_csvs results/per_sample_bbh_Qwen3.5_27B_20260320_1*.csv \
  --output_json results/overthinking_bbh_27b_17seed_20260320_160051.json 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

# math500_8b
echo "[$(date +%H:%M:%S)] Overthinking math500_8b..."
python3 scripts/run_overthinking_aggregate.py \
  --input_csvs results/per_sample_math500_Qwen3_8B_20260320_0*.csv \
  --output_json results/overthinking_math500_8b_9seed_20260320_160051.json 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

# bbh_8b
echo "[$(date +%H:%M:%S)] Overthinking bbh_8b..."
python3 scripts/run_overthinking_aggregate.py \
  --input_csvs results/per_sample_bbh_Qwen3_8B_20260320_0*.csv \
  --output_json results/overthinking_bbh_8b_7seed_20260320_160051.json 2>&1 | tail -3
echo "[$(date +%H:%M:%S)] Done."

echo ""
echo "=== Phase 5: Significance Tests ==="

for tag_budget in "math500_27b 4096 8192.0" "bbh_27b 2048 4096.0" "math500_8b 1024 2048.0" "bbh_8b 512 1024.0"; do
  set -- $tag_budget
  tag=$1; cb=$2; nt=$3
  
  echo "[$(date +%H:%M:%S)] Template Significance ${tag} vs Fixed@${cb}..."
  python3 scripts/run_template_controller_significance.py \
    --rows_csv results/template_controller_rows_${tag}_20260320_160051.csv \
    --compare_budget $cb --lambda_cost 0.15 --norm_tokens $nt \
    --output_json results/template_significance_${tag}_vs_fixed${cb}_20260320_160051.json 2>&1 | tail -3
  echo "[$(date +%H:%M:%S)] Done."
  
  echo "[$(date +%H:%M:%S)] Param Significance ${tag} vs Fixed@${cb}..."
  python3 scripts/run_template_controller_significance.py \
    --rows_csv results/param_controller_rows_${tag}_20260320_160051.csv \
    --compare_budget $cb --lambda_cost 0.15 --norm_tokens $nt \
    --output_json results/param_significance_${tag}_vs_fixed${cb}_20260320_160051.json 2>&1 | tail -3
  echo "[$(date +%H:%M:%S)] Done."
done

echo ""
echo "=== ALL DONE ==="
echo "New result files:"
ls -1 results/*_160051* | wc -l
echo "files with timestamp 160051"
