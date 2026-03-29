#!/bin/bash
# Quick pilot: Compare uncertainty-based vs honest feature controller
set -e

RESULTS_DIR="results"
CSV_DIR="$RESULTS_DIR"

# Find existing per_sample CSVs from 27B experiments
CSVS=$(find "$CSV_DIR" -name "per_sample_Qwen3.5_27B_*.csv" | head -5)

if [ -z "$CSVS" ]; then
    echo "Error: No per_sample CSVs found. Run Phase 1 first."
    exit 1
fi

echo "=== Pilot Test: Uncertainty vs Honest Feature ==="
echo "Using $(echo $CSVS | wc -w) CSV files for quick validation"
echo ""

# Test 1: Honest feature controller (baseline)
echo "[1/2] Running honest feature controller..."
python3 scripts/run_honest_feature_controller.py \
    --input_csvs $CSVS \
    --lambda_cost 0.15 \
    --output_json "$RESULTS_DIR/pilot_honest_feature.json" \
    --output_csv "$RESULTS_DIR/pilot_honest_feature.csv"

# Test 2: Uncertainty-based controller (new method)
echo ""
echo "[2/2] Running uncertainty-based controller..."
python3 scripts/run_uncertainty_controller.py \
    --input_csvs $CSVS \
    --lambda_cost 0.15 \
    --n_bins 4 \
    --output_json "$RESULTS_DIR/pilot_uncertainty.json" \
    --output_csv "$RESULTS_DIR/pilot_uncertainty.csv"

# Compare results
echo ""
echo "=== Comparison ==="
python3 -c "
import json

honest = json.load(open('$RESULTS_DIR/pilot_honest_feature.json'))
uncertainty = json.load(open('$RESULTS_DIR/pilot_uncertainty.json'))

print(f\"Method              Accuracy  Tokens   Utility\")
print(f\"{'='*50}\")
print(f\"Honest Feature      {honest['aggregate']['accuracy']:.4f}    {honest['aggregate']['avg_tokens']:.1f}    {honest['aggregate']['avg_utility']:.4f}\")
print(f\"Uncertainty-based   {uncertainty['aggregate']['accuracy']:.4f}    {uncertainty['aggregate']['avg_tokens']:.1f}    {uncertainty['aggregate']['avg_utility']:.4f}\")
print(f\"\")
print(f\"ΔAccuracy: {uncertainty['aggregate']['accuracy'] - honest['aggregate']['accuracy']:+.4f}\")
print(f\"ΔUtility:  {uncertainty['aggregate']['avg_utility'] - honest['aggregate']['avg_utility']:+.4f}\")
"

echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/pilot_honest_feature.json"
echo "  - $RESULTS_DIR/pilot_uncertainty.json"
