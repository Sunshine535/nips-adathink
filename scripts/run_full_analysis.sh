#!/bin/bash
# Generate all analysis and figures for overthinking paper
set -e

echo "=== Overthinking Analysis Pipeline ==="
echo ""

# Step 1: Mechanism analysis (already done)
echo "[1/4] Mechanism analysis... ✓ (already completed)"

# Step 2: Predictor model
echo "[2/4] Building predictor model..."
python3 scripts/build_overthinking_predictor.py > results/predictor_output.txt 2>&1
echo "  ✓ Saved to results/predictor_output.txt"

# Step 3: Generate figures
echo "[3/4] Generating figures..."
python3 scripts/generate_figures.py
echo "  ✓ Figures in results/figures/"

# Step 4: Case study template
echo "[4/4] Generating case study..."
python3 scripts/generate_case_study.py > results/case_study_template.txt
echo "  ✓ Template in results/case_study_template.txt"

echo ""
echo "=== Analysis Complete ==="
echo "Next: Manual review of case_study_template.txt"
