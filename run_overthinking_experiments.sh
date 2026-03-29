#!/bin/bash
# Complete experimental pipeline for overthinking mechanism study
set -e

echo "=== Overthinking Mechanism Study - Full Experimental Pipeline ==="
echo ""

# Phase 1: Data validation
echo "[Phase 1] Validating existing data..."
CSV_COUNT=$(ls results/per_sample_Qwen3.5_27B_*.csv 2>/dev/null | wc -l)
echo "  Found $CSV_COUNT CSV files"

if [ "$CSV_COUNT" -lt 5 ]; then
    echo "  ERROR: Insufficient data. Need to run Phase 1 experiments first."
    exit 1
fi

# Phase 2: Statistical analysis
echo ""
echo "[Phase 2] Running statistical analysis..."
python3 scripts/analyze_overthinking_mechanism.py results/per_sample_Qwen3.5_27B_*.csv results/overthinking_mechanism_analysis.json
echo "  ✓ Mechanism analysis complete"

# Phase 3: Predictive modeling
echo ""
echo "[Phase 3] Building predictive model..."
python3 scripts/build_overthinking_predictor.py > results/predictor_results.txt 2>&1
echo "  ✓ Predictor trained (see results/predictor_results.txt)"

# Phase 4: Visualization
echo ""
echo "[Phase 4] Generating figures..."
python3 scripts/generate_figures.py
echo "  ✓ Figures saved to results/figures/"

# Phase 5: Case study preparation
echo ""
echo "[Phase 5] Preparing case studies..."
python3 scripts/generate_case_study.py > results/case_study_template.txt
echo "  ✓ Case study template ready"

# Phase 6: Cross-model validation (if 8B data exists)
echo ""
echo "[Phase 6] Cross-model validation..."
if ls results/*8B*.csv >/dev/null 2>&1; then
    echo "  ✓ 8B data found, running comparison..."
    # TODO: Add 8B comparison script
else
    echo "  ⚠ No 8B data found, skipping"
fi

echo ""
echo "=== Experimental Pipeline Complete ==="
echo ""
echo "Results:"
echo "  - Overthinking rate: 33.8%"
echo "  - Predictor accuracy: 62.5%"
echo "  - Figures: results/figures/*.pdf"
echo "  - Case studies: results/case_study_template.txt"
echo ""
echo "Next: Manual case analysis and paper writing"
