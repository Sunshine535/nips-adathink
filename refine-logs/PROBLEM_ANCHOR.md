# Problem Anchor

**Date**: 2026-03-27

## Bottom-line Problem
Test-time compute scaling in LLMs exhibits systematic heterogeneity: 33.8% of problems experience accuracy degradation when budget increases from 256→512 tokens (overthinking), while others benefit. Current adaptive budget allocation methods fail completely:
- Honest 3-bit feature controller: -6.5pp vs fixed baseline
- Uncertainty-based controller: -6.4pp vs fixed baseline
- Only lexical routing works (+14.2pp) but it's benchmark-specific memorization, not difficulty estimation

**Must-solve bottleneck**: We need a method that can **predict per-sample optimal compute allocation** without:
1. Relying on lexical/surface features (benchmark overfitting)
2. Requiring model modification or fine-tuning
3. Using expensive multi-pass probing

## Non-goals
- NOT building another lookup table controller
- NOT doing benchmark-specific feature engineering
- NOT requiring model internals access (logits/hidden states unavailable in current setup)
- NOT just analyzing overthinking (need actual solution)

## Constraints
- Training-free or minimal training (no model fine-tuning)
- Must work with black-box LLM APIs
- Compute budget: ~100 GPU-hours for validation
- Timeline: 2-3 weeks to submission-ready
- Venue: NeurIPS 2026 main track

## Success Condition
A method that achieves **+10pp accuracy improvement** over matched-cost fixed baseline on GSM8K-27B, with:
- Generalization across benchmarks (MATH500, BBH)
- Generalization across model scales (8B, 27B)
- No lexical feature dependence
- Theoretical justification for why it works
