# Paper Plan: AdaThink — Adaptive Test-Time Compute Control for LLMs

## Title
**AdaThink: Adaptive Budget Allocation for Test-Time Compute in Large Language Models**

## Core Claim
A learned budget controller that dynamically allocates reasoning tokens per question significantly outperforms fixed-budget decoding on a quality–cost Pareto frontier across multiple benchmarks and model scales.

## Claims → Evidence Matrix

| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Overthinking exists: longer reasoning does not monotonically improve accuracy | GSM8K-27B 23-seed pooled: Acc@256=0.462 vs Acc@512=0.487, ΔAcc=-0.025 at 2× cost | §4.1 |
| C2 | Template controller beats best fixed budget at matched cost on GSM8K-27B | ΔAcc=+0.142 [0.120, 0.165], ΔTokens=-16.8, ΔUtility=+0.147 [0.125, 0.170] (n=920) | §4.2 |
| C3 | Gains generalize across benchmarks (MATH500, BBH) | MATH500-27B: ΔAcc=+0.121 [0.088, 0.153]; BBH-27B: ΔAcc=+0.113 [0.088, 0.140] | §4.3 |
| C4 | Gains generalize across model scales (8B, 27B) | MATH500-8B: ΔAcc=+0.289 [0.242, 0.339]; BBH-8B: ΔAcc=+0.121 [0.075, 0.168] | §4.3 |
| C5 | Each controller component contributes | Ablation: full > halting-only > no-branch > mid-only across 5 settings | §4.4 |
| C6 | Adaptive control reduces latency at matched accuracy | Wallclock analysis: adaptive latency ≤ max-budget latency | §4.5 |
| C7 | Value-based controller enables fine-grained cost control | Penalty sweep on 8B-think: pen=0.8 gives ΔAcc=+0.046 [0.007, 0.086] at ΔTokens=+11.7 | §4.3 |

## Section Structure

### Abstract (~150 words)
- Problem: overthinking waste in LLM reasoning
- Method: learned budget controller (template + value-based)
- Key result: +12–29% accuracy at matched/lower cost across 3 benchmarks × 2 scales

### §1 Introduction (~1.5 pages)
- Motivation: scaling test-time compute, overthinking phenomenon
- Research questions (3)
- Contributions (4 bullets)

### §2 Related Work (~0.75 pages)
- Test-time compute scaling
- Adaptive computation (early exit, speculative decoding)
- Self-consistency and verification

### §3 Method (~2 pages)
- §3.1 Problem formulation: quality–cost Pareto optimization
- §3.2 Budget controller design: template-based (difficulty → budget mapping)
- §3.3 Training: leave-one-out cross-validation, utility objective
- §3.4 Value-based extension: per-budget correctness prediction + cost-aware action selection
- §3.5 Parametric controller: feature-based linear policy

### §4 Experiments (~3 pages)
- §4.1 Setup: benchmarks, models, baselines, metrics
- §4.2 Main results: template controller vs fixed baselines (Table 1)
- §4.3 Cross-benchmark and cross-scale generalization (Table 2)
- §4.4 Ablation study (Table 3)
- §4.5 Latency analysis (Table 4)
- §4.6 Controller analysis: penalty sweep, oracle gap

### §5 Discussion and Conclusion (~0.5 pages)

### Appendix
- A: Full per-seed results
- B: Reproducibility details
- C: Additional ablations

## Figures and Tables

| ID | Type | Content |
|----|------|---------|
| Fig 1 | Teaser | Overthinking example + Pareto curve sketch |
| Table 1 | Main table | GSM8K results: all controllers vs fixed baselines |
| Table 2 | Cross-benchmark | MATH500 + BBH results across 2 scales |
| Table 3 | Ablation | Component ablation (full/halting/no-branch/max/mid) |
| Table 4 | Latency | Wallclock latency comparison |
| Fig 2 | Method | Controller architecture diagram |
| Fig 3 | Pareto | Quality–cost Pareto frontiers across benchmarks |
| Fig 4 | Penalty sweep | Value controller penalty vs accuracy/cost tradeoff |

## Key Numbers (for quick reference)

### GSM8K-27B (n=920, 23 seeds)
- Template: acc=0.604, tok=269.5 | Fixed256: acc=0.462, tok=286.3
- **ΔAcc=+0.142** [0.120, 0.165], **ΔUtility=+0.147** [0.125, 0.170]

### MATH500-27B (n=680, 17 seeds)
- Template: acc=0.491, tok=4637 | Fixed4096: acc=0.371, tok=3570
- **ΔAcc=+0.121** [0.088, 0.153], **ΔUtility=+0.101** [0.069, 0.134]

### BBH-27B (n=680, 17 seeds)
- Template: acc=0.596, tok=1802 | Fixed2048: acc=0.482, tok=1624
- **ΔAcc=+0.113** [0.088, 0.140], **ΔUtility=+0.107** [0.082, 0.132]

### GSM8K-8B (n=280, 7 seeds)
- Value(pen=0.8): **ΔAcc=+0.046** [0.007, 0.086], ΔTokens=+11.7

### MATH500-8B (n=360, 9 seeds)
- Template: acc=0.375, tok=1464 | Fixed1024: acc=0.086, tok=1030
- **ΔAcc=+0.289** [0.242, 0.339], **ΔUtility=+0.257** [0.212, 0.303]

### BBH-8B (n=280, 7 seeds)
- Template: acc=0.293, tok=585 | Fixed512: acc=0.171, tok=478
- **ΔAcc=+0.121** [0.075, 0.168], **ΔUtility=+0.106** [0.060, 0.151]
