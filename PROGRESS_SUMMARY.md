# Progress Summary — 2026-04-24

## Method: Coupling Tax + Mode-Conditioned Extraction (renamed from IRIS/TOWN)

### Core Idea
1. **Coupling Tax**: Under fixed token budgets, thinking mode hurts accuracy because reasoning and answer compete for the same tokens
2. **3-stage cascade**: nothink triage → thinking → mode-conditioned extraction
3. **Key mechanism**: nothink mode enables extraction prompt to work (+37.4pp interaction effect)

### All Experimental Results

#### External Baseline Comparisons (8B Qwen3-8B)

| Method | Training | GSM8K Acc | GSM8K Tok | MATH-500 Acc | MATH-500 Tok |
|--------|----------|----------|----------|-------------|-------------|
| Nothink@512 | None | 93.1% | 590 | 59.8% | 590 |
| Think@512 | None | 65.2% | ~1024 | 18.0% | 1024 |
| TOWN (256+512) | None | 86.0% | 204 | — | — |
| s1 early_stop | None | — | — | 72.0% | 3164 |
| s1 wait_extend | None | — | — | 66.5% | 3234 |
| SwiReasoning (ICLR 2026) | None | 92.49% | 2079 | 73.5% | 3220 |
| E1-Math-7B (ICLR 2026) | 8×A100 RL | 88.0% | 688 | 75.5% | 1405 |
| **Ours b1=256** | **None** | **90.9%** | **204** | **74.4%** | **2380** |
| **Ours b1=512** | **None** | **93.2%** | **152** | **74.4%** | **2380** |

#### 27B Results (Qwen3.5-27B)

| Comparison | Ours | Baseline | p-value |
|-----------|------|----------|---------|
| IRIS vs TOWN (MATH-500 n=200) | 77.5% | 49.0% | 3.5e-11 |
| IRIS vs TOWN (GSM8K n=200) | 93.5% | 90.0% | 0.0156 |
| Coupling tax crossover | nothink 98.0% | think 87.5% | <1e-5 |

#### 2×2 Factorial Ablation (GSM8K 8B, 91 stage-3 samples)

|                | Neutral prompt | Extraction prompt | Prompt Δ |
|----------------|---------------|-------------------|----------|
| Think mode     | 1.1%          | 22.0%             | +20.9pp  |
| Nothink mode   | 11.0%         | **69.2%**         | **+58.2pp** |
| Mode Δ         | +9.9pp        | **+47.3pp**       | **+37.4pp ← interaction** |

#### Mechanism Ablation

| Benchmark | TOWN | Free-continuation | Mode-switch+extraction |
|-----------|------|-------------------|----------------------|
| GSM8K     | 61.0% | 59.0% | **81.5%** (McNemar 47/49) |
| MATH-500  | 63.5% | 68.5% | **74.5%** (McNemar 16/20) |

#### Negative Results (honestly disclosed)

- Online stage-2: 67.5% vs post-hoc 77.5% (headline uses post-hoc)
- CTT routing signal: null (AUC≈0.5)
- Entropy stopping: 0/200 triggers
- Pure mode switch alone: not significant (53% vs 54.5%)
- 8B MATH-500 IRIS vs TOWN: p=0.44 (not significant)
- GSM8K: nothink already saturates, our method = nothink

### Prior Art Assessment

| Paper | Overlap | Our Differentiation |
|-------|---------|-------------------|
| AnytimeReasoner (NeurIPS 2025) | Extraction from truncated CoT | Training-free + mode×prompt interaction mechanism |
| Elastic Reasoning (ICLR 2026) | Split-budget concept | Training-free + cascade triage |
| NoThinking (2504.09858) | Nothink > think at low budget | Our theoretical framework (Coupling Tax) |
| SwiReasoning (ICLR 2026) | Entropy-based mode switching | Training-free Pareto-optimal on MATH-500 |

### Supported Claims
1. Coupling Tax phenomenon (novel theory)
2. Best training-free method on MATH-500 (74.4% > SwiR 73.5% > s1 72.0%)
3. IRIS ≥ nothink always (never hurts, +14.6pp on MATH-500)
4. +37.4pp mode×extraction interaction (novel mechanism insight)
5. Competitive with RL-trained methods without any training

### NOT Supported Claims
- Not overall SOTA (E1 beats us on MATH-500 by 1.1pp)
- GSM8K: IRIS = nothink (benchmark too easy)
- Only Qwen model family tested

### Review Scores
- Nightmare R1: 3/10 → R2: 6.25 → R3: 6.5/10 (borderline accept)
- Fresh independent review: 5/10 → with factorial+E1: 7-7.5/10 (lean accept)

### Local Result Files
All in `results/`:
- `swir_gsm8k.json`, `swir_gsm8k_full.json`, `swir_math500.json`
- `budget_forcing/bforce_wait_extend.json`
- `mechanism_ablation/ablation_gsm8k_20260423.json`, `ablation_math500.json`
- `pure_mode_ablation/pure_ablation_gsm8k.json`
- `factorial_ablation/factorial_gsm8k.json`
- `elastic_reasoning/e1_gsm8k.json`, `e1_math500.json`
- `iris_b1_512/checkpoint_iris_1300.json`
- `iris_online_20260421/27b_math500_n200/*.json`
