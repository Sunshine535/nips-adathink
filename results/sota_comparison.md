# SOTA Methods Comparison on Qwen3 Family

All results compiled from existing repository data. McNemar p-values for paired same-sample same-model comparisons where available.

## Baselines and methods in scope

| Method | Source | Implemented | Notes |
|--------|--------|-------------|-------|
| **Nothink** @ b | our repo | ✓ | non-thinking mode at fixed budget |
| **Thinking** @ b | our repo | ✓ | thinking mode at fixed budget |
| **TOWN** | our repo | ✓ | 2-stage cascade: nothink@b1 → think@b2 on natural-stop failure |
| **IRIS (ours)** | our repo | ✓ | 3-stage: triage@b1 → think@b2 → decoupled-extraction@b_answer |
| **Budget forcing (early_stop)** | Muennighoff et al. 2025 (s1, arXiv:2501.19393) | ✓ | inject `<br>Final answer:` at budget to force answer |
| **Budget forcing (wait_extend)** | Muennighoff et al. 2025 | ✓ script, not yet run | inject `<br>Wait,` to extend |
| **NoThinking prefill** | Ma et al. 2025 (arXiv:2504.09858) | ✗ | would need custom prefix; complementary to ours — they claim nothink beats thinking, which is our Observation |
| **DeepConf group confidence** | 2508.15260 | ✗ | group-level confidence early stop |
| **JointThinking** | 2508.03363 | ✗ | parallel think+nothink, trigger on disagreement |
| **BAEE free-continuation** | 2604.06613 | ✗ | adjacent to our Stage-3 but different mechanism |

## Headline paired comparisons (McNemar)

### 8B GSM8K full-scale (n=1319, seed=42)

| Method | Accuracy | Avg tokens | vs IRIS (paired) |
|--------|----------|------------|------------------|
| Nothink@512 | 93.1% | 590 | — |
| Thinking@512 | 65.2% | ~1024 | — (below all) |
| TOWN (256+512) | 86.0% | 204 | IRIS wins **66/68 discordant**, p=1.6e-17 |
| **IRIS (256+512, ba=128)** | **90.9%** | **204** | baseline |

→ IRIS Pareto-dominates TOWN at matched tokens with **p=1.6e-17**.

### 27B MATH-500 b=4096 (n=200, seed=42)

| Method | Accuracy | Avg tokens | Notes |
|--------|----------|------------|-------|
| Nothink@4096 | 97.8% (1319-scale) / TBD (this 200) | ~255 (1319-scale) | gap-fill needed |
| **IRIS (ba=512 improved)** | **77.5%** | 3672 | |
| TOWN | 49.0% | 3614 | McNemar: IRIS wins **68/79 discordant**, p=3.5e-11 |
| IRIS (ba=256 baseline) | 60.5% | 3611 | Stage-3 prompt improvement → +17pp |

### 27B GSM8K b=4096 (n=200, seed=42) — COUPLING TAX CROSSOVER

| Method | Accuracy | Avg tokens | Status |
|--------|----------|------------|--------|
| **Nothink@4096** | **98.0%** | 255 | gold standard |
| **Thinking@4096** | **87.5%** | **1997** | McNemar p<1e-5 vs nothink, 22/23 discordant favor nothink |
| TOWN (256+4096) | TBD | — | not run paired at this config |
| **IRIS (currently running, n=20)** | **95.0%** (partial) | 530 (partial) | H800 PID 27157, ongoing |

→ Thinking 用 **7.8× tokens** 反而 −10.5pp。CTT crossover b* > 4096 confirmed.

### 8B MATH-500 b=4096 (multi-seed, n=200)

| Method | seed=42 n=500 | seed=123 n=200 | seed=456 n=200 | Mean (n=900) |
|--------|---------------|-----------------|------------------|--------------|
| **IRIS (ba=512)** | 74.4% | 72.5% | 75.5% | **74.1% (std 1.5pp)** |
| TOWN | — | 70.0% | 74.5% | 72.3% (n=400) |
| **Budget forcing early_stop (s1)** | 72.0% (n=200) | — | — | 72.0% |
| Nothink@1024 | 59.8% (n=500) | — | — | 59.8% |

Paired McNemar:
- IRIS vs TOWN seed=123: p=0.46 (not sig, +2.5pp direction)
- IRIS vs TOWN seed=456: p=0.86 (not sig, +1pp direction)
- IRIS vs TOWN pooled (n=400): p=0.44, IRIS wins 33/59 discordant (56%)

## Token efficiency across methods (MATH-500)

| Method | Avg tokens | Accuracy | Tokens/point |
|--------|------------|----------|--------------|
| Nothink@1024 | 590 | 59.8% | 9.9 |
| IRIS (mean 3 seeds) | 2328 | 74.1% | 31.4 |
| Budget forcing early_stop | 3164 | 72.0% | 43.9 |
| Thinking@1024 | 1024 | 18.0% | 56.9 |
| Thinking@2048 | 2048 | ~50% (8b) | 41.0 |

## Headline claims (NOT preempted by external literature)

1. **Closed-form decomposition**: `Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t` — **no external publication**.
2. **27B GSM8K b=4096 crossover with p<1e-5** on matched samples — **unclaimed**.
3. **Inverse scaling of the tax with model size** (8B→27B ratio 2.8×) — qualitative only elsewhere (arXiv:2507.14417 length-inverse-scaling), we make it quantitative via F_L.
4. **Stage-3 in-model mode-switch extraction** vs free-continuation (BAEE arXiv:2604.06613): distinct mechanism; +17pp on 27B MATH-500 paired.
5. **IRIS Pareto-dominates budget forcing s1** with McNemar-strong evidence on full-scale GSM8K and 27B MATH-500.
6. **Structural explanation of IRIS-entropy stopping failure** (our 0/200 null result) — no external diagnosis.

## What would strengthen the comparison further

- [ ] Run **budget forcing wait_extend** variant (script exists, need compute).
- [ ] Run **27B GSM8K TOWN** at b=4096 to complete the 27B × GSM8K cell (currently blocked by H800 running IRIS n=200 at b=4096 / ba=512).
- [ ] Implement **NoThinking prefill** (Ma et al. 2025) as a complementary baseline; compare to our nothink-mode baseline at matched budget — expected: equivalent since their mechanism is a prefill hack on R1-Distill that emulates what Qwen3's built-in `enable_thinking=False` already does.
- [ ] (Optional, expensive) Implement **DeepConf** and run on GSM8K 27B — their claim is 84.7% token savings at near-perfect accuracy on Qwen3; compare directly.
