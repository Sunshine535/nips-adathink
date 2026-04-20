# Experiment Tracker — Coupling Tax Paper

## Status Legend: ✅ Done | 🔄 Running | ⏳ Auto-Queued | ❌ Blocked

## Pilot Experiments

| ID | Experiment | Status | Key Result | File |
|----|-----------|--------|-----------|------|
| P1 | CDF sufficiency (8B, GSM8K) | ✅ | α_c=98.27%, RMSE=0.18pp | `results/pilot_cdf_sufficiency/` |
| P2 | Portfolio/Pareto (8B, GSM8K) | ✅ | Nothink dominates ≤1024, oracle +0.76pp | `results/pilot_portfolio/` |
| P3 | Entropy stopping (8B, GSM8K) | ✅ | DEAD: 0/90 viable thresholds | `results/iris/threshold_simulation.json` |
| P4 | IRIS mechanism (8B, GSM8K) | ✅ | 100% = TOWN + decoupled answering | `results/iris/mechanism_analysis.json` |
| P5 | MRSD pilot (8B, GSM8K, n=200) | ✅ | **94.0% (+5.0pp), 10 unique wins** | `results/mrsd_pilot/mrsd_*gsm8k*.json` |
| P6 | MRSD pilot (8B, MATH-500, n=200) | 🔄 | 100/200, MRSD=66%(*), NT=46%, +20pp | `results/mrsd_pilot/` |

## Gap-Fill Experiments (Baseline Data)

| ID | Experiment | Status | Key Result | File |
|----|-----------|--------|-----------|------|
| G1 | 8B think/nothink@512 (n=1319) | ✅ | think=56.9%, nt=93.1% | `results/gap_fill/8b_highbudget/` |
| G2 | 8B think/nothink@1024 (n=1319) | ✅ | think=86.1%, nt=93.1% | `results/gap_fill/8b_highbudget/` |
| G3 | 8B think/nothink@2048 (n=1319) | ✅ | **think=93.1%=nothink! b*≈2048** | `results/gap_fill/8b_highbudget/` |
| G4 | 9B nothink/think@256 (n=1319) | ✅ | nt=61.2%, think=3.4% | `results/gap_fill/9b_nothink/` |
| G5 | 9B nothink/think@512 (n=1319) | ✅ | nt=93.2%, think=15.5% | `results/gap_fill/9b_nothink/` |
| G6 | 9B thinking@1024 (n=1319) | 🔄 | 800/1319, acc=41.0% | GPU 0, ETA ~8h |
| G7 | 9B think/nothink@2048 (n=1319) | ⏳ | — | Queued after G6 |

## Full-Scale Experiments (Auto-Queued)

### GPU 1 Queue (`queue_gpu1.sh` — starts after P6)
| ID | Experiment | Status | Priority | Est GPU-h |
|----|-----------|--------|----------|-----------|
| E1 | Split-budget MATH-500 (n=200) | ⏳ | **CRITICAL** | 3 |
| E2 | Split-budget GSM8K (n=200) | ⏳ | HIGH | 2 |
| E3 | MRSD full GSM8K (n=1319) | ⏳ | HIGH | 4 |
| E4 | MRSD full MATH-500 (n=500, Bt=1024) | ⏳ | HIGH | 8 |
| E4b | **MRSD MATH-500 B_think=2048 (n=200)** | ⏳ | **CRITICAL** | 3 |
| E5 | MRSD GSM8K seed=123 (n=200) | ⏳ | MEDIUM | 1 |
| E6 | MRSD MATH-500 seed=123 (n=200, Bt=2048) | ⏳ | MEDIUM | 3 |

### GPU 0 Queue (`queue_gpu0.sh` — starts after G6)
| ID | Experiment | Status | Priority | Est GPU-h |
|----|-----------|--------|----------|-----------|
| E7 | 9B think/nothink@2048 (n=1319) | ⏳ | HIGH | 10 |
| E8 | 27B MRSD GSM8K (n=200) | ⏳ | MEDIUM | 3 |
| E9 | 27B MRSD MATH-500 (n=200) | ⏳ | MEDIUM | 4 |
| E10 | 27B Split-Budget MATH-500 (n=200) | ⏳ | MEDIUM | 4 |

## Auto-Queue Status

```
Auto-queue monitor: ✅ Running (PID 9677)
GPU 0 monitor: waiting for PID 7478 (9B@1024) to finish
GPU 1 monitor: waiting for PID 9058 (MRSD MATH-500 pilot) to finish
```

## Decision Points

- [x] MRSD GSM8K positive → continue with MATH-500
- [x] 8B crossover found: b* ≈ 2048 tokens (think=93.1%=nothink)
- [x] Theory validated: F_L⁻¹(0.931/0.983) ≈ 97th percentile ≈ 2048 ✓
- [ ] MRSD MATH-500 positive → commit to full paper
- [ ] MRSD MATH-500 negative → fall back to pure theory paper

## Running on Server

**Server**: 216.81.245.138:17490  
**GPU 0**: 9B thinking@1024 (800/1319, 17.7GB/80GB, 5% util) → auto-queue after  
**GPU 1**: MRSD MATH-500 pilot (20/200, 16.7GB/80GB, 25% util) → auto-queue after  
**Monitoring**: Cron every 12min (job 4735318e)

## Estimated Timeline

| Time | GPU 0 | GPU 1 |
|------|-------|-------|
| Now | 9B@1024 (800/1319) | MRSD MATH-500 pilot (20/200) |
| +3-4h | E7: 9B@2048 starts | E1: Split-budget MATH-500 starts |
| +6h | E7 running | E2: Split-budget GSM8K |
| +8h | E7 running | E3: MRSD full GSM8K starts |
| +12h | E8: 27B MRSD GSM8K | E3 running |
| +16h | E8 running | E4: MRSD full MATH-500 starts |
| +24h | E9-E10: 27B MATH-500 | E5-E6: Multi-seed |
| +30h | **ALL DONE** | **ALL DONE** |

---
*Last updated: 2026-04-09 05:18 UTC*
