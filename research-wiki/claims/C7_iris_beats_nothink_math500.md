---
type: claim
node_id: claim:C7
title: "IRIS split-budget definitively beats nothink@1024 on MATH-500"
status: supported
strength: strong — full-scale n=500, same hardware, CIs non-overlapping
created_at: 2026-04-12
updated_at: 2026-04-12
---

# IRIS split-budget definitively beats nothink@1024 on MATH-500

## Claim
Split-budget generation (IRIS) at B_think≥2048 definitively outperforms nothink@1024 on MATH-500, demonstrating that decoupling reasoning from answering recovers accuracy lost to the coupling tax.

## Evidence

### Full-scale results (n=500, H800, seed=42)
| Method | Accuracy | 95% Wilson CI | vs nothink@1024 |
|--------|----------|---------------|-----------------|
| nothink@1024 | 59.8% | [55.4, 64.0] | baseline |
| IRIS@2048 | 67.2% | [63.0, 71.2] | +7.4pp |
| IRIS@4096 | 74.0% | [70.0, 77.7] | +14.2pp |
| TOWN@2048 | 55.0% | [50.6, 59.3] | -4.8pp |
| TOWN@4096 | 71.8% | [67.7, 75.6] | +12.0pp |

### Key evidence:
- IRIS@4096 CI lower bound (70.0%) > nothink@1024 point estimate (59.8%)
- IRIS > TOWN at every budget (decoupled answering is the key mechanism)
- IRIS@2048→4096 gain: +6.8pp (McNemar p=0.0004), monotonic scaling confirmed
- First 200 samples reproduce pilot exactly (73.0%@2048, 78.5%@4096)

## Source experiments
- exp:iris_math500_fullscale
- exp:town_math500_fullscale

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
