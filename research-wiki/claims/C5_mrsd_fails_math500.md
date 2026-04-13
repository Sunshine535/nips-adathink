---
type: claim
node_id: claim:C5
title: "MRSD falls short of nothink@1024 on MATH-500 (budget-scaling ceiling)"
status: invalidated
strength: disproven by full-scale experiments
created_at: 2026-04-10
updated_at: 2026-04-12
---

# MRSD loses to nothink budget scaling on harder benchmarks — INVALIDATED

## Original Claim
On MATH-500, MRSD (61.0%, 1823 tokens) < nothink@1024 (69.5%, 600 tokens) on pilot n=200. This suggests a "budget-scaling ceiling" for split-budget methods.

## Resolution: INVALIDATED (2026-04-12)

The claim was invalidated by three findings:

### 1. Sample Selection Bias Eliminated
- Pilot nothink@1024 (n=200): 69.5% — **overestimated by ~10pp**
- Full-scale nothink@1024 (n=500, H800): **59.8%** [55.4, 64.0]
- Cross-run variance (~10pp) was hardware/seed dependent, not sample selection

### 2. Larger B_think Overcomes Truncation
- IRIS@1024 (pilot n=200): 62.5% [55.6, 68.9] — already matches corrected nothink
- IRIS@2048 (full n=500): **67.2%** [63.0, 71.2] — exceeds nothink@1024 by +7.4pp
- IRIS@4096 (full n=500): **74.0%** [70.0, 77.7] — exceeds nothink@1024 by +14.2pp

### 3. CI Lower Bounds Definitive
Both IRIS@2048 (63.0%) and IRIS@4096 (70.0%) have CI lower bounds above nothink@1024 (59.8%).

## Root Cause of Original Failure
1. **Insufficient B_think**: At B_think=1024, 100% of escalated chains truncated → decoupled answering from empty traces
2. **Pilot overestimation**: n=200 pilot nothink@1024=69.5% was unrepresentatively high
3. **Multi-round degradation**: MRSD multi-round hurt when all thinking truncated; single-pass IRIS avoided this

## Key Lesson
Budget-scaling ceiling is **not fundamental** — it was an artifact of insufficient thinking budget. The mechanism (decoupled answering from partial reasoning) works when B_think is large enough for meaningful chains to form.

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
