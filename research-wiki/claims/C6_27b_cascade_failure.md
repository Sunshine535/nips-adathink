---
type: claim
node_id: claim:C6
title: "Cascade (MRSD/TOWN) hurts at 27B scale"
status: under_investigation
strength: preliminary — likely budget-insufficiency, not fundamental
created_at: 2026-04-10
updated_at: 2026-04-10
---

# 27B cascade failure: MRSD and TOWN both below nothink

## Claim
At 27B scale, the cascade (both MRSD and TOWN) performs worse than pure nothink, on both GSM8K and MATH-500.

## Evidence
- GSM8K (n=200): MRSD 60.0% < nothink@256 67.5% — exp:mrsd_27b
- MATH-500 (n=200, FINAL): MRSD 20.0% < nothink@512 23.5% — exp:mrsd_27b
  - 167/200 escalated, 100% hit B_think=1024 ceiling
  - TOWN: 24.5%, IRIS single: 20.0%

## Caveats — ROOT CAUSE NOT YET CONFIRMED
- B_think=512 (GSM8K) and B_think=1024 (MATH-500) are likely too small for 27B
- 27B generates much longer chains → needs proportionally larger B_think
- **B_think=2048 or 4096 NOT YET TESTED at 27B**
- This may be a budget-insufficiency issue, not a fundamental cascade failure

## Open Investigation
1. Does 27B MRSD with B_think=2048 beat nothink?
2. What is the minimum B_think for 27B cascade to help?

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
