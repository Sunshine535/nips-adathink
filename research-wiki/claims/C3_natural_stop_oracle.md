---
type: claim
node_id: claim:C3
title: "Natural stop is a free confidence oracle (99.0% PPV)"
status: supported
strength: strong
created_at: 2026-04-10
updated_at: 2026-04-10
---

# Natural-stop signal: if the chain completes, the answer is almost certainly correct

## Claim
When thinking chains terminate naturally within budget (not truncated), accuracy is ~99%. This provides a free binary confidence signal.

## Evidence
- 8B GSM8K@512: 99.0% accuracy among naturally completed chains
- This is a free signal requiring no logit access or calibration
- Motivates the triage approach in MRSD (Stage 0)

## Connections
[AUTO-GENERATED from graph/edges.jsonl]
