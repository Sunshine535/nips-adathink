# Core A/B/C Comparison

## Setup
- Model: Qwen/Qwen3-8B
- Benchmark: MATH-500, n=100, seed=42
- Config: b1=512, b2_max=4096, b_answer=512
- Server: 3×A100-80GB (216.81.151.54), 1 GPU per variant

## Results

| Variant | Config | Accuracy | Avg Tokens | Key Decisions |
|---------|--------|----------|------------|---------------|
| A. Existing Best Fragment | existing_fragment | 73.0% (73/100) | 2613 | EXTRACT_ALWAYS=42, S2_COMPLETE=12, ACCEPT_S0=46 |
| B. New Method No Gate | rcv_no_gate | 73.0% (73/100) | 2613 | EXTRACT_ALWAYS=42, S2_COMPLETE=12, ACCEPT_S0=46 |
| **C. Full RCV-IRIS** | **full_rcv** | **74.0% (74/100)** | **2613** | EXTRACT_S3=41, S2_COMPLETE=12, ACCEPT_S0=46, **FALLBACK_TOWN=1** |

## Paired Analysis (McNemar A vs C)

- Both correct: 73
- A only correct: 0
- C only correct: 1
- Neither correct: 26
- **Discordant: 1, C wins 1/1**

## Critical Sample: idx=58

- A/B decision: EXTRACT_ALWAYS → extraction failed → **wrong**
- C decision: FALLBACK_TOWN (extractor_margin=0.0) → TOWN parse → **correct**
- The RCV gate correctly detected that extraction would fail (margin=0.0) and fell back to TOWN.

## Interpretation

Per GPT-5.5 criteria:

1. **C > A**: Yes, +1.0pp. C beats A on 1 discordant pair. Direction correct.
2. **C > B**: Yes, same as above (B = A identically).
3. **Effect size**: Small (1 sample out of 100). Gate only triggered once (1/42 Stage3 samples rejected).
4. **Gate selectivity**: Very conservative — tau_recover=0.5 only catches margin=0.0 samples. Most margins are 0.5-1.0.
5. **Token cost**: Identical (2613 tok for all). C runs both strict and soft extraction probes but only counts the used one.

## Decision

**CONTINUE with caveats.**

The RCV mechanism shows correct directional signal: it identifies and avoids one bad extraction. But the effect is very small because:
1. tau_recover=0.5 is too lenient (only catches margin=0.0)
2. At b2=4096, most prefixes ARE recoverable (42/42 Stage3 passed except 1)
3. Need tighter budget (b2=512/1024) to create more non-recoverable prefixes

## Recommended Next Steps

1. Rerun with b2=512 or b2=1024 (more truncation → more gate activity)
2. Sweep tau_recover from 0.3 to 0.8 to find optimal threshold
3. Add Stage0 acceptance verifier with actual model-based check
4. Run on GSM8K for cross-benchmark validation
5. Multi-seed (42/123/456) for stability
