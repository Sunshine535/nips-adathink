# Remaining Risks

## Method / Empirical Risks

### R1. RCV Gate Signal Too Weak at b2=4096 (HIGH)
- Only 1/42 Stage3 extractions rejected
- +1.0pp accuracy on 1 discordant pair (n=100)
- Statistically indistinguishable from noise at n=100
- **Mitigation**: b2=512 experiments running on 3-GPU server

### R2. Single Seed (HIGH)
- All A/B/C results at seed=42 only
- Cannot rule out seed luck
- **Mitigation**: Multi-seed (42/123/456) needed after b512 completes

### R3. Single Benchmark for RCV (MEDIUM)
- RCV A/B/C only tested on MATH-500
- GSM8K may show different pattern (we saw nothink saturates at 93%)
- **Mitigation**: Add GSM8K cross-benchmark after b512

### R4. Thresholds Not Calibrated (MEDIUM)
- `tau_accept=0.7` and `tau_recover=0.5` are hand-chosen
- Only 1 sample triggered FALLBACK_TOWN
- **Mitigation**: Threshold sweep needed after initial evidence

### R5. Stage0 Acceptance Gate Not Differentiated (MEDIUM)
- All 46 natural-stop Stage0 accepts passed the gate
- Gate is effectively no-op for Stage0
- **Mitigation**: Need harder benchmark (27B MATH-500) or lower tau_accept

## Methodological Risks

### R6. Budget Forcing Old Results Contaminated (MEDIUM)
- Old `bforce_*.json` files have undercounted tokens (bug fixed, not re-run)
- Token-efficiency claims against BF are invalid until rerun
- **Mitigation**: Rerun BF with fixed script — flagged in ledger

### R7. Post-hoc vs Online Accounting (HIGH)
- Headline 77.5% is post-hoc (default run_iris.py)
- Online is 67.5%
- Paper currently uses post-hoc without sufficient disclaimer
- **Mitigation**: Must label every IRIS number as online/post-hoc; make online default

### R8. No Official BAEE Comparison (HIGH)
- BAEE code never released (GitHub stub only)
- Novelty risk: reviewers may call RCV "BAEE without RL"
- **Mitigation**: Implement BAEE-style free-continuation baseline ourselves; document code availability

### R9. Only Qwen Family Tested (MEDIUM)
- Cross-architecture generalization not demonstrated
- DeepSeek nothink may not be true mode (prompt-based)
- **Mitigation**: Scope paper to Qwen-family for now; note limitation

### R10. Effect Size vs Reviewer Expectations (HIGH)
- +1.0pp at b4096 on 1 discordant is weak
- Even if b512 shows +3-5pp, is this enough for NeurIPS?
- **Mitigation**: If b512 shows strong signal, also run multi-seed + GSM8K

## Code / Correctness Risks

### R11. Stage0 / Recover Only Ablations Not Run (MEDIUM)
- Code variants added but experiments not executed
- Cannot isolate contribution of each gate component
- **Mitigation**: Schedule ablation runs after main experiments

### R12. benchmarks.py normalize_latex Tests Limited (LOW)
- Tests cover basic cases, but `_replace_frac` brace-counting loop could potentially hang on malformed input
- **Mitigation**: Add adversarial parser tests

### R13. No Deterministic Replay Verification (LOW)
- Have not run same command twice to verify bit-exact output
- Greedy decoding should be deterministic but HW nondeterminism possible
- **Mitigation**: Run replay test

## Claim / Paper Risks

### R14. Related Work Misses Key Papers (HIGH)
- AnytimeReasoner and Elastic Reasoning not discussed in paper
- Reviewer will immediately flag
- **Mitigation**: Rewrite related_final.tex before submission

### R15. Multi-seed Used Unequal n (MEDIUM)
- Prior multiseed_20260419: seed 42 n=500, seed 123/456 n=200
- Std 0.0152 is not valid for mixed-n
- **Mitigation**: Rerun at equal n, or mark as indicative only

### R16. Theorem Claim Too Strong (MEDIUM)
- Paper claims "coupling tax theorem verified"
- It's actually an accounting decomposition, not a predictive theorem
- **Mitigation**: Rename to "accounting framework" in paper rewrite

## Submission Readiness Risks

### R17. Submission Requires Paper Rewrite (HIGH)
- Current draft uses IRIS/MRSD mixed naming
- Uses post-hoc numbers as headline
- Missing critical related work
- **Mitigation**: Full paper rewrite required — not started

### R18. BAEE-like Attack (HIGH)
- Reviewer reads BAEE paper and says "RCV = BAEE + mode switch"
- Our mode-switch synergy (+37.4pp) is the only novel differentiator
- **Mitigation**: Foreground the factorial ablation result; cite BAEE explicitly
