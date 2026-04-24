# GPT-5.5 Pro Report Extraction

## Diagnosis File Used

`/home/tarkoy/nips/nips-adathink/GPT55_DIAGNOSIS.md` (root of repository, ~800 lines)

## Recommended MAIN METHOD PATH

**RCV-IRIS: Recoverability-Calibrated Verifier IRIS**

Upgrade IRIS from "natural-stop / hit-budget triggered cascade" to an "online, auditable, acceptance verifier + prefix recoverability verifier controlled mode/budget/action policy."

Core idea: For each query, maintain online state and use calibrated gates to decide: ACCEPT_STAGE0, THINK_MORE, EXTRACT_STAGE3, FALLBACK_TOWN, or ABSTAIN_LOG_FAILURE.

## Missing Mechanism

**Recoverability-Calibrated Acceptance and Extraction Control**

Current IRIS lacks:
1. Stage0 acceptance verifier (natural stop ≠ correct)
2. Prefix recoverability estimator (truncated CoT may or may not contain answer)
3. Extraction margin estimator (Stage3 may fail on low-recoverability prefixes)
4. Cost-aware routing (unified token accounting)

## Evidence From Positive Results

- Stage3 in 27B MATH high-truncation recovers answers (n50 JSON: acc=0.8)
- Multiseed IRIS MATH-500 stable at 74.1% (std 1.5pp)
- Factorial ablation: +37.4pp mode×prompt interaction
- IRIS beats SwiReasoning on MATH-500 Pareto (+0.9pp, -26% tokens)

## Evidence From Negative Results

- 8B MATH-500 full n500 improved Stage3 only +0.4pp (P04)
- Entropy dynamics NO-GO on both b256 and b512 (P06)
- CTT KL nearly null: AUC=0.535, gap=0.026 (P07)
- Stage0 false accepts are meaningful error source (P08)
- Online Stage2: 67.5% vs post-hoc 77.5% (-10pp gap)

## Evidence From Unstable Results

- Multiseed sample sizes unequal (n=500/200/200)
- Budget forcing token undercount contaminates comparison
- Post-hoc vs online accounting inconsistency

## Evidence From Failed Ablations

- CTT: AUC/null gap doesn't pass → internal KL not useful
- Entropy: b256 and b512 both NO-GO → token entropy wrong signal
- Pure mode switch alone: 53% vs 54.5% not significant

## Why Existing Best Positive Fragment Is Insufficient

"Stage3 extraction + improved prompt" cannot explain:
- 8B MATH negative control (+0.4pp only)
- Entropy/CTT null results
- Stage0 false accepts
- Cannot differentiate from BAEE/Detection-Extraction Gap literature

## Files to Inspect

- `scripts/benchmarks.py` (infinite loop bug)
- `scripts/run_iris.py` (post-hoc default, Stage1 accept logic)
- `scripts/run_budget_forcing.py` (token undercount)
- `scripts/run_nothink_baseline.py` (CLI bug)
- `scripts/iris_online_stage2.py` (should be default)
- All result JSONs for reliability audit

## Files to Edit

- `scripts/benchmarks.py` — fix normalize_latex infinite loop
- `scripts/run_budget_forcing.py` — fix token accounting
- `scripts/run_nothink_baseline.py` — fix --also_thinking default
- `scripts/run_iris.py` — make online Stage2 default, add stage2_mode flag

## Files to Archive

- CTT results → `archive/ctt_negative/`
- Entropy dynamics → keep as negative evidence (already in results/)

## Files to Keep

- All raw result JSONs (freeze, do not modify)
- `NARRATIVE_REPORT.md`, `DATA_PROVENANCE.md`, `PAPER_CLAIM_AUDIT.md`
- Factorial ablation, mechanism ablation, pure mode ablation results

## Files to Keep Only as Baseline

- `scripts/run_nothink_baseline.py` (after fix)
- `scripts/run_budget_forcing.py` (after fix)
- TOWN baseline logic in `run_iris.py`

## Files to Keep Only as Ablation

- Entropy stopping (tau_h, tau_s in run_iris.py)
- CTT pilot
- Learned allocator
- Post-hoc Stage2 mode

## Suspected Bugs

1. **P0**: `benchmarks.py` normalize_latex infinite loop (`while " " in s`)
2. **P0**: `run_budget_forcing.py` returns only initial gen_len, not total
3. **P0**: `run_iris.py` default Stage2 is post-hoc (not deployment-faithful)
4. **P0**: `run_iris.py` Stage1 accepts on natural stop alone (no answer validity check)
5. **P1**: `run_nothink_baseline.py` --also_thinking always True
6. **P1**: IRIS vs MRSD naming confusion in paper

## Required Logging

Per-sample: stage0_verifier_score, recoverability_score, extractor_margin, decision_action, parser_source, retry_used, all token fields (generated/used/forced/injected), model revision, git commit

## Required Minimal Experiments

1. Smoke test (n=2)
2. Metric sanity (pytest)
3. Sample manifest determinism
4. Reproduce entropy NO-GO
5. Reproduce CTT null
6. A/B/C comparison (n=10-100)
7. Stage0 verifier ablation
8. Recoverability gate ablation

## Required Core Comparison

A. Existing Best Positive Fragment Only (current IRIS)
B. New MAIN METHOD Without New Mechanism (online IRIS, no verifier gates)
C. Full RCV-IRIS (with acceptance verifier + recoverability gate)

## Required Baselines

NoThinking, Thinking, TOWN, Budget Forcing (fixed), SwiReasoning, E1-Math-7B, BAEE (if code available)

## Stop / Continue / Pivot Criteria

- **STOP**: If Full RCV-IRIS does not beat both A and B on paired same-sample comparison
- **CONTINUE**: If C > A and C > B consistently
- **PIVOT**: If BAEE dominates with comparable cost and mode-aware split-budget adds no value
