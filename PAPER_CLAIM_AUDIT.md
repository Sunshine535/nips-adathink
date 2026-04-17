# Paper Claim Audit Report

**Date**: 2026-04-17
**Auditor**: GPT-5.4 xhigh (fresh zero-context thread)
**Thread ID**: 019d9931-20dd-75b0-a510-f3f4bcb9810d
**Paper**: "The Coupling Tax" (NeurIPS 2026 submission)
**Difficulty**: nightmare, effort: beast

## Overall Verdict: **FAIL**

Multiple hard internal errors + several headline numbers not verifiable against the supplied evidence bundle.

## Summary

| Status | Count |
|--------|-------|
| exact_match | 3 |
| rounding_ok | 6 |
| ambiguous_mapping | 3 |
| number_mismatch | 4 |
| config_mismatch | 2 |
| scope_overclaim | 1 |
| unsupported_claim | 1 |
| missing_evidence | 3 |
| **Total claims audited** | **24** |

---

## HARD FAILURES (must fix before submission)

### FAIL #19: Arithmetic error in appendix — **49.7pp is wrong, should be 67.2pp**
- **Location**: `appendix_final.tex:805`
- **Paper text**: "99.0% ... 31.8% ... a 49.7pp gap"
- **Actual arithmetic**: 99.0 − 31.8 = **67.2pp**
- **Fix**: Replace 49.7pp → 67.2pp (or re-examine which two numbers should be compared)

### FAIL #20: Full-scale IRIS routing paragraph — multiple number mismatches
- **Location**: `experiments_final.tex:128-162`
- **Paper claims**:
  - "216/500 at 91.7% using only 138 avg tokens" — avg token number wrong
  - "157 samples at 51.0%" — accuracy off
  - "98% of escalated samples converge within 2 rounds" — this is the overall rate, not escalated
- **Evidence**:
  - Stage 1 resolved avg_tokens ≈ **319.1** (not 138)
  - 157 stage-3 samples acc = **51.6%** (not 51.0%)
  - Escalated-sample convergence = 11/16 = **68.8%** on GSM8K (not 98%)
- **Fix**: Re-extract the three numbers from `results/iris_math500_fullscale_b4096/b4096_iris_compact.json` and `results/mrsd_pilot/mrsd_Qwen3_8B_gsm8k...json`; clarify the escalated-subset rate separately from overall rate.

### FAIL #21: Failure-decomposition table caption mismatch
- **Location**: `analysis_final.tex:79`
- **Paper caption**: "on GSM8K (n=1319)"
- **Reality**: The b=512 columns (think@512=57.5%, nothink@512=93.4%) come from n=200 subset (per footnote)
- **Fix**: Either (a) split the table into two panels with distinct n, or (b) update the caption/columns to remove the n=200 rows, or (c) re-run the 512 columns at full scale.

### FAIL #22: Routing baseline table internal source mixing
- **Location**: `appendix_final.tex:813-839`
- **Paper row**: "Think@512 = 56.9%, 460 tok, 109 recoveries, 403 regrets"
- **Evidence** in `results/uncertainty_router/routing_baselines_metrics.json`:
  - think_only row shows **65.2%, 477.3 avg tokens** (not 56.9/460)
  - Recoveries/regrets (109/403) are correct for that file
- **Fix**: The 56.9/460 numbers come from an unlisted run that used different think@512 configuration. Either (a) use 65.2/477.3 from the listed file, or (b) clearly cite the other file source and state why two different think@512 numbers appear.

### FAIL #23: DeepSeek GSM8K utilization internal inconsistency
- **Location**: `appendix_final.tex:350-361` vs `results/paper_figures/table_token_utilization.tex:21-24`
- **Text says**: 43.7% utilization (447/1024)
- **Table says**: 38.3%
- These are **internally inconsistent** — independent of any evidence file.
- **Fix**: Reconcile to one canonical value (compute from same source).

### FAIL #24: Pilot Wilson CI — several intervals don't reproduce
- **Location**: `experiments_final.tex:78, 86, 89, 91, 92`
- **Paper CIs** (pilot GSM8K row): e.g. "89.0 [84.5, 93.0]", "94.0 [90.5, 97.0]"
- **Evidence** (Wilson recomputed from n=200 counts):
  - 89.0% → [83.9, 92.6] (paper: [84.5, 93.0])
  - 94.0% → [89.8, 96.5] (paper: [90.5, 97.0])
- **Fix**: Recompute Wilson intervals from actual correct counts or document which CI method was used.

---

## CONFIG / AGGREGATION ISSUES (WARN)

### WARN #6: Sign-test p-value specificity
- **Location**: `experiments_final.tex:103`
- **Paper claims**: "10 unique wins, 0 unique losses (exact sign test p=0.001)"
- **Exact values**: one-sided p = 0.0009766; two-sided p = 0.0019531
- Paper's "p=0.001" is correct only for **one-sided** sign test. State which.
- **Fix**: Specify "one-sided" or change to "p=0.002" for two-sided.

### WARN #8: IRIS vs TOWN McNemar on escalated samples
- **Location**: `experiments_final.tex:140, 149`
- **Paper**: "p < 10⁻⁶"
- **Evidence**: MRSD vs TOWN p = 1.95e-08 ✓; IRIS vs TOWN p = **1.40e-06** (slightly > 1e-6)
- **Fix**: Change IRIS-vs-TOWN claim to "p < 2×10⁻⁶" or compute with sign-test instead.

### WARN #9: MATH-500 MRSD rescored mapping
- **Location**: `experiments_final.tex:78-95, 104`
- **Paper values**: MRSD=61.0%, nothink-within-MRSD-pilot=42.0%
- **Listed file `mrsd_Qwen3_8B_math500_*.json`** has: MRSD=**59.0%**, nothink=**40.5%**
- **Unlisted file `results/mrsd_pilot/mrsd_math500_rescored.json`** matches the paper values.
- **Fix**: Cite the rescored file explicitly or remove "rescored" numbers in favor of the listed file's values.

---

## MISSING EVIDENCE (cannot verify with supplied bundle)

### MISSING #4: MATH-500 nothink@1024 = 59.8% and +7.4pp / +14.2pp deltas
- **Location**: abstract, intro, experiments, discussion, conclusion (many places)
- **Paper values**: 59.8%, +7.4pp, +14.2pp
- **Status**: No raw MATH-500 nothink@1024 full-scale file was in the supplied bundle
- **Fix**: Either locate and cite the actual raw file (likely exists in `results/gap_fill/` or similar), or note the data provenance.

### MISSING #13: 8B GSM8K @512 numbers (93.1%, 56.9%, etc.)
- **Location**: introduction, analysis, discussion, conclusion (many places)
- **Paper values**: nothink@512=93.1%, think@512=56.9%, natural_stop=37.4%, α_c=99.0%, α_t=31.8%
- **Status**: Supplied `results_kun/fulltest/summary_gsm8k_*.json` gives think@512 = **65.2%** (different run/config)
- **Note**: The auditor identified a nearby unlisted file `results/gap_fill/8b_highbudget/nothink_baseline_Qwen3-8B_gsm8k_20260407_121035.json` matches paper numbers.
- **Fix**: Include the gap-fill files in the bundle; ensure paper cites them consistently.

### MISSING #15 + #16 + #17: 9B, BBH, DeepSeek claims
- All three topics have no raw evidence files in the supplied bundle.
- **Fix**: Locate and include the actual files in future audits.

---

## EXACT/ROUNDING MATCHES (verified ✓)

### ✅ #1: IRIS@b2048 = 67.2%, IRIS@b4096 = 74.0% (MATH-500)
- Exact: 336/500 = 67.2%, 370/500 = 74.0%
- Wilson CIs reproduce exactly: [63.0, 71.2] and [70.0, 77.7]

### ✅ #2: +6.8pp cross-budget gain, McNemar p=0.0004
- 74.0 − 67.2 = +6.8pp ✓
- Exact McNemar p = 0.000374 → rounds to 0.0004 ✓

### ✅ #3: IRIS vs TOWN MATH-500 gains
- +12.2pp @b2048, +2.2pp @b4096 — exact
- Token counts 1573/1590 (b2048), 2401/2565 (b4096) — rounding OK

### ✅ #5: MRSD pilot GSM8K 94.0%, 235 tok, +5.0pp
### ✅ #7: 106 escalated MATH samples breakdown (+25.4pp, +32.1pp)
### ✅ #10: 27B MRSD numbers (60.0/67.5/62.5/20.0/23.5/24.5)
### ✅ #11: nothink@128/256, think@128/256 — exact match to fullset JSON
### ✅ #12: 98.6% truncation, 11.2% in nothink — per-sample rates reproduce exactly
### ✅ #14: 27B GSM8K both modes (9.9/65.1/95.5/3.6/7.9/18.3) — all match recovered summary

---

## Priority Fix List (for paper revision)

| Priority | Issue | Fix type | Effort |
|----------|-------|----------|--------|
| **P0 (hard error)** | #19 arithmetic | Edit: 49.7 → 67.2 | 1 min |
| **P0 (hard error)** | #23 DeepSeek internal inconsistency | Reconcile | 10 min |
| **P0 (data integrity)** | #24 Wilson CI mismatch | Recompute CIs | 30 min |
| **P0 (data integrity)** | #22 routing baseline source mixing | Disambiguate | 30 min |
| **P0 (caption)** | #21 analysis table n=1319/n=200 mix | Split table or caption | 15 min |
| **P0 (major text)** | #20 full-scale IRIS numbers (138, 51.0, 98%) | Re-extract values | 30 min |
| **P1 (precision)** | #6 sign-test p direction | Specify one-sided | 2 min |
| **P1 (precision)** | #8 IRIS McNemar > 1e-6 | Weaken to < 2e-6 | 2 min |
| **P1 (sourcing)** | #9 MATH rescored file | Cite or replace | 15 min |
| **P2 (verification)** | #4 #13 #15 #16 #17 missing evidence | Include files in bundle | — |

---

## Bottom Line

**Verdict: FAIL.** The paper has:
- **1 arithmetic error** (off by 17.5pp)
- **1 internal inconsistency** (DeepSeek utilization)  
- **3 data sourcing issues** (table mixing, Wilson CIs, rescored file)
- **1 caption/data mismatch** (n=1319 vs n=200)
- **3 number mismatches** in the IRIS routing paragraph
- **5 claim-evidence pairs** not verifiable without additional files

**The core headline results ARE supported**: IRIS MATH-500 67.2%/74.0%, +12.2pp over TOWN, 27B MRSD numbers, 8B baseline at @128/@256, 27B both-mode results, MRSD pilot wins. But the paper needs **at least P0 fixes** before submission to avoid reviewer objections.

**Next step**: Execute P0 fixes (estimated ~2 hours), then re-run audit.
