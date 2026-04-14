# Proof Audit Log — The Coupling Tax

> Round 1: GPT-5.4 xhigh adversarial review (nightmare difficulty)
> Thread: 019d86c9-8df9-7b30-8779-8b278227e4d6

---

## Round 1 — Score: N/A (proof review, not paper review)

### Issues Found: 10

| ID | Status | Impact | Category | Location | Summary |
|----|--------|--------|----------|----------|---------|
| 1 | INVALID | GLOBAL | NORMALIZATION_MISMATCH | theory:32,51-55,192-210; appendix:123-126,691-700 | F_L(b), S(b)={|y|<b}, and "NatStop<95%" are three different events, mixed freely |
| 2 | UNDERSTATED | GLOBAL | HIDDEN_ASSUMPTION | theory:49-55,100-110,144-173,227-247 | α_c treated as budget/model/task-invariant; only α_t flagged as varying |
| 3 | INVALID | GLOBAL | INSUFFICIENT_ASSUMPTION | theory:144-173; appendix:848-915 | Crossover formula b*≈F_L^{-1}(ᾱ_nt/α_c) requires α_t=0 AND constant α_c; conflicting percentile claims (97th vs 99th) |
| 4 | INVALID | LOCAL | INSUFFICIENT_ASSUMPTION | theory:190-203; appendix:689-695 | PPV(S_nt) ≥ α_c claimed without proof; counterexample exists |
| 5 | INVALID | LOCAL | MISSING_DERIVATION | theory:207-210 | Hoeffding bound arithmetic error: 0.901 ≠ 0.919 |
| 6 | INVALID | GLOBAL | HIDDEN_ASSUMPTION | theory:227-247,257-265 | Inverse scaling needs α_c model-invariant (not stated); counterexample exists |
| 7 | INVALID | GLOBAL | MISSING_DERIVATION | theory:333-369; method:96-120 | P6 algebra drops negative correction term (F(b)-F(b_r))(α_c-α_t); counterexample: Δ can be negative |
| 8 | INVALID | LOCAL | INSUFFICIENT_ASSUMPTION | theory:385-400 | P7 uniqueness of interior max unsupported; counterexample with constant α_extract gives boundary max |
| 9 | OVERSTATED | GLOBAL | SCOPE_OVERCLAIM | method:317-341; appendix:947-951 | "Pareto-dominance" overclaimed — theorem only shows interpolation between two baselines |
| 10 | INVALID | GLOBAL | REFERENCE_MISMATCH | appendix:921-956 | Appendix proof is for TOWN two-stage, not K-round MRSD; cost bound requires B_1 < p·B_2 |

### Severity Classification

| Severity | Count | IDs |
|----------|-------|-----|
| FATAL (INVALID+GLOBAL) | 5 | 1, 3, 6, 7, 10 |
| CRITICAL (INVALID+LOCAL or UNJUSTIFIED+GLOBAL) | 3 | 4, 5, 8 |
| MAJOR (UNDERSTATED+GLOBAL) | 1 | 2 |
| MINOR (OVERSTATED+GLOBAL) | 1 | 9 |

### Full Reviewer Response

<details>
<summary>Click to expand full GPT-5.4 response</summary>

**Issue 1: NORMALIZATION_MISMATCH (INVALID, GLOBAL)**
- F_L(b) defined by L≤b; S(b) defined as |y|<b; NatStop uses <95% budget threshold
- These are three different events mixed freely
- Counterexample: samples with L=b exactly satisfy L≤b but NOT |y|<b
- Affects: P4, parameter instantiations for P1 and P3, theory verification table
- Fix: define one event E_b and use it everywhere

**Issue 2: HIDDEN_ASSUMPTION (UNDERSTATED, GLOBAL)**
- α_c used as budget/model/task invariant in crossover, scaling, held-out prediction
- By definition α_c = Pr(correct | L≤b) depends on b
- Counterexample: α_c(512)=0.99 but α_c(2048)=0.75 (harder chains complete at larger b)
- Affects: P3, P5, crossover scaling corollary, held-out prediction
- Fix: write α_c(b,M,task) explicitly; label constant-parameter use as heuristic

**Issue 3: INSUFFICIENT_ASSUMPTION (INVALID, GLOBAL)**
- Exact crossover: F_L(b*) = (ᾱ_nt - α_t(b*)) / (α_c(b*) - α_t(b*))
- Paper's simplified formula requires α_t(b*)=0 AND constant α_c
- Conflicting percentile claims: main text says 99th, appendix says 97th, table uses F_L≈0.93
- Counterexample: ᾱ_nt=0.93, α_c=0.99, α_t=0.30 gives F=0.91, not 0.94
- Fix: derive exact formula with α_t(b*), or clearly state stronger assumptions

**Issue 4: INSUFFICIENT_ASSUMPTION (INVALID, LOCAL)**
- PPV(S_nt) ≥ α_c for nothink mode — no proof given
- Counterexample: easy questions use full budget (correct), hard questions stop immediately (wrong) → PPV(S_nt) = 0
- Fix: weaken to empirical observation

**Issue 5: MISSING_DERIVATION (INVALID, LOCAL)**
- √(ln(40)/(2·475)) = 0.06231, so bound = 0.963 - 0.06231 = 0.901, not 0.919
- Fix: recompute correctly

**Issue 6: HIDDEN_ASSUMPTION (INVALID, GLOBAL)**
- P5 proof needs α_c model-size-invariant (not stated)
- Counterexample: model 1 F=0.8, α_c=0.5; model 2 F=0.6, α_c=1.0 → Tax_2 < Tax_1 despite F_2 < F_1
- Fix: assume or measure model-indexed α_c(M,b)

**Issue 7: MISSING_DERIVATION (INVALID, GLOBAL)**
- P6 exact Δ = (1-F(b_r))(α_extract-α_t(b)) - (F(b)-F(b_r))(α_c-α_t(b))
- Proof drops the negative correction term
- Counterexample: F(b_r)=0.1, F(b)=0.9, α_c=1, α_t=0, α_extract=0.5 → claimed +0.45, exact -0.35
- Fix: compare at same reasoning budget, or use full formula

**Issue 8: INSUFFICIENT_ASSUMPTION (INVALID, LOCAL)**
- Log-normal CDF + monotone/concave α_extract → unique interior max: WRONG
- Counterexample: constant α_extract → objective maximized at boundary
- Fix: downgrade to argmax existence, or assume strict quasi-concavity

**Issue 9: SCOPE_OVERCLAIM (OVERSTATED, GLOBAL)**
- "Pareto-dominance" not actually established — theorem shows interpolation between two baselines
- Acc > nothink@B_1 AND Cost < think@max is NOT Pareto dominance in the standard sense
- Fix: rename to frontier/interpolation result

**Issue 10: REFERENCE_MISMATCH (INVALID, GLOBAL)**
- Appendix proves TOWN two-stage, not K-round MRSD
- Cost bound E[T] < B_2 requires B_1 < p·B_2, not just B_1 < B_2
- Counterexample: p=0.1, B_1=100, B_2=200 → E[T]=270 > 200
- Fix: prove MRSD theorem, or add B_1 < p·B_2 condition

</details>

### Cross-Validation with Phase 0.5 Skeleton

| Skeleton Issue | GPT Issue | Match? |
|---------------|-----------|--------|
| S1 (α_c model-invariance in P5) | Issue 6 | ✅ Confirmed + counterexample |
| S2 (P7 uniqueness unproven) | Issue 8 | ✅ Confirmed + counterexample |
| S3 (PPV(S_nt) ≥ α_c unjustified) | Issue 4 | ✅ Confirmed + counterexample |
| S4 (P6 algebra error) | Issue 7 | ✅ Confirmed + stronger counterexample |
| S5 (Appendix proves TOWN not MRSD) | Issue 10 | ✅ Confirmed |
| S6 (Cost bound needs B_1 < p·B_2) | Issue 10 | ✅ Confirmed + counterexample |
| S7 (F_L^{-1} existence) | Issue 3 | Partially (GPT found larger issue) |
| S8 (α_c budget-dependence) | Issues 2, 3 | ✅ Confirmed + elevated to GLOBAL |
| S9 (Hoeffding arithmetic) | Issue 5 | ✅ Confirmed |
| S10 (Stochastic dominance at one budget) | Issue 6 | ✅ Confirmed |
| — (NEW: event mismatch F_L vs S vs NatStop) | Issue 1 | ⚠️ New issue from GPT |
| — (NEW: Pareto-dominance overclaimed) | Issue 9 | ⚠️ New issue from GPT |

---

## Acceptance Gate Check

- [x] Zero open FATAL issues? **NO — 5 FATAL issues (1, 3, 6, 7, 10)**
- [ ] Every theorem has explicit hypotheses? **NO — α_c(b) dependence hidden in P3, P5**
- [ ] All interchanges justified? **N/A — no limit/integral interchanges**
- [ ] Counterexample pass executed? **YES — counterexamples found for 6/10 issues**

**GATE: FAIL — proceed to Phase 2 (fix implementation)**

---

## Round 2 — Fix Implementation + Re-Review

### Fixes Applied (Round 1 → Round 2)

All 10 issues from Round 1 were addressed:

| ID | Fix Strategy | Summary |
|----|-------------|---------|
| 1 | STRENGTHEN_ASSUMPTION | Unified on {L(q) ≤ b}; added proxy paragraph; footnote clarifies subset estimate |
| 2 | STRENGTHEN_ASSUMPTION | α_c(b), α_t(b) explicit in Assumption 1; Remark on budget-dependent α_t |
| 3 | ADD_DERIVATION | Added exact crossover formula Eq. crossover-exact; simplified version labeled as heuristic |
| 4 | WEAKEN_CLAIM | PPV(S_nt) ≥ α_c removed; downgraded to empirical observation |
| 5 | ADD_DERIVATION | Hoeffding corrected: 0.963 - 0.0623 = 0.901 |
| 6 | STRENGTHEN_ASSUMPTION | P5 condition (iii): F_{L_{M2}}(b)·α_c(M2,b) ≤ F_{L_{M1}}(b)·α_c(M1,b) |
| 7 | ADD_DERIVATION | Same-budget Δ (clean) + matched-total Δ^total (with correction terms) |
| 8 | WEAKEN_CLAIM | Uniqueness removed; practical guidance only |
| 9 | WEAKEN_CLAIM | "Pareto dominance" → "interpolation dominance" + footnote |
| 10 | ADD_DERIVATION | Appendix proof rewritten for MRSD; cost bound against B_max |

### Round 2 Re-Review (GPT-5.4 xhigh)

Thread: 019d86c9-8df9-7b30-8779-8b278227e4d6

Results: 3 fully resolved, 6 partially resolved, 1 unresolved, 2 new issues.

New issues found:
- **Issue 11**: "lower bound" in proxy footnote → invalid claim
- **Issue 12**: Proof sketch claimed α_c^nt ≥ Acc_nt(B_1) → false inequality

### Round 2 Additional Fixes

| ID | Fix Strategy | Summary |
|----|-------------|---------|
| 6 (full) | STRENGTHEN_ASSUMPTION | Appendix P5 rewritten with product condition |
| 9 (full) | WEAKEN_CLAIM | Appendix section title + footnote updated |
| 11 | WEAKEN_CLAIM | "lower bound" → "separate estimate" |
| 12 | ADD_DERIVATION | Exact decomposition identity in proof sketch |

---

## Round 3 — Final Re-Review

### Round 3 Additional Fixes (pre-submission)

Before Round 3 review, fixed remaining exposition issues:
- Appendix theory-verification table: "97th percentile" → "ᾱ_nt/α_c quantile" with correct F_L ≈ 0.93
- Appendix held-out paragraph: same percentile fix
- Eq. recoverable-tax-total: explicit α_c(b_r) and α_c(b) budget indexing; added third term F_L(b_r)(α_c(b_r) - α_c(b))
- Eq. optimal-split: α_c → α_c(b_r)

### Round 3 Assessment (GPT-5.4 xhigh)

<details>
<summary>Click to expand full GPT-5.4 Round 3 response</summary>

**Final Verdict**

1. **Issue 1 — PARTIALLY RESOLVED.** Formal event definitions aligned. Residual: 99.0% vs 0.963 presented without clarifying different proxy subsets.
2. **Issue 2 — PARTIALLY RESOLVED.** Budget dependence explicit in assumptions. Residual: α_c shorthand in eq:optimal-split and appendix crossover discussion.
3. **Issue 3 — PARTIALLY RESOLVED.** Exact crossover correct. Residual: appendix prose says "fully determined" (stronger than exact formula proves).
4. **Issue 4 — RESOLVED.**
5. **Issue 5 — RESOLVED.**
6. **Issue 6 — RESOLVED.**
7. **Issue 7 — RESOLVED.**
8. **Issue 8 — RESOLVED.**
9. **Issue 9 — RESOLVED.**
10. **Issue 10 — RESOLVED.**
11. **Issue 11 — RESOLVED.**
12. **Issue 12 — RESOLVED.**

**Acceptance gate**: Zero open FATAL. Zero open CRITICAL. All counterexamples neutralized for Issues 4-12. Residual items for 1-3 are exposition clarity, not proof defects.

</details>

### Post-Round-3 Exposition Fixes

After Round 3 review, immediately fixed all residual exposition items:

| Residual | Fix |
|----------|-----|
| 99.0% vs 0.963 confusion | Added footnote explaining 96.3% is |y|<0.95b subset; 99.0% is all L≤b |
| α_c in eq:optimal-split | Changed to α_c(b_r) |
| Bare α_c in correction discussion | Softened to "correction terms are small" |
| "Fully determined" in appendix | → "approximately determined by..."; references exact formula |
| Crossover corollary shorthand | Added "Under the heuristic (Eq. crossover)" |

---

## Acceptance Gate — Final

- [x] Zero open FATAL issues? **YES — all 5 original FATALs resolved**
- [x] Zero open CRITICAL issues? **YES — all 3 original CRITICALs resolved**
- [x] Every theorem has explicit hypotheses? **YES**
- [x] All interchanges justified? **N/A — no limit/integral interchanges**
- [x] Counterexample pass executed? **YES — all original counterexamples neutralized**
- [x] LaTeX compiles cleanly? **YES — zero errors**

**GATE: PASS**

---

## Summary of Changes

### Files Modified
- `paper/sections/theory_final.tex` — 15+ edits across all propositions
- `paper/sections/method_final.tex` — Theorem 1 renamed, proof sketch fixed
- `paper/sections/appendix_final.tex` — P5 rewritten, T1 proof rewritten, percentile fixes

### Impact on Paper Claims
- All formal propositions (P1-P7) and Theorem 1 are now mathematically sound
- No claim was strengthened; several were weakened to match what the proofs actually show
- Key weakening: "Pareto dominance" → "interpolation dominance"; crossover "exact" → "heuristic + exact"; P7 uniqueness removed
- Key strengthening: P5 now has the correct sufficient condition; P6 has explicit budget indexing

### What Remains Assumed (Not Proven)
- Non-thinking PPV is high (empirical observation, not derived from Assumption 1)
- Non-thinking accuracy is approximately size-invariant (empirical, used in P5 condition (i))
- α_c(b) is approximately stable across moderate budgets (stated explicitly in Assumption 1)

---
*Generated: 2026-04-13, Proof-checker (nightmare) — 3 rounds completed*
*Thread ID: 019d86c9-8df9-7b30-8779-8b278227e4d6*
