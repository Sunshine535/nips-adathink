# Proof Audit Log — The Coupling Tax (Round 2)

> Fresh independent review: GPT-5.4 xhigh adversarial (nightmare difficulty, beast effort)
> Thread: 019d8b2c-e8a1-7a41-b2e7-599130088867
> Date: 2026-04-14
> Reviewer: Codex GPT-5.4 (xhigh reasoning), 3 rounds

---

## Round 1 — 16 Issues Found

| ID | Status | Impact | Category | Location | Summary |
|----|--------|--------|----------|----------|---------|
| 1 | INVALID | GLOBAL | INSUFFICIENT_ASSUMPTION | theory:150-177 | Prop 3 "exact" condition uses ᾱ_nt instead of Acc_nt(b*); needs saturation hypothesis |
| 2 | INVALID | GLOBAL | NORMALIZATION_MISMATCH | theory:183-191 | Crossover uses 0.955/0.963 (mixed proxy/true α_c); numeric error |
| 3 | INVALID | GLOBAL | HIDDEN_ASSUMPTION | theory:386-445, method:97-116 | Split accuracy assumes completed chains bypass extraction (unstated) |
| 4 | INVALID | GLOBAL | UNJUSTIFIED_ASSERTION | method:252-270, app:987-997 | Cost formula uses = where ≤ justified; comparison against budget cap not baseline |
| 5 | INVALID | GLOBAL | CASE_INCOMPLETE | method:318-353 | Theorem 1 needs 0<p<1, not just p>0 (p=1 ⟹ equality) |
| 6 | INVALID | GLOBAL | LOGICAL_GAP | theory:271-299 | Prop 5 proof cites F_L ordering, not condition (iii) product inequality |
| 7 | INVALID | GLOBAL | UNPROVEN_SUBCLAIM | theory:309-318 | Cor 2 unproven from Prop 5 (fixed-budget ≠ quantile ordering) |
| 8 | OVERSTATED | GLOBAL | SCOPE_OVERCLAIM | theory:325-353, method:57-63 | "Sole mechanism" overclaim — tax possible with zero truncation if α_c < Acc_nt |
| 9 | INVALID | LOCAL | STOCHASTIC_MODE_CONFUSION | theory:221-255 | Hoeffding bounds proxy PPV, not α_c; sampling model unstated |
| 10 | INVALID | LOCAL | REFERENCE_MISMATCH | theory:191-198 | Cor 1 MATH-500 assumes b_sat=1024 but saturation not reached |
| 11 | INVALID | GLOBAL | CASE_INCOMPLETE | theory:160-166 | F_L^{-1} undefined when ᾱ_nt/α_c > 1; no existence condition |
| 12 | UNCLEAR | GLOBAL | HIDDEN_ASSUMPTION | method:320-353, app:972-976 | α_extract^hard undefined for multi-round MRSD |
| 13 | INVALID | LOCAL | NORMALIZATION_MISMATCH | method:332-340, app:1010-1016 | Conflicting α_nt^hard (≈0% pilot vs 33.0% full-scale decomposition) |
| 14 | OVERSTATED | LOCAL | LOGICAL_GAP | theory:421-426 | Eq. 8 discussion omits third term in sufficiency condition |
| 15 | UNJUSTIFIED | GLOBAL | HIDDEN_ASSUMPTION | theory:25-33 | L(q)∈ℕ assumes Pr(L<∞)=1 (almost-sure termination) |
| 16 | INVALID | LOCAL | IMPLICATION_REVERSAL | app:999-1004 | "Strictly inside the convex hull" claim is false |

### Severity Classification

| Severity | Count | IDs |
|----------|-------|-----|
| FATAL (INVALID+GLOBAL) | 8 | 1, 2, 3, 4, 5, 6, 7, 11 |
| CRITICAL | 5 | 9, 10, 13, 15, 16 |
| MAJOR | 1 | 8 |
| MINOR | 2 | 12, 14 |

---

## Fix Summary (3 Rounds)

### Round 1→2 Fixes (16 original issues)

| ID | Fix Strategy | Summary |
|----|-------------|---------|
| 1 | STRENGTHEN_ASSUMPTION | Eq. crossover-exact now uses Acc_nt(b*); heuristic gated by b*≥b_sat |
| 2 | ADD_DERIVATION | Removed 0.963; uses α_c(512)=0.990 giving quantile ≈0.965 |
| 3 | STRENGTHEN_ASSUMPTION | Added bypass hypothesis + footnote explaining natural-stop detection |
| 4 | WEAKEN_CLAIM | Cost claim (2) → "lower than B_max"; footnote clarifies vs budget cap |
| 5 | STRENGTHEN_ASSUMPTION | Changed p>0 → 0<p<1 |
| 6 | ADD_DERIVATION | Proof now explicitly cites conditions (i), (ii), (iii) |
| 7 | STRENGTHEN_ASSUMPTION | Added model-invariant ᾱ_nt/α_c + global stochastic dominance |
| 8 | WEAKEN_CLAIM | "Sole" → "dominant mechanism when α_c ≥ ᾱ_nt" + footnote |
| 9 | WEAKEN_CLAIM | Hoeffding now bounds "proxy PPV" not α_c; i.i.d. assumption stated |
| 10 | WEAKEN_CLAIM | MATH-500: b_sat ≥ 2048; only claim b* > 2048 |
| 11 | ADD_DERIVATION | Generalized inverse defined; footnote: crossover DNE when ᾱ_nt > α_c |
| 12 | ADD_DERIVATION | α_extract^hard defined in theorem as full MRSD escalated branch accuracy |
| 13 | ADD_DERIVATION | Method uses full-scale data (33.0% vs 62.8%), consistent decomposition |
| 14 | ADD_DERIVATION | Sufficient condition rewritten to reflect all three terms |
| 15 | STRENGTHEN_ASSUMPTION | Added Pr(L<∞)=1 footnote to Def 1 |
| 16 | WEAKEN_CLAIM | Convex hull claim removed |

### Round 2→3 Fixes (4 new issues + residuals)

| ID | Fix Strategy | Summary |
|----|-------------|---------|
| NEW-1 | ADD_DERIVATION | 27B crossover notes small sample; assumes model-invariant α_c |
| NEW-2 | ADD_DERIVATION | Algorithm 1: natural-stop bypass in ALL rounds (Stage 1 + k≥2) |
| NEW-3 | WEAKEN_CLAIM | "Improves on at least one objective" → specific claims |
| NEW-4 | WEAKEN_CLAIM | Footnote: Pr(L<∞)=1 as explicit assumption, not architecture claim |
| Issue 4 (residual) | ADD_DERIVATION | ≤ propagated to proof chain in method + appendix |
| Issue 8 (residual) | WEAKEN_CLAIM | Second "sole mechanism" instance fixed |
| Issue 9 (residual) | WEAKEN_CLAIM | Appendix Hoeffding aligned with main text proxy PPV |
| Issue 13 (residual) | ADD_DERIVATION | Appendix: ≥ 62.8% with lower-bound note; definition aligned |
| Issue 14 (true fix) | ADD_DERIVATION | Sufficient condition correctly stated with all terms |

### Final Round Fixes (3 remaining)

| ID | Fix Strategy | Summary |
|----|-------------|---------|
| Issue 12 (residual) | ADD_DERIVATION | Appendix proof definition aligned with theorem statement |
| Issue 13 (app residual) | ADD_DERIVATION | Appendix empirical: ≥ 62.8% with TOWN lower-bound note |
| NEW-4 (footnote) | WEAKEN_CLAIM | Removed false EOS justification; states as pure assumption |

---

## Acceptance Gate — Final

- [x] Zero open FATAL issues? **YES — all 8 original FATALs resolved**
- [x] Zero open CRITICAL issues? **YES — all 5 original CRITICALs resolved**
- [x] Every theorem has explicit hypotheses? **YES (with Prop 6 bypass now in algorithm)**
- [x] All interchanges justified? **N/A — no limit/integral interchanges**
- [x] All O/Θ/o have parameter dependence? **YES (N/A — none in proofs)**
- [x] Counterexample pass: all neutralized? **YES — all 16 original counterexamples addressed**

**GATE: PASS**

---

## Files Modified

- `paper/sections/theory_final.tex` — 12 edits: Def 1 (a.s. termination), Prop 3 (exact condition, quantile fix, existence), Cor 1 (MATH-500), Prop 5 proof (cite condition iii), Cor 2 (conditions), "sole mechanism" ×2, Prop 6 (bypass assumption), Eq. 8 discussion, Hoeffding proxy, 27B crossover
- `paper/sections/method_final.tex` — 8 edits: "dominant mechanism", Thm 1 (p condition, definitions, cost claim, data), Algorithm 1 (natural-stop bypass all rounds), cost formula (≤), interpolation dominance text
- `paper/sections/appendix_final.tex` — 5 edits: Hoeffding proxy, Cor 2 restated, cost bound (≤ + conditions), convex hull removed, empirical instantiation aligned

## Impact on Paper Claims

- All formal propositions (P1-P7) and Theorem 1 are mathematically sound
- No claim was strengthened; 7 were weakened:
  - "Sole mechanism" → "dominant mechanism"
  - "Pareto dominance" → "interpolation" (from previous audit)
  - P3 exact formula corrected; heuristic labeled with existence condition
  - Cost bound against B_max (not expected cost of baseline)
  - P7 uniqueness removed (from previous audit)
  - α_extract^hard ≥ 62.8% (lower bound)
  - Hoeffding bounds proxy PPV only

## What Remains Assumed (Not Proven)

1. Pr(L < ∞) = 1 (explicit assumption, Def 1 footnote)
2. α_c approximately model-invariant (used in crossover example, Cor 2)
3. Non-thinking PPV is high (empirical observation, not derived from Assumption 1)
4. Non-thinking accuracy approximately size-invariant (Prop 5 condition (i))
5. α_c(b) approximately stable across moderate budgets (stated in Assumption 1)
6. Net recovery condition for MRSD (validated with TOWN lower bound ≥62.8%)

---
*Generated: 2026-04-14, Proof-checker (nightmare, beast) — 3 rounds completed*
*Thread ID: 019d8b2c-e8a1-7a41-b2e7-599130088867*
