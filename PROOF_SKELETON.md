# Proof Skeleton: The Coupling Tax

> Auto-generated Phase 0.5 of proof-checker (nightmare difficulty)
> Covers: theory_final.tex, method_final.tex, appendix_final.tex

---

## 1. Dependency DAG

```
Definitions:
  D1: def:chain-length        (theory:25-34)   — L(q), F_L(t)
  D2: def:truncation           (theory:36-39)   — ρ(b) = 1 - F_L(b)
  D3: def:coupled-split        (theory:314-321) — coupled vs split generation

Assumptions:
  A1: asm:binary               (theory:49-56)   — α_c, α_t binary outcome structure

Propositions:
  P1: prop:acc-decomp          (theory:64-73)   — Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t
  P2: prop:thinking-tax        (theory:83-98)   — Tax ≈ Acc_nt(b) - F_L(b)·α_c
  P3: prop:crossover           (theory:144-158) — b* ≈ F_L^{-1}(ᾱ_nt / α_c)
  P4: prop:oracle              (theory:190-204) — PPV(S) = α_c
  P5: prop:inverse-scaling     (theory:227-238) — Tax(M2,b) ≥ Tax(M1,b) under stoch. dom.
  P6: prop:recoverable-tax     (theory:333-352) — Δ_split = (1-F_L(b_r))·(α_extract - α_t)
  P7: prop:optimal-split       (theory:385-401) — Unique interior max under log-normal F_L

Corollaries:
  C1: cor:multiplier           (theory:175-183) — γ = b*/b_sat
  C2: cor:crossover-scaling    (theory:257-265) — b*(M2) ≥ b*(M1)

Theorems:
  T1: thm:mrsd-pareto          (method:317-341) — MRSD Pareto-dominance

Remarks:
  R1: rem:alpha-t              (theory:100-111) — Budget-dependent α_t
  R2: rem:split-bridge         (theory:278-289) — Two levers: increase F_L or increase α_t
  R3: Information-theoretic    (theory:213-220) — Entropy interpretation of natural stop

Equations (method_final.tex):
  E1: eq:coupling-constraint   (method:34-37)   — |Z| + |A| ≤ b
  E2: eq:split-acc             (method:100-106) — α_split(B_r, B_a)
  E3: eq:split-gain            (method:110-115) — Δ = (1-F_L(B_r))·(α_extract - α_t)
  E4: eq:mrsd-cost             (method:257-261) — E[T_MRSD]
  E5: eq:net-recovery          (method:320-323) — α_extract^hard > α_nt^hard
```

### Dependency Edges

```
D1 → D2 (truncation rate defined from CDF)
D1, A1 → P1 (accuracy decomposition)
P1 → P2 (thinking tax as special case with α_t=0)
P1, P2 → P3 (crossover as quantile of F_L)
A1 → P4 (oracle PPV = α_c by definition)
P2 → P5 (tax comparison across model sizes)
P1, D3 → P6 (split vs coupled accuracy difference)
P6 → P7 (optimization over b_r given F_L log-normal)
P3 → C1 (budget multiplier from crossover)
P3, P5 → C2 (crossover scaling with model size)
E1 → E2 (split-budget removes coupling constraint)
E2, P6 → E3 (gain formula from split accuracy)
E2, E4, E5 → T1 (Pareto dominance from cost + accuracy bounds)
```

### Cycle Check: **NO CYCLES DETECTED**

The DAG is strictly layered: D→A→P1→P2→P3→C1/C2, P1→P6→P7, and P1+E→T1. No forward references.

---

## 2. Assumption Ledger

### A1: Binary Outcome Structure (asm:binary)

| Hypothesis | Where verified | Status |
|-----------|---------------|--------|
| α_c = Pr(correct \| L ≤ b) ∈ (0,1] | theory:59-61: α_c = 99.0% (8B, b=512, n=1319) | ✅ VERIFIED (GSM8K) |
| α_t = Pr(correct \| L > b) ≪ α_c | theory:59-61: α_t = 31.8% at b=512 | ✅ VERIFIED (GSM8K) |
| α_c budget-independent | NOT EXPLICITLY VERIFIED — α_c varies: 99.0% (b=512), 100% (b=256, n=200), 78.7% (b=2048 MATH-500) | ⚠️ CONCERN |
| α_t budget-independent | VIOLATED — rem:alpha-t (theory:100-111) explicitly acknowledges α_t varies with b | ⚠️ ACKNOWLEDGED |

**Usage-minimal set**: P1 requires only that the partition {L≤b, L>b} is exhaustive and the conditional accuracies are well-defined — which is trivially true. The stronger claim α_c ≈ const, α_t ≈ const is needed for P3 and held-out prediction but is NOT formally stated as an assumption.

### Net Recovery Condition (E5, used by T1)

| Hypothesis | Where verified | Status |
|-----------|---------------|--------|
| α_extract^hard > α_nt^hard | method:332-334: 87.5% vs ≈0% (GSM8K pilot n=200) | ✅ VERIFIED (GSM8K) |
| p > 0 (early-stop rate positive) | method:325-326, method:189: p=0.888 on GSM8K | ✅ VERIFIED |

### Stochastic Dominance (used by P5)

| Hypothesis | Where verified | Status |
|-----------|---------------|--------|
| F_{L_{M2}}(b) ≤ F_{L_{M1}}(b) ∀b | theory:251: 8B NS=37.4%, 9B≈0.5%, 27B=0.7% at b=512 | ⚠️ PARTIAL (checked at one budget, claimed ∀b) |
| Acc_nt(b) approx size-invariant | theory:252: 93.1%, 93.2%, 95.5% | ✅ APPROXIMATELY VERIFIED |

### Log-Normal F_L (used by P7)

| Hypothesis | Where verified | Status |
|-----------|---------------|--------|
| F_L is log-normal CDF | theory:397: "as empirically observed" | ⚠️ NO FORMAL FIT TEST SHOWN |
| α_extract monotone non-decreasing in b_r | theory:399: claimed | ⚠️ NOT EMPIRICALLY VERIFIED |
| α_extract concave in b_a | theory:400: claimed | ⚠️ NOT EMPIRICALLY VERIFIED |

---

## 3. Typed Symbol Table

| Symbol | Type | Depends on | Defined at |
|--------|------|-----------|-----------|
| L(q) | ℕ (natural chain length for question q) | model M, question q | theory:27-31 |
| F_L(t) | [0,1] → [0,1], CDF of L over Q | model M, distribution Q | theory:32-33 |
| ρ(b) | [0,1], truncation rate | b, F_L | theory:38-39 |
| b | ℕ, token budget | — | theory:38 |
| α_c | (0,1], accuracy of completed chains | model M, budget b (claimed const, actually varies) | theory:52-53 |
| α_t | [0,1], residual accuracy of truncated chains | model M, budget b (explicitly varies) | theory:55 |
| b* | ℕ, crossover budget | F_L, ᾱ_nt, α_c | theory:147-149 |
| ᾱ_nt | (0,1], saturated nothink accuracy | model M | theory:153-155 |
| b_sat | ℕ, nothink saturation budget | model M | theory:164-166 |
| γ | ℝ₊, budget multiplier | b*, b_sat | theory:177 |
| S(b) | event {\|y\| < b} | b | theory:193 |
| PPV(S) | [0,1] | α_c | theory:196-198 |
| Tax(M,b) | [0,1], thinking tax | M, b, F_L, α_c, Acc_nt | theory:235-236 |
| L_M | distribution on ℕ | model size M | theory:229 |
| Z | token sequence, reasoning trace | — | theory:316-317 |
| A | token sequence, answer | — | theory:316-317 |
| b_r | ℕ, reasoning budget | — | theory:319 |
| b_a | ℕ, answer budget | — | theory:319 |
| α_extract(b_r,b_a) | [0,1], extraction accuracy | b_r, b_a, model | theory:335-337 |
| Δ_split(b) | [0,1], split-budget gain | b_r, F_L, α_extract, α_t | theory:341-344 |
| B_r | ℕ, reasoning budget (method) | — | method:75 |
| B_a | ℕ, answer budget (method) | — | method:78 |
| B_1 | ℕ, triage budget | — | method:139 |
| K | ℕ, max rounds | — | method:141 |
| p | [0,1], early-stop rate at B_1 | B_1, model | method:252-253 |
| t̄_1 | ℝ₊, avg cost of early-stop queries | B_1, model | method:262-263 |
| K̄ | ℝ₊, avg refinement rounds | — | method:253-254 |
| α_c^nt | (0,1], nothink accuracy on easy queries | — | appendix:924 |
| α_nt^hard | [0,1], nothink accuracy on hard queries | — | appendix:926-927 |
| α_th^hard | [0,1], thinking accuracy on hard queries | — | appendix:929-930 |
| α_extract^hard | [0,1], extraction accuracy on hard queries | — | method:322 |

### Consistency check:
- **α_c** is used as budget-independent in P3 and P4 but acknowledged as budget-dependent in R1 and theory verification table (α_c=1.000@b=256 vs 0.990@b=512 vs 0.787@b=2048 MATH-500). **POTENTIAL ISSUE for P3 derivation**.
- **α_t** is explicitly budget-dependent (R1) — the decomposition P1 is correct as an identity at each fixed b, but prediction across budgets requires assuming α_t ≈ const.
- **b_r vs B_r**: Both notations used (lowercase in theory, uppercase in method). Semantically equivalent. Minor notation inconsistency.

---

## 4. Canonical Quantified Statements

### P1: Accuracy Decomposition
```
∀ model M, ∀ budget b ∈ ℕ, ∀ question distribution Q:
  Under A1 (binary outcome) with budget-specific α_c(b), α_t(b):
  Acc_think(b) = F_L(b)·α_c(b) + (1-F_L(b))·α_t(b)
  [Exact accounting identity, no approximation]
```

### P2: Thinking Tax
```
∀ model M, ∀ budget b ∈ ℕ:
  Setting α_t = 0 (first-order approximation):
  Tax(b) ≡ Acc_nt(b) - Acc_think(b) ≈ Acc_nt(b) - F_L(b)·α_c
  When F_L(b) ≪ 1: Tax(b) ≈ Acc_nt(b)
  [Approximate; depends on α_t = 0 being a good approximation]
```

### P3: Crossover Budget
```
∃ b* ∈ ℕ such that Acc_think(b*) = Acc_nt(b*), and:
  b* ≈ F_L^{-1}(ᾱ_nt / α_c)
  [Under: α_t = 0 approximation, Acc_nt saturated at ᾱ_nt for b ≥ b_sat,
   α_c approximately constant across b]
  [Uniqueness not proven — requires F_L strictly increasing, which holds for
   continuous CDFs but needs stating]
```

### P4: Oracle Precision
```
∀ budget b, for thinking mode:
  PPV(S(b)) = Pr(correct | |y| < b) = α_c
  [By definition of α_c = Pr(correct | L ≤ b)]
∀ budget b, for nothink mode:
  PPV(S_nt(b)) ≥ α_c
  [Claimed: early completion selects easier questions]
  [NOT PROVEN — the inequality requires that nothink accuracy on easy
   questions exceeds thinking accuracy on completed chains, which is
   plausible but not formally established]
```

### P5: Inverse Scaling
```
∀ M1, M2 with M2 > M1, ∀ budget b ∈ ℕ:
  IF L_{M2} ≥_{st} L_{M1} (stochastic dominance)
  AND Acc_nt(b) approximately constant across M:
  THEN Tax(M2, b) ≥ Tax(M1, b)
  [Under α_t = 0 approximation]
  [α_c is implicitly assumed model-size-invariant — NOT STATED]
```

### P6: Recoverable Coupling Tax
```
∀ budget b = b_r + b_a, b_r ≤ b:
  Δ_split(b) = (1-F_L(b_r))·(α_extract(b_r,b_a) - α_t(b))
              + F_L(b_r)·α_c - F_L(b)·α_c
  [The simplified form Δ = (1-F_L(b_r))·(α_extract - α_t) is claimed
   to hold when b_r = b, but stated generally as Eq. recoverable-tax.
   See micro-claim analysis for the gap.]
```

### P7: Optimal Budget Allocation
```
Given total B ∈ ℕ:
  b_r* = argmax_{b_r ∈ [0,B]} F_L(b_r)·α_c + (1-F_L(b_r))·α_extract(b_r, B-b_r)
  IF F_L is log-normal CDF
  AND α_extract is non-decreasing in b_r and concave in b_a:
  THEN b_r* is a unique interior maximum
  [NO PROOF PROVIDED — stated without derivation]
  [Uniqueness requires strict concavity of objective, which needs
   more than monotonicity + concavity of components]
```

### T1: MRSD Pareto-Dominance
```
IF α_extract^hard > α_nt^hard (net recovery condition)
AND p > 0 (early-stop rate positive):
THEN:
  (1) Acc_MRSD > Acc_nt(B_1)
  (2) E[T_MRSD] < B_1 + K(B_r + B_a)
  [Pareto-dominates nothink@B_1 on accuracy, think@max on cost]
  [CAVEAT in theorem statement: dominance is over nothink@B_1 only]
```

---

## 5. Micro-Claim Inventory

### MC-1: P1 proof (theory:76-81)
**Context**: D1, D2, A1
**⊢ Goal**: Acc_think(b) = F_L(b)·α_c + (1-F_L(b))·α_t
**Rule**: Law of total probability, conditioning on {L ≤ b}
**Side-conditions**: {L ≤ b} and {L > b} partition the sample space ✓ (trivially true)
**Status**: ✅ VALID — straightforward application of LTP

### MC-2: P2 derivation (theory:87-97)
**Context**: P1 with α_t = 0
**⊢ Goal**: Tax ≈ Acc_nt(b) - F_L(b)·α_c
**Rule**: Substitution of α_t = 0 into P1, then subtraction from Acc_nt
**Side-conditions**: α_t = 0 is a "first-order approximation" — when is this valid?
**Status**: ⚠️ The claim "when F_L(b) ≪ 1, the tax approaches Acc_nt(b) itself" is correct only if α_t ≈ 0, which is acknowledged. The ≈ sign makes this non-rigorous but honest.

### MC-3: P3 proof — crossover (theory:160-173)
**Context**: P2 (with α_t=0), Acc_nt saturated at ᾱ_nt
**⊢ Goal**: b* = F_L^{-1}(ᾱ_nt/α_c)
**Rule**: Set Acc_think(b*) = Acc_nt(b*), solve for F_L(b*)
**Side-conditions**:
  - F_L^{-1} exists (requires F_L strictly increasing) — ⚠️ NOT STATED for discrete distributions
  - Acc_nt(b) = ᾱ_nt for b ≥ b_sat — verified empirically (theory:165-166)
  - α_t = 0 — approximation, acknowledged
  - α_c constant across b — ⚠️ IMPLICIT, violated on MATH-500
**Status**: ⚠️ APPROXIMATELY CORRECT — the "≈" in b* ≈ F_L^{-1}(...) absorbs errors, but the proof doesn't bound the approximation error

### MC-4: P4 — Oracle PPV = α_c (theory:190-204)
**Context**: A1, definition of S(b) = {|y| < b}
**⊢ Goal**: PPV(S) = α_c
**Rule**: By definition, S(b) ↔ {L ≤ b} for thinking mode, and α_c = Pr(correct | L ≤ b)
**Side-conditions**:
  - S(b) = {|y| < b} is the same event as {L ≤ b} — ⚠️ SUBTLE: |y| is actual output length, L is "natural chain length". If model is truncated at b, |y| = b when L > b and |y| = L when L ≤ b. So |y| < b ↔ L < b (strict inequality). The definition uses L ≤ b (weak). This creates an off-by-one: samples with L = b exactly satisfy L ≤ b but NOT |y| < b.
**Status**: ⚠️ MINOR — Off-by-one between |y| < b and L ≤ b. Pr(L = b exactly) ≈ 0 for continuous-like distributions, so negligible in practice.

### MC-5: P4 — Nothink PPV ≥ α_c (theory:201-203)
**Context**: None stated
**⊢ Goal**: PPV(S_nt) ≥ α_c
**Rule**: "early completion in non-thinking mode selects for easier questions where model accuracy is higher"
**Side-conditions**: This is an UNJUSTIFIED ASSERTION — no proof given
**Status**: ⚠️ UNJUSTIFIED — The claim is intuitively reasonable but formally requires: (i) a notion of "easiness", (ii) that nothink early-stop selects for easy questions, (iii) that easy questions have accuracy ≥ α_c. None proven.

### MC-6: P5 proof (theory:240-247)
**Context**: P2 (thinking tax formula)
**⊢ Goal**: Tax(M2,b) ≥ Tax(M1,b)
**Rule**: Stochastic dominance → F_{L_{M2}}(b) ≤ F_{L_{M1}}(b), so -F_{L_{M2}}·α_c ≥ -F_{L_{M1}}·α_c
**Side-conditions**:
  - Acc_nt(b) approximately constant across M — verified (93.1%, 93.2%, 95.5%)
  - α_c approximately constant across M — ⚠️ IMPLICIT, NOT STATED
  - α_t = 0 approximation — acknowledged
**Status**: ⚠️ HIDDEN ASSUMPTION — α_c model-size-invariance is used but not stated. If α_c differs across model sizes, the proof breaks.

### MC-7: P6 proof (theory:354-369)
**Context**: P1, D3
**⊢ Goal**: Δ_split = (1-F_L(b_r))·(α_extract - α_t) [at b_r = b]
**Rule**: Subtract coupled from split accuracy
**Derivation check**:
  Acc_split = F_L(b_r)·α_c + (1-F_L(b_r))·α_extract
  Acc_coupled = F_L(b)·α_c + (1-F_L(b))·α_t
  Δ = (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + (F_L(b_r) - F_L(b))·α_c
  = (1-F_L(b_r))·(α_extract - α_t) + (F_L(b_r) - F_L(b))·α_c + (1-F_L(b_r))·α_t - (1-F_L(b))·α_t
  Wait, let me redo:
  Δ = [F_L(b_r)·α_c + (1-F_L(b_r))·α_extract] - [F_L(b)·α_c + (1-F_L(b))·α_t]
  = (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + (F_L(b_r)-F_L(b))·α_c

  The proof says: = (1-F_L(b_r))(α_extract - α_t) + F_L(b_r)·α_c - F_L(b)·α_c
  Check: (1-F_L(b_r))·α_extract - (1-F_L(b_r))·α_t + (F_L(b_r)-F_L(b))·α_c
  = (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + [(1-F_L(b))-(1-F_L(b_r))]·α_t + (F_L(b_r)-F_L(b))·α_c
  = (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + (F_L(b_r)-F_L(b))·α_t + (F_L(b_r)-F_L(b))·α_c
  = (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + (F_L(b_r)-F_L(b))·(α_t + α_c)

  This does NOT match the proof's claimed decomposition. Let me re-check the proof text:
  "Δ_split = (1 - F_L(b_r))(α_extract - α_t) + F_L(b_r)·α_c - F_L(b)·α_c"

  Expanding: (1-F_L(b_r))·α_extract - (1-F_L(b_r))·α_t + (F_L(b_r)-F_L(b))·α_c

  But the original Δ is:
  (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + (F_L(b_r)-F_L(b))·α_c

  So the proof's version has -(1-F_L(b_r))·α_t where the actual has -(1-F_L(b))·α_t.
  Difference: (1-F_L(b))·α_t - (1-F_L(b_r))·α_t = (F_L(b_r)-F_L(b))·α_t

  Since b_r ≤ b → F_L(b_r) ≤ F_L(b) → this difference is ≤ 0.
  So the proof's decomposition UNDERSTATES the actual Δ by (F_L(b)-F_L(b_r))·α_t.

  **This is an ALGEBRA ERROR in the proof.**

**Status**: ❌ **ALGEBRA ERROR** — The proof's intermediate decomposition uses α_t at budget b_r instead of budget b. The correct expression is:
  Δ = (1-F_L(b_r))·α_extract - (1-F_L(b))·α_t + (F_L(b_r)-F_L(b))·α_c
  The proof claims:
  Δ = (1-F_L(b_r))·(α_extract - α_t) + (F_L(b_r)-F_L(b))·α_c
  These differ by (F_L(b)-F_L(b_r))·α_t ≥ 0.

  However, the FINAL claimed result (Eq. recoverable-tax, at b_r = b) is:
  Δ = (1-F_L(b))·(α_extract - α_t)
  At b_r = b, the algebra error vanishes because F_L(b_r) = F_L(b).

  **SEVERITY**: The intermediate step is wrong but the headline result is correct at b_r = b. The general formula (for b_r < b) as stated in the theorem IS wrong by the missing term.

### MC-8: P7 — uniqueness of interior maximum (theory:385-401)
**Context**: F_L log-normal, α_extract monotone in b_r, concave in b_a
**⊢ Goal**: Unique interior maximum of Eq. optimal-split
**Rule**: NONE — claimed without proof
**Side-conditions**: Log-normality not formally tested, monotonicity/concavity not verified
**Status**: ❌ UNJUSTIFIED — No proof of uniqueness. For uniqueness of interior max, need strict concavity of objective. The composition of log-normal CDF (which is strictly increasing, concave after the mode) with the sum of two terms does not trivially yield strict concavity.

### MC-9: T1 proof — accuracy bound (method:343-348, appendix:921-957)
**Context**: p > 0, net recovery condition
**⊢ Goal**: Acc_MRSD > Acc_nt(B_1)
**Rule**: Decompose by easy/hard, use net recovery for hard subset
**Proof check**:
  Acc_MRSD = p·α_c^nt + (1-p)·α_th^hard (appendix:929-930)
  Wait — the proof in the appendix uses α_th^hard, but MRSD uses α_extract^hard (split-budget), not plain thinking accuracy. Let me check...

  The method section's theorem statement says "α_extract^hard" in the net recovery condition but the appendix proof (line 929-930) writes:
  Acc_TOWN = p·α_c^nt + (1-p)·α_th^hard

  This conflates TOWN (cascade without decoupling) with MRSD (with decoupling). The theorem is about MRSD but the appendix proof is about TOWN.

**Status**: ⚠️ **SCOPE CONFUSION** — The full proof in appendix:921-957 proves Pareto-dominance for TOWN (using α_th^hard = thinking accuracy), not MRSD (which uses α_extract^hard). The method section's theorem (Theorem 1) claims MRSD Pareto-dominance using the net recovery condition with α_extract^hard, but the proof doesn't distinguish these.

  The confusion works out because the proof structure is the same regardless of which α is used for the hard subset, as long as it exceeds α_nt^hard. But the proof should be clear about which method it's proving dominance for.

### MC-10: T1 proof — cost bound (method:349-352, appendix:942-951)
**Context**: p > 0, early-stop queries cheaper
**⊢ Goal**: E[T] < B_1 + K(B_r + B_a) (method) or < B_2 (appendix)
**Proof check (appendix version)**:
  E[T] = p·t̄_1 + (1-p)(B_1 + t̄_2)
  Claims: "< p·B_1 + (1-p)(B_1 + B_2) = B_1 + (1-p)B_2 < B_2"
  Step 1: t̄_1 < B_1 ✓ (early-stop means fewer tokens)
  Step 2: t̄_2 ≤ B_2 ✓ (generation capped at budget)
  Step 3: B_1 + (1-p)B_2 < B_2 ⟺ B_1 < pB_2 ⟺ B_1/B_2 < p
  This requires B_1 < p·B_2. With B_1=256, p=0.888, B_2=512: 256 < 0.888×512 = 454.7 ✓
  But this is NOT universally true — it depends on the specific parameters.

**Status**: ⚠️ **HIDDEN PARAMETER DEPENDENCE** — The cost bound requires B_1 < p·B_2, which is parameter-dependent and not stated as a condition. The proof says "the last inequality holds because p > 0 and B_1 < B_2" but this is INSUFFICIENT — you need B_1/B_2 < p specifically.

### MC-11: Hoeffding confidence bound (theory:207-211)
**Context**: n=475 natural-stop samples, α̂_c = 0.963
**⊢ Goal**: 95% lower bound α_c ≥ 0.919
**Rule**: Hoeffding's inequality
**Check**: ε = √(ln(2/0.05)/(2·475)) = √(ln(40)/950) = √(3.689/950) = √0.003883 = 0.0623
  Lower bound = 0.963 - 0.0623 = 0.901
  Paper claims 0.919. Let me recheck: ln(2/0.05) = ln(40) ≈ 3.689.
  Actually wait — the one-sided Hoeffding bound is: Pr(X̂ - X ≥ ε) ≤ exp(-2nε²)
  For 95% one-sided: exp(-2·475·ε²) = 0.05 → ε = √(ln(20)/950) = √(2.996/950) = √0.003154 = 0.0562
  Lower bound = 0.963 - 0.0562 = 0.907.

  Paper uses √(ln(2/0.05)/(2·475)). ln(2/0.05) = ln(40) ≈ 3.689.
  √(3.689/950) = √0.003883 = 0.0623.
  0.963 - 0.0623 = 0.901, not 0.919.

  For α_c = 0.919: ε = 0.963 - 0.919 = 0.044.
  exp(-2·475·0.044²) = exp(-2·475·0.001936) = exp(-1.839) ≈ 0.159.
  This is only an 84.1% confidence bound, not 95%.

  **Wait** — the text says α̂_c = 0.963, but the confidence bound uses Hoeffding on bounded [0,1] random variables. Let me check the exact formula again.

  Standard two-sided Hoeffding: Pr(|X̂-μ| ≥ t) ≤ 2exp(-2nt²)
  For 95% CI: 2exp(-2·475·t²) = 0.05 → t = √(ln(40)/(2·475)) = √(3.689/950) = 0.0623
  So 95% CI is [0.963-0.0623, 0.963+0.0623] = [0.901, 1.025]
  Lower bound would be 0.901, not 0.919.

  The paper writes: "0.963 - √{ln(2/0.05)/(2·475)} = 0.919"
  This would require √(3.689/950) = 0.044.
  But 0.044² = 0.001936, 0.001936 × 950 = 1.839, which is NOT 3.689.
  √(3.689/950) = 0.0623, giving 0.963 - 0.0623 = 0.901.

**Status**: ❌ **ARITHMETIC ERROR** — The computed bound 0.963 - √(ln(2/0.05)/(2·475)) ≈ 0.901, not 0.919 as claimed. The error is in the arithmetic, not the method.

---

## 6. Limit-Order Map

| Statement | Limit | Uniformity | Location |
|-----------|-------|-----------|----------|
| "F_L(b) ≪ 1" implies tax ≈ Acc_nt(b) | As b → 0 | Fixed M, Q | theory:96-97 |
| α_t → 0 at low budgets | As b → 0 | Not specified | theory:55, R1 |
| "crossover budget" b* | Where Acc_think = Acc_nt | Fixed M, Q | theory:147 |
| Chain-length stochastic dominance | Fixed b, comparison across M | Claimed ∀b | theory:231 |
| "the coupling tax vanishes" | As b → ∞ (F_L(b) → 1) | Fixed M, Q | theory:378-380 |
| Log-normal tail behavior | Affects b* location | Not analyzed | theory:397 |
| Budget multiplier γ | Ratio, fixed | Varies with task difficulty | theory:177-183 |

### Key Uniformity Concerns:
1. α_c is treated as constant but varies across b (0.787 to 1.000) — non-uniform
2. Stochastic dominance L_{M2} ≥_{st} L_{M1} checked at single budget point, claimed universally
3. The "≈" in crossover formula has unspecified error bounds

---

## Summary of Issues Found in Phase 0.5

| ID | Category | Severity | Location | Description |
|----|----------|----------|----------|-------------|
| S1 | HIDDEN_ASSUMPTION | MAJOR | P5 proof | α_c model-size-invariance used but not stated |
| S2 | MISSING_DERIVATION | MAJOR | P7 | Uniqueness of optimal split claimed without proof |
| S3 | UNJUSTIFIED_ASSERTION | MINOR | P4 nothink | PPV(S_nt) ≥ α_c claimed without proof |
| S4 | LOGICAL_GAP | CRITICAL | P6 proof | Algebra error in general Δ_split derivation (correct at b_r=b) |
| S5 | SCOPE_OVERCLAIM | MINOR | T1 proof | Appendix proves TOWN dominance, theorem claims MRSD |
| S6 | HIDDEN_ASSUMPTION | MAJOR | T1 cost proof | Requires B_1 < p·B_2, not just B_1 < B_2 |
| S7 | UNJUSTIFIED_ASSERTION | MINOR | P3 proof | F_L^{-1} existence requires F_L strictly increasing (unstated) |
| S8 | CONSTANT_DEPENDENCE_HIDDEN | MINOR | Multiple | α_c treated as constant but varies 0.787-1.000 across budgets |
| S9 | MISSING_DERIVATION | MINOR | MC-11 | Hoeffding bound arithmetic: 0.901 ≠ 0.919 |
| S10 | HIDDEN_ASSUMPTION | MINOR | P5 | Stochastic dominance checked at one budget, claimed ∀b |

---
*Generated: 2026-04-13, Phase 0.5 of proof-checker (nightmare)*
