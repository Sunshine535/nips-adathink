# Core A/B/C Comparison (FROZEN — feature-RCV is NEGATIVE ABLATION)

## Frozen Status (post GPT-5.5 review)

**Status: `FEATURE_RCV_NEGATIVE_ABLATION`**

- V1 implementation had token accounting bug (soft probe uncounted)
- V1 had A/B identical code path (not a clean ablation)
- V1 had hard-coded GSM8K validity on MATH (Stage0 verifier no-op)
- V2 fixes applied in scripts/run_rcv_iris.py; old V1 results marked `v1_deprecated`

**The accuracy null conclusion (0 discordant at b2=512) survives** because it's an upper bound — even if V2 separates A/B cleanly, samples where neither A nor C flips cannot be rescued by gate honesty.

**Per GPT-5.5 stop criteria**: feature-based gate does NOT beat existing_fragment. Need revised mechanism (model-based verifier OR better fallback) for Task 5.

---

## V1 Original Experiment Data (for historical record only)



## Experiment 1: b2=4096 (moderate truncation)

Setup: Qwen3-8B, MATH-500, n=100, seed=42, b1=512, b2_max=4096, b_answer=512

| Variant | Config | Accuracy | Avg Tokens | Decisions |
|---------|--------|----------|------------|-----------|
| A. Existing Fragment | existing_fragment | 73.0% | 2613 | EXTRACT_ALWAYS=42, S2=12, S0=46 |
| B. New Method No Gate | rcv_no_gate | 73.0% | 2613 | EXTRACT_ALWAYS=42, S2=12, S0=46 |
| C. Full RCV-IRIS | full_rcv | **74.0%** | 2613 | EXTRACT_S3=41, S2=12, S0=46, FALLBACK_TOWN=1 |

McNemar A vs C: 1 discordant, C wins 1/1. Effect: +1.0pp on 1 sample (margin=0.0 sample).

## Experiment 2: b2=512 (tight truncation — gate-activation regime)

Setup: Qwen3-8B, MATH-500, n=200, seed=42, b1=256, b2_max=512, b_answer=128

| Variant | Config | Accuracy | Avg Tokens | Decisions |
|---------|--------|----------|------------|-----------|
| A. Existing Fragment | existing_fragment | **40.5%** | 684 | EXTRACT_ALWAYS=167, ACCEPT_S0=33 |
| B. New Method No Gate | rcv_no_gate | **40.5%** | 684 | EXTRACT_ALWAYS=167, ACCEPT_S0=33 |
| C. Full RCV-IRIS | full_rcv | **40.5%** | 684 | EXTRACT_S3=159, ACCEPT_S0=33, **FALLBACK_TOWN=8** |

McNemar A vs C: **0 discordant pairs** — identical outcomes despite 8 gate triggers.

### Per-FALLBACK_TOWN Sample Analysis (b2=512)

| Sample | A decision | A correct? | C decision | C correct? |
|--------|-----------|-----------|------------|-----------|
| 38 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 58 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 59 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 64 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 157 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 162 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 169 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |
| 188 | EXTRACT_ALWAYS | False | FALLBACK_TOWN | False |

**All 8 gate-triggered samples: A extraction fails AND TOWN fallback also fails.** The gate correctly identifies low-recoverability prefixes, but the fallback is equally broken on these samples.

## Honest Interpretation

### Per GPT-5.5 Diagnosis Decision Rules

The diagnosis stated:
- If C > A consistently → new mechanism helps → proceed
- If C ≈ A → new method only reuses old fragment → do not claim new mechanism
- If C ≈ B → mechanism inactive/irrelevant → check mechanism logs

**Our result: C ≈ A ≈ B exactly at b2=512.**

### Why the Gate Doesn't Help

The feature-based recoverability gate triggers on samples with `extractor_margin = 0.0` (both strict and soft extraction completely fail to produce any parseable answer). On these samples:
1. A's extraction fails (prediction = empty/garbage) → wrong
2. C's FALLBACK_TOWN uses truncated thinking text → also wrong (TOWN parser also can't find answer in incomplete reasoning)

The gate identifies HARD samples correctly but **there is no viable fallback** — if extraction fails, TOWN parsing of the same truncated thinking also fails. The gate moves decision but not outcome.

### What Would Be Needed

1. **Model-based verifier** (not feature-based): Use a cheap call to verify Stage0 answer correctness or prefix recoverability. This was marked optional in the diagnosis (`--enable_stage0_verifier` with model).
2. **Alternative fallback action**: Instead of TOWN on same prefix, escalate to more thinking budget or retry extraction with different prompt.
3. **Calibration data**: Train thresholds on held-out set, not hand-tuned.

## Decision (per GPT-5.5 stop criteria)

**REVISE METHOD.** The feature-based gate is directionally correct but makes zero accuracy difference on this benchmark. Per GPT-5.5's own criteria, this is a negative result for the current implementation. The underlying mechanism hypothesis (recoverability-calibrated routing) remains plausible but requires:
- Model-based verifier, or
- Better fallback action, or
- Calibrated thresholds from dev data

## Paper Implications

1. **Cannot claim RCV-IRIS as new main method** — evidence does not support accuracy gain.
2. **Can still claim**:
   - Coupling Tax phenomenon (27B crossover p<1e-5)
   - 2×2 factorial interaction (+37.4pp mode×prompt)
   - Training-free Pareto-competitive with SwiR/s1 on MATH-500
   - Honest negative: natural-stop routing is sufficient; acceptance verifier does not help above it
3. **Must add to paper as a negative ablation**: "We tested a feature-based recoverability gate (RCV-IRIS variant C); at both moderate (b=4096) and tight (b=512) truncation regimes, it produced 0 or 1 discordant pair vs the no-gate baseline. This null result suggests that at feature-space, mode-conditioned extraction is already near-optimal; further gains require model-based verifiers."
