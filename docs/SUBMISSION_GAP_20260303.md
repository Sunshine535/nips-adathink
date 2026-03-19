# Submission Gap Report (2026-03-03)

## Overall
- If we pick **one main paper now**, `01_adathink` is the only direction close to NeurIPS submission bar.
- If we insist on **all six directions simultaneously投稿**, the project is still far from submission-ready.

## Readiness by Track
| Track | Readiness (estimate) | Why |
|---|---:|---|
| `01_adathink` | 72% | 27B + 8B-think replicated, paired CIs positive, matched-cost evidence now significant at 7 seeds (`n=280`) |
| `02_trace_hallu` | 25% | only offline pilot on GSM8K-derived traces, no real claim-level benchmark |
| `03_noisepo` | 22% | only synthetic-noise pilot, no full preference-model training/eval |
| `04_unirag_policy` | 24% | budget-policy proxy exists, no real retriever/reranker/citation stack |
| `05_text2subspace` | 20% | low-rank proxy only, no text-to-adapter generator over checkpoints |
| `06_templatebank_pp` | 23% | lexical-template pilot only, no full extraction/instantiation pipeline |

## `01_adathink` Remaining Gap to Submission
1. Missing required ablations (high priority):
   - `halting-only` vs full controller
   - `no-branch` vs full controller
   - verifier on/off with full cost accounting
2. Missing out-of-domain validation:
   - current strong evidence is GSM8K-centric; need at least one extra family (e.g., MATH/BBH-style) under matched-cost protocol.
3. Missing latency wall-clock analysis:
   - current tables emphasize token-cost; need runtime/throughput curves and matched-latency view.
4. Missing final paper package:
   - camera-ready figures/tables, statistical appendix, deterministic rerun script.

## Fastest Path to One Submit-Ready Paper (`01_adathink`)
1. Run 3 ablation groups on current 7-seed setting (same split protocol).
2. Add 1 out-of-domain dataset with minimum 3 seeds.
3. Freeze one primary operating point (`penalty=0.8`) + one quality-first point (`penalty=0.0`) for main table.
4. Produce final paper draft with reproducibility appendix and artifact map.

## Practical Estimate
- For **single-paper target (`01_adathink`)**: about **2-3 weeks** of focused runs + writing.
- For **all six tracks each to paper-grade**: about **3-5 months** (parallel team), much longer for single-person serial execution.
