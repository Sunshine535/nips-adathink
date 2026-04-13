# Query Pack — The Coupling Tax (IRIS/MRSD)

> Compressed context for ideation. Max 8000 chars. Auto-generated 2026-04-12.

## Project Direction

NeurIPS 2026: "The Coupling Tax" — fixed token budget下thinking mode在所有≤2048预算全面劣于nothink mode。提出MRSD split-budget方法（IRIS为1-round变体）。核心发现：coupling tax（推理链与答案耦合导致截断浪费）。Round 4 review: 7.5/10 best-paper scale. 差距：cross-family DeepSeek replication (running).

## Top 5 Gaps (ranked)

**G5** [in progress] Cross-family validation — DeepSeek-R1-Distill-Llama-8B full replication running on A100+H800. Expected 24-36h. This is THE gap to best paper.

**G4** [unresolved] Minimum B_think per model scale. 27B needs larger B_think but threshold未确定. Scoped as future work.

**G7** [unresolved] Coupling tax on non-reasoning tasks (open-ended, code, agentic) completely untested. Acknowledged as limitation.

**G1** [resolved] Truncation-waste decomposition: Acc_think(b) = α_c·F_L(b) + α_t·(1-F_L(b)).

**G3** [resolved] Split-budget overcomes coupling tax: IRIS@4096=74.0% > nothink@1024=59.8% on MATH-500 (n=500, same hardware).

## Paper Clusters

**Cluster 1: CoT Foundations** — wei2022_chain, snell2024_scaling. CoT effectiveness and optimal compute. We show that under budget constraints, more thinking often hurts.

**Cluster 2: Overthinking & Waste** — chen2024_overthinking, muennighoff2025_s1. LLMs overthink easy problems. We extend: on hard problems under budget, thinking wastes tokens via coupling.

**Cluster 3: Adaptive Methods** — adathink2025, nvidia2025_pareto, deepseekr1_2025. Adaptive token allocation and Pareto-optimal reasoning. IRIS adds difficulty-aware triage + decoupled answering.

## Failed/Negative Ideas

**idea:001 (partial failure at 27B)**: Cascade hurts at 27B with B_think=512 — think@512=18.3% too weak. Net recovery condition fails. LESSON: cascade needs B_think large enough for fallback to add value. 27B requires B_think≥1024.

**claim:C5 INVALIDATED**: Original "MRSD loses to nothink on MATH-500" was artifact of (1) insufficient B_think=1024 (100% truncation), (2) pilot n=200 overestimated nothink by ~10pp. Full-scale with B_think≥2048 definitively wins.

## Top Papers (ranked)

1. **wei2022_chain** [core] — CoT baseline. No budget-constrained analysis.
2. **snell2024_scaling** [core] — Optimal test-time compute. Assumes thinking always helps.
3. **chen2024_overthinking** [core] — Overthinking on easy tasks. We extend to hard tasks under budget.
4. **deepseekr1_2025** [core] — R1 reasoning model. Cross-family validation target (exp running).
5. **muennighoff2025_s1** [related] — Budget forcing. Complementary to truncation analysis.
6. **adathink2025** [related] — Adaptive thinking allocation. Different approach to same problem.
7. **nvidia2025_pareto** [related] — Pareto-optimal reasoning.

## Active Chains

- idea:001 → exp:iris_math500_fullscale [WORKS: 74.0%] → claim:C7 [STRONG] → G3 resolved
- claim:C1 → exp:bbh [n=1187] → claim:C8 [cross-task validation] → BBH in main text
- claim:C1 → exp:deepseek_replication [RUNNING] → G5 pending → THE gap to best paper

## Open Unknowns

1. Does coupling tax curve replicate on DeepSeek-R1? (Running: exp:deepseek_replication)
2. Does IRIS work on DeepSeek-R1? (Depends on Q1)
3. What is minimum B_think for 27B cascade? (G4, scoped as future work)
4. Does coupling tax exist on non-reasoning tasks? (G7, acknowledged as limitation)

---
*Auto-generated. Last updated: 2026-04-12*
