# Gap Map

Identified field gaps with stable IDs. Ranked by: unresolved + linked ideas + failed experiments.

| ID | Gap | Status | Linked Ideas | Linked Papers |
|----|-----|--------|-------------|---------------|
| G1 | No theory for truncation waste under budget-constrained CoT | **resolved** | idea:001 | paper:wei2022_chain, paper:snell2024_scaling |
| G2 | No systematic study of thinking vs non-thinking at matched budgets | **resolved** | idea:001 | paper:chen2024_overthinking |
| G3 | Unknown whether split-budget generation can overcome the coupling tax | **resolved** | idea:001 | — |
| G4 | Unknown minimum B_think for cascade to help at each model scale | **unresolved** | — | — |
| G5 | No cross-family validation of coupling tax (only Qwen tested) | **in progress** | — | paper:deepseekr1_2025 |
| G6 | Budget-scaling ceiling hypothesis: when does split-budget beat nothink scaling? | **resolved** | idea:001 | — |
| G7 | No study of coupling tax on non-reasoning tasks (open-ended, code, agentic) | **unresolved** | — | — |

## Resolution Notes (2026-04-12)

**G3 → resolved**: IRIS@4096 achieves 74.0% [70.0, 77.7] on MATH-500 full-scale (n=500), definitively beating nothink@1024 (59.8%) by +14.2pp with non-overlapping CIs. Split-budget generation works.

**G5 → in progress**: DeepSeek-R1-Distill-Llama-8B full replication running on A100 (GSM8K) + H800 (MATH-500). Results expected in 24-36h. Partial pilot data (n=40/80) shows consistent utilization collapse.

**G6 → resolved**: The "budget-scaling ceiling" was an artifact of insufficient B_think (1024) and pilot sample bias. Full-scale experiments show monotonic improvement with B_think: 62.5%→67.2%→74.0% as B_think goes from 1024→2048→4096.

## Key Unresolved Gaps

**G4**: We know 27B needs larger B_think, but don't know the minimum. Need B_think sweep (512, 1024, 2048, 4096) at each model size. Scoped as future work in paper.

**G7**: Our scoping explicitly excludes non-reasoning tasks, but reviewers may ask. Acknowledged as limitation.

---
*Updated: 2026-04-12*
