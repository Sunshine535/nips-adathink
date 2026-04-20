# Reviewer Memory

Persistent reviewer memory across `/auto-review-loop` rounds. Each round appends — never delete. Fed back to GPT every round so it tracks its own suspicions.

## Prior history (medium difficulty, pre-2026-04-21)

- Previous 4 rounds at medium difficulty scored: 8.0 → 8.5 → 7.5 (rescored to best-paper scale)
- Final medium-difficulty score: 7.5/10 ("Strong accept, not best paper yet")
- Key pending items: DeepSeek full replication, DeepSeek IRIS experiments
- DeepSeek MATH-500 full-scale n=500 nothink + thinking data integrated
- Pivoted narrative from CTT hero to Stage-3 decoupled extraction (CTT mini-pilot null)

## Round 0 — external human-reviewer concerns (2026-04-20, entering Round 1 nightmare)

External critique triggered this loop. Unresolved concerns the nightmare reviewer should verify:

- **Claim-audit state**: `PAPER_CLAIM_AUDIT.md` promoted to fresh PASS (commit `e03617c`, 2026-04-20). Prior 2026-04-17 Round-2 FAIL archived to `archive/pre_coupling_tax_pivot/PAPER_CLAIM_AUDIT_R2_FAIL_20260417.md`. Verify this holds on disk.
- **Online vs post-hoc Stage 2**: `scripts/iris_online_stage2.py` exists + `--online_stage2` flag in `run_iris.py` (commit `a73e6c9`), but headline IRIS numbers (8B GSM8K 90.9%, 27B MATH-500 77.5%, multi-seed) are from the ORIGINAL post-hoc runner. Efficiency claims are effective-token accounting, not wall-clock. Watch for "deployment-faithful" being mis-applied.
- **8B MATH-500 saturation**: Stage-3 improvement collapses to +0.4pp at n=500 and is non-significant on seed 123/456 (p=0.46, 0.86). Verify paper does not claim universal improvement.
- **CTT null**: AUC≈0.5 on both 8B and 27B GSM8K. CTT must be presented strictly as ablation, never as method.
- **477 vs 460 seed-mixing**: fix commit `fb5e334` reconciled 5 occurrences to 460 when paired with 56.9%. Verify no 477 remains mispaired.
- **Stage-3 table 8B GSM8K row**: baseline corrected 90.0% → 89.0% with Δ+4.0pp (was +3.0pp).
- **NEW evidence available to verify**: 27B GSM8K IRIS @ b=4096 n=200 run finished on H800 at 2026-04-20 22:02 UTC. checkpoint_iris_200.json shows acc=93.5%, avg_tok=1182, stages {1:122, 2:50, 3:28}. TOWN baseline started but not yet complete.

## Patterns being tracked

- Evidence inflation: same-sample paired vs full-scale-pilot mixing.
- Reproducibility gap: every table cell must trace to a specific JSON file.
- "Online / adaptive / deployment-faithful" must not apply to post-hoc trace-truncation accounting.
- Claim-audit must match the latest manuscript, not stale snapshots.
