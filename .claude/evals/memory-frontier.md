# Memory Frontier Evaluation

## Capability Evals

- [x] Retrieve 8/8 representative memories at rank 5 or better.
- [x] Achieve MRR >= 0.75 on the isolated fixture.
- [x] Prevent cross-project scope leakage between `alpha` and `beta`.
- [x] Preserve deterministic capture idempotency.
- [x] Resolve a conflict and pass lifecycle audit.
- [x] Create and verify a restorable SQLite backup.
- [x] Pass 3 consecutive identical evaluation runs.

## Regression Evals

- [x] Existing platform suite remains green.
- [x] Golden retrieval remains 4/4.
- [x] Vault sync does not modify the isolated fixture outside the test root.

## Current Thresholds

- Recall@5: >= 0.90
- MRR: >= 0.75
- Scope isolation: 100%
- pass^3: 100%
- Lifecycle audit: `ok=true`
- Backup verification: `verified=true`
