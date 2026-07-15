# Codex Memory Platform

Canonical Codex-owned memory runtime for the Obsidian vault.

The versioned source mirror is hosted at `Yhvoid0101/-`; local databases,
vault content, credentials, logs, and temporary work are excluded from Git.

## Capabilities

- SQLite WAL + FTS5 durable index
- project scope, confidence, importance, recency scoring
- note backlinks and duplicate-title conflict detection
- redacted observation capture and evidence-based promotion
- backup, golden retrieval evaluation, and MCP stdio server
- isolated CPU semantic worker with JSONL supervision and FTS-only last-resort fallback
- durable autonomy control plane with idempotent jobs, restart recovery, bounded retries, and evidence-gated evolution
- automatic failure observation, repeated-pattern analysis, and bounded evolution candidate generation
- authenticated GitHub read-only snapshot of repository metadata, open issues, and pull requests
- independent from Hermes Chroma, BM25, HTTP services, and memory service

## Commands

```powershell
& D:\hermes\.venv\Scripts\python.exe src\codex_memory_platform.py sync
& D:\hermes\.venv\Scripts\python.exe src\codex_memory_platform.py status
& D:\hermes\.venv\Scripts\python.exe src\codex_memory_platform.py evaluate
& D:\hermes\.venv\Scripts\python.exe src\codex_memory_platform.py verify-backup <backup-db-path>
& D:\hermes\.venv\Scripts\python.exe src\codex_memory_platform.py serve
& D:\hermes\.venv\Scripts\python.exe src\codex_autonomy.py status
& D:\hermes\.venv\Scripts\python.exe src\codex_autonomy.py worker
& D:\hermes\.venv\Scripts\python.exe src\codex_autonomy.py github-check
& D:\hermes\.venv\Scripts\python.exe src\codex_autonomy.py evolution-check
```

`worker` only executes the fixed memory check, GitHub snapshot, and evolution
analysis jobs. The
GitHub integration uses the existing `gh` credential store and writes the
redacted local snapshot to `D:\hermes\codex_memory_store\github\last_snapshot.json`.
It cannot comment, merge, push, edit workflows, or execute arbitrary commands.

Every worker outcome is stored as a redacted control event. Repeated low-risk
failures are automatically verified and promoted when their evidence matches;
the resulting bounded retry policy changes later scheduling. Code, credentials,
deployment, and external write behavior never self-modify.
