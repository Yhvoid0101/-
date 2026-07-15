# Codex Memory Platform

Canonical Codex-owned memory runtime for the Obsidian vault.

## Capabilities

- SQLite WAL + FTS5 durable index
- project scope, confidence, importance, recency scoring
- note backlinks and duplicate-title conflict detection
- redacted observation capture and evidence-based promotion
- backup, golden retrieval evaluation, and MCP stdio server
- isolated CPU semantic worker with JSONL supervision and FTS-only last-resort fallback
- durable autonomy control plane with idempotent jobs, restart recovery, bounded retries, and evidence-gated evolution
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
```
