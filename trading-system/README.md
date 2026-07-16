# Trading System Snapshot

This directory is the Codex-managed, auditable snapshot of the TRAE Work CN trading system.

## Canonical source

- Original source: `/home/lmy/hermes_v6/sandbox_trading` in WSL Ubuntu
- Snapshot date: 2026-07-16 UTC
- The original WSL tree is preserved and is the source of truth for future refreshes.

## Snapshot policy

Every source file is represented in `manifests/inventory.json` with its relative path, size, modification time, and SHA-256. Files copied under `content/` are the unique, maintainable snapshot: source code, project configuration, automation, tests, and documentation under GitHub's file-size envelope.

Generated runtime data, large logs, caches, model artifacts, backups, archives, and possible credential files are not active repository content. They are not silently discarded: their paths, hashes, sizes, classification, and exclusion reason remain in the inventory. Exact-content duplicates are represented by `manifests/duplicates.json`.

## Future optimization workflow

1. Treat `content/` as the versioned review surface; never edit the WSL source by assumption.
2. Reproduce behavior from real runtime logs and reports, preserving a redacted baseline before changing one variable.
3. Run deterministic tests, static checks, and the production gate before any runtime canary.
4. Compare the same workload and protected metrics, then retain or roll back the candidate with evidence.
5. Refresh the snapshot from WSL only through an auditable inventory/duplicate pass.

Secrets, tokens, private keys, and raw credentials must never be committed. Runtime data stays in the canonical WSL tree or an explicitly approved artifact store.
