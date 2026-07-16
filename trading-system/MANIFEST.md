# Unified Trading System Source

This directory is the single active Codex source for the TRAE Work CN trading system.

- `runtime/`: the WSL canonical trading runtime snapshot.
- `legacy-tools/`: C-drive scripts/tools/utilities that are not exact duplicates of the canonical runtime.
- `docs/`: redacted TRAE design and workflow context.
- `manifests/`: four-source audit and selected-file provenance.

The WSL source remains the runtime authority during this migration. C:, D: and TRAE are reference inputs only and are not active project roots. D: was verified as an incomplete one-file mirror and contributes nothing. Raw TRAE sessions, caches, built-ins, credentials and runtime data remain local and are never copied into the active root.

All future changes must start from this directory, update the manifest, pass preflight/tests/production verification, and then refresh the GitHub `trading-system/` mirror. No second active copy is permitted.
