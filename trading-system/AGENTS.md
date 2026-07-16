# Unified Trading System Rules

- This directory is the only active Codex project root for the trading system.
- `runtime/` is the canonical executable source; `legacy-tools/` is reference tooling and must not silently become runtime dependencies.
- Before editing, search the unified root and read the relevant provenance record.
- Preserve real runtime feedback and use one bounded change per experiment.
- Run tests, static checks, production gates and independent verification before claiming completion.
- Never copy secrets, raw TRAE sessions, caches, large runtime data or credentials.
- Keep `manifests/` updated whenever source provenance changes.
