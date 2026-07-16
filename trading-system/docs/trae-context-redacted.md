# TRAE Trading Context (Redacted)

This note is a redacted, reusable summary of the TRAE Work CN project context. Raw project memory and session JSONL remain local and are represented by the four-source audit manifest.

## Source-of-truth decision

The TRAE project memory identifies the WSL `hermes_v6/sandbox_trading` tree as the unique trading-code source. The C: Hermes tree and the D: link are not independent complete trading sources.

## Reusable workflow constraints

- Start from the existing CodeGraph and full-file index before editing.
- Preserve real runtime data and distinguish source, test, report, cache and backup artifacts.
- Use a plan, preflight checks, tests, production gates and independent verification before claiming a result.
- Treat the trading system as a coupled evolution, strategy, risk, data and execution system; avoid unverified bulk rewrites.
- Record failures with the exact command, error, hypothesis, evidence and next experiment.
- Use real runtime feedback for optimization and retain only changes that improve protected metrics.

## Privacy boundary

Raw session transcripts, user profile data, local caches, credentials and built-in TRAE assets are not copied into this public repository. Their presence and classification are recorded in `manifests/four_sources_audit.json`.
