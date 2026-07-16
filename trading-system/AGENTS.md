# Trading System Agent Rules

- The canonical runtime source is `/home/lmy/hermes_v6/sandbox_trading` in WSL; `content/` is the versioned GitHub snapshot.
- Before edits, identify the runtime symptom, reproduction command, acceptance threshold, and relevant existing implementation.
- Search existing code, reports, and manifests before creating a new component.
- Make one bounded change at a time and preserve a redacted runtime baseline.
- Run tests and the production gate before claiming completion. A passing unit test alone is not runtime evidence.
- Never commit secrets, credentials, private keys, raw tokens, or unreviewed generated runtime data.
- Record failures with the exact command, error, hypothesis, evidence, and next experiment.
