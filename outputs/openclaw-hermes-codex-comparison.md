# OpenClaw and Hermes Adaptation Evidence

## Source snapshots

- OpenClaw: `openclaw/openclaw`, commit `5bbf30b`, inspected `src/cron`, `src/agents`, `src/auto-reply`, and `docs/gateway/restart-recovery.md`.
- Hermes: `NousResearch/hermes-agent`, commit `d2c81eb`, inspected `cron/jobs.py`, `cron/scheduler.py`, `agent/memory_manager.py`, `agent/learn_prompt.py`, `agent/learning_graph.py`, and `tools/skill_provenance.py`.

## Adopted

| Capability | Upstream evidence | Codex implementation |
|---|---|---|
| Durable queue | OpenClaw persists cron, background tasks, deliveries, and recovery markers in SQLite | `src/codex_autonomy.py` `jobs` table |
| Restart recovery | OpenClaw reclaims stale running work and retries with bounded exponential backoff | `recover_orphans`, `fail`, `max_attempts` |
| Idempotency | OpenClaw uses durable dispatch identifiers and delivery claims | unique `idempotency_key` |
| Low-risk 24/7 loop | Both projects use resident scheduler/heartbeat evidence | Windows task `Codex Autonomous Control Plane` every 10 minutes |
| Evidence-gated learning | Hermes tracks memory/skill provenance and separates review from foreground writes | `evolution_candidates`, `verify_candidate`, risk/confidence gate |
| Single memory boundary | Hermes has a manager; Codex already has Native Memory + Obsidian | adapter stores control state only; Native Memory remains authoritative |

## Deliberate boundaries

- No arbitrary shell payloads are executed from durable jobs.
- No autonomous repository edits, commits, releases, or deployments are enabled.
- High-risk or low-confidence candidates cannot be promoted automatically.
- OpenClaw channel adapters and Hermes external terminal backends require separate credentials and are not silently claimed as installed.

## Verification

- Project tests: `29 passed`.
- Autonomous SQLite status: `integrity_check=ok`, one scheduled worker job succeeded.
- Windows task: `LastTaskResult=0`, next run scheduled.
