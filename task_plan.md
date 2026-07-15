# Codex Memory Platform Plan

## Goal

Build a Codex-owned, MCP-accessible Obsidian memory platform with scoped retrieval, lifecycle promotion, evaluation, and verified automation.

## Acceptance Criteria

- [x] Defined
- [x] Implemented core runtime and MCP server
- [x] Verified CLI, unit tests, golden retrieval, backup, and live MCP handshake
- [x] Integrate native MCP after Codex app restart and complete recovery drill
- [x] Reload the MCP process after CPU semantic implementation and verify the live tool reports the semantic provider

## Reliability Evolution: CPU Resource Isolation

- [x] Record the observed concurrent ONNX allocation failure and separate it from host-client crash reports
- [x] Serialize semantic model operations across threads and processes
- [x] Verify concurrent requests, MCP errors, and live platform health after process reload
- [x] Promote the evidence-backed workflow and sync the daily memory
- [x] Isolate ONNX in a supervised worker and prove repeated lifecycle requests
- [x] Prove one real worker JSONL request returns a 1024-dimensional BGE-M3 vector
- [x] Prove worker operation inside the semantic lock and repeated restart/recovery without process termination
- [x] Reload the live Codex MCP only after the lifecycle gate passes

## Host `/wham` Polling Root Cause

- [x] Confirmed `goals=false` is unrelated to the host `current tasks` query
- [x] Located the unconditional query in the installed Codex webview bundle
- [x] Added a reversible, version-aware patch tool with fingerprint verification
- [blocked] Apply patch to Microsoft Store `app.asar`: WindowsApps ACL returns `EPERM`
- [ ] Replace the vendor build through a supported elevated app update/deployment path, then verify zero `/wham/tasks/list` requests

## Automatic Memory Orchestration

- [x] Add one idempotent orchestrator for sync, curate, quality, evaluation, gate, backup, and reporting
- [x] Run the orchestrator end to end and verify every stage exit code
- [x] Attach the existing 01:00 Windows task to the orchestrator and verify the next run metadata
- [x] Record the workflow as a reusable Obsidian decision

## Encoding and Governance Hardening

- [x] Normalize the damaged daily note to strict UTF-8 without deleting its content
- [x] Make automation and sync reads replacement-safe and all writes UTF-8 without BOM
- [x] Align the capability gate with the canonical orchestrator action
- [x] Verify the scheduled task itself returns `LastTaskResult=0` and writes a `PASS` report

## Continuous Distilled Memory Closure

- [x] Add an append-only redacted provenance journal for changed project artifacts
- [x] Add a structured distilled-memory inbox with deterministic event ids and deduplication
- [x] Add a lightweight 15-minute watchdog for incremental sync, health checks, and failure alerts
- [x] Extend the 01:00 run to back up and verify the provenance journal with the Native Memory database
- [x] Verify watchdog and daily task entry points independently, including recovery after a forced failed run
- [x] Promote the lossless-distilled memory workflow to Obsidian

## Hermes Writer-Race Hardening

- [x] Replace the minute-based auto-restart watchdog with a read-only health guard
- [x] Reconcile the 455 Chroma orphan IDs with a redacted snapshot and verified backup
- [x] Add bounded incremental BM25 reconciliation for count-only mismatches
- [x] Make Hermes memory-service writes atomic and fail closed on unreadable stores
- [x] Fix the undefined recovery timestamp in `index_delta.py`
- [blocked] Activate the patched `semantic_hook.py` inside the already-running Hermes indexer; requires a normal Hermes service restart or computer reboot, which is intentionally not forced by Codex

## Codex Primary Boundary

- [x] Remove Hermes health and reconcile from the Codex primary completion path
- [x] Keep Hermes health guard and incremental reconcile as an independent sidecar
- [x] Verify full Codex Run passes with Native status, golden evaluation, Vault quality, capability gate, and backups while Hermes is not a required dependency

## Lifecycle And Task-End Closure

- [x] Add SQLite lifecycle fields and migrations for active, superseded, conflicted, expired, importance, validity, provenance, and project scope
- [x] Implement supersede, conflict, resolve, merge, expire, lifecycle explanation, and invariant audit operations in CLI and MCP
- [x] Add regression coverage for lifecycle transitions, merge preservation, MCP tool discovery, and audit invariants
- [x] Add structured `codex_task_end.ps1` intake with task metadata, artifact SHA-256 hashes, deterministic event ids, and redacted summaries
- [x] Wire real Codex `turn-ended` notify configuration through a wrapper that preserves the existing notifier and drains the memory event
- [x] Verify explicit task-end capture, inbox drain, Obsidian append, Native replay idempotency, and lifecycle audit
- [x] Document host notifications without metadata as limited-context events; full context requires the explicit task-end metadata entrypoint

## Automatic Rollout-Derived Task Memory

- [x] Read-only scan of Codex state SQLite and rollout JSONL for terminal task events
- [x] Produce bounded, redacted, ASCII-safe packets with deterministic event ids
- [x] Add idempotent state tracking and automatic inbox drain into Obsidian and Native Memory
- [x] Integrate capture into the 15-minute watchdog
- [x] Register and verify the independent 2-minute automatic task capture schedule
- [x] Add the automatic capture script and schedule to the capability gate

## Frontier Upgrade

- [x] Define and run the isolated multi-dimensional memory evaluation harness
- [x] Expand production retrieval evaluation beyond four golden cases
- [x] Reduce Vault quality warnings to zero or classify every remaining warning with an owner and disposition
- [x] Add Native MCP singleton/process supervision and worker exit-state normalization
- [x] Add verified restore drills and retention/expiration policy checks
- [x] Re-run all gates and publish a scored frontier report

## Current User-Scoped Backup Closure

- [x] Add a DPAPI backup protector bound to the current Windows user, with no secrets written to disk or memory notes
- [x] Add an independent verifier that decrypts into an isolated temporary directory and validates SHA-256, SQLite integrity, counts, and tamper rejection
- [x] Integrate encrypted backup and verification into the canonical automatic Run without making Hermes a Codex completion dependency
- [x] Execute fresh tests, the frontier evaluation, Vault quality, capability gate, evolution audit, full Run, and Watchdog
- [x] Sync a redacted evidence summary and residual-risk statement to Obsidian

### Verification Boundary

- Project tests: `24 passed`; SQLite probe compiled successfully.
- Full Run: `PASS`, 16 stages; final Native 477/477, SQLite integrity `ok`, Golden `8/8`, Frontier `PASS`, Vault `errors=0 warnings=0`, final DPAPI decrypt verification `verified=true` for 477/477.
- Watchdog: `PASS`; automatic task capture scanned 12 historical tasks with no failures.
- Windows tasks: automatic capture, watchdog, and 01:00 consolidation each report `LastTaskResult=0` and zero missed runs.
- Production gate: `RELEASED`, 3 real checks passed, 5 explicit skips caused by missing snapshots, unspecified cross-environment target, or manual-only branches; no failed check.
- Residual risks: vendor `/wham` polling patch remains blocked by Microsoft Store ACL; DPAPI recovery is intentionally bound to the same Windows user profile. CodeGraph now has a real retrying stdio health probe and passed initialize/tools/status checks.

## Project Production-Gate Environment Closure

- [x] Install and verify the missing local gate dependencies (`coverage`, `hypothesis`) in the Codex runtime.
- [x] Add a project-scoped gate adapter and scope file; remove the generic Hermes hardcoding from this project's gate path.
- [x] Add real contract, black-box, branch, mutation, snapshot, fuzz, dependency/resource, and cross-venv checks.
- [x] Stabilize CodeGraph MCP lifecycle with a health probe, singleton guard, and recovery path; verify status and context twice.
- [x] Run the project gate with zero `EXPLICIT_SKIP`, then rerun tests, automation, capability gate, evolution audit, and sync the evidence.
- [x] Fix the UTF-8 no-BOM task-end inbox parsing boundary and verify both the original failing packet and a new Chinese packet end to end.

## Search-First Capability Routing

- [x] Audit installed skills, MCP services, verified lifecycle hook, and scheduled automation instead of treating their presence as proof of use.
- [x] Align the resident Native Memory resource profile with the verified E3-1230v3 policy.
- [x] Add a search-first router that selects required skills and MCPs from task intent and records the required research decision.
- [x] Add a daily capability-routing audit to the canonical automation; failure blocks the Run instead of silently degrading.
- [x] Integrate the capability router into `D:\hermes\codex_preflight.ps1` so every command-bearing preflight records and validates its route automatically.
- [x] Verify router modes, daily audit, project tests, production gate, and a fresh canonical Run.

### Routing Verification

- Router feature mode: `PASS`; required search-first, iterative retrieval, deployment/verification skills, and CodeGraph/full-file/native-memory MCPs.
- Router regression mode: `PASS`; required systematic debugging, TDD, verification, and search-first routes.
- Daily capability audit: `PASS`; required skills and MCPs found, 4 scheduled tasks Ready with `LastResult=0`, verified turn-ended hook, and the 2-thread/128-token resource profile.
- Canonical Run: `PASS`, 17 stages; project tests `24 passed`; project production gate `PASS`, 8/8 checks, `explicit_skips=0`; Native `478/478`, Vault `errors=0 warnings=0`, and DPAPI restore verified `478/478`.

### Final Closure Evidence

- Project gate: `PASS`, 8/8 checks, `explicit_skips=0`; branch 36.49% against a documented 35% baseline, mutation 6/6, fuzz 100 examples, dependency resources 115/115, cross-environment 16/16 in both venvs.
- Canonical Run: `PASS`, 16 stages; CodeGraph health, project gate, Native status/evaluation, Vault quality, and DPAPI backup verification all passed.
- CodeGraph health: real server `codegraph 0.1.0`, 10 tools discovered, `codegraph_status` tool call passed on the indexed 1002-file WSL project.
- Dependencies pinned in `requirements.txt` and `requirements-gate.txt`; `coverage`, `hypothesis`, `pytest`, `PyYAML`, `numpy`, and `pyflakes` are present and `pip check` is clean.

## OpenClaw and Hermes Adaptation

- [x] Inspect upstream source snapshots and record commits and implementation evidence.
- [x] Implement Codex-owned durable queue, idempotency, restart recovery, bounded retry, and dead-letter state.
- [x] Implement evidence-gated memory/skill/workflow candidate lifecycle with risk and confidence limits.
- [x] Attach a fixed low-risk worker to Windows Task Scheduler and verify a real successful run.
- [x] Publish comparison evidence and the Codex adaptation boundary to Obsidian.
- [ ] Add channel-specific adapters only when credentials, sandbox, and delivery contracts are independently verified.

## GitHub Read-Only Control Plane

- [x] Add an authenticated, fixed-scope repository snapshot through `gh`
- [x] Persist repository metadata, open issues, and open pull requests locally
- [x] Enqueue the GitHub check idempotently in the low-risk worker
- [x] Add CLI and unit coverage for the snapshot and worker routing
- [x] Verify live snapshot against `Yhvoid0101/-` without any write operation
- [x] Verify local tests and remote GitHub Actions quality matrix

### GitHub Verification Evidence

- Live snapshot: `github-check` returned `PASS`; repository `Yhvoid0101/-`; 0 open issues; 0 open pull requests.
- Worker: first invocation succeeded; immediate second invocation was `idle`, confirming time-bucket idempotency.
- Local tests: `33 passed in 3.75s`.
- Scope: read-only `gh api`, `gh issue list`, and `gh pr list`; no comments, merges, pushes, workflow edits, or arbitrary shell payloads.

## Codex Self-Evolution Loop

- [x] Record every autonomous job outcome as a bounded redacted control event
- [x] Aggregate repeated failures into deterministic, evidence-linked candidates
- [x] Run evolution analysis as a fixed low-risk worker job and CLI command
- [x] Automatically verify and promote only low-risk candidates with matching evidence
- [x] Apply promoted bounded retry policy to later matching failures without manual confirmation
- [x] Keep code, credentials, deployment, and external writes outside automatic self-modification
- [x] Add regression coverage for failure observation and candidate generation
- [ ] Run a multi-day soak test proving the loop changes a later routing decision without manual intervention
