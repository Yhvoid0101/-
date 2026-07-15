# Progress

## 2026-07-14 CPU Resource Isolation

- Observed failure: concurrent `memory_sync` and `memory_search` produced `ONNXRuntime bad allocation`.
- Current hypothesis: model initialization/encoding needs one shared operation gate on the low-memory CPU profile.
- Implementation: add an in-process thread gate plus a cross-process lock around every ONNX encode operation.
- Verification: 7 tests pass, cross-process lock regression passes, live probe reports the bounded-release policy, and capability/evolution/Vault gates pass.
- Residual activation: the Codex host must reload the MCP process before a real `memory_search`/`memory_evaluate` confirms native-session release under the resident desktop process.
- Next: after reload, run live semantic search and golden evaluation, then mark the reliability phase complete.

## 2026-07-15 Host Polling Investigation

- Root cause: installed webview bundle `codex-api-D61kVUkT.js` gates one task hook on `authMethod`, but `open-project-setup-dialog-D_quWjec.js` creates the `current tasks` query with `enabled: true`, `refetchIntervalInBackground: true`, and 15s/60s polling.
- `features.goals=false` is not read by that query. No public config key was found in the installed config schema or current `config.toml`.
- A reversible patch tool was added at `tools/harden_codex_host_wham.js` and its PowerShell wrapper. Verify-only result is `UNPATCHED`; write attempt failed with `EPERM` from the Microsoft Store package ACL, so the host was not modified.
- No new ChatGPT Application Error event occurred in the final 30-minute verification window.
- Residual risk: the host may continue polling and can still reach the known timeout/crash path. This phase is blocked until the same build is replaced or patched through a supported elevated deployment route.

## 2026-07-15 Native Semantic Worker Evolution

- Implemented a supervised JSONL semantic worker boundary in the canonical runtime. The MCP process no longer loads ONNX when using the default `process` mode; worker EOF, timeout, and error are captured and the FTS path remains available.
- Unit verification: `9 passed`; live MCP handshake passed with `semantic_resource_policy=isolated-worker-serialized-release`.
- Real isolated worker probe passed once: BGE-M3 returned a 1024-dimensional vector with exit code 0 and no stderr.
- Real CLI search and a second worker lifecycle probe did not pass; the database recorded `semantic worker exited before response` and the second probe returned no usable exit output. The worker change is therefore not complete and must not be reported as fully activated.
- Resource evidence: one old Native Memory server had a 477MB working set and was stopped as a targeted stale duplicate; Hermes index service was not touched. The remaining lifecycle failure needs a new hypothesis around repeated worker startup/teardown and lock interaction.

## 2026-07-15 Restart Recovery Complete

- After Codex restart, the live MCP loaded the isolated worker implementation.
- Final live status: `semantic_health=ready`, `semantic_worker_active=true`, `semantic_worker_restarts=0`, `semantic_session_loaded=false`, `last_semantic_error=''`.
- Two sequential live semantic searches reused the same worker; golden evaluation passed `4/4`.
- Native Memory database remains `440/440` notes and semantic vectors with `integrity_check=ok`. Historical `semantic_failures=5` remains as audit history; it is not an active failure.

## 2026-07-15 Automatic Orchestration and Encoding Hardening

- The first real `Run` reached all memory stages but was correctly blocked by one invalid UTF-8 sequence in `daily/Codex/2026-07-15.md`; the note was repaired with replacement decoding and rewritten as UTF-8 without BOM.
- `codex_memory_automation.ps1` and `codex_memory_sync.ps1` now read with explicit UTF-8 replacement semantics and write through .NET UTF-8 without BOM. Existing daily notes are normalized before appends.
- Manual end-to-end run passed in 24.66 seconds: sync, curate, incremental native sync, status, golden evaluation `4/4`, Vault quality `errors=0`, capability gate, backup, and backup verification all returned exit code `0`.
- The first scheduled-task launch exposed a stale capability-gate rule expecting the retired maintenance script. The gate now requires the canonical orchestrator action and still rejects unknown actions.
- Independent scheduled-task rerun passed: `LastTaskResult=0`, `NumberOfMissedRuns=0`, and `automation_last.json.status=PASS`; the daily trigger remains 01:00.
- Final scheduled-task rerun after the canonical-root sync fix also passed with `LastTaskResult=0`; the latest report contains 449 notes, 4/4 golden evaluation, Vault quality `errors=0`, verified backup, and capability gate PASS.
- The workflow observation was captured and promoted to `Hermes/Memory/Codex/Promoted/2026-07-15-5a329325.md`; Native Memory now reports 449 notes, 449 FTS documents, and 449 semantic vectors.
- Residual risk: Vault has 129 historical unresolved-link warnings; they are non-blocking and were not rewritten. The Codex host `/wham` polling issue remains a separate vendor-build blocker.

## 2026-07-15 Continuous Distilled Memory Closure Checkpoint

- Objective: retain redacted, structured source packets with hashes while presenting concise Obsidian summaries; run incremental health checks throughout the day and full consolidation at 01:00.
- Acceptance criteria: journal writes are append-only and deduplicated by source path plus hash; inbox events are redacted and durable; watchdog is independently scheduled and observable; daily backup includes both Native Memory DB and journal; failures produce a state file and alert without deleting source packets.
- Current hypothesis: the missing 24-hour behavior is an orchestration gap, not a retrieval-model gap. A lightweight scheduled watchdog plus an append-only provenance layer can close it within the current Windows/Codex constraints without loading BGE-M3 every 15 minutes.
- Completed before mutation: required Obsidian memory read, Native Memory status/search, existing task verification, and CodeGraph limitation preserved from prior evidence.
- Next action: implement journal and inbox primitives first, then add Watch mode and a separate scheduled task.

## 2026-07-15 Continuous Distilled Memory Closure Completed

- Provenance journal is live at `D:\hermes\memory_store\codex\journal\project-artifacts.jsonl`; it stores redacted full artifact content, source paths, hashes, and deterministic artifact event ids. Initial backfill added 9 events.
- Distilled inbox is live at `D:\hermes\memory_store\codex\inbox`; event ids are content-derived and Native capture accepts them idempotently. A replay test returned `already_captured` and the queue now reports `queued=0`.
- `codex_memory_watch.ps1` is scheduled every 15 minutes with a shared lock, 01:00 guard window, incremental sync, curate, Hermes reconcile, Native health, Vault quality, and alert state. Task entry test passed with `LastTaskResult=0` and no missed runs.
- Existing `Codex Retrieval Rebuild Watchdog` was changed from repeated `start_service.vbs` launches to a read-only Hermes health guard; the legacy script now delegates to that guard.
- Hermes recovery: an empty `memory.json` was restored only from the validated backup `pre_rebuild_20260711_004506`; 455 Chroma orphans were snapshot-redacted and removed, restoring 479/479/479 consistency. Later vault updates were reconciled to the current consistent count 554.
- Added bounded `codex_hermes_health_reconcile.ps1`: only count mismatches trigger incremental `backup_restore.py --sync`; corrupt formats remain BLOCKED and source-preserving. Full-run and watchdog both call it before health gates.
- Hardened `memory_service.ps1` with atomic UTF-8 replacement writes and fail-closed unreadable-store handling; fixed `index_delta.py` undefined `ts` in corrupt JSON recovery.
- Verification: Codex tests `10 passed`; Hermes recovery/index tests `12 passed`; full orchestrator `PASS` with 10 stages, golden `4/4`, Vault `errors=0`, Native DB backup verified, provenance journal backup hash verified, Hermes health consistent, capability gate PASS.
- Residual risk: the already-running Hermes index_service process may retain old Python code until its normal restart/reboot; the new read-only guard plus bounded reconcile prevents the old writer from silently blocking the Codex loop. Historical Vault warnings remain 129 and Native historical semantic failures remain 6; current semantic error is empty.

## 2026-07-15 Live Activation Boundary

- Final read-only audit found the old resident Hermes indexer can still rewrite BM25 after a successful reconcile: current observed mismatch was `memory=555, chroma=555, bm25=777` after the last daily event settled.
- The durable fix is implemented in `semantic_hook.py`, `memory_service.ps1`, and `index_delta.py`; the existing live Python process cannot load changed modules without a normal restart.
- `HermesVaultIndexer` is now enabled with a boot trigger and the patched service action. Codex did not force-stop either resident Hermes process.
- Until the next normal reboot/service restart, `codex_hermes_health_reconcile.ps1` and both watchdogs provide bounded protection, but live Hermes consistency is conditional rather than permanently activated.

## 2026-07-15 Codex/Hermes Boundary Correction

- Root coupling found: the Codex primary orchestrator explicitly called Hermes reconcile and passed `-RequireHermesHealthy` to capability gate, so a Hermes-sidecar mismatch blocked an otherwise healthy Codex Native Memory run.
- Corrected architecture: Codex primary now depends only on Obsidian sync, Native Memory status/evaluation, Vault quality, Codex capability checks, and Codex DB/journal backups. Hermes health guard and reconcile remain independent monitoring/repair tasks.
- Verification after decoupling: full Codex Run passed with Native `451/451`, golden `4/4`, Vault `errors=0`, capability gate PASS, Native backup verified, and journal backup hash verified. Hermes is no longer a primary completion dependency.
- Current Hermes health after Codex restart was independently observed `ok` and consistent; any later legacy-writer mismatch is sidecar state and cannot invalidate Codex's dedicated memory loop.

## 2026-07-15 Current-User Encrypted Backup Closure

- Added `D:\hermes\codex_backup_protect.ps1`, which validates the native manifest, packages the Native DB, manifest, and provenance journal, and protects the package with Windows DPAPI `CurrentUser` scope.
- Added `D:\hermes\codex_backup_verify_encrypted.ps1`, which decrypts only into an isolated temporary directory, verifies member hashes and manifest counts, and runs the checked-in `tools\verify_sqlite_backup.py` probe for SQLite integrity.
- The first automation attempt exposed a Windows PowerShell `python -c` quoting failure; the unchanged inline probe was not retried. It was replaced by the file-based probe and the failure was recorded in `findings.md`.
- Real verification: latest encrypted backup returned `status=PASS`, `verified=true`, `scope=CurrentUser`, `integrity_check=ok`, `notes=475`, `fts_docs=475`, `member_count=3`, `journal_verified=true`.
- Tamper verification: changing the encrypted payload was rejected by DPAPI with `CryptographicException: The data is invalid.` The outer harness did not emit its summary, so the harness limitation remains recorded; the verifier itself rejected the tampered copy.
- Full automatic Run passed with 13 stages and the DPAPI verifier as the final stage. Independent Watchdog passed with zero capture failures. Native Memory remains independent of Hermes health.
- Final residual risks: CodeGraph MCP returned `Transport closed`; production gate released with 5 explicit skips from its generic configuration; DPAPI recovery requires the same Windows user profile.

## 2026-07-15 Lifecycle And Task-End Closure

- Implemented SQLite lifecycle migrations for observations and conflicts, preserving the existing database and duplicate-title conflict records.
- Added supersede, conflict, resolve, merge, expire, effective-memory explanation, and invariant audit operations through CLI and MCP. Verification: `12 passed`.
- Added `D:\hermes\codex_task_end.ps1` with structured task metadata, redacted summaries, deterministic event ids, artifact SHA-256 hashes, and provenance hashes. The inbox drain now consumes legacy and v2 events safely.
- Wired `config.toml` `turn-ended` notify through `D:\hermes\codex_turn_notify.ps1`, which preserves the prior notifier and invokes the memory bridge. Explicit task-end e2e returned `captured`, drain returned `processed=1`, Obsidian contained the event, and Native replay returned `already_captured`.
- Native lifecycle audit returned `ok=true`; task-end packets remain `captured` until the existing high-confidence curator promotes them, so raw task events are not silently treated as durable decisions.
- Notify verification: `CODEX_TASK_EVENT_JSON` produced a complete `metadataSource=environment-json` event and drained successfully. Direct positional JSON appended to the PowerShell wrapper was not accepted reliably by this installation and is intentionally not treated as verified.
- Residual risk: without `CODEX_TASK_EVENT_JSON` or the explicit task-end parameters, the fallback event is intentionally limited-context; existing legacy notifier output also requires the host's own payload when run manually.

## 2026-07-15 Semantic Resource Recovery And Final Gate

- Root cause: the only pending embedding was the 98 KB current daily note; a stale Codex Native semantic worker reached approximately 1.8 GB working set and caused ONNX bad allocation.
- Fix: added `bounded_semantic_text` with head/tail preservation before BGE-M3 encoding, targeted only the confirmed Codex Native worker for cleanup, and left Hermes processes untouched.
- Verification after the changed hypothesis: Native sync returned `semantic_error=null` and `452/452`; platform tests `13 passed`; serial automation Check `PASS`; full automation `PASS` with 10 stages, `453/453`, Golden `4/4`, Vault `errors=0`, capability gate PASS, verified Native backup, and verified journal hash.
- Final lifecycle MCP probe returned handshake `ok`, all 7 lifecycle tools discovered, and `memory_audit=ok`. Evolution audit returned `PASS`. The verified Curator observation was promoted to `Hermes/Memory/Codex/Promoted/2026-07-15-67eab8d7.md`.
- Historical semantic failure count is 11 for audit continuity; current `last_semantic_error` is empty and `semantic_health=ready`.

## 2026-07-15 Automatic Rollout-Derived Task Memory Completed

- Added `tools/codex_auto_memory_capture.py` and `D:\hermes\codex_auto_memory_capture.ps1`.
- Collector reads Codex state SQLite plus rollout JSONL, skips active tasks, redacts secrets, bounds summaries, and writes deterministic ASCII-safe inbox packets.
- Integrated the collector into `codex_memory_watch.ps1` and registered `Codex Automatic Task Memory Capture` every 2 minutes with catch-up and single-instance settings.
- Hardened inbox drain to preserve packet bytes and fixed promoted-note naming to use the full sanitized observation id.
- Repaired 11 historical task memories from the collided promoted path into 11 unique Obsidian files.
- Verification: collector 4 passed; platform 14 passed; full suite 18 passed; Watch PASS; Evolution Audit PASS; Capability Gate PASS; full Run PASS with Golden 4/4, Vault errors=0, verified Native backup, and verified journal hash.

## 2026-07-15 Duplicate Title Conflicts Resolved

- Root cause: duplicate-title detection was correct, but promoted task notes and historical R8/Audit notes had non-unique display titles.
- Fix: promoted titles now use the full sanitized observation id; historical conflict notes were migrated to unique time/id suffixes without deleting content.
- Verification: platform tests 14 passed; full suite 18 passed; Native sync reports `conflicts=0`, `open_conflicts=0`, notes `467`, embeddings `467`, integrity `ok`; full Run PASS with Golden 4/4 and verified backup.

## 2026-07-15 Semantic Resource Recovery Completed

- Found two resident Codex Native MCP trees after restart and a 1.89 GB semantic worker under the newer tree.
- Stopped only the stale Codex tree and semantic worker; Hermes processes were not touched.
- Serial bounded sync restored `semantic_health=ready` and cleared `last_semantic_error`.
- Automatic Run and Watch now enforce the verified E3-1230v3 profile: 2 CPU threads and max sequence length 128.

## 2026-07-15 Final Evidence After Task-End Promotion

- Explicit task-end metadata was captured with `metadataSource=explicit-parameters`, `artifactCount=3`, and `drainExitCode=0`; the Watchdog promoted it to a redacted Obsidian lesson.
- Final Full Run passed with 16 stages after that promotion: Native 477 notes, 477 FTS documents, 477 semantic vectors, 45 observations, zero open conflicts, SQLite integrity `ok`, Golden 8/8, Frontier `PASS`, Vault `errors=0 warnings=0`.
- Final encrypted backup directory: `D:\hermes\codex_memory_store\backups\20260715_123548`; DPAPI `CurrentUser` restore verified 477/477 with three members and journal verification.
- Final Check, capability gate, evolution audit, and Watchdog all passed. Remaining risks are unchanged and explicitly documented.

## 2026-07-15 Production-Gate Environment and CodeGraph Recovery Closure

- Added pinned runtime and gate dependencies: `requirements.txt`, `requirements-gate.txt`, `coverage`, `hypothesis`, `pytest`, `PyYAML`, `numpy`, and `pyflakes`; the canonical venv reports clean `pip check`.
- Added `tools/production_gate_adapter.py`, `.production-gate/scope.json`, two executable golden snapshots, and `tools/run_codex_production_gate.ps1`. The adapter runs all eight checks against Codex Memory rather than the generic Hermes trading hardcodes.
- Final project gate: `PASS`, `explicit_skips=0`; contract, blackbox, branch 36.49%/35 baseline, mutation 6/6, snapshot 2/2, fuzz 100 examples, dependency/resource 115/115, and cross-environment 16/16 in both isolated venvs.
- Added `tools/codegraph_mcp_health.py` and `tools/run_codegraph_health.ps1`. The probe uses a singleton lock, bounded retries, real stdio initialize, initialized notification, tools/list, and codegraph_status calls; it never kills Codex-managed or Hermes processes.
- Integrated CodeGraph health into both canonical Run and Watch. Final Run passed with 16 stages; CodeGraph returned server 0.1.0, 10 tools, and status call PASS. Final Watch also passed.
- Root causes fixed: generic gate was hardcoded to Hermes; missing gate dependencies caused isolated test failures; crossenv accepted identical failures and zero-test runs; adapter parsed nested JSON fragments; CodeGraph lacked a lifecycle health/retry boundary; unused `shutil` and scanner stdlib/package aliases were corrected.
- Final Native status remains ready with SQLite integrity `ok`, 472 notes/FTS/semantic docs, zero open conflicts, and empty current semantic error. Remaining host vendor `/wham` ACL limitation and same-user DPAPI recovery boundary are documented, not hidden.

## 2026-07-15 UTF-8 Task-End Drain Recovery

- Observed failure: `codex_task_end.ps1` queued a valid UTF-8 JSON packet, but inbox drain returned `drainExitCode=1` with `Invalid object passed in, ':' or '}' expected. (629)`.
- Root cause: `codex_memory_drain_inbox.ps1` used Windows PowerShell default-encoding `Get-Content -Raw` for a UTF-8 no-BOM packet; explicit UTF-8 decoding and Python `json.loads` both proved the packet itself was valid.
- Fix: changed inbox JSON parsing to the existing `Read-Utf8Replacement` helper, preserving byte-safe UTF-8 handling across the PowerShell/Python boundary.
- Regression evidence: the original queued packet processed with `processed=1, failed=[]`; a new Chinese task-end packet returned `captured` with `drainExitCode=0`; project tests `24 passed`; production gate `PASS` with 8/8 checks and `explicit_skips=0`.

## 2026-07-15 Search-First Capability Routing

- Observed gap: skills, MCPs, tasks, and the `turn-ended` hook were installed, but no single executable route proved that required capabilities were selected for a task or still healthy.
- Decision: use a bounded router and daily audit rather than an unbounded autonomous coding loop. The loop may inspect, sync, curate, and verify; code changes, commits, deployments, and external actions remain explicit task work with production gates.
- Implementation in progress: align the resident Native Memory profile to the verified two-thread CPU limit, route feature/integration work through search-first, and block the daily canonical Run if required skills, MCPs, hook, router, or scheduled tasks are missing.

## 2026-07-15 Search-First Capability Routing Verified

- Resident MCP configuration now matches the verified E3-1230v3 profile: `CODEX_MEMORY_CPU_THREADS=2` and `CODEX_MEMORY_MAX_LENGTH=128`.
- Added `D:\hermes\codex_task_router.ps1`: feature/integration work is routed through search-first with a durable research record; bug, security, deployment, and test intents select their matching skills and MCPs. The router reports `BLOCKED` when a required capability is absent.
- Added `D:\hermes\codex_capability_route_audit.ps1` and inserted it into canonical Run immediately after CodeGraph health. A missing core skill/MCP/task, broken turn-ended hook, absent router, or unstable resource profile now blocks daily full automation.
- Verification: two distinct router intents passed with different routes; audit passed with six required skills, six required MCPs, and four Ready scheduled tasks; project tests `24 passed`; fresh canonical Run passed all 17 stages, including project production gate 8/8 with no explicit skip, Native `478/478`, Vault `errors=0 warnings=0`, and DPAPI restore verification `478/478`.
- Latest full Run completed in `104.94s`, improving on the earlier `116.30s` run while retaining the same 17-stage verification chain.
- Boundary: only the documented `turn-ended` lifecycle hook is configured because it is proven in this installation. The pre-edit constraint is enforced by the router plus preflight and project guidance; no undocumented host hook syntax was added. The background loop monitors and verifies, but does not autonomously commit, deploy, or alter external systems.
- Automatic preflight integration is now active: any command-bearing preflight invokes the router with `-Record`, returns the route in its JSON result, and blocks when the required skill/MCP route cannot be satisfied.
