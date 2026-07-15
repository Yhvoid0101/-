"""Deterministic, isolated evaluation for the Codex memory platform."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


SOURCE = Path(__file__).parents[1] / "src" / "codex_memory_platform.py"


def load_platform():
    os.environ["CODEX_MEMORY_SEMANTIC"] = "disabled"
    spec = importlib.util.spec_from_file_location("frontier_eval_platform", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_note(vault: Path, relative: str, title: str, body: str, scope: str = "global") -> None:
    path = vault / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"id: eval-{relative.replace('/', '-').replace('\\', '-').replace('.md', '')}\n"
        f"title: {title}\n"
        "type: knowledge\n"
        "created: 2026-07-15T00:00:00+00:00\n"
        "updated: 2026-07-15T00:00:00+00:00\n"
        "source: codex\n"
        "tags: [eval]\n"
        "status: active\n"
        "category: Knowledge\n"
        f"scope: {scope}\n"
        "---\n\n"
        f"# {title}\n\n{body}\n",
        encoding="utf-8",
    )


def build_fixture(vault: Path) -> None:
    topics = [
        ("alpha-api.md", "Alpha API Contract", "REST pagination, error envelopes, and versioned endpoints.", "alpha"),
        ("alpha-auth.md", "Alpha Authentication", "OAuth refresh rotation, session expiry, and secure token handling.", "alpha"),
        ("alpha-tests.md", "Alpha Testing Workflow", "pytest unit tests, integration tests, and coverage gates.", "alpha"),
        ("alpha-deploy.md", "Alpha Deployment", "health checks, rollback, backups, and release verification.", "alpha"),
        ("beta-api.md", "Beta API Contract", "GraphQL schema evolution and resolver compatibility.", "beta"),
        ("beta-auth.md", "Beta Authentication", "service accounts, scopes, and audit events.", "beta"),
        ("beta-tests.md", "Beta Testing Workflow", "property tests, fuzzing, and snapshot verification.", "beta"),
        ("beta-deploy.md", "Beta Deployment", "container rollout, readiness probes, and blue green release.", "beta"),
        ("global-memory.md", "Memory Governance", "redaction, retention, conflict resolution, and provenance.", "global"),
        ("global-recovery.md", "Recovery Protocol", "backup verification, restore drills, and residual risk.", "global"),
    ]
    for relative, title, body, scope in topics:
        write_note(vault, relative, title, body, scope)


def retrieval_metrics(platform: Any) -> dict[str, Any]:
    cases = [
        ("pagination error envelope", "Alpha API Contract"),
        ("OAuth session expiry", "Alpha Authentication"),
        ("pytest coverage integration", "Alpha Testing Workflow"),
        ("rollback health check", "Alpha Deployment"),
        ("GraphQL resolver schema", "Beta API Contract"),
        ("service account audit", "Beta Authentication"),
        ("fuzz property snapshot", "Beta Testing Workflow"),
        ("container readiness rollout", "Beta Deployment"),
    ]
    ranks: list[int] = []
    details: list[dict[str, Any]] = []
    for query, expected in cases:
        results = platform.search(query, limit=5)
        titles = [item["title"] for item in results]
        rank = next((index + 1 for index, title in enumerate(titles) if expected.lower() in title.lower()), 0)
        ranks.append(rank)
        details.append({"query": query, "expected": expected, "rank": rank, "titles": titles})
    hits = sum(rank > 0 for rank in ranks)
    reciprocal = sum((1 / rank) for rank in ranks if rank) / len(ranks)
    alpha_results = platform.search("authentication", limit=10, scope="alpha")
    isolation = all("Beta" not in item["title"] for item in alpha_results)
    return {
        "cases": len(cases),
        "hits_at_5": hits,
        "recall_at_5": round(hits / len(cases), 4),
        "mrr": round(reciprocal, 4),
        "scope_isolation": isolation,
        "details": details,
    }


def lifecycle_metrics(platform: Any) -> dict[str, Any]:
    first = platform.capture("temporary eval observation", scope="eval", event_id="eval-idempotent-1", confidence=0.9)
    replay = platform.capture("temporary eval observation replay", scope="eval", event_id="eval-idempotent-1", confidence=0.9)
    left = platform.capture("eval decision left", scope="eval", event_id="eval-left", confidence=0.9)
    right = platform.capture("eval decision right", scope="eval", event_id="eval-right", confidence=0.9)
    conflict = platform.memory_conflict([left["id"], right["id"]], title="eval conflict")
    resolved = platform.memory_resolve(conflict["conflict_id"], left["id"], "deterministic eval winner")
    audit = platform.memory_audit()
    expiring = platform.capture("temporary expiring eval observation", scope="eval", event_id="eval-expiring-1")
    platform.promote(expiring["id"])
    expiration = platform.memory_expire([expiring["id"]])
    return {
        "idempotent_replay": replay["status"] == "already_captured",
        "conflict_resolved": resolved["status"] == "resolved",
        "audit_ok": audit["ok"],
        "expiration": expiration["count"] == 1,
        "first_id": first["id"],
    }


def run_evaluation(root: Path | None = None) -> dict[str, Any]:
    platform = load_platform()
    owned_temp = None
    if root is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="codex-frontier-eval-", ignore_cleanup_errors=True)
        root = Path(owned_temp.name)
    vault = root / "vault"
    store = root / "store"
    vault.mkdir(parents=True, exist_ok=True)
    build_fixture(vault)
    platform.VAULT = vault
    platform.STORE = store
    platform.DB = store / "codex_memory.db"
    platform.LOCK = store / ".lock"
    platform.sync()
    runs = [retrieval_metrics(platform) for _ in range(3)]
    lifecycle = lifecycle_metrics(platform)
    backup_source = platform.backup()
    backup = platform.verify_backup(backup_source["path"])
    restored_path = root / "restored" / "codex_memory.db"
    restored_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_source["path"], restored_path)
    shutil.copy2(backup_source["manifest"], restored_path.with_name("manifest.json"))
    backup["restore_drill"] = platform.verify_backup(str(restored_path))["verified"]
    stable = all(run == runs[0] for run in runs)
    result = {
        "name": "codex-memory-frontier",
        "status": "PASS" if (
            runs[0]["recall_at_5"] >= 0.9
            and runs[0]["mrr"] >= 0.75
            and runs[0]["scope_isolation"]
            and stable
            and all(lifecycle.values())
            and backup["verified"]
            and backup["restore_drill"]
        ) else "BLOCKED",
        "pass_at_3": stable,
        "retrieval": runs[0],
        "stability_runs": runs,
        "lifecycle": lifecycle,
        "backup": backup,
    }
    if owned_temp:
        platform.close_semantic_worker()
        gc.collect()
        owned_temp.cleanup()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_evaluation()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
