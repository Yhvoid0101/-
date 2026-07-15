"""Preserve unresolved legacy links while keeping active related links valid."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter


QUALITY = Path(__file__).with_name("vault_quality_gate.py")
spec = importlib.util.spec_from_file_location("vault_quality_gate", QUALITY)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def build_reference_index(vault: Path) -> set[str]:
    quality.VAULT = vault
    keys: set[str] = set()
    for path in vault.rglob("*.md"):
        if ".obsidian" in path.parts:
            continue
        metadata, _ = quality.parse_note(path)
        if metadata:
            keys.update(quality.note_reference_keys(path, metadata))
    return keys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remediate(vault: Path, backup_root: Path) -> dict[str, Any]:
    known = build_reference_index(vault)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / timestamp
    changes: list[dict[str, Any]] = []
    for path in vault.rglob("*.md"):
        if ".obsidian" in path.parts or "Archive" in path.parts:
            continue
        post = frontmatter.load(path)
        related = post.metadata.get("related", []) or []
        if not isinstance(related, list):
            related = [related]
        resolved = []
        unresolved = []
        for item in related:
            if not isinstance(item, str) or quality.reference_variants(item) & known:
                resolved.append(item)
            else:
                unresolved.append(item)
        if not unresolved:
            continue
        backup_path = backup_dir / path.relative_to(vault)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        legacy = post.metadata.get("related_legacy", []) or []
        if not isinstance(legacy, list):
            legacy = [legacy]
        post.metadata["related"] = resolved
        post.metadata["related_legacy"] = sorted(set(str(item) for item in [*legacy, *unresolved]))
        path.write_text(frontmatter.dumps(post), encoding="utf-8")
        changes.append({
            "path": str(path.relative_to(vault)),
            "moved": unresolved,
            "backup": str(backup_path),
            "beforeSha256": sha256(backup_path),
            "afterSha256": sha256(path),
        })
    log = backup_root / "vault-link-remediation.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": timestamp, "changes": changes}, ensure_ascii=False) + "\n")
    return {"status": "PASS", "changed_files": len(changes), "moved_links": sum(len(item["moved"]) for item in changes), "backup": str(backup_dir), "journal": str(log), "changes": changes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", type=Path, default=Path(r"D:\hermes\hermes_obsidian_vault"))
    parser.add_argument("--backup-root", type=Path, default=Path(r"D:\hermes\memory_store\codex\vault_link_remediation"))
    args = parser.parse_args()
    result = remediate(args.vault, args.backup_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
