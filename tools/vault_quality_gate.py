"""Portable vault parsing and reference helpers used by link remediation."""
from __future__ import annotations

import re
from pathlib import Path

import frontmatter

VAULT = Path.cwd()


def parse_note(path: Path) -> tuple[dict, str] | tuple[None, str]:
    try:
        post = frontmatter.load(path)
        return dict(post.metadata), post.content
    except Exception as exc:
        return None, f"frontmatter parse error: {exc}"


def reference_variants(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    raw = value.strip()
    if raw.startswith("[[") and raw.endswith("]]" ):
        raw = raw[2:-2]
    raw = raw.split("|", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if raw.lower().endswith(".md"):
        raw = raw[:-3]
    if not raw:
        return set()
    normalized = re.sub(r"[^a-z0-9]+", "", raw.casefold())
    return {raw.casefold(), normalized} - {""}


def note_reference_keys(path: Path, metadata: dict) -> set[str]:
    keys: set[str] = set()
    try:
        keys.update(reference_variants(path.relative_to(VAULT).as_posix()))
    except ValueError:
        pass
    keys.update(reference_variants(path.stem))
    for field in ("id", "title"):
        keys.update(reference_variants(metadata.get(field)))
    return keys
