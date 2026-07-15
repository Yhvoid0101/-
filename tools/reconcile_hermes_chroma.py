from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import chromadb


MEMORY_PATH = Path(r"D:\hermes\memory_store\memory.json")
PERSIST_DIR = Path(r"D:\hermes\memory_store\chroma")
SNAPSHOT_DIR = Path(r"D:\hermes\memory_store\hermes_chroma_orphans")
COLLECTION_NAME = "hermes_memory"


def redact(value: str) -> str:
    value = re.sub(r"(?i)(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})", "[REDACTED]", value)
    return re.sub(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s]+", r"\1=[REDACTED]", value)


def run(apply: bool) -> dict:
    memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8-sig"))
    expected = {str(entry["id"]) for entry in memory.get("entries", [])}
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    data = collection.get()
    ids = [str(item) for item in data.get("ids", [])]
    orphan_ids = sorted(set(ids) - expected)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot = SNAPSHOT_DIR / f"orphans_{timestamp}.jsonl"
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    index = {item: pos for pos, item in enumerate(ids)}
    with snapshot.open("w", encoding="utf-8", newline="\n") as handle:
        for orphan_id in orphan_ids:
            pos = index[orphan_id]
            handle.write(json.dumps({
                "id": orphan_id,
                "document": redact(str(documents[pos])) if pos < len(documents) else "",
                "metadata": metadatas[pos] if pos < len(metadatas) else {},
                "redaction": "applied",
            }, ensure_ascii=False) + "\n")
    before = collection.count()
    after = before
    if apply and orphan_ids:
        collection.delete(ids=orphan_ids)
        after = collection.count()
    return {
        "memory_ids": len(expected),
        "chroma_before": before,
        "orphans": len(orphan_ids),
        "snapshot": str(snapshot),
        "applied": apply,
        "chroma_after": after,
        "reconciled": after == len(expected) if apply else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.apply), ensure_ascii=False))
