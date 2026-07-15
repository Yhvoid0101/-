"""Verify the SQLite payload extracted from a Codex backup."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_sqlite_backup.py DB_PATH")
    db_path = Path(sys.argv[1])
    if not db_path.is_file():
        raise SystemExit(f"database not found: {db_path}")
    db = sqlite3.connect(db_path)
    try:
        result = {
            "integrity_check": db.execute("PRAGMA integrity_check").fetchone()[0],
            "notes": db.execute("SELECT COUNT(*) FROM notes").fetchone()[0],
            "fts_docs": db.execute("SELECT COUNT(*) FROM notes_fts").fetchone()[0],
        }
    finally:
        db.close()
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
