import importlib.util
from pathlib import Path

import frontmatter


MODULE = Path(__file__).parents[1] / "tools" / "remediate_vault_links.py"
spec = importlib.util.spec_from_file_location("remediate_vault_links", MODULE)
remediator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remediator)


def test_unresolved_links_are_preserved_as_legacy_and_backed_up(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "note.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "---\nid: note-id\ntitle: Note\ntype: knowledge\ncreated: 2026-01-01\nupdated: 2026-01-01\nsource: test\ntags: []\nstatus: active\ncategory: Knowledge\nrelated: [note-id, missing-note]\n---\n\nbody\n",
        encoding="utf-8",
    )
    result = remediator.remediate(vault, tmp_path / "backup")
    assert result["changed_files"] == 1
    updated = frontmatter.load(note)
    assert updated.metadata["related"] == ["note-id"]
    assert updated.metadata["related_legacy"] == ["missing-note"]
    assert len(list((tmp_path / "backup").rglob("note.md"))) == 1
