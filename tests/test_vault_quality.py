import importlib.util
from pathlib import Path


MODULE = Path(__file__).parents[1] / "tools" / "vault_quality_gate.py"
spec = importlib.util.spec_from_file_location("vault_quality_gate", MODULE)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def test_reference_variants_support_wiki_links_and_paths():
    variants = quality.reference_variants("[[Hermes/Workflow/Codex/Automatic Task Memory Capture#section|workflow]]")
    assert "hermes/workflow/codex/automatic task memory capture" in variants
    assert "hermesworkflowcodexautomatictaskmemorycapture" in variants


def test_archived_note_reference_is_known(tmp_path, monkeypatch):
    monkeypatch.setattr(quality, "VAULT", tmp_path)
    path = tmp_path / "Hermes" / "Archive" / "old-note.md"
    path.parent.mkdir(parents=True)
    metadata = {"id": "old-id", "title": "Old Note"}
    keys = quality.note_reference_keys(path, metadata)
    assert "old-id" in keys
    assert "old note" in keys
