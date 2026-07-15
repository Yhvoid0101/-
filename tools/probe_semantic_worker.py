"""Run one real semantic-worker request and preserve stdout/stderr/exit evidence."""
import json
import os
import subprocess
import sys
import importlib.util
from pathlib import Path

root = Path(__file__).parents[1]
server = root / "src" / "codex_memory_platform.py"
env = os.environ.copy()
env.update({
    "CODEX_MEMORY_SEMANTIC": "required",
    "CODEX_MEMORY_SEMANTIC_WORKER": "child",
    "CODEX_MEMORY_CPU_THREADS": "2",
    "CODEX_MEMORY_MAX_LENGTH": "128",
})
request = json.dumps({"texts": ["Codex semantic memory isolation"], "batch_size": 1}, ensure_ascii=False) + "\n"
process = subprocess.Popen(
    [sys.executable, str(server), "semantic-worker"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding="utf-8",
    errors="replace",
    env=env,
)
stdout, stderr = process.communicate(request, timeout=180)
print(json.dumps({"exit_code": process.returncode, "stdout": stdout, "stderr": stderr}, ensure_ascii=False))
if process.returncode != 0 or '"ok": true' not in stdout:
    raise SystemExit(1)

spec = importlib.util.spec_from_file_location("codex_memory_platform", server)
platform = importlib.util.module_from_spec(spec)
spec.loader.exec_module(platform)
platform.SEMANTIC_WORKER_COMMAND = [sys.executable, str(server), "semantic-worker"]
client = platform.SemanticWorkerClient(timeout=180)
try:
    vectors = client.encode(["Codex semantic memory isolation"], batch_size=1)
    print(json.dumps({"client": "PASS", "dimensions": len(vectors[0]), "active": client.active}, ensure_ascii=False))
finally:
    client.close()

locked_client = platform.SemanticWorkerClient(timeout=180)
try:
    with platform.semantic_operation(timeout=10):
        vectors = locked_client.encode(["Codex semantic memory isolation"], batch_size=1)
    print(json.dumps({"locked_client": "PASS", "dimensions": len(vectors[0])}, ensure_ascii=False))
finally:
    locked_client.close()
