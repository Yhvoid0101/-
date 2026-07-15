"""Live MCP handshake probe for the configured Codex native memory server."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SERVER = ROOT / "src" / "codex_memory_platform.py"
env = os.environ.copy()
env["CODEX_MEMORY_VAULT"] = r"D:\hermes\hermes_obsidian_vault"
env["CODEX_MEMORY_STORE"] = r"D:\hermes\codex_memory_store"
requests = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "memory_status", "arguments": {}}},
]
proc = subprocess.Popen([sys.executable, str(SERVER), "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
output, _ = proc.communicate("".join(json.dumps(item) + "\n" for item in requests), timeout=15)
responses = [json.loads(line) for line in output.splitlines() if line.strip()]
tools = {tool["name"] for tool in responses[1]["result"]["tools"]}
status = responses[2]["result"]["structuredContent"]
assert responses[0]["result"]["serverInfo"]["name"] == "codex-native-memory"
assert {"memory_status", "memory_search", "memory_sync", "memory_capture", "memory_promote", "memory_curate", "memory_evaluate"} <= tools
assert status["integrity_check"] == "ok"
assert status["semantic_index_docs"] == status["notes"]
assert status["semantic_provider"] == "onnx-bge-m3-cpu"
assert status["semantic_resource_policy"] == "isolated-worker-serialized-release"
assert status["semantic_session_loaded"] is False
assert status["semantic_worker_active"] is False
print(json.dumps({"handshake": "PASS", "tools": sorted(tools), "status": status}, ensure_ascii=False))
