"""Real CodeGraph MCP stdio health probe with bounded retries."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import time
from typing import Any


COMMAND = [
    "wsl.exe",
    "-d",
    "Ubuntu",
    "--",
    "bash",
    "-lc",
    "cd /home/lmy/hermes_v6 && exec /home/lmy/hermes_v6/.codegraph/bundles/linux-x64-0.9.4/bin/codegraph serve --mcp --path /home/lmy/hermes_v6",
]


def _readline(proc: subprocess.Popen[str], timeout: float) -> str:
    output: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            output.put(proc.stdout.readline() if proc.stdout else "")
        except Exception as exc:  # pragma: no cover - process boundary
            output.put(f"__READ_ERROR__:{exc}")

    import threading

    threading.Thread(target=read, daemon=True).start()
    try:
        line = output.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(f"CodeGraph MCP response exceeded {timeout:.1f}s") from exc
    if line.startswith("__READ_ERROR__:"):
        raise RuntimeError(line)
    if not line:
        raise RuntimeError("CodeGraph MCP closed stdout before response")
    return line.strip()


def _request(proc: subprocess.Popen[str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    if proc.stdin is None:
        raise RuntimeError("CodeGraph MCP stdin is unavailable")
    proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
    proc.stdin.flush()
    line = _readline(proc, timeout)
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("CodeGraph MCP returned a non-object response")
    return value


def _attempt(timeout: float) -> dict[str, Any]:
    proc = subprocess.Popen(
        COMMAND,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        initialized = _request(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "codex-codegraph-health", "version": "1.0"},
                },
            },
            timeout,
        )
        if "result" not in initialized or initialized["result"].get("serverInfo", {}).get("name") != "codegraph":
            raise RuntimeError(f"invalid initialize response: {initialized}")
        if proc.stdin is None:
            raise RuntimeError("CodeGraph MCP stdin closed after initialize")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
        proc.stdin.flush()
        listed = _request(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}, timeout)
        tools = listed.get("result", {}).get("tools", [])
        if not isinstance(tools, list) or not tools:
            raise RuntimeError(f"CodeGraph tools/list returned no tools: {listed}")
        status_call = _request(
            proc,
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "codegraph_status", "arguments": {}}},
            timeout,
        )
        if "result" not in status_call or status_call.get("result", {}).get("isError"):
            raise RuntimeError(f"CodeGraph status tool failed: {status_call}")
        return {
            "status": "PASS",
            "server": initialized["result"].get("serverInfo"),
            "tool_count": len(tools),
            "status_call": "PASS",
        }
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
            proc.wait(timeout=3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    failures: list[str] = []
    for attempt in range(1, max(1, args.attempts) + 1):
        try:
            result = _attempt(args.timeout)
            result.update({"attempt": attempt, "attempts": args.attempts})
            print(json.dumps(result, ensure_ascii=True))
            return 0
        except Exception as exc:
            failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < args.attempts:
                time.sleep(min(2.0, 0.25 * attempt))
    print(json.dumps({"status": "FAIL", "attempts": args.attempts, "failures": failures}, ensure_ascii=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
