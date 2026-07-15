"""Project-scoped production gate for Codex Memory Platform.

Each check executes a real project entry point or a real verification runner.
The adapter exists because the generic gate runner historically targeted an
unrelated WSL trading project and could not prove this project's behavior.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PYTHON = Path(sys.executable)
SRC = ROOT / "src" / "codex_memory_platform.py"
TEST_FILE = ROOT / "tests" / "test_platform.py"
WORK = ROOT / "work" / "production-gate"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The host is deliberately low-resource. Keep every adapter child bounded.
os.environ["CODEX_MEMORY_CPU_THREADS"] = "2"
os.environ["CODEX_MEMORY_MAX_LENGTH"] = "128"


def _run(args: list[str], timeout: int = 300) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _json_from_output(output: str) -> Any:
    candidates: list[Any] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(output):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            candidates.append(value)
    if candidates:
        structured = [
            value for value in candidates
            if isinstance(value, dict) and "status" in value
        ]
        return max(structured or candidates, key=lambda value: len(json.dumps(value, default=str)))
    raise ValueError(f"no JSON object in output: {output[-500:]}")


def _pass(name: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS", **details}


def _fail(name: str, reason: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": "FAIL", "reason": reason, **details}


def contract_check() -> dict[str, Any]:
    required = [ROOT / "PROJECT.json", ROOT / "AGENTS.md", ROOT / "task_plan.md", SRC, TEST_FILE]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        return _fail("contract", "required project files are missing", missing=missing)
    try:
        project = json.loads((ROOT / "PROJECT.json").read_text(encoding="utf-8-sig"))
        ast.parse(SRC.read_text(encoding="utf-8"))
        sys.path.insert(0, str(ROOT))
        from src import codex_memory_platform as platform

        status = platform.status()
        audit = platform.memory_audit()
    except Exception as exc:
        return _fail("contract", f"runtime contract probe failed: {type(exc).__name__}: {exc}")
    if project.get("slug") != "codex-memory-platform":
        return _fail("contract", "PROJECT.json slug mismatch")
    if status.get("integrity_check") != "ok" or status.get("open_conflicts") != 0:
        return _fail("contract", "Native Memory status contract failed", status=status)
    if not audit.get("ok"):
        return _fail("contract", "lifecycle contract failed", audit=audit)
    return _pass("contract", notes=status.get("notes"), integrity=status.get("integrity_check"))


def blackbox_check() -> dict[str, Any]:
    probes = {
        "status": [str(PYTHON), str(SRC), "status"],
        "evaluate": [str(PYTHON), str(SRC), "evaluate"],
        "search": [str(PYTHON), str(SRC), "search", "DPAPI", "--limit", "10"],
    }
    outputs: dict[str, Any] = {}
    for name, command in probes.items():
        code, stdout, stderr = _run(command, timeout=180)
        if code != 0:
            return _fail("blackbox", f"{name} exited {code}", stdout=stdout[-1000:], stderr=stderr[-1000:])
        try:
            outputs[name] = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return _fail("blackbox", f"{name} returned invalid JSON: {exc}")
    if outputs["status"].get("semantic_health") != "ready":
        return _fail("blackbox", "status did not report semantic_health=ready", status=outputs["status"])
    if outputs["evaluate"].get("failed") != 0:
        return _fail("blackbox", "golden evaluation failed", evaluation=outputs["evaluate"])
    titles = [item.get("title", "") for item in outputs["search"]]
    if not any("Encrypted Backup" in title for title in titles):
        return _fail("blackbox", "search did not return the durable backup workflow", titles=titles)
    return _pass("blackbox", probes=list(probes), result_count=len(outputs["search"]))


def branch_check() -> dict[str, Any]:
    WORK.mkdir(parents=True, exist_ok=True)
    coverage_json = WORK / "coverage.json"
    for command in (
        [str(PYTHON), "-m", "coverage", "erase"],
        [str(PYTHON), "-m", "coverage", "run", "--branch", "-m", "pytest", "-q"],
        [str(PYTHON), "-m", "coverage", "json", "-o", str(coverage_json)],
    ):
        code, stdout, stderr = _run(command, timeout=300)
        if code != 0:
            return _fail("branch", f"coverage command exited {code}", stdout=stdout[-1500:], stderr=stderr[-1500:])
    report = json.loads(coverage_json.read_text(encoding="utf-8"))
    files = report.get("files", {})
    source_data = files.get(str(SRC).replace("\\", "/")) or files.get(str(SRC))
    if source_data is None:
        source_data = next((data for path, data in files.items() if path.endswith("codex_memory_platform.py")), None)
    if not source_data:
        return _fail("branch", "coverage report does not contain the canonical source")
    summary = source_data.get("summary", {})
    total_branches = int(summary.get("num_branches", 0))
    covered_branches = int(summary.get("covered_branches", 0))
    percent = (covered_branches / total_branches * 100.0) if total_branches else 100.0
    # The canonical module also contains independently black-boxed MCP/ONNX
    # process boundaries. Their runtime behavior is gated separately above;
    # this threshold covers the in-process branch baseline without counting
    # those process-only paths as silently exercised by pytest.
    threshold = 35.0
    if total_branches == 0 or percent < threshold:
        return _fail("branch", "branch coverage below project threshold", percent=percent, threshold=threshold, summary=summary)
    return _pass("branch", branch_percent=round(percent, 2), threshold=threshold, covered=covered_branches, total=total_branches)


def mutation_check() -> dict[str, Any]:
    runner = Path(os.environ.get("CODEX_MUTATION_RUNNER", "C:\\Users\\Administrator.SK-20260514VARK\\.agents\\skills\\production-gate\\mutation_runner.py"))
    command = [str(PYTHON), str(runner), "--target", str(SRC), "--test", str(TEST_FILE), "--function", "bounded_semantic_text"]
    code, stdout, stderr = _run(command, timeout=300)
    if code != 0:
        return _fail("mutation", f"mutation runner exited {code}", stdout=stdout[-2000:], stderr=stderr[-2000:])
    try:
        report = _json_from_output(stdout)
    except ValueError as exc:
        return _fail("mutation", str(exc), stdout=stdout[-2000:])
    if report.get("status") != "PASS" or report.get("survived"):
        return _fail("mutation", "mutants survived or mutation runner failed", report=report)
    return _pass("mutation", killed=report.get("killed"), total=report.get("total_mutants"), kill_rate=report.get("kill_rate"))


def snapshot_check() -> dict[str, Any]:
    runner = Path(os.environ.get("CODEX_SNAPSHOT_RUNNER", "C:\\Users\\Administrator.SK-20260514VARK\\.agents\\skills\\golden-snapshot\\snapshot_runner.py"))
    snapshot_dir = ROOT / ".production-gate" / "snapshots"
    code, stdout, stderr = _run([str(PYTHON), str(runner), "--snapshot-dir", str(snapshot_dir)], timeout=120)
    if code != 0:
        return _fail("snapshot", f"snapshot runner exited {code}", stdout=stdout[-2000:], stderr=stderr[-2000:])
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return _fail("snapshot", f"invalid snapshot JSON: {exc}")
    if report.get("status") != "PASS":
        return _fail("snapshot", "golden snapshot comparison failed", report=report)
    return _pass("snapshot", files=len(list(snapshot_dir.glob("*.json"))), report=report)


def fuzz_check() -> dict[str, Any]:
    try:
        from hypothesis import given, settings, strategies as st
        from src.codex_memory_platform import bounded_semantic_text, extract_links, redact
    except Exception as exc:
        return _fail("fuzz", f"Hypothesis runtime unavailable: {type(exc).__name__}: {exc}")

    examples = {"count": 0}

    @given(text=st.text(max_size=160), limit=st.integers(min_value=1, max_value=160))
    @settings(max_examples=100, deadline=None)
    def check_bounded(text: str, limit: int) -> None:
        examples["count"] += 1
        safe = redact(text)
        result = bounded_semantic_text(text, limit)
        if len(safe) <= limit:
            assert result == safe
        else:
            half = limit // 2
            expected = safe[:half] + "\n[semantic middle truncated]\n" + safe[-half:]
            assert result == expected

    @given(text=st.text(max_size=160))
    @settings(max_examples=100, deadline=None)
    def check_text_helpers(text: str) -> None:
        assert isinstance(redact(text), str)
        assert isinstance(extract_links(text), list)

    try:
        check_bounded()
        check_text_helpers()
    except Exception as exc:
        return _fail("fuzz", f"property test failed: {type(exc).__name__}: {exc!r}", examples=examples["count"])
    return _pass("fuzz", examples=examples["count"])


def dependency_check() -> dict[str, Any]:
    runner = Path(os.environ.get("CODEX_DEPENDENCY_RUNNER", "C:\\Users\\Administrator.SK-20260514VARK\\.agents\\skills\\dependency-lifecycle\\lifecycle_runner.py"))
    code, stdout, stderr = _run([str(PYTHON), str(runner), str(ROOT), str(TEST_FILE)], timeout=300)
    if code != 0:
        try:
            report = _json_from_output(stdout)
        except ValueError:
            report = {"stdout": stdout[-2000:], "stderr": stderr[-2000:]}
        return _fail("dependency", "dependency/resource lifecycle check failed", report=report)
    try:
        report = _json_from_output(stdout)
    except ValueError as exc:
        return _fail("dependency", str(exc), stdout=stdout[-2000:])
    if report.get("status") != "PASS":
        return _fail("dependency", "dependency lifecycle reported failure", report=report)
    return _pass("dependency", summary=report.get("summary"), resources=report.get("resource_tracking"))


def cross_environment_check() -> dict[str, Any]:
    runner_path = Path(os.environ.get("CODEX_CROSSENV_RUNNER", "C:\\Users\\Administrator.SK-20260514VARK\\.agents\\skills\\cross-environment\\crossenv_runner.py"))
    spec = importlib.util.spec_from_file_location("codex_crossenv_runner", runner_path)
    if spec is None or spec.loader is None:
        return _fail("cross_env", "could not load cross-environment runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run(
        TEST_FILE,
        ROOT / "requirements-gate.txt",
        ROOT / "requirements-gate.txt",
        work_dir=WORK / "venvs",
    )
    if result.get("status") != "PASS":
        return _fail("cross_env", "two isolated environments produced different results", report=result)
    comparison = result.get("comparison", {})
    prod = comparison.get("prod", {})
    test = comparison.get("test", {})
    if min(int(prod.get("passed", 0)), int(test.get("passed", 0))) <= 0:
        return _fail("cross_env", "cross-environment run reported zero executed tests", report=result)
    if int(prod.get("failed", 0)) != 0 or int(test.get("failed", 0)) != 0:
        return _fail("cross_env", "cross-environment tests reported failures", report=result)
    return _pass("cross_env", comparison=comparison, prod_venv=result.get("prod_venv"), test_venv=result.get("test_venv"))


CHECKS = {
    "contract": contract_check,
    "blackbox": blackbox_check,
    "branch": branch_check,
    "mutation": mutation_check,
    "snapshot": snapshot_check,
    "fuzz": fuzz_check,
    "dependency": dependency_check,
    "cross_env": cross_environment_check,
}


def main() -> int:
    requested = sys.argv[1:] or list(CHECKS)
    unknown = [name for name in requested if name not in CHECKS]
    if unknown:
        print(json.dumps({"status": "FAIL", "unknown_checks": unknown}))
        return 1
    results = []
    for name in requested:
        try:
            results.append(CHECKS[name]())
        except Exception as exc:
            results.append(_fail(name, f"uncaught adapter error: {type(exc).__name__}: {exc}"))
    failed = [item for item in results if item.get("status") != "PASS"]
    report = {
        "project": "codex-memory-platform",
        "status": "PASS" if not failed else "FAIL",
        "explicit_skips": 0,
        "checks": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
