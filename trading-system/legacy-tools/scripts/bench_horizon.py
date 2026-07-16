import asyncio
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()

LONG_TASKS = [
    "Write a 200-word essay about artificial intelligence development",
    "Analyze the code complexity of a Python factorial function",
    "Create a test project structure with 5 files",
]


async def run_horizon_test():
    print("=" * 70)
    print("Long-Horizon Persistence Test (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.orchestrator import HermesOrchestrator

    orch = HermesOrchestrator(api_key=os.environ["NVAPI_API_KEY"])
    await orch.initialize()
    await orch.start()

    try:
        import psutil

        has_psutil = True
        process = psutil.Process()
    except ImportError:
        has_psutil = False

    results = []

    for i, task in enumerate(LONG_TASKS * 3):
        print(f"  [{i+1}/9] {task[:50]}...")
        mem_before = process.memory_info().rss / 1024 / 1024 if has_psutil else 0
        start = time.time()

        try:
            result = await orch.intelligent_execute(task)
            success = result.success
            error = None
        except Exception as e:
            success = False
            error = str(e)[:100]

        elapsed = time.time() - start
        mem_after = process.memory_info().rss / 1024 / 1024 if has_psutil else 0

        results.append(
            {
                "task": task,
                "success": success,
                "error": error,
                "elapsed_sec": elapsed,
                "mem_delta_mb": mem_after - mem_before,
            }
        )
        print(f"    {'OK' if success else 'FAIL'} {elapsed:.1f}s mem={mem_after:.1f}MB")

    await orch.stop()

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    consecutive_failures = 0
    max_consecutive = 0
    for r in results:
        if not r["success"]:
            consecutive_failures += 1
            max_consecutive = max(max_consecutive, consecutive_failures)
        else:
            consecutive_failures = 0

    success_rate = success_count / total * 100
    systemic_crash = max_consecutive >= 3
    passed = success_rate >= 80 and not systemic_crash

    print(f"\n  Horizon: {success_count}/{total} ({success_rate:.1f}%)")
    print(f"  Max consecutive failures: {max_consecutive}")
    print(f"  Systemic crash: {systemic_crash}")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    report = {
        "total_tasks": total,
        "success_count": success_count,
        "success_rate": success_rate,
        "max_consecutive_failures": max_consecutive,
        "systemic_crash_detected": systemic_crash,
        "passed": passed,
        "results": results,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "horizon_bench_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    asyncio.run(run_horizon_test())
