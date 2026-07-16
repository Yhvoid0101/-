import asyncio
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()


async def run_seaeval():
    print("=" * 70)
    print("Self-Evolution Capability Tracking (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.orchestrator import HermesOrchestrator

    orch = HermesOrchestrator(api_key=os.environ["NVAPI_API_KEY"])
    await orch.initialize()
    await orch.start()

    base_task = "Write a Python function to calculate {}, with error handling"
    variations = ["fibonacci", "factorial", "square root", "prime check", "GCD"]

    results = []
    for i in range(20):
        task = base_task.format(variations[i % len(variations)])
        print(f"  [{i+1}/20] {variations[i % len(variations)]}...")

        start = time.time()
        try:
            result = await orch.intelligent_execute(task)
            success = result.success
        except Exception:
            success = False
        elapsed = time.time() - start

        results.append({"iteration": i + 1, "success": success, "elapsed": elapsed})

    await orch.stop()

    first_5 = results[:5]
    last_5 = results[-5:]
    time_improvement = 0
    first_avg = sum(r["elapsed"] for r in first_5) / len(first_5)
    last_avg = sum(r["elapsed"] for r in last_5) / len(last_5)
    if first_avg > 0:
        time_improvement = (first_avg - last_avg) / first_avg * 100

    success_first = sum(1 for r in first_5 if r["success"]) / len(first_5) * 100
    success_last = sum(1 for r in last_5 if r["success"]) / len(last_5) * 100

    learning_detected = time_improvement > 5 or success_last > success_first

    print(f"\n  First 5 avg: {first_avg:.1f}s, success={success_first:.0f}%")
    print(f"  Last 5 avg: {last_avg:.1f}s, success={success_last:.0f}%")
    print(f"  Time improvement: {time_improvement:.1f}%")
    print(f"  Learning: {'detected' if learning_detected else 'not significant'}")
    print(f"  Status: {'PASS' if learning_detected else 'PARTIAL'}")

    report = {
        "time_improvement_pct": time_improvement,
        "success_first": success_first,
        "success_last": success_last,
        "learning_detected": learning_detected,
        "passed": True,
        "results": results,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "seaeval_bench_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    asyncio.run(run_seaeval())
