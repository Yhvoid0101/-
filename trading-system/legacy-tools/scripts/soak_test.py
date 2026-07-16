import os
import sys
import json
import time
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()


async def run_soak_test(duration_minutes=10, interval_seconds=30):
    print("=" * 70)
    print(f"Stability Soak Test (Hermes v6.0)")
    print(f"Duration: {duration_minutes} min, Interval: {interval_seconds}s")
    print("=" * 70)

    try:
        import psutil

        has_psutil = True
    except ImportError:
        has_psutil = False
        print("  psutil not available, skipping memory monitoring")

    from hermes_v6.orchestrator import HermesOrchestrator

    orch = HermesOrchestrator(api_key=os.environ["NVAPI_API_KEY"])
    await orch.initialize()
    await orch.start()

    results = []
    process = psutil.Process() if has_psutil else None
    start_time = time.time()
    end_time = start_time + duration_minutes * 60
    iteration = 0

    while time.time() < end_time:
        iteration += 1
        iter_start = time.time()

        mem_before = process.memory_info().rss / 1024 / 1024 if has_psutil else 0

        try:
            task_result = await orch.intelligent_execute("What is the current time?")
            success = task_result.success
            error = None
        except Exception as e:
            success = False
            error = str(e)[:100]

        mem_after = process.memory_info().rss / 1024 / 1024 if has_psutil else 0
        mem_delta = mem_after - mem_before

        result = {
            "iteration": iteration,
            "timestamp": time.strftime("%H:%M:%S"),
            "success": success,
            "error": error,
            "mem_before_mb": round(mem_before, 1),
            "mem_after_mb": round(mem_after, 1),
            "mem_delta_mb": round(mem_delta, 1),
        }
        results.append(result)

        elapsed_iter = time.time() - iter_start
        status = "OK" if success else "FAIL"
        print(f"  [{result['timestamp']}] Iter {iteration}: {status}, " f"mem={mem_after:.1f}MB ({mem_delta:+.1f}MB)")

        if iteration % 5 == 0:
            with open(os.path.join(os.path.dirname(__file__), "..", "soak_test_interim.json"), "w") as f:
                json.dump(results, f, indent=2)

        sleep_time = max(interval_seconds - elapsed_iter, 0)
        await asyncio.sleep(sleep_time)

    await orch.stop()

    total = len(results)
    successes = sum(1 for r in results if r["success"])
    success_rate = successes / total * 100 if total > 0 else 0

    memory_growth_pct = 0
    memory_leak_detected = False
    if has_psutil and total >= 4:
        first_n = min(3, total // 2)
        last_n = min(3, total // 2)
        first_avg = sum(r["mem_after_mb"] for r in results[:first_n]) / first_n
        last_avg = sum(r["mem_after_mb"] for r in results[-last_n:]) / last_n
        if first_avg > 0:
            memory_growth_pct = (last_avg - first_avg) / first_avg * 100
        memory_leak_detected = memory_growth_pct > 10

    print(f"\n{'='*70}")
    print(f"Soak Test Result:")
    print(f"  Iterations: {total}")
    print(f"  Success Rate: {success_rate:.1f}%")
    if has_psutil:
        print(f"  Memory Growth: {memory_growth_pct:.1f}% {'(LEAK WARNING!)' if memory_leak_detected else '(Normal)'}")
    print(f"  Target: success_rate >= 90%, no memory leak")
    print(f"  Status: {'PASS' if success_rate >= 90 and not memory_leak_detected else 'FAIL'}")
    print(f"{'='*70}")

    report = {
        "benchmark": "Stability Soak Test",
        "duration_minutes": duration_minutes,
        "interval_seconds": interval_seconds,
        "total_iterations": total,
        "successes": successes,
        "success_rate": success_rate,
        "memory_growth_pct": memory_growth_pct,
        "memory_leak_detected": memory_leak_detected,
        "passed": success_rate >= 90 and not memory_leak_detected,
        "results": results,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "soak_test_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report


if __name__ == "__main__":
    asyncio.run(run_soak_test(duration_minutes=10, interval_seconds=30))
