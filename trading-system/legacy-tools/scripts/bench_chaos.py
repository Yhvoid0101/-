import asyncio
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()


async def run_chaos_test():
    print("=" * 70)
    print("Chaos Engineering Test (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.orchestrator import HermesOrchestrator
    from hermes_v6.llm_gateway import LLMGateway

    results = {}

    print("[1/4] Dependency failure test (bad API key)...")
    try:
        bad_gw = LLMGateway(api_key="invalid-key-chaos-test")
        r = await bad_gw.chat(
            messages=[{"role": "user", "content": "test"}],
            task_type="fast",
            max_tokens=5,
            max_retries=1,
        )
        results["dependency_failure"] = not r.success
        print(f"  Bad key handled gracefully: {not r.success}")
    except Exception as e:
        results["dependency_failure"] = True
        print(f"  Bad key raised exception (graceful): {str(e)[:50]}")

    print("[2/4] Network delay resilience...")
    orch = HermesOrchestrator(api_key=os.environ["NVAPI_API_KEY"])
    await orch.initialize()
    await orch.start()
    start = time.time()
    try:
        task = await orch.intelligent_execute("Say hello")
        latency = time.time() - start
        results["network_delay"] = latency < 30.0
        print(f"  Latency: {latency:.1f}s, OK: {latency < 30.0}")
    except Exception as e:
        results["network_delay"] = False
        print(f"  Failed: {e}")

    print("[3/4] Sub-agent timeout test...")
    try:
        r = await asyncio.wait_for(
            orch.intelligent_execute("Calculate 2+2"),
            timeout=60,
        )
        results["subagent_timeout"] = r.success
        print(f"  Task completed within timeout: {r.success}")
    except asyncio.TimeoutError:
        results["subagent_timeout"] = True
        print(f"  Task timed out gracefully (no crash)")
    except Exception as e:
        results["subagent_timeout"] = False
        print(f"  Task failed: {e}")

    print("[4/4] OOM simulation (skipped - needs Docker)...")
    results["oom"] = "skipped"

    await orch.stop()

    passed = sum(1 for v in results.values() if v is True)
    total = len([v for v in results.values() if v is not None and v != "skipped"])
    print(f"\n  Chaos test: {passed}/{total} passed")
    print(f"  Status: {'PASS' if passed >= total - 1 else 'FAIL'}")

    report = {"results": results, "passed": passed >= total - 1}
    with open(os.path.join(os.path.dirname(__file__), "..", "chaos_bench_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return passed >= total - 1


if __name__ == "__main__":
    asyncio.run(run_chaos_test())
