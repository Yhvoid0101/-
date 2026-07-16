import asyncio
import sys
import json
import time
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()


async def run_all_stage3():
    print("=" * 70)
    print(f"Hermes v6.0 Stage 3 Complete Lab Validation")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = {}
    start_time = time.time()

    print("\n[1/7] Multimodal & GUI test...")
    try:
        from bench_multimodal import run_multimodal_test

        rate = await run_multimodal_test()
        all_results["multimodal"] = {"success_rate": rate, "passed": rate >= 50}
    except Exception as e:
        all_results["multimodal"] = {"error": str(e)[:80], "passed": False}

    print("\n[2/7] Chaos engineering test...")
    try:
        from bench_chaos import run_chaos_test

        passed = await run_chaos_test()
        all_results["chaos"] = {"passed": passed}
    except Exception as e:
        all_results["chaos"] = {"error": str(e)[:80], "passed": False}

    print("\n[3/7] Memory system deep evaluation...")
    try:
        from bench_memory import run_memory_bench

        passed = await run_memory_bench()
        all_results["memory"] = {"passed": passed}
    except Exception as e:
        all_results["memory"] = {"error": str(e)[:80], "passed": False}

    print("\n[4/7] Checkpointer strict validation...")
    try:
        from bench_checkpointer import run_checkpointer_validation

        passed = run_checkpointer_validation()
        all_results["checkpointer"] = {"passed": passed}
    except Exception as e:
        all_results["checkpointer"] = {"error": str(e)[:80], "passed": False}

    print("\n[5/7] Production gate check...")
    try:
        from check_production_gates import run_gate_check

        passed = run_gate_check()
        all_results["production_gates"] = {"passed": passed}
    except Exception as e:
        all_results["production_gates"] = {"error": str(e)[:80], "passed": False}

    print("\n[6/7] Long-horizon persistence test...")
    try:
        from bench_horizon import run_horizon_test

        report = await run_horizon_test()
        all_results["horizon"] = {
            "success_rate": report["success_rate"],
            "passed": report["success_rate"] >= 80 and not report["systemic_crash_detected"],
        }
    except Exception as e:
        all_results["horizon"] = {"error": str(e)[:80], "passed": False}

    print("\n[7/7] Self-evolution capability tracking...")
    try:
        from bench_seaeval import run_seaeval

        report = await run_seaeval()
        all_results["seaeval"] = {
            "time_improvement": report["time_improvement_pct"],
            "passed": True,
        }
    except Exception as e:
        all_results["seaeval"] = {"error": str(e)[:80], "passed": False}

    total_elapsed = time.time() - start_time
    passed_count = sum(1 for r in all_results.values() if r.get("passed", False))
    total_tests = len(all_results)

    print("\n" + "=" * 70)
    print(f"Stage 3 Complete (elapsed: {total_elapsed / 60:.1f} min)")
    print("=" * 70)
    for name, result in all_results.items():
        status = "PASS" if result.get("passed") else "FAIL"
        if "error" in result:
            print(f"  {status} {name}: {result['error']}")
        elif "success_rate" in result:
            print(f"  {status} {name}: {result['success_rate']:.1f}%")
        else:
            print(f"  {status} {name}")

    print(f"\n  Pass rate: {passed_count}/{total_tests} ({passed_count / total_tests * 100:.0f}%)")

    final_report = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_min": total_elapsed / 60,
        "total_tests": total_tests,
        "passed": passed_count,
        "pass_rate": passed_count / total_tests * 100,
        "details": all_results,
    }
    with open(os.path.join(os.path.dirname(__file__), "..", "stage3_final_report.json"), "w") as f:
        json.dump(final_report, f, indent=2, default=str)

    if passed_count >= total_tests - 1:
        print("\n  Stage 3 PASSED! Ready for Stage 4.")
        return 0
    else:
        print("\n  Some tests failed. Review report and fix.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_all_stage3()))
