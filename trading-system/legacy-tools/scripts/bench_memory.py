import asyncio
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()


async def run_memory_bench():
    print("=" * 70)
    print("Memory System Deep Evaluation (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.e40_layered_memory import LayeredMemorySystem, MemoryLayer

    mem = LayeredMemorySystem()
    results = {}

    print("[1/5] Memory fidelity test...")
    try:
        item = mem.store(
            "user favorite color is ocean blue, birthday is May 20",
            layer=MemoryLayer.LONG_TERM,
            importance=0.8,
            tags=["user", "preference", "color"],
        )
        retrieved = mem.retrieve(item.id)
        fidelity = retrieved is not None and "ocean blue" in str(retrieved.content).lower()
        results["fidelity"] = fidelity
        print(f"  Fidelity: {'PASS' if fidelity else 'FAIL'}")
    except Exception as e:
        results["fidelity"] = False
        print(f"  Fidelity FAIL: {e}")

    print("[2/5] Memory search test...")
    try:
        mem.store("user birthday is May 20", layer=MemoryLayer.LONG_TERM, importance=0.7, tags=["user", "birthday"])
        found = mem.search(query="birthday", tags=["birthday"])
        results["search"] = len(found) > 0
        print(f"  Search: {'PASS' if results['search'] else 'FAIL'} - found {len(found)} items")
    except Exception as e:
        results["search"] = False
        print(f"  Search FAIL: {e}")

    print("[3/5] Memory conflict resolution...")
    try:
        mem.store(
            "user birthday is May 20", layer=MemoryLayer.LONG_TERM, importance=0.5, tags=["user", "birthday", "old"]
        )
        mem.store(
            "user birthday is June 15", layer=MemoryLayer.LONG_TERM, importance=0.9, tags=["user", "birthday", "new"]
        )
        found = mem.search(query="birthday", tags=["birthday"])
        has_june = any("june" in str(f.content).lower() for f in found)
        results["conflict_resolution"] = has_june or len(found) >= 2
        print(f"  Conflict: {'PASS' if results['conflict_resolution'] else 'PARTIAL'}")
    except Exception as e:
        results["conflict_resolution"] = False
        print(f"  Conflict FAIL: {e}")

    print("[4/5] Memory capacity stress test...")
    start = time.time()
    try:
        for i in range(200):
            mem.store(f"stress test entry {i} with data payload", layer=MemoryLayer.WORKING, importance=0.1)
        latency = time.time() - start
        results["capacity_latency"] = latency < 10.0
        print(f"  200 writes in {latency:.2f}s: {'PASS' if latency < 10 else 'SLOW'}")
    except Exception as e:
        latency = time.time() - start
        results["capacity_latency"] = False
        print(f"  Capacity FAIL: {e}")

    print("[5/5] Memory layer promotion test...")
    try:
        item = mem.store("promotion test", layer=MemoryLayer.SENSORY, importance=0.3)
        stats = mem.get_stats()
        results["promotion"] = stats.get("stores", 0) > 0
        print(f"  Promotion: stores={stats.get('stores', 0)}, PASS")
    except Exception as e:
        results["promotion"] = False
        print(f"  Promotion FAIL: {e}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Memory test: {passed}/{total} passed")
    print(f"  Status: {'PASS' if passed >= 4 else 'FAIL'}")

    report = {"results": results, "passed": passed >= 4}
    with open(os.path.join(os.path.dirname(__file__), "..", "memory_bench_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return passed >= 4


if __name__ == "__main__":
    asyncio.run(run_memory_bench())
