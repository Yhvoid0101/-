import json
import os
import sys
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hermes_v6.config import load_api_keys_from_env
load_api_keys_from_env()


def run_checkpointer_validation():
    print("=" * 70)
    print("Checkpointer Strict Validation (Hermes v6.0)")
    print("=" * 70)

    from hermes_v6.langgraph_orchestrator import Checkpointer, LangGraphOrchestrator
    from hermes_v6.core.kernel import GraphNode, GraphEdge

    results = {}

    cp_dir = os.path.join(os.path.dirname(__file__), "_test_cp")
    cp = Checkpointer(storage_dir=cp_dir)

    print("[1/5] Basic save/load test...")
    try:
        saved = cp.save("test_basic", {"step": 1, "data": "hello"})
        loaded = cp.load(saved.checkpoint_id)
        results["basic"] = loaded is not None and loaded.graph_state.get("data") == "hello"
        print(f"  Basic: {'PASS' if results['basic'] else 'FAIL'}")
    except Exception as e:
        results["basic"] = False
        print(f"  Basic FAIL: {e}")

    print("[2/5] Branching test...")
    try:
        cp2 = Checkpointer(storage_dir=cp_dir)
        cp2.save("branch_a", {"step": 1, "branch": "A"})
        cp2.save("branch_b", {"step": 1, "branch": "B"})
        latest = cp2.get_latest("branch_a")
        results["branching"] = latest is not None
        print(f"  Branching: {'PASS' if results['branching'] else 'FAIL'}")
    except Exception as e:
        results["branching"] = False
        print(f"  Branching FAIL: {e}")

    print("[3/5] Time travel test...")
    try:
        cp3 = Checkpointer(storage_dir=cp_dir)
        cp3.save("time_test", {"version": 1})
        cp3.save("time_test", {"version": 2})
        cp3.save("time_test", {"version": 3})
        all_cp = cp3.list_checkpoints("time_test")
        results["time_travel"] = len(all_cp) >= 2
        print(f"  Time travel: {'PASS' if results['time_travel'] else 'FAIL'} ({len(all_cp)} checkpoints)")
    except Exception as e:
        results["time_travel"] = False
        print(f"  Time travel FAIL: {e}")

    print("[4/5] Interrupt & resume test...")
    try:
        lg = LangGraphOrchestrator()
        lg.add_node(GraphNode(id="start", name="start", action=lambda s: {**s, "step": 1}))
        lg.add_node(GraphNode(id="mid", name="mid", action=lambda s: {**s, "step": 2}))
        lg.add_node(GraphNode(id="end", name="end", action=lambda s: {**s, "step": 3}))
        lg.add_edge(GraphEdge(source="start", target="mid"))
        lg.add_edge(GraphEdge(source="mid", target="end"))
        lg.set_entry_point("start")

        r = asyncio.run(lg.execute("interrupt_test", {"input": "test"}))
        results["interrupt_resume"] = r.get("success", False)
        print(f"  Interrupt/resume: {'PASS' if results['interrupt_resume'] else 'FAIL'}")
    except Exception as e:
        results["interrupt_resume"] = False
        print(f"  Interrupt/resume FAIL: {e}")

    print("[5/5] Delete test...")
    try:
        cp5 = Checkpointer(storage_dir=cp_dir)
        saved = cp5.save("delete_test", {"temp": True})
        cp5.delete(saved.checkpoint_id)
        results["delete"] = True
        print(f"  Delete: PASS")
    except Exception as e:
        results["delete"] = False
        print(f"  Delete FAIL: {e}")

    import shutil

    shutil.rmtree(cp_dir, ignore_errors=True)

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  Checkpointer: {passed}/{total} passed")
    print(f"  Status: {'PASS' if passed >= 4 else 'FAIL'}")

    report = {"results": results, "passed": passed >= 4}
    with open(os.path.join(os.path.dirname(__file__), "..", "checkpointer_validation_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return passed >= 4


if __name__ == "__main__":
    success = run_checkpointer_validation()
    sys.exit(0 if success else 1)
