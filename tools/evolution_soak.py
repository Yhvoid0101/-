"""Run an isolated deterministic soak test for the Codex evolution control plane."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("codex_autonomy", ROOT / "src" / "codex_autonomy.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def run(cycles: int) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="codex-evolution-soak-") as temp:
        store = MODULE.AutonomyStore(Path(temp) / "soak.db")
        executed = 0
        replayed = 0
        for index in range(cycles):
            def side_effect(index=index):
                nonlocal executed
                executed += 1
                return {"cycle": index, "result": "persisted"}

            first = store.run_durable_step(f"soak-{index}", "side-effect", {"cycle": index}, side_effect)
            second = store.run_durable_step(f"soak-{index}", "side-effect", {"cycle": index}, side_effect)
            replayed += int(second["replayed"])
            if first["replayed"] or not second["replayed"]:
                raise RuntimeError(f"replay invariant failed at cycle {index}")
            store.enqueue("soak-job", {"cycle": index}, idempotency_key=f"soak-job:{index}", max_attempts=1)
            if index % 3 == 0:
                store.run_once(lambda job: (_ for _ in ()).throw(RuntimeError("injected soak failure")))
            else:
                store.run_once(lambda job: None)
            store.evolution_cycle(cycle_key=f"soak-evolution-{index}")
            audit = store.audit_invariants()
            if audit["status"] != "PASS":
                raise RuntimeError(json.dumps(audit, ensure_ascii=False))
        result = {"status": "PASS", "cycles": cycles, "side_effect_executions": executed,
                  "replayed_steps": replayed, "audit": store.audit_invariants()}
        del store
        gc.collect()
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    args = parser.parse_args()
    if args.cycles < 1 or args.cycles > 10000:
        parser.error("--cycles must be between 1 and 10000")
    print(json.dumps(run(args.cycles), ensure_ascii=False, indent=2))
