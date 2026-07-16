#!/usr/bin/env python3
"""五模块联合验证"""
import sys
sys.path.insert(0, ".")

from core.oxygent_adapter import OxyDAGOrchestrator
from core.evomaster_loop import EvoMasterLoop
from core.magma_memory import MagmaMemoryEngine
from core.parallax_security import ParallaxSandbox, TrustLevel
from core.metaclaw_dual_loop import MetaClawDualLoop

print("=== Five Module Integration Test ===")

dag = OxyDAGOrchestrator()
em = EvoMasterLoop()
magma = MagmaMemoryEngine()
sandbox = ParallaxSandbox(trust_level=TrustLevel.TRUSTED)
mc = MetaClawDualLoop()

print("All 5 modules instantiated OK")

dag.register_component("test", "tool", lambda **kw: {"ok": True})
r1 = dag.execute("test", task="hello")
print(f"OxyGent: {r1['success']}")

r2 = em.discover("test", domain="test", experiment_iterations=1, multi_hypothesis=False)
print(f"EvoMaster: {r2['total_hypotheses']} hypotheses")

mid = magma.store("test memory", semantic_tags=["test"], importance=0.7)
print(f"MAGMA store: {mid}")

r4 = sandbox.reason("analyze code")
print(f"Parallax: {r4['action']}")

mc.notify_task_completed({"success": True, "task": "test"})
print("MetaClaw: notified")

print("ALL FIVE MODULES WORKING TOGETHER!")
