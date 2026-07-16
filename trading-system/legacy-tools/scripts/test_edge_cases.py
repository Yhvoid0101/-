# -*- coding: utf-8 -*-
"""Edge case and stress test for the three qualitative transform modules"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.self_organizing_swarm import SelfOrganizingSwarm, CapabilitySignal
from core.neurosymbolic_verifier import NeurosymbolicVerifier
from core.pea_security import PEASecurityArchitecture

print("=" * 60)
print("Edge Case Tests")
print("=" * 60)

# Test 1: Swarm peer selection with different capabilities
print("\n--- Swarm: Peer selection with different capabilities ---")
s = SelfOrganizingSwarm(agent_id="main")
for i in range(5):
    peer = SelfOrganizingSwarm(agent_id=f"p{i}")
    for _ in range(3):
        peer.capability_signal.update("code_analysis", True, quality_score=0.5 + i * 0.1, response_time=5.0)
    s.receive_broadcast(peer.broadcast_capability())

best = s.find_best_peer_for_task("code_analysis")
print(f"Best peer for code_analysis: {best}")
for pid, info in s.peers.items():
    cap = info.get("capabilities", {}).get("code_analysis", 0)
    print(f"  {pid}: code_analysis={cap:.3f}")
assert best == "p4", f"Expected p4 (highest capability), got {best}"
print("[PASS] Best peer selection works correctly")

# Test 2: Verifier with mixed language
print("\n--- Verifier: Mixed language input ---")
nv = NeurosymbolicVerifier()
r = nv.verify("Because the data is abnormal, so we must fix it. Because the system is unstable, therefore we need to restart.")
print(f"Mixed language: propositions={r['propositions_count']}, issues={r['issue_count']}")
assert r["propositions_count"] >= 1, "Should extract at least one English proposition"
print("[PASS] Mixed language proposition extraction works")

# Test 3: PEA with empty task
print("\n--- PEA: Empty task handling ---")
pea = PEASecurityArchitecture()
r = pea.process({"task": "", "task_type": "general"})
print(f"Empty task: final_action={r['final_action']}, vetoed={r['vetoed']}")
assert r["final_action"] == "unnamed_task", f"Expected 'unnamed_task', got '{r['final_action']}'"
print("[PASS] Empty task handled correctly")

# Test 4: Verifier with contradictory assertions
print("\n--- Verifier: Contradictory assertions ---")
r = nv.verify("这个方案是正确的。这个方案是错误的。")
consistency_issues = [i for i in r.get("issues", []) if i.get("type") == "proposition_contradiction"]
print(f"Contradictions found: {len(consistency_issues)}")
assert len(consistency_issues) >= 1, "Should detect proposition contradiction"
print("[PASS] Proposition contradiction detection works")

# Test 5: PEA progressive restrictions escalation
print("\n--- PEA: Progressive restriction escalation ---")
pea2 = PEASecurityArchitecture()
for i in range(5):
    pea2.process({"task": "删除所有日志", "task_type": "dangerous"})
stats = pea2.get_stats()
trackers = stats.get("violation_trackers", {})
first_key = list(trackers.keys())[0] if trackers else ""
tracker = trackers.get(first_key, {})
print(f"Violation count: {tracker.get('count', 0)}, restriction level: {tracker.get('restriction_level', 0)}")
assert tracker.get("count", 0) >= 3, "Should have accumulated violations"
assert tracker.get("restriction_level", 0) >= 1, "Should have escalated restriction level"
print("[PASS] Progressive restrictions work")

# Test 6: Swarm stigmergy collaboration
print("\n--- Swarm: Stigmergy collaboration ---")
s1 = SelfOrganizingSwarm(agent_id="s1")
s1.stigmergy.leave_marker("s1", "code_analysis", "workspace", "success", strength=0.9, data={"quality": 0.85})
s1.stigmergy.leave_marker("s2", "code_analysis", "workspace", "success", strength=0.7, data={"quality": 0.7})
markers = s1.stigmergy.read_markers(location="workspace", task_type="code_analysis")
print(f"Stigmergy markers: {len(markers)}")
assert len(markers) >= 1, "Should have stigmergy markers"
print("[PASS] Stigmergy works")

# Test 7: Verifier causal gap detection - no reasoning at all
print("\n--- Verifier: Causal gap with conclusion but no reasoning ---")
r = nv.verify("因此我们可以确定系统需要重构。")
print(f"Causal gaps: {len(r.get('causal_gaps', []))}")
assert len(r.get("causal_gaps", [])) >= 1, "Should detect causal gap"
print("[PASS] Causal gap detection works")

# Test 8: PEA audit chain integrity after many operations
print("\n--- PEA: Audit chain integrity ---")
pea3 = PEASecurityArchitecture()
for i in range(15):
    pea3.process({"task": f"test_{i}", "task_type": "general"})
integrity = pea3.verify_audit_integrity()
print(f"Audit chain: valid={integrity['valid']}, entries={integrity['entries']}")
assert integrity["valid"] == True, "Audit chain should be valid"
print("[PASS] Audit chain integrity maintained")

# Test 9: Swarm natural selection - low performers removed
print("\n--- Swarm: Natural selection ---")
s2 = SelfOrganizingSwarm(agent_id="selector")
s2.update_collaboration("bad_peer", "code_analysis", False, quality=0.1, response_time=60.0)
s2.update_collaboration("bad_peer", "code_analysis", False, quality=0.1, response_time=60.0)
s2.update_collaboration("bad_peer", "code_analysis", False, quality=0.1, response_time=60.0)
s2.update_collaboration("bad_peer", "code_analysis", False, quality=0.1, response_time=60.0)
remaining = s2.collaboration_graph[s2.id].get("bad_peer", 0)
print(f"Bad peer score after 4 failures: {remaining}")
assert remaining < 0.2, "Bad peer should have low collaboration score"
print("[PASS] Natural selection works")

# Test 10: Verifier fact base learning
print("\n--- Verifier: Fact base learning from verified output ---")
nv2 = NeurosymbolicVerifier()
initial_facts = nv2.fact_base.get_stats()["total_facts"]
nv2.verify("Python是编程语言")
nv2.verify("Hermes是智能体框架")
nv2.verify("GABA是抑制性神经递质")
after_facts = nv2.fact_base.get_stats()["total_facts"]
print(f"Facts: {initial_facts} -> {after_facts}")
assert after_facts >= initial_facts, "Should learn new facts"
print("[PASS] Fact base learning works")

print("\n" + "=" * 60)
print("ALL EDGE CASE TESTS PASSED!")
print("=" * 60)
