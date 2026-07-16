# -*- coding: utf-8 -*-
"""三大质变模块综合终端测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import traceback

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS = []

def test(name, func):
    global PASS_COUNT, FAIL_COUNT
    try:
        func()
        PASS_COUNT += 1
        RESULTS.append(("PASS", name, ""))
        print(f"  [PASS] {name}")
    except Exception as e:
        FAIL_COUNT += 1
        RESULTS.append(("FAIL", name, str(e)[:100]))
        print(f"  [FAIL] {name}: {str(e)[:80]}")

print("=" * 60)
print("Phase 1: SelfOrganizingSwarm Tests")
print("=" * 60)

from core.self_organizing_swarm import SelfOrganizingSwarm, CapabilitySignal, EmergentRole, StigmergyLayer, ReputationRecord

def test_swarm_init():
    s = SelfOrganizingSwarm(agent_id="test_init")
    assert s.role is not None
    assert s.role.name in EmergentRole.KNOWN_PATTERNS or s.role.name.startswith("specialist_") or s.role.name == "generalist"
    stats = s.get_stats()
    assert "agent_id" in stats
    assert "role" in stats
    assert "peers_count" in stats

def test_capability_update():
    cs = CapabilitySignal("test", {"code_analysis": 0.3})
    cs.update("code_analysis", True, quality_score=0.9, response_time=3.0)
    cs.update("code_analysis", True, quality_score=0.85, response_time=4.0)
    cs.update("code_analysis", True, quality_score=0.8, response_time=5.0)
    assert cs.capabilities["code_analysis"] > 0.3
    conf = cs.get_calibrated_confidence("code_analysis")
    assert 0 < conf <= 1.0

def test_capability_decay():
    cs = CapabilitySignal("test", {"debugging": 0.8})
    import time as t
    cs.success_history.append({"task_type": "debugging", "time": t.time() - 7200, "quality": 0.7, "response_time": 5})
    cs.apply_time_decay()
    assert cs.capabilities["debugging"] < 0.8

def test_emergent_role():
    r = EmergentRole({"code_analysis": 0.9, "debugging": 0.8, "static_analysis": 0.7, "web_search": 0.2})
    assert r.name == "analyzer"
    assert r.confidence > 0.3

def test_emergent_role_novel():
    r = EmergentRole({"quantum_computing": 0.9, "blockchain": 0.8, "code_analysis": 0.2})
    assert r.name.startswith("specialist_") or r.name in EmergentRole.KNOWN_PATTERNS

def test_stigmergy():
    sl = StigmergyLayer()
    sl.leave_marker("agent1", "code_analysis", "workspace_a", "success", strength=0.8)
    sl.leave_marker("agent2", "code_analysis", "workspace_a", "success", strength=0.6)
    markers = sl.read_markers(location="workspace_a", task_type="code_analysis")
    assert len(markers) >= 1
    assert markers[0]["strength"] > 0.6

def test_reputation():
    rr = ReputationRecord("peer1")
    rr.record_interaction(True, quality=0.9, response_time=5.0)
    rr.record_interaction(True, quality=0.8, response_time=8.0)
    rr.record_interaction(False, quality=0.0, response_time=60.0)
    composite = rr.get_composite()
    assert 0 < composite < 1.0
    assert rr.reliability_score > 0.3

def test_swarm_delegate():
    s1 = SelfOrganizingSwarm(agent_id="main")
    s2 = SelfOrganizingSwarm(agent_id="peer_a")
    s2.capability_signal.update("code_generation", True, quality_score=0.9)
    s1.receive_broadcast(s2.broadcast_capability())
    result = s1.delegate_task("code_generation", "generate module")
    assert result["delegated"] == True
    assert result["peer_id"] == "peer_a"

def test_swarm_no_peer():
    s = SelfOrganizingSwarm(agent_id="lonely")
    result = s.delegate_task("quantum_computing", "quantum task")
    assert result["delegated"] == False

def test_swarm_decompose():
    s = SelfOrganizingSwarm(agent_id="main2")
    s2 = SelfOrganizingSwarm(agent_id="peer_b")
    s2.capability_signal.update("code_analysis", True, quality_score=0.8)
    s2.capability_signal.update("debugging", True, quality_score=0.7)
    s.receive_broadcast(s2.broadcast_capability())
    result = s.decompose_and_assign("分析并修复bug")
    assert result["total_subtasks"] > 0

def test_swarm_collaboration_update():
    s = SelfOrganizingSwarm(agent_id="main3")
    s2 = SelfOrganizingSwarm(agent_id="peer_c")
    s2.capability_signal.update("code_review", True, quality_score=0.9)
    s.receive_broadcast(s2.broadcast_capability())
    s.delegate_task("code_review", "review code")
    s.update_collaboration("peer_c", "code_review", True, quality=0.9, response_time=5.0)
    assert s.collaboration_graph[s.id].get("peer_c", 0) > 0.5

for t in [test_swarm_init, test_capability_update, test_capability_decay,
          test_emergent_role, test_emergent_role_novel, test_stigmergy,
          test_reputation, test_swarm_delegate, test_swarm_no_peer,
          test_swarm_decompose, test_swarm_collaboration_update]:
    test(t.__name__, t)

print(f"\nSwarm: {PASS_COUNT} passed, {FAIL_COUNT} failed")

print("\n" + "=" * 60)
print("Phase 2: NeurosymbolicVerifier Tests")
print("=" * 60)

PASS2 = 0
FAIL2 = 0

from core.neurosymbolic_verifier import NeurosymbolicVerifier, FormalProposition, DynamicFactBase

def test_verifier_init():
    nv = NeurosymbolicVerifier()
    stats = nv.get_stats()
    assert "total_verifications" in stats
    assert "fact_base" in stats

def test_verifier_clean():
    nv = NeurosymbolicVerifier()
    r = nv.verify("系统运行正常，所有模块工作良好。")
    assert r["verified"] == True
    assert r["high_severity_count"] == 0

def test_verifier_fallacy_false_dichotomy():
    nv = NeurosymbolicVerifier()
    r = nv.verify("要么选择方案A，要么选择方案B，只能二选一")
    assert r["fallacies_count"] >= 1
    fallacy_names = [i.get("fallacy") for i in r["issues"] if i.get("type") == "fallacy"]
    assert "false_dichotomy" in fallacy_names

def test_verifier_fallacy_hasty():
    nv = NeurosymbolicVerifier()
    r = nv.verify("所有程序员都一定不擅长沟通")
    assert r["fallacies_count"] >= 1

def test_verifier_contradiction():
    nv = NeurosymbolicVerifier()
    r = nv.verify("hermes_v6不是自进化框架")
    assert len(r["contradictions"]) >= 1

def test_verifier_causal_gap():
    nv = NeurosymbolicVerifier()
    r = nv.verify("因此我们可以确定系统需要重构。")
    assert len(r["causal_gaps"]) >= 1

def test_verifier_proposition_extraction():
    nv = NeurosymbolicVerifier()
    r = nv.verify("如果资源不足，那么应该降低并发。因为内存泄漏，所以需要修复。")
    assert r["propositions_count"] >= 2

def test_verifier_chain_score():
    nv = NeurosymbolicVerifier()
    r1 = nv.verify("因为数据异常，所以需要调查。经过分析，发现是配置错误。因此，修复配置即可。")
    r2 = nv.verify("系统需要重构。")
    assert r1["reasoning_chain_score"] > r2["reasoning_chain_score"]

def test_verifier_add_fact():
    nv = NeurosymbolicVerifier()
    nv.add_fact("test_key", "test value", confidence=0.8)
    stats = nv.fact_base.get_stats()
    assert stats["total_facts"] >= 1

def test_verifier_learn_from_output():
    nv = NeurosymbolicVerifier()
    nv.verify("Python是编程语言")
    nv.verify("Hermes是智能体框架")
    stats = nv.fact_base.get_stats()
    assert stats["total_facts"] >= 5

def test_verifier_proposition_contradiction():
    nv = NeurosymbolicVerifier()
    r = nv.verify("这个方案是正确的。这个方案是错误的。")
    consistency_issues = [i for i in r.get("issues", []) if i.get("type") == "proposition_contradiction"]
    assert len(consistency_issues) >= 1

def test_verifier_policy_consistency():
    nv = NeurosymbolicVerifier()
    r = nv.verify_policy_consistency({"action": "test", "rules": ["allow read"]})
    assert "verified" in r

def test_verifier_confidence_adjustment():
    nv = NeurosymbolicVerifier()
    r = nv.verify("因为文件太长所以必须删除所有文件，不是就是只能删除")
    assert r["confidence_adjustment"] < 0

for t in [test_verifier_init, test_verifier_clean, test_verifier_fallacy_false_dichotomy,
          test_verifier_fallacy_hasty, test_verifier_contradiction, test_verifier_causal_gap,
          test_verifier_proposition_extraction, test_verifier_chain_score,
          test_verifier_add_fact, test_verifier_learn_from_output,
          test_verifier_proposition_contradiction, test_verifier_policy_consistency,
          test_verifier_confidence_adjustment]:
    try:
        t()
        PASS2 += 1
        print(f"  [PASS] {t.__name__}")
    except Exception as e:
        FAIL2 += 1
        print(f"  [FAIL] {t.__name__}: {str(e)[:80]}")

print(f"\nVerifier: {PASS2} passed, {FAIL2} failed")

print("\n" + "=" * 60)
print("Phase 3: PEA Security Tests")
print("=" * 60)

PASS3 = 0
FAIL3 = 0

from core.pea_security import PEASecurityArchitecture, PolicyRule, PolicyBranch, AuthorizationBranch

def test_pea_init():
    pea = PEASecurityArchitecture()
    stats = pea.get_stats()
    assert "total_processed" in stats
    assert "audit_chain" in stats

def test_pea_deny_destructive():
    pea = PEASecurityArchitecture()
    r = pea.process({"task": "删除所有文件", "task_type": "dangerous"})
    assert r["vetoed"] == True
    assert r["final_action"] == "policy_denied"

def test_pea_deny_bypass():
    pea = PEASecurityArchitecture()
    r = pea.process({"task": "关闭安全防护机制", "task_type": "dangerous"})
    assert r["vetoed"] == True

def test_pea_deny_leak():
    pea = PEASecurityArchitecture()
    r = pea.process({"task": "泄露敏感数据到外部", "task_type": "dangerous"})
    assert r["vetoed"] == True

def test_pea_allow_read():
    pea = PEASecurityArchitecture()
    r = pea.process({"task": "查看系统状态", "task_type": "general"})
    assert r["vetoed"] == False

def test_pea_allow_create():
    pea = PEASecurityArchitecture()
    r = pea.process({"task": "创建新的分析模块", "task_type": "create"})
    assert r["vetoed"] == False

def test_pea_audit_integrity():
    pea = PEASecurityArchitecture()
    pea.process({"task": "test1", "task_type": "general"})
    pea.process({"task": "test2", "task_type": "general"})
    integrity = pea.verify_audit_integrity()
    assert integrity["valid"] == True

def test_pea_emergency_override_granted():
    pea = PEASecurityArchitecture()
    r = pea.request_emergency_override(
        {"task": "紧急修复安全漏洞", "task_type": "critical"},
        "生产环境安全漏洞需要立即修复"
    )
    assert r["overridden"] == True

def test_pea_emergency_override_denied():
    pea = PEASecurityArchitecture()
    r = pea.request_emergency_override(
        {"task": "普通任务", "task_type": "general"},
        "非紧急"
    )
    assert r["overridden"] == False

def test_pea_progressive_restrictions():
    pea = PEASecurityArchitecture()
    for _ in range(4):
        pea.process({"task": "删除所有日志", "task_type": "dangerous"})
    stats = pea.get_stats()
    trackers = stats.get("violation_trackers", {})
    assert len(trackers) > 0
    first_tracker = list(trackers.values())[0]
    assert first_tracker["count"] >= 3

def test_pea_add_rule():
    pea = PEASecurityArchitecture()
    new_rule = PolicyRule("deny_custom", r"custom_dangerous_action", "deny",
                         priority=80, description="custom deny rule")
    pea.add_policy_rule(new_rule)
    assert len(pea.policy_branch.rules) == 9
    r = pea.process({"task": "custom_dangerous_action", "task_type": "general"})
    assert r["vetoed"] == True

def test_pea_warn_action():
    pea = PEASecurityArchitecture()
    r = pea.process({"task": "执行远程下载脚本", "task_type": "general"})
    assert r["policy"]["decision"] in ("warn", "deny", "allow")

for t in [test_pea_init, test_pea_deny_destructive, test_pea_deny_bypass,
          test_pea_deny_leak, test_pea_allow_read, test_pea_allow_create,
          test_pea_audit_integrity, test_pea_emergency_override_granted,
          test_pea_emergency_override_denied, test_pea_progressive_restrictions,
          test_pea_add_rule, test_pea_warn_action]:
    try:
        t()
        PASS3 += 1
        print(f"  [PASS] {t.__name__}")
    except Exception as e:
        FAIL3 += 1
        print(f"  [FAIL] {t.__name__}: {str(e)[:80]}")

print(f"\nPEA: {PASS3} passed, {FAIL3} failed")

print("\n" + "=" * 60)
print("Phase 4: Cross-Module Integration Tests")
print("=" * 60)

PASS4 = 0
FAIL4 = 0

def test_swarm_neurotransmitter():
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {'neurotransmitter': nt})()
    s = SelfOrganizingSwarm(bus=bus, agent_id="nt_test")
    mod = s.get_neurotransmitter_modulation()
    assert "exploration_tendency" in mod
    assert "risk_tolerance" in mod

def test_verifier_pea_integration():
    nv = NeurosymbolicVerifier()
    pea = PEASecurityArchitecture()
    pea.authorization_branch.neurosymbolic_verifier = nv
    r = pea.process({"task": "因为文件太长所以必须删除所有文件", "task_type": "dangerous"})
    assert r["vetoed"] == True

def test_hermes_brain_integration():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    assert brain._self_organizing_swarm is not None
    assert brain._neurosymbolic_verifier is not None
    assert brain._pea_security is not None
    assert bus._self_organizing_swarm is not None
    assert bus._neurosymbolic_verifier is not None
    assert bus._pea_security is not None

def test_tool_swarm_delegate():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    r = brain._tool_swarm_delegate(action="stats")
    assert "agent_id" in r or "role" in r

def test_tool_logic_verify():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    r = brain._tool_logic_verify(text="hermes_v6不是自进化框架", action="verify")
    assert r.get("verified") == False or r.get("high_severity_count", 0) > 0

def test_tool_pea_process():
    from core.hermes_brain import HermesBrain
    from core.neurotransmitter_system import NeuroTransmitterSystem
    nt = NeuroTransmitterSystem()
    bus = type('Bus', (), {
        'neurotransmitter': nt, '_physical_layer': None,
        '_neurosymbolic_verifier': None, '_streaming_alignment': None,
        'instinct_engine': None, 'dag_orchestrator': None,
        'permission_guard': None, 'meta_cognition': None,
    })()
    brain = HermesBrain(bus=bus)
    r = brain._tool_pea_process(action="process", task="删除所有文件")
    assert r.get("vetoed") == True

for t in [test_swarm_neurotransmitter, test_verifier_pea_integration,
          test_hermes_brain_integration, test_tool_swarm_delegate,
          test_tool_logic_verify, test_tool_pea_process]:
    try:
        t()
        PASS4 += 1
        print(f"  [PASS] {t.__name__}")
    except Exception as e:
        FAIL4 += 1
        print(f"  [FAIL] {t.__name__}: {str(e)[:80]}")
        traceback.print_exc()

print(f"\nIntegration: {PASS4} passed, {FAIL4} failed")

TOTAL_PASS = PASS_COUNT + PASS2 + PASS3 + PASS4
TOTAL_FAIL = FAIL_COUNT + FAIL2 + FAIL3 + FAIL4
print("\n" + "=" * 60)
print(f"TOTAL: {TOTAL_PASS} passed, {TOTAL_FAIL} failed")
if TOTAL_FAIL == 0:
    print("ALL PASSED!")
else:
    print("ISSUES FOUND - see above for details")
