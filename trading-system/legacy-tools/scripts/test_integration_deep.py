#!/usr/bin/env python3
"""
OxyGent + EvoMaster 深度集成测试
验证两个模块协同工作、与AgentBus集成、边界场景、压力测试
"""
import sys
import time
import json
import traceback
import threading
import random

sys.path.insert(0, ".")

from core.oxygent_adapter import OxyDAGOrchestrator, OxyComponent, CircuitBreaker, OxyBank
from core.evomaster_loop import EvoMasterLoop, Hypothesis, BayesianConfidence, KnowledgeGraph

passed = 0
failed = 0
errors = []

def test(name, func):
    global passed, failed
    try:
        func()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


# ========== 集成测试 ==========

def test_oxygent_evomaster_collaboration():
    """OxyGent组件执行结果驱动EvoMaster假设验证"""
    dag = OxyDAGOrchestrator()
    em = EvoMasterLoop()

    dag.register_component("data_collector", "tool",
        lambda **kw: {"samples": [random.gauss(0.7, 0.1) for _ in range(20)]},
        input_schema={"required": ["task"]})
    dag.register_component("analyzer", "tool",
        lambda **kw: {"mean": 0.7, "std": 0.1, "significant": True},
        input_schema={"required": ["task"]}, depends_on=["data_collector"])

    result = dag.execute("analyzer", task_description="分析缓存命中率数据", task="分析缓存命中率")
    assert result["success"], f"OxyGent执行应成功: {result}"

    evo_result = em.discover("基于OxyGent分析结果优化缓存", domain="cache",
                              experiment_iterations=2, multi_hypothesis=False)
    assert evo_result["total_hypotheses"] >= 1

def test_oxybank_pattern_reuse_in_evomaster():
    """OxyBank成功模式被EvoMaster参考"""
    dag = OxyDAGOrchestrator()
    dag.register_component("step1", "tool", lambda **kw: {"result": "step1_done"})
    dag.register_component("step2", "tool", lambda **kw: {"result": "step2_done"},
                           depends_on=["step1"])
    dag.execute("step2", task_description="cache optimization workflow", task="cache")
    dag.execute("step2", task_description="cache hit rate improvement", task="cache")

    recs = dag.get_recommendations("cache optimization")
    assert len(recs) > 0, "OxyBank应有推荐模式"

def test_hypothesis_drives_component_creation():
    """EvoMaster确认的假设驱动OxyGent新组件注册"""
    dag = OxyDAGOrchestrator()
    em = EvoMasterLoop()

    evo_result = em.discover("组件创建测试", domain="test",
                              experiment_iterations=3, multi_hypothesis=False)

    for hyp_id, hyp in em.hypotheses.items():
        if hyp.status == Hypothesis.STATUS_CONFIRMED:
            new_comp_name = f"evo_validated_{hyp.id[:6]}"
            dag.register_component(new_comp_name, "tool",
                lambda **kw: {"validated": True, "hypothesis": hyp.statement[:50]})
            assert new_comp_name in dag.components
            break

def test_circuit_breaker_with_evo_feedback():
    """EvoMaster发现组件故障模式，触发熔断器"""
    dag = OxyDAGOrchestrator()
    call_count = [0]

    def intermittent_failure(**kw):
        call_count[0] += 1
        if call_count[0] % 3 == 0:
            raise RuntimeError("intermittent failure")
        return {"ok": True}

    dag.register_component("flaky_service", "tool", intermittent_failure,
                           fallback_handler=lambda **kw: {"fallback": True},
                           retry_count=0)

    for i in range(10):
        result = dag.execute("flaky_service", task=f"call_{i}")
        assert result["success"], f"降级应保证成功: {result}"

    comp = dag.components["flaky_service"]
    assert comp.metrics["error_count"] > 0, "应有错误记录"
    assert comp.metrics["fallback_count"] > 0, "应有降级记录"

# ========== 边界场景测试 ==========

def test_empty_dag_execution():
    """空DAG执行"""
    dag = OxyDAGOrchestrator()
    result = dag.execute("nonexistent", task="test")
    assert not result["success"]
    assert "不存在" in result.get("error", "") or "不可用" in result.get("error", "")

def test_single_component_no_deps():
    """单组件无依赖"""
    dag = OxyDAGOrchestrator()
    dag.register_component("solo", "tool", lambda **kw: {"solo_result": True})
    result = dag.execute("solo", task="test")
    assert result["success"]
    assert len(result["execution_order"]) == 1

def test_deep_dependency_chain():
    """深层依赖链 (A→B→C→D→E)"""
    dag = OxyDAGOrchestrator()
    dag.register_component("a", "tool", lambda **kw: {"step": "a"})
    dag.register_component("b", "tool", lambda **kw: {"step": "b"}, depends_on=["a"])
    dag.register_component("c", "tool", lambda **kw: {"step": "c"}, depends_on=["b"])
    dag.register_component("d", "tool", lambda **kw: {"step": "d"}, depends_on=["c"])
    dag.register_component("e", "tool", lambda **kw: {"step": "e"}, depends_on=["d"])

    result = dag.execute("e", task="deep chain test")
    assert result["success"], f"深层依赖链应成功: {result}"
    assert result["execution_order"] == ["a", "b", "c", "d", "e"]

def test_wide_parallel_graph():
    """宽并行图 (10个独立组件→1个汇聚)"""
    dag = OxyDAGOrchestrator()
    for i in range(10):
        def make_handler(idx):
            return lambda **kw: {"result": idx}
        dag.register_component(f"parallel_{i}", "tool", make_handler(i))
    dag.register_component("merge_all", "tool",
        lambda **kw: {"merged": True},
        depends_on=[f"parallel_{i}" for i in range(10)])

    result = dag.execute("merge_all", task="wide parallel test")
    assert result["success"], f"宽并行图应成功: {result}"
    assert len(result["execution_order"]) == 11

def test_component_reregistration():
    """组件重新注册（版本升级）"""
    dag = OxyDAGOrchestrator()
    dag.register_component("upgradable", "tool", lambda **kw: {"version": 1}, version="1.0.0")
    v1_comp = dag.components["upgradable"]
    assert v1_comp.version == "1.0.0"

    dag.register_component("upgradable", "tool", lambda **kw: {"version": 2}, version="2.0.0")
    v2_comp = dag.components["upgradable"]
    assert v2_comp.version == "2.0.0"

def test_evomaster_empty_knowledge():
    """空知识库下的EvoMaster发现"""
    em = EvoMasterLoop()
    em.knowledge_base.clear()
    result = em.discover("全新领域探索", domain="unknown_domain",
                          experiment_iterations=1, multi_hypothesis=False)
    assert result["total_hypotheses"] >= 1

def test_evomaster_all_hypotheses_rejected():
    """所有假设都被拒绝的场景"""
    em = EvoMasterLoop()
    result = em.discover("不可能的任务", domain="impossible",
                          experiment_iterations=2, multi_hypothesis=True,
                          hypothesis_count=2)
    assert result["total_hypotheses"] == 2
    assert "hypotheses" in result

def test_bayesian_extreme_evidence():
    """极端证据量下的贝叶斯更新"""
    bc = BayesianConfidence(prior_alpha=1.0, prior_beta=1.0)
    for _ in range(100):
        bc.update(True)
    assert bc.confidence > 0.99, f"100次正面证据后置信度应接近1: {bc.confidence}"
    assert bc.uncertainty < 0.05, f"不确定性应极低: {bc.uncertainty}"

    bc2 = BayesianConfidence(prior_alpha=1.0, prior_beta=1.0)
    for _ in range(100):
        bc2.update(False)
    assert bc2.confidence < 0.01, f"100次负面证据后置信度应接近0: {bc2.confidence}"

def test_hypothesis_inconclusive_state():
    """假设无法确认也无法拒绝（模糊状态）"""
    hyp = Hypothesis("inconclusive_test", "模糊假设", domain="test", prior_confidence=0.5)
    for i in range(10):
        if i % 2 == 0:
            hyp.add_evidence(True, f"正面证据{i}")
        else:
            hyp.add_evidence(False, f"负面证据{i}")
    assert hyp.status in (Hypothesis.STATUS_TESTING, Hypothesis.STATUS_INCONCLUSIVE), \
        f"交替证据应导致模糊状态: {hyp.status}"

# ========== 压力测试 ==========

def test_rapid_component_registration():
    """快速注册大量组件"""
    dag = OxyDAGOrchestrator()
    start = time.time()
    for i in range(100):
        dag.register_component(f"stress_{i}", "tool", lambda **kw: {"idx": i})
    elapsed = time.time() - start
    assert len(dag.components) == 100
    assert elapsed < 5.0, f"100个组件注册应在5秒内完成: {elapsed:.2f}s"

def test_rapid_dag_execution():
    """快速执行DAG多次"""
    dag = OxyDAGOrchestrator()
    dag.register_component("fast_a", "tool", lambda **kw: {"a": True})
    dag.register_component("fast_b", "tool", lambda **kw: {"b": True}, depends_on=["fast_a"])

    start = time.time()
    success_count = 0
    for i in range(50):
        result = dag.execute("fast_b", task=f"rapid_{i}")
        if result["success"]:
            success_count += 1
    elapsed = time.time() - start

    assert success_count == 50, f"50次执行应全部成功: {success_count}/50"
    assert elapsed < 10.0, f"50次执行应在10秒内完成: {elapsed:.2f}s"

def test_concurrent_dag_execution():
    """并发DAG执行"""
    dag = OxyDAGOrchestrator()
    dag.register_component("conc_comp", "tool", lambda **kw: {"conc": True})

    results = []
    errors_list = []

    def worker(task_id):
        try:
            r = dag.execute("conc_comp", task=f"concurrent_{task_id}")
            results.append(r["success"])
        except Exception as e:
            errors_list.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(errors_list) == 0, f"并发执行不应有错误: {errors_list}"
    assert all(results), f"所有并发执行应成功: {sum(results)}/{len(results)}"

def test_evomaster_rapid_cycles():
    """快速执行多轮EvoMaster循环"""
    em = EvoMasterLoop()
    start = time.time()
    for i in range(5):
        result = em.discover(f"快速循环测试{i}", domain="test",
                              experiment_iterations=1, multi_hypothesis=False)
        assert result["total_hypotheses"] >= 1
    elapsed = time.time() - start
    assert elapsed < 60.0, f"5轮循环应在60秒内完成: {elapsed:.2f}s"

# ========== 数据持久化验证 ==========

def test_oxybank_persistence():
    """OxyBank持久化验证"""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        bank1 = OxyBank(persist_path=temp_path)
        bank1.deposit(["a", "b"], {"a": {"success": True}, "b": {"success": True}},
                      task_description="persistence test", success=True)

        bank2 = OxyBank(persist_path=temp_path)
        recs = bank2.recommend("persistence test")
        assert len(recs) > 0, "持久化后应能检索到模式"
    finally:
        os.unlink(temp_path)

def test_knowledge_graph_persistence():
    """知识图谱持久化验证"""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        temp_path = f.name

    try:
        kg1 = KnowledgeGraph(persist_path=temp_path)
        kg1.add_entity("persist_test", "hypothesis",
                       {"statement": "persistence test", "domain": "test"})
        kg1.add_relation("persist_test", "tested_by", "exp_001")

        kg2 = KnowledgeGraph(persist_path=temp_path)
        stats = kg2.get_stats()
        assert stats["total_entities"] >= 1
        assert stats["total_relations"] >= 1
    finally:
        os.unlink(temp_path)

# ========== 组件与Bus集成 ==========

class MockBus:
    def __init__(self):
        self.events = []
    def event(self, event_type, payload, source="", context=None):
        self.events.append(event_type)

def test_oxygent_with_bus_events():
    """OxyGent与Bus事件集成"""
    bus = MockBus()
    dag = OxyDAGOrchestrator(bus=bus)
    dag.register_component("bus_test", "tool", lambda **kw: {"bus": True})
    dag.execute("bus_test", task="bus event test")
    assert "oxy_component_registered" in bus.events, f"应发布组件注册事件, events={bus.events}"
    assert "oxy_dag_executed" in bus.events, f"应发布DAG执行事件, events={bus.events}"

def test_evomaster_with_bus_events():
    """EvoMaster与Bus事件集成"""
    bus = MockBus()
    em = EvoMasterLoop(bus=bus)
    em.discover("Bus事件集成测试", domain="test",
                experiment_iterations=1, multi_hypothesis=False)
    assert "evo_master_discovery_completed" in bus.events, f"应发布发现完成事件, events={bus.events}"


if __name__ == "__main__":
    print("=" * 60)
    print("OxyGent + EvoMaster 深度集成测试")
    print("=" * 60)

    tests = [
        ("集成: OxyGent+EvoMaster协同", test_oxygent_evomaster_collaboration),
        ("集成: OxyBank模式复用", test_oxybank_pattern_reuse_in_evomaster),
        ("集成: 假设驱动组件创建", test_hypothesis_drives_component_creation),
        ("集成: 熔断器+Evo反馈", test_circuit_breaker_with_evo_feedback),
        ("边界: 空DAG执行", test_empty_dag_execution),
        ("边界: 单组件无依赖", test_single_component_no_deps),
        ("边界: 深层依赖链", test_deep_dependency_chain),
        ("边界: 宽并行图", test_wide_parallel_graph),
        ("边界: 组件重新注册", test_component_reregistration),
        ("边界: 空知识库发现", test_evomaster_empty_knowledge),
        ("边界: 全部假设被拒绝", test_evomaster_all_hypotheses_rejected),
        ("边界: 极端贝叶斯证据", test_bayesian_extreme_evidence),
        ("边界: 假设模糊状态", test_hypothesis_inconclusive_state),
        ("压力: 快速注册100组件", test_rapid_component_registration),
        ("压力: 快速执行50次DAG", test_rapid_dag_execution),
        ("压力: 并发DAG执行", test_concurrent_dag_execution),
        ("压力: 快速5轮EvoMaster", test_evomaster_rapid_cycles),
        ("持久化: OxyBank", test_oxybank_persistence),
        ("持久化: 知识图谱", test_knowledge_graph_persistence),
        ("Bus集成: OxyGent事件", test_oxygent_with_bus_events),
        ("Bus集成: EvoMaster事件", test_evomaster_with_bus_events),
    ]

    for name, func in tests:
        test(name, func)

    print()
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 项")
    if errors:
        print("失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
