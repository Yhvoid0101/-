#!/usr/bin/env python3
"""EvoMaster 科学进化循环全面终端测试"""
import sys
import time
import json
import traceback
import os

sys.path.insert(0, ".")

from core.evomaster_loop import (
    EvoMasterLoop, Hypothesis, BayesianConfidence,
    KnowledgeGraph, StatisticalAnalyzer, ExperimentSandbox,
)

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


def test_basic_import():
    assert EvoMasterLoop is not None
    assert Hypothesis is not None
    assert BayesianConfidence is not None
    assert KnowledgeGraph is not None
    assert StatisticalAnalyzer is not None
    assert ExperimentSandbox is not None

def test_bayesian_confidence():
    bc = BayesianConfidence(prior_alpha=1.0, prior_beta=1.0)
    assert abs(bc.confidence - 0.5) < 0.01, f"初始置信度应为0.5, 实际{bc.confidence}"
    bc.update(True)
    assert bc.confidence > 0.5, f"正面证据后置信度应上升, 实际{bc.confidence}"
    bc.update(False)
    assert bc.confidence < 0.7, f"负面证据后置信度应下降, 实际{bc.confidence}"
    assert bc.uncertainty > 0, "不确定性应大于0"

def test_bayesian_convergence():
    bc = BayesianConfidence(prior_alpha=1.0, prior_beta=1.0)
    for _ in range(10):
        bc.update(True, weight=1.0)
    assert bc.confidence > 0.9, f"10次正面证据后置信度应>0.9, 实际{bc.confidence}"
    assert bc.uncertainty < 0.15, f"不确定性应降低, 实际{bc.uncertainty}"

def test_hypothesis_lifecycle():
    hyp = Hypothesis("test_001", "测试假设", domain="test")
    assert hyp.status == Hypothesis.STATUS_PROPOSED
    assert abs(hyp.confidence - 0.5) < 0.01
    hyp.add_evidence(True, "实验1成功")
    assert hyp.status == Hypothesis.STATUS_TESTING
    assert hyp.confidence > 0.5
    hyp.add_evidence(True, "实验2成功", weight=2.0)
    assert hyp.confidence > 0.7
    if hyp.confidence >= 0.85:
        assert hyp.status == Hypothesis.STATUS_CONFIRMED

def test_hypothesis_rejection():
    hyp = Hypothesis("test_002", "会被拒绝的假设", domain="test", prior_confidence=0.3)
    for i in range(5):
        hyp.add_evidence(False, f"实验{i+1}失败")
    assert hyp.status == Hypothesis.STATUS_REJECTED, f"5次失败后应被拒绝, status={hyp.status}"
    assert hyp.confidence < 0.2

def test_hypothesis_to_dict():
    hyp = Hypothesis("test_003", "序列化测试", domain="test")
    hyp.add_evidence(True, "证据1")
    d = hyp.to_dict()
    assert "id" in d
    assert "statement" in d
    assert "confidence" in d
    assert "bayesian" in d
    assert "evidence_for_count" in d
    assert d["evidence_for_count"] == 1

def test_knowledge_graph():
    kg = KnowledgeGraph()
    kg.add_entity("hyp_001", "hypothesis", {"statement": "test", "domain": "cache", "status": "confirmed"})
    kg.add_entity("exp_001", "experiment", {"success": True, "domain": "cache"})
    kg.add_relation("exp_001", "tests", "hyp_001")
    stats = kg.get_stats()
    assert stats["total_entities"] >= 2
    assert stats["total_relations"] >= 1

def test_knowledge_graph_cross_domain():
    kg = KnowledgeGraph()
    kg.add_entity("hyp_cache", "hypothesis", {"statement": "cache optimization", "domain": "cache", "status": "confirmed"})
    kg.add_entity("hyp_memory", "hypothesis", {"statement": "memory retrieval", "domain": "memory", "status": "confirmed"})
    patterns = kg.find_cross_domain_patterns("cache")
    assert isinstance(patterns, list)

def test_statistical_analyzer():
    sa = StatisticalAnalyzer()
    sample_a = [0.3, 0.4, 0.5, 0.6, 0.7]
    sample_b = [0.6, 0.7, 0.8, 0.9, 1.0]
    result = sa.t_test(sample_a, sample_b)
    assert "significant" in result
    assert "p_value" in result
    assert "t_statistic" in result
    assert "mean_diff" in result

def test_statistical_analyzer_insufficient_samples():
    sa = StatisticalAnalyzer()
    result = sa.t_test([1.0], [2.0])
    assert result["significant"] is False
    assert "样本量不足" in result.get("error", "")

def test_effect_size():
    sa = StatisticalAnalyzer()
    sample_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    sample_b = [6.0, 7.0, 8.0, 9.0, 10.0]
    es = sa.effect_size(sample_a, sample_b)
    assert es != 0, "效应量不应为0"

def test_experiment_sandbox():
    sandbox = ExperimentSandbox(timeout=10)
    safe_code = """
import json
result = {"value": 42}
print(json.dumps(result))
"""
    result = sandbox.execute(safe_code, "test_safe_001")
    assert result["success"], f"安全代码应执行成功: {result}"
    assert result["data"]["value"] == 42

def test_experiment_sandbox_dangerous():
    sandbox = ExperimentSandbox()
    dangerous_code = "import os; os.system('echo hack')"
    result = sandbox.execute(dangerous_code, "test_danger_001")
    assert not result["success"], "危险代码应被拒绝"
    assert "安全检查失败" in result.get("error", "")

def test_experiment_sandbox_with_seed():
    sandbox = ExperimentSandbox(timeout=10)
    code = """
import json, random
values = [random.random() for _ in range(5)]
print(json.dumps({"values": values}))
"""
    result1 = sandbox.execute(code, "seed_test_1", seed=42)
    result2 = sandbox.execute(code, "seed_test_2", seed=42)
    assert result1["success"] and result2["success"]
    assert result1["data"]["values"] == result2["data"]["values"], "相同seed应产生相同结果"

def test_evomaster_discover():
    em = EvoMasterLoop()
    result = em.discover(
        "如何提升缓存命中率？",
        domain="cache",
        experiment_iterations=2,
        multi_hypothesis=False,
    )
    assert "research_question" in result
    assert "total_hypotheses" in result
    assert "confirmed_hypotheses" in result
    assert "rejected_hypotheses" in result
    assert result["total_hypotheses"] >= 1

def test_evomaster_multi_hypothesis():
    em = EvoMasterLoop()
    result = em.discover(
        "如何优化API延迟？",
        domain="api",
        experiment_iterations=3,
        multi_hypothesis=True,
        hypothesis_count=3,
    )
    assert result["total_hypotheses"] == 3, f"应生成3个假设, 实际{result['total_hypotheses']}"
    assert "hypotheses" in result
    for h in result["hypotheses"]:
        assert "confidence" in h
        assert "status" in h
        assert "bayesian" in h

def test_evomaster_knowledge_persistence():
    em = EvoMasterLoop()
    em.discover("测试知识持久化", domain="test", experiment_iterations=1, multi_hypothesis=False)
    assert len(em.knowledge_base) > 0, "知识库应有记录"

def test_evomaster_knowledge_graph_integration():
    em = EvoMasterLoop()
    em.discover("知识图谱集成测试", domain="test", experiment_iterations=2, multi_hypothesis=False)
    kg_stats = em.knowledge_graph.get_stats()
    assert kg_stats["total_entities"] > 0, "知识图谱应有实体"
    assert kg_stats["total_relations"] > 0, "知识图谱应有关系"

def test_evomaster_stats():
    em = EvoMasterLoop()
    em.discover("统计测试", domain="test", experiment_iterations=1, multi_hypothesis=False)
    stats = em.get_stats()
    assert "total_hypotheses" in stats
    assert "knowledge_entries" in stats
    assert "knowledge_graph" in stats
    assert "cycle_count" in stats

def test_evomaster_daemon():
    em = EvoMasterLoop()
    em.start_daemon(domains=["test_domain"], interval=1)
    assert em._daemon_running is True
    time.sleep(2)
    em.stop_daemon()
    assert em._daemon_running is False

def test_hypothesis_evidence_accumulation():
    hyp = Hypothesis("evi_test", "证据累积测试", domain="test")
    assert len(hyp.evidence_for) == 0
    assert len(hyp.evidence_against) == 0
    hyp.add_evidence(True, "支持证据1")
    hyp.add_evidence(True, "支持证据2")
    hyp.add_evidence(False, "反对证据1")
    assert len(hyp.evidence_for) == 2
    assert len(hyp.evidence_against) == 1
    assert hyp.experiment_count == 3

def test_cross_domain_transfer():
    em = EvoMasterLoop()
    em.discover("缓存优化", domain="cache", experiment_iterations=3, multi_hypothesis=False)
    em.discover("记忆检索", domain="memory", experiment_iterations=3, multi_hypothesis=False)
    patterns = em.knowledge_graph.find_cross_domain_patterns("memory")
    assert isinstance(patterns, list)

def test_sandbox_timeout():
    sandbox = ExperimentSandbox(timeout=2)
    timeout_code = """
import time
time.sleep(10)
print('should not reach here')
"""
    result = sandbox.execute(timeout_code, "timeout_test")
    assert not result["success"], "超时代码应失败"
    assert "超时" in result.get("error", "")

def test_sandbox_cleanup():
    sandbox = ExperimentSandbox()
    sandbox.execute("print('cleanup test')", "cleanup_test")
    sandbox.cleanup("cleanup_test")
    sandbox_path = sandbox.sandbox_dir / "exp_cleanup_test"
    assert not sandbox_path.exists(), "清理后沙箱目录应不存在"


if __name__ == "__main__":
    print("=" * 60)
    print("EvoMaster 科学进化循环全面终端测试")
    print("=" * 60)

    tests = [
        ("基础导入", test_basic_import),
        ("贝叶斯置信度-基础", test_bayesian_confidence),
        ("贝叶斯置信度-收敛", test_bayesian_convergence),
        ("假设生命周期", test_hypothesis_lifecycle),
        ("假设拒绝", test_hypothesis_rejection),
        ("假设序列化", test_hypothesis_to_dict),
        ("知识图谱-基础", test_knowledge_graph),
        ("知识图谱-跨域", test_knowledge_graph_cross_domain),
        ("统计检验-t检验", test_statistical_analyzer),
        ("统计检验-样本不足", test_statistical_analyzer_insufficient_samples),
        ("统计检验-效应量", test_effect_size),
        ("实验沙箱-安全执行", test_experiment_sandbox),
        ("实验沙箱-危险代码", test_experiment_sandbox_dangerous),
        ("实验沙箱-确定性种子", test_experiment_sandbox_with_seed),
        ("EvoMaster-单假设发现", test_evomaster_discover),
        ("EvoMaster-多假设对比", test_evomaster_multi_hypothesis),
        ("EvoMaster-知识持久化", test_evomaster_knowledge_persistence),
        ("EvoMaster-知识图谱集成", test_evomaster_knowledge_graph_integration),
        ("EvoMaster-统计信息", test_evomaster_stats),
        ("EvoMaster-守护进程", test_evomaster_daemon),
        ("假设证据累积", test_hypothesis_evidence_accumulation),
        ("跨域迁移", test_cross_domain_transfer),
        ("沙箱超时", test_sandbox_timeout),
        ("沙箱清理", test_sandbox_cleanup),
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
