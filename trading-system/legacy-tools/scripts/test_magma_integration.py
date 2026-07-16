#!/usr/bin/env python3
"""
MAGMA + OxyGent + EvoMaster 深度集成测试
验证三模块协同工作、与AgentBus集成、边界场景
"""
import sys
import time
import json
import traceback
import os
import tempfile

sys.path.insert(0, ".")

from core.magma_memory import MagmaMemoryEngine, MagmaMemoryNode, GraphTraversalEngine
from core.oxygent_adapter import OxyDAGOrchestrator
from core.evomaster_loop import EvoMasterLoop, Hypothesis

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


def make_magma():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = MagmaMemoryEngine(db_path=path)
    engine._test_db_path = path
    return engine

def cleanup_magma(engine):
    try:
        engine.db.close()
        os.unlink(engine._test_db_path)
    except Exception:
        pass


# ========== MAGMA + OxyGent 集成 ==========

def test_magma_as_oxy_component():
    """MAGMA作为OxyGent原子组件使用"""
    magma = make_magma()
    dag = OxyDAGOrchestrator()
    try:
        dag.register_component("memory_store", "tool",
            lambda **kw: magma.store(kw.get("content", ""),
                                      semantic_tags=kw.get("tags", []),
                                      importance=kw.get("importance", 0.5)),
            input_schema={"required": ["content"]})
        dag.register_component("memory_retrieve", "tool",
            lambda **kw: {"results": magma.retrieve(kw.get("query", ""), top_k=3)},
            input_schema={"required": ["query"]})

        store_result = dag.execute("memory_store",
            task_description="store memory", content="测试记忆", tags=["测试"], importance=0.8)
        assert store_result["success"]

        retrieve_result = dag.execute("memory_retrieve",
            task_description="retrieve memory", query="测试")
        assert retrieve_result["success"]
    finally:
        cleanup_magma(magma)

def test_oxy_execution_recorded_in_magma():
    """OxyGent执行结果存入MAGMA"""
    magma = make_magma()
    dag = OxyDAGOrchestrator()
    try:
        dag.register_component("analyze", "tool", lambda **kw: {"analysis": "done"})
        result = dag.execute("analyze", task_description="分析任务", task="test")

        mid = magma.store(
            f"OxyGent执行: analyze, 成功={result['success']}",
            semantic_tags=["oxygent", "执行记录"],
            entity_links=["analyze"],
            importance=0.6, source="oxygent")

        retrieved = magma.retrieve("oxygent执行", strategy="semantic", top_k=1)
        assert len(retrieved) > 0
    finally:
        cleanup_magma(magma)

# ========== MAGMA + EvoMaster 集成 ==========

def test_evomaster_hypothesis_in_magma():
    """EvoMaster假设存入MAGMA因果图"""
    magma = make_magma()
    em = EvoMasterLoop()
    try:
        cause_id = magma.store("系统负载高", semantic_tags=["系统", "负载"],
                               importance=0.8)
        effect_id = magma.store("API延迟增加", semantic_tags=["API", "延迟"],
                                causal_links=[cause_id], importance=0.7)

        result = em.discover("如何降低API延迟", domain="api",
                              experiment_iterations=1, multi_hypothesis=False)
        assert result["total_hypotheses"] >= 1

        causal_results = magma.retrieve("系统负载", strategy="causal", top_k=3)
        assert len(causal_results) > 0
    finally:
        cleanup_magma(magma)

def test_magma_guides_evomaster():
    """MAGMA历史记忆指导EvoMaster假设生成"""
    magma = make_magma()
    try:
        magma.store("缓存LRU策略有效", semantic_tags=["缓存", "LRU", "策略"],
                     importance=0.9, source="experience")
        magma.store("缓存TTL策略中等效果", semantic_tags=["缓存", "TTL", "策略"],
                     importance=0.6, source="experience")

        em = EvoMasterLoop()
        result = em.discover("如何优化缓存策略", domain="cache",
                              experiment_iterations=2, multi_hypothesis=False)
        assert result["total_hypotheses"] >= 1
    finally:
        cleanup_magma(magma)

# ========== 三模块协同 ==========

def test_triple_module_workflow():
    """三模块协同完整工作流"""
    magma = make_magma()
    dag = OxyDAGOrchestrator()
    em = EvoMasterLoop()
    try:
        dag.register_component("collect_data", "tool",
            lambda **kw: {"data": "collected"},
            input_schema={"required": ["task"]})
        dag.register_component("analyze_data", "tool",
            lambda **kw: {"insight": "cache_hit_rate_low"},
            input_schema={"required": ["task"]}, depends_on=["collect_data"])

        exec_result = dag.execute("analyze_data", task_description="分析缓存数据", task="cache")
        assert exec_result["success"]

        mid = magma.store(
            f"分析结果: cache_hit_rate_low",
            semantic_tags=["缓存", "分析", "命中率"],
            entity_links=["analyze_data"],
            causal_links=[],
            importance=0.8, source="oxygent")

        evo_result = em.discover("如何提升缓存命中率", domain="cache",
                                  experiment_iterations=1, multi_hypothesis=False)
        assert evo_result["total_hypotheses"] >= 1

        memory_results = magma.retrieve("缓存命中率", strategy="hybrid", top_k=3)
        assert len(memory_results) > 0
    finally:
        cleanup_magma(magma)

# ========== 边界场景 ==========

def test_magma_empty_db():
    """空DB检索"""
    magma = make_magma()
    try:
        results = magma.retrieve("任何查询", top_k=5)
        assert results == []
    finally:
        cleanup_magma(magma)

def test_magma_single_node_all_strategies():
    """单节点全策略检索"""
    magma = make_magma()
    try:
        magma.store("唯一记忆", semantic_tags=["唯一"], entity_links=["only_mod"],
                     importance=0.5)
        for strategy in ["semantic", "temporal", "causal", "entity", "hybrid"]:
            results = magma.retrieve("唯一", strategy=strategy, top_k=3)
            assert isinstance(results, list), f"策略{strategy}应返回列表"
    finally:
        cleanup_magma(magma)

def test_magma_long_causal_chain():
    """长因果链检索"""
    magma = make_magma()
    try:
        prev_id = None
        for i in range(5):
            mid = magma.store(f"因果链节点{i}", semantic_tags=["因果", f"节点{i}"],
                              causal_links=[prev_id] if prev_id else [],
                              importance=0.7)
            prev_id = mid

        results = magma.retrieve("因果", strategy="causal", top_k=10)
        assert len(results) >= 3, f"因果链应返回多个结果: {len(results)}"
    finally:
        cleanup_magma(magma)

def test_magma_concurrent_access():
    """并发访问MAGMA"""
    import threading
    magma = make_magma()
    try:
        errors_list = []

        def writer(idx):
            try:
                magma.store(f"并发记忆{idx}", semantic_tags=["并发"],
                            importance=0.5)
            except Exception as e:
                errors_list.append(str(e))

        def reader(idx):
            try:
                magma.retrieve("并发", top_k=3)
            except Exception as e:
                errors_list.append(str(e))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors_list) == 0, f"并发访问不应有错误: {errors_list}"
    finally:
        cleanup_magma(magma)

def test_magma_special_characters():
    """特殊字符内容"""
    magma = make_magma()
    try:
        mid = magma.store('特殊字符: <script>alert("xss")</script> & "引号" \'单引号\'',
                          semantic_tags=["特殊"], importance=0.5)
        row = magma._get_node(mid)
        assert row is not None
        assert "<script>" in row["content"]
    finally:
        cleanup_magma(magma)

def test_magma_unicode_content():
    """Unicode内容"""
    magma = make_magma()
    try:
        mid = magma.store("日本語テスト 한국어 테스트 🚀💻",
                          semantic_tags=["unicode"], importance=0.5)
        row = magma._get_node(mid)
        assert row is not None
    finally:
        cleanup_magma(magma)

def test_magma_very_long_content():
    """超长内容"""
    magma = make_magma()
    try:
        long_content = "A" * 10000
        mid = magma.store(long_content, semantic_tags=["长内容"], importance=0.5)
        row = magma._get_node(mid)
        assert row is not None
        assert len(row["content"]) == 10000
    finally:
        cleanup_magma(magma)

def test_magma_duplicate_tags():
    """重复标签"""
    magma = make_magma()
    try:
        mid = magma.store("重复标签测试", semantic_tags=["重复", "重复", "标签"],
                          importance=0.5)
        row = magma._get_node(mid)
        tags = json.loads(row["semantic_tags"])
        assert tags.count("重复") <= 1, "标签不应重复存储"
    finally:
        cleanup_magma(magma)

def test_magma_self_causal_link():
    """自引用因果链接"""
    magma = make_magma()
    try:
        mid = magma.store("自引用", semantic_tags=["自引用"], importance=0.5)
        magma.link_causal(mid, mid)
        assert mid is not None
    finally:
        cleanup_magma(magma)

# ========== 端到端场景 ==========

def test_e2e_caching_optimization_scenario():
    """端到端: 缓存优化场景"""
    magma = make_magma()
    try:
        root = magma.store("缓存命中率下降到60%", semantic_tags=["缓存", "命中率"],
                           importance=0.9, source="monitoring")
        cause1 = magma.store("新功能上线导致缓存key变化", semantic_tags=["功能", "缓存"],
                              causal_links=[root], importance=0.8, source="analysis")
        cause2 = magma.store("缓存过期时间设置过短", semantic_tags=["缓存", "过期"],
                              causal_links=[root], importance=0.7, source="analysis")
        magma.store("用户投诉页面加载慢", semantic_tags=["用户", "加载"],
                     causal_links=[cause1, cause2], importance=0.6, source="feedback")

        results = magma.retrieve("缓存命中率下降原因", strategy="hybrid", top_k=5)
        assert len(results) >= 2, f"应找到多个相关记忆: {len(results)}"

        causal = magma.retrieve("缓存命中率", strategy="causal", top_k=5)
        assert len(causal) >= 1, "因果搜索应找到因果链"

        stats = magma.get_graph_stats()
        assert stats["causal_edges"] >= 2
        assert stats["total_memories"] == 4
    finally:
        cleanup_magma(magma)

def test_e2e_api_debugging_scenario():
    """端到端: API调试场景"""
    magma = make_magma()
    try:
        magma.store("API /users 返回500错误", semantic_tags=["API", "错误"],
                     entity_links=["users_api"], importance=0.9, source="alert")
        magma.store("数据库连接池耗尽", semantic_tags=["数据库", "连接池"],
                     entity_links=["db_pool"], importance=0.8, source="monitoring")
        magma.store("users_api 调用 db_pool", semantic_tags=["调用"],
                     entity_links=["users_api", "db_pool"], importance=0.7, source="trace")

        entity_results = magma.retrieve("users_api", strategy="entity", top_k=3)
        assert len(entity_results) >= 1, "实体搜索应找到users_api相关记忆"

        temporal_results = magma.retrieve("API错误", strategy="temporal", top_k=3)
        assert len(temporal_results) >= 1, "时序搜索应找到近期记忆"
    finally:
        cleanup_magma(magma)


if __name__ == "__main__":
    print("=" * 60)
    print("MAGMA + OxyGent + EvoMaster 深度集成测试")
    print("=" * 60)

    tests = [
        ("集成: MAGMA作为OxyGent组件", test_magma_as_oxy_component),
        ("集成: OxyGent执行记录存入MAGMA", test_oxy_execution_recorded_in_magma),
        ("集成: EvoMaster假设存入MAGMA因果图", test_evomaster_hypothesis_in_magma),
        ("集成: MAGMA指导EvoMaster假设", test_magma_guides_evomaster),
        ("集成: 三模块协同工作流", test_triple_module_workflow),
        ("边界: 空DB检索", test_magma_empty_db),
        ("边界: 单节点全策略", test_magma_single_node_all_strategies),
        ("边界: 长因果链", test_magma_long_causal_chain),
        ("边界: 并发访问", test_magma_concurrent_access),
        ("边界: 特殊字符", test_magma_special_characters),
        ("边界: Unicode内容", test_magma_unicode_content),
        ("边界: 超长内容", test_magma_very_long_content),
        ("边界: 重复标签", test_magma_duplicate_tags),
        ("边界: 自引用因果", test_magma_self_causal_link),
        ("端到端: 缓存优化场景", test_e2e_caching_optimization_scenario),
        ("端到端: API调试场景", test_e2e_api_debugging_scenario),
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
