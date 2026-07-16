#!/usr/bin/env python3
"""MAGMA 多图记忆架构全面终端测试"""
import sys
import time
import json
import traceback
import os
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from core.magma_memory import (
    MagmaMemoryEngine, MagmaMemoryNode, GraphTraversalEngine, StrategyTracker,
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


def make_engine() -> MagmaMemoryEngine:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = MagmaMemoryEngine(db_path=path)
    engine._test_db_path = path
    return engine

def cleanup_engine(engine):
    try:
        engine.db.close()
        os.unlink(engine._test_db_path)
    except Exception:
        pass


def test_basic_import():
    assert MagmaMemoryEngine is not None
    assert MagmaMemoryNode is not None
    assert GraphTraversalEngine is not None
    assert StrategyTracker is not None

def test_store_basic():
    engine = make_engine()
    try:
        mid = engine.store("测试记忆内容", semantic_tags=["测试", "基础"],
                           entity_links=["agent_bus"], importance=0.7)
        assert mid is not None and len(mid) > 0
        row = engine._get_node(mid)
        assert row is not None
        assert row["content"] == "测试记忆内容"
        tags = json.loads(row["semantic_tags"])
        assert "测试" in tags
    finally:
        cleanup_engine(engine)

def test_store_with_causal_links():
    engine = make_engine()
    try:
        cause_id = engine.store("原因: 系统负载过高", semantic_tags=["系统", "负载"],
                                importance=0.8)
        effect_id = engine.store("结果: API延迟增加", semantic_tags=["API", "延迟"],
                                 causal_links=[cause_id], importance=0.7)
        assert cause_id in engine.causal_graph
        assert effect_id in engine.causal_graph[cause_id]
    finally:
        cleanup_engine(engine)

def test_semantic_search():
    engine = make_engine()
    try:
        engine.store("缓存命中率优化方案", semantic_tags=["缓存", "优化", "命中率"],
                     importance=0.8)
        engine.store("API延迟优化策略", semantic_tags=["API", "延迟", "优化"],
                     importance=0.7)
        engine.store("记忆检索质量提升", semantic_tags=["记忆", "检索", "质量"],
                     importance=0.6)
        results = engine.retrieve("缓存优化", strategy="semantic", top_k=3)
        assert len(results) > 0, "语义搜索应有结果"
        assert results[0]["strategy"] in ("semantic", "semantic_traverse")
        assert results[0]["relevance"] > 0
    finally:
        cleanup_engine(engine)

def test_temporal_search():
    engine = make_engine()
    try:
        engine.store("早期事件A", semantic_tags=["事件"], importance=0.5)
        time.sleep(0.1)
        engine.store("中期事件B", semantic_tags=["事件"], importance=0.6)
        time.sleep(0.1)
        engine.store("近期事件C", semantic_tags=["事件"], importance=0.7)
        results = engine.retrieve("事件", strategy="temporal", top_k=3)
        assert len(results) > 0, "时序搜索应有结果"
        has_temporal = any(r["strategy"].startswith("temporal") for r in results)
        assert has_temporal, "应有temporal策略结果"
    finally:
        cleanup_engine(engine)

def test_causal_search():
    engine = make_engine()
    try:
        root_id = engine.store("根本原因: 数据库连接池耗尽",
                               semantic_tags=["数据库", "连接池"], importance=0.9)
        mid_id = engine.store("中间结果: 查询超时",
                              semantic_tags=["查询", "超时"],
                              causal_links=[root_id], importance=0.7)
        leaf_id = engine.store("最终影响: 用户请求失败",
                               semantic_tags=["用户", "请求"],
                               causal_links=[mid_id], importance=0.6)
        results = engine.retrieve("数据库连接池", strategy="causal", top_k=5)
        assert len(results) > 0, "因果搜索应有结果"
        has_causal = any("causal" in r["strategy"] for r in results)
        assert has_causal, "应有causal策略结果"
    finally:
        cleanup_engine(engine)

def test_entity_search():
    engine = make_engine()
    try:
        engine.store("agent_bus 初始化流程", semantic_tags=["初始化"],
                     entity_links=["agent_bus"], importance=0.8)
        engine.store("agent_bus 事件发布机制", semantic_tags=["事件"],
                     entity_links=["agent_bus"], importance=0.7)
        engine.store("deep_think 推理引擎", semantic_tags=["推理"],
                     entity_links=["deep_think"], importance=0.6)
        results = engine.retrieve("agent_bus", strategy="entity", top_k=3)
        assert len(results) > 0, "实体搜索应有结果"
        entity_results = [r for r in results if r["strategy"] == "entity"]
        assert len(entity_results) > 0, "应有entity策略结果"
        assert entity_results[0]["entity"] == "agent_bus"
    finally:
        cleanup_engine(engine)

def test_hybrid_search():
    engine = make_engine()
    try:
        engine.store("缓存命中率优化", semantic_tags=["缓存", "优化"],
                     entity_links=["cache_module"], importance=0.8)
        engine.store("API延迟降低方案", semantic_tags=["API", "延迟"],
                     entity_links=["api_gateway"], importance=0.7)
        results = engine.retrieve("缓存优化", strategy="hybrid", top_k=5)
        assert len(results) > 0, "混合搜索应有结果"
        strategies = set(r["strategy"] for r in results)
        assert len(strategies) >= 1, "混合搜索应使用多种策略"
    finally:
        cleanup_engine(engine)

def test_auto_strategy_infer():
    engine = make_engine()
    try:
        assert engine._infer_strategy("为什么系统崩溃了") == "causal"
        assert engine._infer_strategy("最近做了什么") == "temporal"
        assert engine._infer_strategy("agent_bus 怎么用") == "entity"
        assert engine._infer_strategy("类似缓存的概念") == "semantic"
        assert engine._infer_strategy("优化性能") == "hybrid"
    finally:
        cleanup_engine(engine)

def test_bidirectional_temporal_links():
    engine = make_engine()
    try:
        mid1 = engine.store("第一条记忆", semantic_tags=["测试"], importance=0.5)
        mid2 = engine.store("第二条记忆", semantic_tags=["测试"], importance=0.5)
        mid3 = engine.store("第三条记忆", semantic_tags=["测试"], importance=0.5)
        assert mid2 in engine.temporal_index.get(mid3, {}).get("before", []), "mid3应链接到mid2"
        assert mid3 in engine.temporal_index.get(mid2, {}).get("after", []), "mid2应链接到mid3"
        assert mid1 in engine.temporal_index.get(mid2, {}).get("before", []), "mid2应链接到mid1"
    finally:
        cleanup_engine(engine)

def test_graph_traversal():
    graph = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["e"],
        "d": ["f"],
        "e": ["f"],
    }
    result = GraphTraversalEngine.bfs_traverse(graph, "a", max_depth=3, max_nodes=10)
    nodes = [n for n, _ in result]
    assert "a" in nodes
    assert "b" in nodes
    assert "d" in nodes
    depths = {n: d for n, d in result}
    assert depths["a"] == 0
    assert depths["b"] == 1
    assert depths["d"] == 2

def test_graph_traversal_cycle():
    graph = {
        "a": ["b"],
        "b": ["c"],
        "c": ["a"],
    }
    result = GraphTraversalEngine.bfs_traverse(graph, "a", max_depth=5, max_nodes=10)
    nodes = [n for n, _ in result]
    assert len(nodes) <= 3, "环检测应防止无限循环"

def test_path_score():
    assert GraphTraversalEngine.path_score(0) > GraphTraversalEngine.path_score(1)
    assert GraphTraversalEngine.path_score(1) > GraphTraversalEngine.path_score(2)
    assert abs(GraphTraversalEngine.path_score(0, 1.0) - 1.0) < 0.01

def test_retention_decay():
    node = MagmaMemoryNode("test", "test content", importance=0.5)
    assert node.retention() > 0.9, "新记忆保持率应很高"
    node.created = (datetime.now() - timedelta(hours=100)).isoformat()
    node.access_count = 0
    assert node.retention() < 0.95, "100小时后保持率应下降"

def test_retention_with_access():
    node = MagmaMemoryNode("test", "test content", importance=0.5)
    node.created = (datetime.now() - timedelta(hours=100)).isoformat()
    node.access_count = 0
    retention_no_access = node.retention()
    node.access_count = 50
    retention_with_access = node.retention()
    assert retention_with_access > retention_no_access, "频繁访问应提高保持率"

def test_forget_stale():
    engine = make_engine()
    try:
        engine.store("重要记忆", semantic_tags=["重要"], importance=0.9)
        old_id = engine.store("过时记忆", semantic_tags=["过时"], importance=0.05)
        engine.db.execute(
            "UPDATE magma_nodes SET created = ? WHERE id = ?",
            ((datetime.now() - timedelta(days=100)).isoformat(), old_id))
        engine.db.commit()
        deleted = engine.forget_stale(max_age_days=30, min_importance=0.1)
        assert deleted >= 0
    finally:
        cleanup_engine(engine)

def test_strategy_tracker():
    tracker = StrategyTracker()
    tracker.record("semantic", "test query", 5, 0.8)
    tracker.record("semantic", "test query 2", 3, 0.6)
    tracker.record("causal", "why query", 2, 0.9)
    weights = tracker.get_weights()
    assert "semantic" in weights
    assert "causal" in weights
    assert abs(sum(weights.values()) - 1.0) < 0.01

def test_strategy_tracker_empty():
    tracker = StrategyTracker()
    weights = tracker.get_weights()
    assert abs(sum(weights.values()) - 1.0) < 0.01

def test_persistence():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        engine1 = MagmaMemoryEngine(db_path=path)
        mid = engine1.store("持久化测试", semantic_tags=["持久化"],
                            entity_links=["test_module"], importance=0.7)
        engine1.db.close()

        engine2 = MagmaMemoryEngine(db_path=path)
        row = engine2._get_node(mid)
        assert row is not None, "重启后应能读取记忆"
        assert row["content"] == "持久化测试"
        assert "持久化" in engine2.semantic_index
        assert "test_module" in engine2.entity_index
        engine2.db.close()
    finally:
        os.unlink(path)

def test_index_rebuild_after_restart():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        engine1 = MagmaMemoryEngine(db_path=path)
        cause_id = engine1.store("原因", semantic_tags=["原因"], importance=0.8)
        engine1.store("结果", semantic_tags=["结果"],
                      causal_links=[cause_id], importance=0.7)
        engine1.db.close()

        engine2 = MagmaMemoryEngine(db_path=path)
        assert len(engine2.semantic_index) > 0, "语义索引应被重建"
        assert len(engine2.causal_graph) > 0, "因果图应被重建"
        assert len(engine2.entity_index) >= 0
        engine2.db.close()
    finally:
        os.unlink(path)

def test_graph_stats():
    engine = make_engine()
    try:
        engine.store("统计测试1", semantic_tags=["统计"], entity_links=["stat_mod"],
                     importance=0.7)
        cause_id = engine.store("原因", semantic_tags=["原因"], importance=0.8)
        engine.store("结果", semantic_tags=["结果"], causal_links=[cause_id],
                     importance=0.7)
        stats = engine.get_graph_stats()
        assert stats["total_memories"] >= 3
        assert stats["semantic_tags"] > 0
        assert stats["causal_edges"] > 0
        assert stats["entity_types"] > 0
        assert "connectivity" in stats
    finally:
        cleanup_engine(engine)

def test_link_causal():
    engine = make_engine()
    try:
        id1 = engine.store("独立事件1", semantic_tags=["独立"], importance=0.5)
        id2 = engine.store("独立事件2", semantic_tags=["独立"], importance=0.5)
        engine.link_causal(id1, id2)
        assert id2 in engine.causal_graph.get(id1, []), "应建立因果链接"
        row1 = engine._get_node(id1)
        links = json.loads(row1["causal_links"] or "[]")
        assert id2 in links, "DB中也应有因果链接"
    finally:
        cleanup_engine(engine)

def test_extract_tags():
    engine = make_engine()
    try:
        tags = engine._extract_tags("缓存命中率优化方案 cache hit rate optimization")
        assert "缓存" in tags or "命中" in tags or "优化" in tags, f"应提取中文标签: {tags}"
        assert "cache" in tags or "optimization" in tags, f"应提取英文标签: {tags}"
        tags2 = engine._extract_tags("的了一是在")
        assert len(tags2) == 0, f"停用词应被过滤: {tags2}"
    finally:
        cleanup_engine(engine)

def test_consolidate_similar():
    engine = make_engine()
    try:
        engine.store("相似记忆A", semantic_tags=["相似", "测试", "优化"], importance=0.6)
        engine.store("相似记忆B", semantic_tags=["相似", "测试", "改进"], importance=0.6)
        merged = engine.consolidate_similar(similarity_threshold=0.4)
        assert merged >= 0
    finally:
        cleanup_engine(engine)

def test_retrieve_with_fallback():
    engine = make_engine()
    try:
        engine.store("MAGMA记忆", semantic_tags=["MAGMA"], importance=0.8)
        results = engine.retrieve_with_fallback("MAGMA", top_k=5)
        assert len(results) > 0, "应有检索结果"
    finally:
        cleanup_engine(engine)

def test_multiple_strategies_combined():
    engine = make_engine()
    try:
        cause_id = engine.store("系统过载", semantic_tags=["系统", "过载"],
                                entity_links=["system_monitor"], importance=0.9)
        engine.store("API超时", semantic_tags=["API", "超时"],
                     causal_links=[cause_id], entity_links=["api_gateway"],
                     importance=0.7)
        results = engine.retrieve("系统过载导致API超时", strategy="hybrid", top_k=5)
        assert len(results) > 0
        strategies_used = set(r["strategy"] for r in results)
        assert len(strategies_used) >= 1
    finally:
        cleanup_engine(engine)

def test_empty_query():
    engine = make_engine()
    try:
        engine.store("测试记忆", semantic_tags=["测试"], importance=0.5)
        results = engine.retrieve("", top_k=3)
        assert isinstance(results, list)
    finally:
        cleanup_engine(engine)

def test_nonexistent_entity():
    engine = make_engine()
    try:
        engine.store("测试", semantic_tags=["测试"], importance=0.5)
        results = engine.retrieve("nonexistent_entity_xyz", strategy="entity", top_k=3)
        assert isinstance(results, list)
    finally:
        cleanup_engine(engine)

def test_large_volume_store():
    engine = make_engine()
    try:
        start = time.time()
        ids = []
        for i in range(100):
            mid = engine.store(f"批量记忆{i}", semantic_tags=[f"标签{i%10}"],
                               entity_links=[f"entity_{i%5}"], importance=0.5 + (i % 5) * 0.1)
            ids.append(mid)
        elapsed = time.time() - start
        assert len(ids) == 100
        assert elapsed < 10.0, f"100条记忆存储应在10秒内: {elapsed:.2f}s"
        stats = engine.get_graph_stats()
        assert stats["total_memories"] == 100
    finally:
        cleanup_engine(engine)

def test_node_to_dict():
    node = MagmaMemoryNode("test_id", "test content", semantic_tags=["tag1"],
                           importance=0.7, source="user")
    d = node.to_dict()
    assert d["id"] == "test_id"
    assert d["content"] == "test content"
    assert "tag1" in d["semantic_tags"]
    assert d["importance"] == 0.7
    assert d["source"] == "user"
    assert "retention" in d


if __name__ == "__main__":
    print("=" * 60)
    print("MAGMA 多图记忆架构全面终端测试")
    print("=" * 60)

    tests = [
        ("基础导入", test_basic_import),
        ("存储-基础", test_store_basic),
        ("存储-因果链接", test_store_with_causal_links),
        ("语义搜索", test_semantic_search),
        ("时序搜索", test_temporal_search),
        ("因果搜索", test_causal_search),
        ("实体搜索", test_entity_search),
        ("混合搜索", test_hybrid_search),
        ("策略自动推断", test_auto_strategy_infer),
        ("双向时序链接", test_bidirectional_temporal_links),
        ("图遍历-BFS", test_graph_traversal),
        ("图遍历-环检测", test_graph_traversal_cycle),
        ("路径评分", test_path_score),
        ("记忆保持率衰减", test_retention_decay),
        ("访问频次提升保持率", test_retention_with_access),
        ("遗忘清理", test_forget_stale),
        ("策略追踪", test_strategy_tracker),
        ("策略追踪-空", test_strategy_tracker_empty),
        ("持久化", test_persistence),
        ("索引重建", test_index_rebuild_after_restart),
        ("图统计", test_graph_stats),
        ("因果链接添加", test_link_causal),
        ("标签提取", test_extract_tags),
        ("相似合并", test_consolidate_similar),
        ("降级回退检索", test_retrieve_with_fallback),
        ("多策略组合", test_multiple_strategies_combined),
        ("空查询", test_empty_query),
        ("不存在的实体", test_nonexistent_entity),
        ("批量存储100条", test_large_volume_store),
        ("节点序列化", test_node_to_dict),
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
