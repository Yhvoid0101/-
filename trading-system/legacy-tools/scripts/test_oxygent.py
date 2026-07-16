#!/usr/bin/env python3
"""OxyGent 适配器全面终端测试"""
import sys
import time
import traceback

sys.path.insert(0, ".")

from core.oxygent_adapter import (
    OxyDAGOrchestrator, OxyComponent, CircuitBreaker, SchemaValidator, OxyBank,
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
    assert OxyDAGOrchestrator is not None
    assert OxyComponent is not None
    assert CircuitBreaker is not None
    assert SchemaValidator is not None
    assert OxyBank is not None

def test_component_registration():
    dag = OxyDAGOrchestrator()
    oxy_id = dag.register_component(
        "test_comp", "tool",
        lambda **kw: {"output": f"hello {kw.get('task', '')}"},
        input_schema={"required": ["task"], "properties": {"task": {"type": "str"}}},
    )
    assert oxy_id is not None
    assert "test_comp" in dag.components
    assert dag.components["test_comp"].lifecycle == OxyComponent.LIFECYCLE_ACTIVE

def test_dag_execution():
    dag = OxyDAGOrchestrator()
    dag.register_component(
        "analyze", "tool",
        lambda **kw: {"analysis": f"analyzed: {kw.get('task', '')}"},
        input_schema={"required": ["task"]},
    )
    dag.register_component(
        "fix", "tool",
        lambda **kw: {"fix": f"fixed: {kw.get('task', '')}"},
        input_schema={"required": ["task"]},
        depends_on=["analyze"],
    )
    dag.register_component(
        "verify", "tool",
        lambda **kw: {"verified": True},
        input_schema={"required": ["task"]},
        depends_on=["fix"],
    )
    result = dag.execute("verify", task_description="test task", task="test task")
    assert result["success"], f"DAG执行失败: {result}"
    assert "analyze" in result["execution_order"]
    assert "fix" in result["execution_order"]
    assert "verify" in result["execution_order"]
    assert result["pattern_id"] is not None

def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.5)
    assert cb.state == CircuitBreaker.CLOSED
    assert cb.allow() is True
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitBreaker.OPEN
    assert cb.allow() is False
    time.sleep(0.6)
    assert cb.allow() is True
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitBreaker.CLOSED

def test_circuit_breaker_in_component():
    dag = OxyDAGOrchestrator()
    call_count = [0]

    def failing_handler(**kw):
        call_count[0] += 1
        raise RuntimeError("intentional failure")

    def fallback_handler(**kw):
        return {"fallback": True}

    dag.register_component(
        "flaky", "tool", failing_handler,
        fallback_handler=fallback_handler,
        retry_count=1,
    )
    result = dag.execute("flaky", task="test")
    assert result["success"], f"降级执行应成功: {result}"
    comp_result = result.get("results", {}).get("flaky", {})
    assert comp_result.get("fallback") is True, f"应使用降级: comp_result={comp_result}"

def test_schema_validation():
    data = {"task": "hello", "count": "5"}
    schema = {
        "required": ["task"],
        "properties": {"task": {"type": "str"}, "count": {"type": "int"}},
    }
    valid, err = SchemaValidator.validate_input(data, schema)
    assert valid, f"Schema验证应成功(含类型转换): {err}"
    assert isinstance(data["count"], int), f"count应被转为int: {type(data['count'])}"

def test_schema_validation_missing_required():
    data = {}
    schema = {"required": ["task"]}
    valid, err = SchemaValidator.validate_input(data, schema)
    assert not valid, "缺少必填字段应失败"
    assert "task" in err

def test_schema_validation_defaults():
    data = {"task": "hello"}
    schema = {"required": ["task"], "defaults": {"verbose": False, "max_retries": 3}}
    valid, err = SchemaValidator.validate_input(data, schema)
    assert valid
    assert data["verbose"] is False
    assert data["max_retries"] == 3

def test_oxybank_deposit_and_recommend():
    bank = OxyBank()
    bank.deposit(["a", "b", "c"], {"a": {"success": True}, "b": {"success": True}, "c": {"success": True}},
                 task_description="cache hit rate optimization", success=True)
    bank.deposit(["x", "y"], {"x": {"success": True}, "y": {"success": True}},
                 task_description="memory retrieval quality", success=True)
    recs = bank.recommend("cache optimization", top_k=2)
    assert len(recs) > 0, "OxyBank应能推荐模式"
    assert recs[0]["task_keywords"] is not None

def test_cycle_detection():
    dag = OxyDAGOrchestrator()
    dag.register_component("a", "tool", lambda **kw: {"ok": True})
    dag.register_component("b", "tool", lambda **kw: {"ok": True}, depends_on=["a"])
    try:
        dag.register_component("a_new", "tool", lambda **kw: {"ok": True}, depends_on=["b"])
        dag.execution_graph["a"] = ["a_new"]
        cycle = dag._detect_cycle()
        assert cycle is not None, "应检测到循环依赖"
    except ValueError:
        pass

def test_component_lifecycle():
    dag = OxyDAGOrchestrator()
    dag.register_component("lifecycle_test", "tool", lambda **kw: {"ok": True})
    comp = dag.components["lifecycle_test"]
    assert comp.lifecycle == OxyComponent.LIFECYCLE_ACTIVE
    comp.deprecate()
    assert comp.lifecycle == OxyComponent.LIFECYCLE_DEPRECATED
    comp.remove()
    assert comp.lifecycle == OxyComponent.LIFECYCLE_REMOVED
    assert not comp.is_available()

def test_component_unregistration():
    dag = OxyDAGOrchestrator()
    dag.register_component("temp_comp", "tool", lambda **kw: {"ok": True})
    assert "temp_comp" in dag.components
    dag.unregister_component("temp_comp")
    assert "temp_comp" not in dag.components

def test_health_status():
    dag = OxyDAGOrchestrator()
    dag.register_component("healthy_comp", "tool", lambda **kw: {"ok": True},
                           health_check=lambda: True)
    health = dag.get_health_status()
    assert "healthy_comp" in health
    assert health["healthy_comp"]["healthy"] is True

def test_execution_history():
    dag = OxyDAGOrchestrator()
    dag.register_component("hist_comp", "tool", lambda **kw: {"ok": True})
    dag.execute("hist_comp", task_description="history test", task="test")
    history = dag.get_execution_history()
    assert len(history) > 0

def test_stats():
    dag = OxyDAGOrchestrator()
    dag.register_component("stat_comp", "tool", lambda **kw: {"ok": True})
    stats = dag.get_stats()
    assert stats["total_components"] >= 1
    assert stats["active_components"] >= 1

def test_parallel_execution():
    dag = OxyDAGOrchestrator()
    dag.register_component("parallel_a", "tool", lambda **kw: {"result": "a"})
    dag.register_component("parallel_b", "tool", lambda **kw: {"result": "b"})
    dag.register_component("merge", "tool", lambda **kw: {"merged": True},
                           depends_on=["parallel_a", "parallel_b"])
    result = dag.execute("merge", task_description="parallel test", task="test")
    assert result["success"], f"并行执行应成功: {result}"

def test_output_schema_validation():
    dag = OxyDAGOrchestrator()
    dag.register_component(
        "output_test", "tool",
        lambda **kw: {"status": "ok", "data": [1, 2, 3]},
        output_schema={"required": ["status"]},
    )
    result = dag.execute("output_test", task="test")
    assert result["success"]

def test_component_diagnostics():
    dag = OxyDAGOrchestrator()
    dag.register_component("diag_comp", "tool", lambda **kw: {"ok": True},
                           version="2.0.0", tags=["test", "diagnostic"])
    result = dag.execute("diag_comp", task="test")
    comp = dag.components["diag_comp"]
    diag = comp.get_diagnostics()
    assert diag["version"] == "2.0.0"
    assert "test" in diag["tags"]
    assert diag["lifecycle"] == "active"
    assert "p50_latency_ms" in diag["metrics"]
    assert "p95_latency_ms" in diag["metrics"]
    assert "p99_latency_ms" in diag["metrics"]


if __name__ == "__main__":
    print("=" * 60)
    print("OxyGent 适配器全面终端测试")
    print("=" * 60)

    tests = [
        ("基础导入", test_basic_import),
        ("组件注册", test_component_registration),
        ("DAG执行", test_dag_execution),
        ("熔断器基础", test_circuit_breaker),
        ("熔断器+降级", test_circuit_breaker_in_component),
        ("Schema验证-类型转换", test_schema_validation),
        ("Schema验证-缺少必填", test_schema_validation_missing_required),
        ("Schema验证-默认值", test_schema_validation_defaults),
        ("OxyBank存入+推荐", test_oxybank_deposit_and_recommend),
        ("循环依赖检测", test_cycle_detection),
        ("组件生命周期", test_component_lifecycle),
        ("组件卸载", test_component_unregistration),
        ("健康状态", test_health_status),
        ("执行历史", test_execution_history),
        ("统计信息", test_stats),
        ("并行执行", test_parallel_execution),
        ("输出Schema验证", test_output_schema_validation),
        ("组件诊断信息", test_component_diagnostics),
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
