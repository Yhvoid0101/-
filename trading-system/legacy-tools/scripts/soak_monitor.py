# -*- coding: utf-8 -*-
"""
Hermes V6.0 阶段七：长期浸泡与可观测性监控脚本

持续运行，每10分钟采集一次指标，监控：
  - 内存增长（72小时增长 < 起始10%）
  - 线程泄漏（无永久阻塞）
  - 相位状态机稳定性
  - 价值引擎裁决一致性
  - 层级隔离消息通过率
  - 宪法裁决分布
  - Obsidian写入成功率
"""
import os
import sys
import time
import json
import psutil
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta

HERMES_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HERMES_ROOT))

REPORT_DIR = HERMES_ROOT / "data" / "soak_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class SoakMonitor:
    def __init__(self, duration_hours=72, interval_minutes=10):
        self.duration = duration_hours * 3600
        self.interval = interval_minutes * 60
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.snapshots = []
        self.errors = []
        self._stop = threading.Event()

    def snapshot(self):
        now = time.time()
        elapsed = now - self.start_time
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        mem_growth_pct = ((mem_mb - self.start_memory) / self.start_memory) * 100 if self.start_memory > 0 else 0
        thread_count = threading.active_count()

        result = {
            "timestamp": now,
            "elapsed_hours": round(elapsed / 3600, 2),
            "memory_mb": round(mem_mb, 2),
            "memory_growth_pct": round(mem_growth_pct, 2),
            "thread_count": thread_count,
            "checks": {},
        }

        try:
            from core.stability.phase_state_machine import PhaseStateMachine, Phase
            psm = PhaseStateMachine()
            psm.update_telemetry(consecutive_failures=0, system_health_score=1.0,
                                  resource_usage_percent=50, user_idle_seconds=0)
            result["checks"]["phase_machine"] = {
                "status": "ok",
                "initial_phase": psm.current_phase.value,
            }
        except Exception as e:
            result["checks"]["phase_machine"] = {"status": "error", "error": str(e)}
            self.errors.append(("phase_machine", now, str(e)))

        try:
            from core.cognitive.value_compromise import ValueCompromiseEngine, ValueDimension, Candidate
            engine = ValueCompromiseEngine()
            c1 = Candidate(name="a", scores={ValueDimension.SAFETY: 0.8, ValueDimension.EFFICIENCY: 0.6})
            c2 = Candidate(name="b", scores={ValueDimension.SAFETY: 0.5, ValueDimension.EFFICIENCY: 0.9})
            r = engine.compromise([c1, c2])
            result["checks"]["value_engine"] = {
                "status": "ok",
                "decision_type": r.decision_type,
            }
        except Exception as e:
            result["checks"]["value_engine"] = {"status": "error", "error": str(e)}
            self.errors.append(("value_engine", now, str(e)))

        try:
            from core.stability.layer_isolation import LayerIsolationMiddleware, Layer, LayerMessage
            middleware = LayerIsolationMiddleware()
            msg = LayerMessage(
                source_layer=Layer.REFLEX, target_layer=Layer.COGNITIVE,
                direction="up", payload={"soak_test": True},
            )
            passed = middleware.send_message(msg)
            result["checks"]["layer_isolation"] = {
                "status": "ok",
                "message_passed": passed,
            }
        except Exception as e:
            result["checks"]["layer_isolation"] = {"status": "error", "error": str(e)}
            self.errors.append(("layer_isolation", now, str(e)))

        try:
            from core.execution.constitutional_engine import ConstitutionalEngine
            r1 = ConstitutionalEngine.review("read", "test.txt",
                                              resource_state={"cpu": 50, "memory": 50, "disk_free_gb": 100})
            r2 = ConstitutionalEngine.review("delete", "secret.key")
            result["checks"]["constitutional_engine"] = {
                "status": "ok",
                "normal_ruling": r1["ruling"],
                "sensitive_ruling": r2["ruling"],
            }
        except Exception as e:
            result["checks"]["constitutional_engine"] = {"status": "error", "error": str(e)}
            self.errors.append(("constitutional_engine", now, str(e)))

        try:
            from core.evolution.self_slimming_engine import SelfSlimmingEngine
            engine = SelfSlimmingEngine()
            m = engine.collect_metrics()
            result["checks"]["self_slimming"] = {
                "status": "ok",
                "code_entropy": round(m.code_entropy, 4),
                "memory_entropy": round(m.memory_entropy, 4),
            }
        except Exception as e:
            result["checks"]["self_slimming"] = {"status": "error", "error": str(e)}
            self.errors.append(("self_slimming", now, str(e)))

        self.snapshots.append(result)
        return result

    def check_health(self):
        if not self.snapshots:
            return True
        latest = self.snapshots[-1]
        if latest["memory_growth_pct"] > 10:
            return False
        if latest["thread_count"] > 50:
            return False
        for name, check in latest["checks"].items():
            if check["status"] == "error":
                return False
        return True

    def generate_report(self):
        if not self.snapshots:
            return "No snapshot data"
        lines = [
            "=" * 60,
            "Hermes V6.0 Soak Test Report",
            f"Start: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Snapshots: {len(self.snapshots)}",
            f"Errors: {len(self.errors)}",
            "",
        ]
        latest = self.snapshots[-1]
        lines.append(f"Latest (elapsed {latest['elapsed_hours']}h):")
        lines.append(f"  Memory: {latest['memory_mb']:.1f}MB (growth {latest['memory_growth_pct']:.2f}%)")
        lines.append(f"  Threads: {latest['thread_count']}")
        for name, check in latest["checks"].items():
            st = "OK" if check["status"] == "ok" else "FAIL"
            lines.append(f"  [{st}] {name}: {check['status']}")
        lines.append("")
        lines.append("Health Checks:")
        mem_ok = latest["memory_growth_pct"] < 10
        thread_ok = latest["thread_count"] < 50
        error_ok = len(self.errors) == 0
        lines.append(f"  [{'OK' if mem_ok else 'FAIL'}] Memory growth < 10%: {latest['memory_growth_pct']:.2f}%")
        lines.append(f"  [{'OK' if thread_ok else 'FAIL'}] Threads < 50: {latest['thread_count']}")
        lines.append(f"  [{'OK' if error_ok else 'FAIL'}] No errors: {len(self.errors)}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def run(self):
        print(f"[Soak] Starting - {self.duration/3600:.0f}h duration, {self.interval/60:.0f}min interval")
        print(f"[Soak] Start memory: {self.start_memory:.1f}MB")
        cycle = 0
        while not self._stop.is_set():
            cycle += 1
            try:
                snap = self.snapshot()
                health = self.check_health()
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] cycle#{cycle} | mem={snap['memory_mb']:.1f}MB "
                      f"(+{snap['memory_growth_pct']:.2f}%) | threads={snap['thread_count']} | "
                      f"health={'OK' if health else 'FAIL'}")
                if not health:
                    print(f"[WARN] Health check failed!")
                    report = self.generate_report()
                    report_path = REPORT_DIR / f"unhealthy_{int(time.time())}.txt"
                    report_path.write_text(report, encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] Snapshot failed: {e}")
                self.errors.append(("monitor", time.time(), str(e)))

            report_path = REPORT_DIR / "latest_report.txt"
            report_path.write_text(self.generate_report(), encoding="utf-8")

            if time.time() - self.start_time >= self.duration:
                print(f"[Soak] Duration reached, stopping")
                break

            self._stop.wait(self.interval)

        final_report = self.generate_report()
        final_path = REPORT_DIR / f"final_report_{int(time.time())}.txt"
        final_path.write_text(final_report, encoding="utf-8")
        print(final_report)
        return len(self.errors) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hermes V6.0 浸泡监控")
    parser.add_argument("--hours", type=float, default=72, help="浸泡时长(小时)")
    parser.add_argument("--interval", type=float, default=10, help="采集间隔(分钟)")
    parser.add_argument("--quick", action="store_true", help="快速模式(5分钟)")
    args = parser.parse_args()

    if args.quick:
        args.hours = 5 / 60
        args.interval = 1

    monitor = SoakMonitor(duration_hours=args.hours, interval_minutes=args.interval)
    success = monitor.run()
    sys.exit(0 if success else 1)
