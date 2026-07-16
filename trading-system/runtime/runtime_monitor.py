#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hermes 交易系统演化循环运行时监控仪表盘

本模块与 run_sandbox.py 演化循环并行运行，提供实时可见性：
1. 所有门控触发统计（调用次数、调整、boost、reduce、adjust_rate）
2. 数据源健康状态（来自 data/auto_fetched/health.json）
3. 演化进度（代数、最佳适应度、多样性、种群规模）
4. 系统资源（CPU、内存、磁盘使用率）
5. 错误/告警监控（扫描日志中的错误）
6. 周期性报告（每 100 轮或 5 分钟）

仅使用 Python 标准库，避免与交易系统产生循环导入。
"""

import os
import sys
import json
import time
import re
import argparse
import logging
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('RuntimeMonitor')


class RuntimeMonitor:
    """演化循环运行时监控器

    通过解析日志文件、健康状态文件、种群文件等，
    提供对 Hermes 交易系统演化循环的实时监控能力。
    """

    # 门控名称列表（按 Layer 顺序）
    GATE_NAMES = [
        'FearGreedGate',
        'IRGate',
        'OBGate',
        'SPGate',
        'SMGate',
        'URPDGate',
        'DecayGate',
        'SOPRGate',
        'LSRGate',
    ]

    # 门控动作正则：匹配 "GateName ACTION" 或 "GateName ACTION #N:"
    # ACTION 可能值：BOOST / REDUCE / BOOST_HEAVY / REDUCE_HEAVY / BOOST_LIGHT / REDUCE_LIGHT
    GATE_ACTION_RE = re.compile(
        r'\b(?P<gate>FearGreedGate|IRGate|OBGate|SPGate|SMGate|URPDGate|DecayGate|SOPRGate|LSRGate)'
        r'\s+(?P<action>BOOST(?:_HEAVY|_LIGHT)?|REDUCE(?:_HEAVY|_LIGHT)?)'
    )

    # 错误日志模式（排除 WARNING/INFO/DEBUG 行中的 error 字样）
    ERROR_RE = re.compile(r'\b(ERROR|CRITICAL|FATAL|Traceback|Exception)\b', re.IGNORECASE)

    def __init__(self, log_file, data_dir, output_dir, interval=30):
        """
        初始化监控器

        Args:
            log_file: 演化循环日志文件路径（stdout/stderr 重定向文件）
            data_dir: 数据目录路径（包含 auto_fetched/、sandbox_trading/ 等）
            output_dir: 报告输出目录
            interval: 刷新间隔（秒）
        """
        self.log_file = Path(log_file) if log_file else None
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        self.output_dir = Path(output_dir) if output_dir else Path('data/monitoring_reports')
        self.interval = interval

        # 关键文件路径
        self.health_file = self.data_dir / 'auto_fetched' / 'health.json'
        self.sandbox_dir = self.data_dir / 'sandbox_trading'
        self.audit_log = self.sandbox_dir / 'audit.log'
        self.evolution_stats_file = self.sandbox_dir / 'evolution_stats.jsonl'

        # 运行状态
        self._running = False
        self._thread = None
        self._start_time = None
        self._round_count = 0
        self._last_report_time = 0

        # 缓存上一轮状态（用于检测 fitness 下降等趋势）
        self._last_best_fitness = None
        self._last_health_status = {}

        # 确保输出目录存在
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass  # 输出目录创建失败不致命，后续保存报告时会再次尝试

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def start(self):
        """启动监控循环（非阻塞，在后台线程运行）"""
        if self._running:
            logger.warning("监控器已在运行")
            return
        self._running = True
        self._start_time = datetime.now()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"监控器已启动，刷新间隔 {self.interval} 秒")

    def stop(self):
        """优雅停止监控"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("监控器已停止")

    def _monitor_loop(self):
        """监控主循环（在后台线程中运行）"""
        while self._running:
            try:
                self._round_count += 1
                self.print_dashboard()

                # 每 100 轮或每 5 分钟生成一次报告
                now_ts = time.time()
                if (self._round_count % 100 == 0) or (now_ts - self._last_report_time >= 300):
                    report = self.generate_report()
                    self._save_report(report)
                    self._last_report_time = now_ts

            except Exception as e:
                logger.error(f"监控循环出错: {e}", exc_info=True)

            # 分段睡眠，便于快速响应停止信号
            for _ in range(self.interval):
                if not self._running:
                    break
                time.sleep(1)

    # ------------------------------------------------------------------
    # 数据采集方法
    # ------------------------------------------------------------------

    def parse_gate_stats(self):
        """
        解析日志文件，统计各门控的触发次数

        日志中的典型格式：
            Layer1 FearGreedGate BOOST #10: ...
            Layer1 IRGate REDUCE #1: ...
            Layer8 DecayGate REDUCE_HEAVY #5: ...
            SOPRGate BOOST: ...
            LSRGate REDUCE: ...

        Returns:
            dict: {gate_name: {'BOOST': N, 'REDUCE': N, 'REDUCE_HEAVY': N,
                               'REDUCE_LIGHT': N, 'Total': N}}
        """
        # 初始化所有门控的统计字典
        stats = {}
        for gate_name in self.GATE_NAMES:
            stats[gate_name] = {
                'BOOST': 0,
                'REDUCE': 0,
                'REDUCE_HEAVY': 0,
                'REDUCE_LIGHT': 0,
                'Total': 0,
            }

        if not self.log_file or not self.log_file.exists():
            return stats

        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = self.GATE_ACTION_RE.search(line)
                    if match:
                        gate = match.group('gate')
                        action = match.group('action').upper()
                        if gate in stats:
                            if action in stats[gate]:
                                stats[gate][action] += 1
                            elif 'REDUCE' in action:
                                # 未知 REDUCE 变体，归入 REDUCE
                                stats[gate]['REDUCE'] += 1
                            else:
                                stats[gate]['BOOST'] += 1
                            stats[gate]['Total'] += 1
        except OSError as e:
            logger.error(f"解析门控统计失败（读取日志文件出错）: {e}")

        return stats

    def check_data_health(self):
        """
        读取 health.json，返回数据源健康状态

        health.json 结构：
            {
                "sources": [
                    {"name": "fear_greed", "status": "healthy",
                     "last_success": 1783868202, "last_error": null, ...},
                    ...
                ],
                "summary": {"overall": "degraded", "total": 6, ...}
            }

        Returns:
            dict: {source_name: {'status': str, 'age_str': str,
                                 'last_fetch': datetime|None, 'error': str|None}}
        """
        result = {}

        if not self.health_file.exists():
            return result

        try:
            with open(self.health_file, 'r', encoding='utf-8') as f:
                health_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"读取 health.json 失败: {e}")
            return result

        now = datetime.now()
        sources = health_data.get('sources', [])

        for src in sources:
            if not isinstance(src, dict):
                continue
            name = src.get('name', 'unknown')
            status = src.get('status', 'unknown')
            last_error = src.get('last_error')

            # 优先用 last_success，其次 last_failure
            last_ts = src.get('last_success') or src.get('last_failure')
            last_fetch = None
            if last_ts is not None:
                try:
                    last_fetch = datetime.fromtimestamp(float(last_ts))
                except (ValueError, OSError, TypeError):
                    last_fetch = None

            # 计算距上次抓取的时间
            age_str = 'unknown'
            if last_fetch:
                age = now - last_fetch
                age_str = self._format_timedelta(age)
                # 自动修正状态：超过 10 分钟视为 down，超过 5 分钟视为 degraded
                if status == 'healthy' and age > timedelta(minutes=10):
                    status = 'down'
                elif status == 'healthy' and age > timedelta(minutes=5):
                    status = 'degraded'

            result[name] = {
                'status': status,
                'last_fetch': last_fetch,
                'age_str': age_str,
                'error': last_error,
            }

        return result

    def get_evolution_progress(self):
        """
        读取最新的 population_gen*.jsonl 文件，获取演化进度

        Returns:
            dict: {'generation': N, 'population_size': N, 'best_fitness': F,
                   'diversity': F, 'avg_fitness': F}
        """
        result = {
            'generation': 0,
            'population_size': 0,
            'best_fitness': 0.0,
            'diversity': 0.0,
            'avg_fitness': 0.0,
        }

        if not self.sandbox_dir.exists():
            return result

        # 查找最新的 population 文件
        pop_files = sorted(self.sandbox_dir.glob('population_gen*.jsonl'))
        if pop_files:
            latest_pop_file = pop_files[-1]
            try:
                # 从文件名提取代数
                gen_match = re.search(r'population_gen(\d+)', latest_pop_file.name)
                if gen_match:
                    result['generation'] = int(gen_match.group(1))

                # 读取种群个体（每行一个 JSON）
                individuals = []
                with open(latest_pop_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                individuals.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

                result['population_size'] = len(individuals)

                # 提取适应度
                fitnesses = []
                for ind in individuals:
                    fitness = ind.get('fitness_score') or ind.get('fitness') or ind.get('score') or ind.get('eval_score') or ind.get('gt_score')
                    if fitness is not None:
                        try:
                            fitnesses.append(float(fitness))
                        except (ValueError, TypeError):
                            pass

                if fitnesses:
                    result['best_fitness'] = max(fitnesses)
                    result['avg_fitness'] = sum(fitnesses) / len(fitnesses)
                    # 多样性 = 适应度标准差 / |平均值|
                    if len(fitnesses) > 1 and result['avg_fitness'] != 0:
                        mean = result['avg_fitness']
                        variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
                        result['diversity'] = (variance ** 0.5) / abs(mean)
            except OSError as e:
                logger.error(f"读取种群文件失败: {e}")

        # 尝试从 evolution_stats.jsonl 获取更精确的统计（最后一行）
        if self.evolution_stats_file.exists():
            try:
                last_line = None
                with open(self.evolution_stats_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            last_line = line

                if last_line:
                    stats = json.loads(last_line)
                    # 用 evolution_stats 中的数据覆盖（更权威）
                    if 'generation' in stats:
                        result['generation'] = stats['generation']
                    if 'best_fitness' in stats:
                        result['best_fitness'] = stats['best_fitness']
                    if 'diversity' in stats:
                        result['diversity'] = stats['diversity']
                    if 'population_size' in stats:
                        result['population_size'] = stats['population_size']
                    if 'avg_fitness' in stats:
                        result['avg_fitness'] = stats['avg_fitness']
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"读取 evolution_stats.jsonl 失败: {e}")

        return result

    def get_system_resources(self):
        """
        获取系统资源使用情况

        优先使用 psutil（如果可用），否则回退到 /proc/stat 等。

        Returns:
            dict: {'cpu_percent': F,
                   'memory': {'used': N, 'total': N, 'percent': F},
                   'disk': {'used': N, 'total': N, 'percent': F}}
        """
        result = {
            'cpu_percent': 0.0,
            'memory': {'used': 0, 'total': 0, 'percent': 0.0},
            'disk': {'used': 0, 'total': 0, 'percent': 0.0},
        }

        try:
            result['cpu_percent'] = self._get_cpu_percent()
            result['memory'] = self._get_memory_info()
            result['disk'] = self._get_disk_info()
        except Exception as e:
            logger.error(f"获取系统资源失败: {e}")

        return result

    def _get_cpu_percent(self):
        """获取 CPU 使用率（Linux 读取 /proc/stat，Windows 用 wmic）"""
        # 优先尝试 psutil
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            pass

        # Linux: 读取 /proc/stat 两次计算差值
        if os.path.exists('/proc/stat'):
            try:
                with open('/proc/stat', 'r') as f:
                    line1 = f.readline()
                fields1 = list(map(int, line1.split()[1:]))
                idle1 = fields1[3]
                total1 = sum(fields1)

                time.sleep(0.1)

                with open('/proc/stat', 'r') as f:
                    line2 = f.readline()
                fields2 = list(map(int, line2.split()[1:]))
                idle2 = fields2[3]
                total2 = sum(fields2)

                total_diff = total2 - total1
                idle_diff = idle2 - idle1
                if total_diff > 0:
                    return (1 - idle_diff / total_diff) * 100
            except (OSError, ValueError, IndexError):
                pass

        # Windows: 使用 wmic（备用）
        if sys.platform == 'win32':
            try:
                output = subprocess.check_output(
                    ['wmic', 'cpu', 'get', 'loadpercentage'],
                    universal_newlines=True, timeout=5,
                )
                lines = [l.strip() for l in output.split('\n') if l.strip()]
                if len(lines) > 1:
                    return float(lines[1])
            except (subprocess.SubprocessError, ValueError, OSError):
                pass

        return 0.0

    def _get_memory_info(self):
        """获取内存信息（Linux 读取 /proc/meminfo）"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {'used': mem.used, 'total': mem.total, 'percent': mem.percent}
        except ImportError:
            pass

        if os.path.exists('/proc/meminfo'):
            try:
                meminfo = {}
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        parts = line.split(':')
                        if len(parts) == 2:
                            key = parts[0].strip()
                            value = int(parts[1].split()[0]) * 1024  # kB -> bytes
                            meminfo[key] = value

                total = meminfo.get('MemTotal', 0)
                available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
                used = total - available
                percent = (used / total * 100) if total > 0 else 0
                return {'used': used, 'total': total, 'percent': percent}
            except (OSError, ValueError):
                pass

        return {'used': 0, 'total': 0, 'percent': 0.0}

    def _get_disk_info(self):
        """获取磁盘使用信息"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {'used': disk.used, 'total': disk.total, 'percent': disk.percent}
        except ImportError:
            pass

        # Unix: 使用 os.statvfs
        try:
            stat = os.statvfs('/')
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            percent = (used / total * 100) if total > 0 else 0
            return {'used': used, 'total': total, 'percent': percent}
        except (OSError, AttributeError):
            pass

        # Windows: 使用 ctypes 调用 GetDiskFreeSpaceEx
        if sys.platform == 'win32':
            try:
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p('/'), None,
                    ctypes.pointer(total_bytes),
                    ctypes.pointer(free_bytes),
                )
                total = total_bytes.value
                used = total - free_bytes.value
                percent = (used / total * 100) if total > 0 else 0
                return {'used': used, 'total': total, 'percent': percent}
            except (OSError, AttributeError):
                pass

        return {'used': 0, 'total': 0, 'percent': 0.0}

    # ------------------------------------------------------------------
    # 告警检查
    # ------------------------------------------------------------------

    def check_alerts(self):
        """
        检查告警条件

        告警规则：
        1. 门控调用 100+ 次但从未进行任何调整（BOOST/REDUCE 均为 0）
        2. 数据源状态为 down
        3. 最佳适应度下降
        4. 日志错误数量超过阈值（>50 告警，>10 警告）

        Returns:
            list: [{'level': 'ALERT'/'WARN'/'OK', 'message': str}]
        """
        alerts = []

        # 1. 检查门控异常
        gate_stats = self.parse_gate_stats()
        gate_anomaly_found = False
        for gate_name in self.GATE_NAMES:
            s = gate_stats.get(gate_name, {})
            total = s.get('Total', 0)
            adjusted = (s.get('BOOST', 0) + s.get('REDUCE', 0)
                        + s.get('REDUCE_HEAVY', 0) + s.get('REDUCE_LIGHT', 0))
            if total >= 100 and adjusted == 0:
                alerts.append({
                    'level': 'ALERT',
                    'message': f"{gate_name} 已被调用 {total} 次但从未进行任何调整",
                })
                gate_anomaly_found = True
        if not gate_anomaly_found:
            alerts.append({'level': 'OK', 'message': '未检测到门控异常'})

        # 2. 检查数据源宕机
        health = self.check_data_health()
        for source, info in sorted(health.items()):
            if info['status'] == 'down':
                alerts.append({
                    'level': 'ALERT',
                    'message': f"{source} 已宕机（上次抓取: {info['age_str']}前）",
                })
            elif info['status'] == 'degraded':
                alerts.append({
                    'level': 'WARN',
                    'message': f"{source} 状态降级（上次抓取: {info['age_str']}前）",
                })

        # 3. 检查 fitness 下降
        progress = self.get_evolution_progress()
        current_fitness = progress.get('best_fitness')
        if (current_fitness is not None and current_fitness > 0
                and self._last_best_fitness is not None
                and self._last_best_fitness > 0):
            if current_fitness < self._last_best_fitness:
                alerts.append({
                    'level': 'WARN',
                    'message': (f"最佳适应度下降: {self._last_best_fitness:.2f} "
                                f"-> {current_fitness:.2f}"),
                })
        self._last_best_fitness = current_fitness

        # 4. 检查日志错误数量
        error_count = self._count_log_errors()
        if error_count > 50:
            alerts.append({
                'level': 'ALERT',
                'message': f"日志错误数量过高: {error_count} 条",
            })
        elif error_count > 10:
            alerts.append({
                'level': 'WARN',
                'message': f"日志错误数量较多: {error_count} 条",
            })

        return alerts

    def _count_log_errors(self):
        """统计日志中的错误行数"""
        if not self.log_file or not self.log_file.exists():
            return 0

        try:
            count = 0
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if self.ERROR_RE.search(line):
                        count += 1
            return count
        except OSError:
            return 0

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def generate_report(self):
        """
        生成 Markdown 格式的监控报告

        Returns:
            str: Markdown 报告内容
        """
        now = datetime.now()
        uptime = now - self._start_time if self._start_time else timedelta(0)

        gate_stats = self.parse_gate_stats()
        health = self.check_data_health()
        progress = self.get_evolution_progress()
        resources = self.get_system_resources()
        alerts = self.check_alerts()

        lines = []
        lines.append("# Hermes 交易系统监控报告")
        lines.append("")
        lines.append(f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**运行时长**: {self._format_timedelta(uptime)}")
        lines.append(f"**监控轮次**: {self._round_count}")
        lines.append("")

        # 演化进度
        lines.append("## 演化进度")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 代数 | {progress['generation']} |")
        lines.append(f"| 种群规模 | {progress['population_size']} |")
        lines.append(f"| 最佳适应度 | {progress['best_fitness']:.4f} |")
        lines.append(f"| 平均适应度 | {progress['avg_fitness']:.4f} |")
        lines.append(f"| 多样性 | {progress['diversity']:.4f} |")
        lines.append("")

        # 门控统计
        lines.append("## 门控统计")
        lines.append("")
        _gate_total_all = sum(s.get('Total', 0) for s in gate_stats.values())
        if _gate_total_all == 0:
            lines.append("> ⚠️ **quiet模式**: 门控动作(BOOST/REDUCE)为INFO/DEBUG级日志,--quiet模式下不输出,故统计全0。")
            lines.append("> 如需门控统计,请去掉 `--quiet` 参数运行沙盘。门控本身正常工作(见audit.log的risk/pre_trade_rejected)。")
            lines.append("")
        lines.append("| 门控 | BOOST | REDUCE | REDUCE_HEAVY | REDUCE_LIGHT | Total |")
        lines.append("|------|-------|--------|--------------|--------------|-------|")
        for gate_name in self.GATE_NAMES:
            s = gate_stats.get(gate_name, {})
            lines.append(
                f"| {gate_name} | {s.get('BOOST', 0)} | {s.get('REDUCE', 0)} | "
                f"{s.get('REDUCE_HEAVY', 0)} | {s.get('REDUCE_LIGHT', 0)} | "
                f"{s.get('Total', 0)} |"
            )
        lines.append("")

        # 数据源健康
        lines.append("## 数据源健康")
        lines.append("")
        if health:
            lines.append("| 数据源 | 状态 | 上次抓取 | 错误 |")
            lines.append("|--------|------|----------|------|")
            status_emoji = {
                'healthy': '✅', 'degraded': '⚠️',
                'down': '❌', 'unknown': '❓',
            }
            for source in sorted(health.keys()):
                info = health[source]
                emoji = status_emoji.get(info['status'], '❓')
                err = info.get('error') or '-'
                lines.append(
                    f"| {source} | {emoji} {info['status']} | "
                    f"{info['age_str']} | {err} |"
                )
        else:
            lines.append("无健康数据")
        lines.append("")

        # 系统资源
        lines.append("## 系统资源")
        lines.append("")
        mem = resources['memory']
        disk = resources['disk']
        lines.append("| 资源 | 使用 | 总量 | 百分比 |")
        lines.append("|------|------|------|--------|")
        lines.append(f"| CPU | - | - | {resources['cpu_percent']:.1f}% |")
        lines.append(
            f"| 内存 | {self._format_bytes(mem['used'])} | "
            f"{self._format_bytes(mem['total'])} | {mem['percent']:.1f}% |"
        )
        lines.append(
            f"| 磁盘 | {self._format_bytes(disk['used'])} | "
            f"{self._format_bytes(disk['total'])} | {disk['percent']:.1f}% |"
        )
        lines.append("")

        # 告警
        lines.append("## 告警")
        lines.append("")
        for alert in alerts:
            lines.append(f"- **[{alert['level']}]** {alert['message']}")
        lines.append("")

        return '\n'.join(lines)

    def _save_report(self, report_content):
        """保存报告到文件"""
        now = datetime.now()
        filename = f"report_{now.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.output_dir / filename

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"报告已保存: {filepath}")
        except OSError as e:
            logger.error(f"保存报告失败: {e}")

    # ------------------------------------------------------------------
    # 控制台仪表盘
    # ------------------------------------------------------------------

    def print_dashboard(self):
        """打印当前状态到控制台（刷新式显示）"""
        now = datetime.now()
        uptime = now - self._start_time if self._start_time else timedelta(0)

        # 清屏
        if sys.platform != 'win32':
            os.system('clear')
        else:
            os.system('cls')

        print("=" * 60)
        print("=== Hermes 交易系统监控 ===")
        print(f"时间: {now.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"运行时长: {self._format_timedelta(uptime)}")
        print("=" * 60)

        # 演化进度
        progress = self.get_evolution_progress()
        print("\n--- 演化进度 ---")
        print(f"代数: {progress['generation']} | "
              f"种群: {progress['population_size']} | "
              f"最佳适应度: {progress['best_fitness']:.2f} | "
              f"多样性: {progress['diversity']:.4f}")

        # 门控统计
        gate_stats = self.parse_gate_stats()
        _gate_total_all = sum(s.get('Total', 0) for s in gate_stats.values())
        print("\n--- 门控统计 (来自日志) ---")
        if _gate_total_all == 0:
            print("  [quiet模式] 门控动作未记录到日志(BOOST/REDUCE为INFO/DEBUG级,--quiet不输出)。门控正常工作,见audit.log。")
        for gate_name in self.GATE_NAMES:
            s = gate_stats.get(gate_name, {})
            heavy = s.get('REDUCE_HEAVY', 0)
            heavy_str = f"HEAVY={heavy:<4}" if heavy > 0 else ""
            print(f"{gate_name + ':':<16} BOOST={s.get('BOOST', 0):<6} "
                  f"REDUCE={s.get('REDUCE', 0):<6}{heavy_str} "
                  f"Total={s.get('Total', 0)}")

        # 数据源健康
        health = self.check_data_health()
        print("\n--- 数据源健康 ---")
        if health:
            status_emoji = {
                'healthy': '✅', 'degraded': '⚠️',
                'down': '❌', 'unknown': '❓',
            }
            for source in sorted(health.keys()):
                info = health[source]
                emoji = status_emoji.get(info['status'], '❓')
                print(f"{source + ':':<18} {emoji} {info['status']:<10} "
                      f"(上次抓取: {info['age_str']}前)")
        else:
            print("无健康数据（health.json 不存在）")

        # 系统资源
        resources = self.get_system_resources()
        mem = resources['memory']
        disk = resources['disk']
        print("\n--- 系统资源 ---")
        print(f"CPU: {resources['cpu_percent']:.1f}% | "
              f"内存: {self._format_bytes(mem['used'])}/{self._format_bytes(mem['total'])} "
              f"({mem['percent']:.1f}%) | "
              f"磁盘: {self._format_bytes(disk['used'])}/{self._format_bytes(disk['total'])} "
              f"({disk['percent']:.1f}%)")

        # 告警
        alerts = self.check_alerts()
        print("\n--- 告警 ---")
        for alert in alerts:
            print(f"[{alert['level']}] {alert['message']}")

        print("\n" + "=" * 60)
        print(f"下次刷新: {self.interval} 秒后 | 按 Ctrl+C 停止")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_bytes(size):
        """格式化字节数为人类可读格式"""
        size = float(size)
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(size) < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}PB"

    @staticmethod
    def _format_timedelta(td):
        """格式化时间间隔为人类可读格式"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='Hermes 交易系统演化循环运行时监控',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 runtime_monitor.py --log-file /path/to/evolution.log --interval 30
  python3 runtime_monitor.py --once --log-file /path/to/evolution.log
  python3 runtime_monitor.py --log-file /path/to/evolution.log \\
      --data-dir data --output-dir data/monitoring_reports
        """,
    )
    parser.add_argument('--log-file', type=str, default=None,
                        help='演化循环日志文件路径')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='数据目录路径（默认: data）')
    parser.add_argument('--output-dir', type=str, default='data/monitoring_reports',
                        help='报告输出目录（默认: data/monitoring_reports）')
    parser.add_argument('--interval', type=int, default=30,
                        help='刷新间隔秒数（默认: 30）')
    parser.add_argument('--once', action='store_true',
                        help='仅打印一次当前状态并退出')

    args = parser.parse_args()

    # 创建监控器实例
    monitor = RuntimeMonitor(
        log_file=args.log_file,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        interval=args.interval,
    )

    if args.once:
        # 单次模式：打印状态并退出
        monitor._start_time = datetime.now()
        monitor._round_count = 1
        monitor.print_dashboard()

        # 在 --once 模式下也生成一份报告
        report = monitor.generate_report()
        report_path = (monitor.output_dir /
                       f"report_once_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        try:
            monitor.output_dir.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存至: {report_path}")
        except OSError as e:
            logger.error(f"保存报告失败: {e}")
    else:
        # 持续监控模式
        monitor.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n收到中断信号，正在停止...")
            monitor.stop()


if __name__ == '__main__':
    main()
