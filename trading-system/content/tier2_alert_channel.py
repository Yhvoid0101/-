# -*- coding: utf-8 -*-
"""Tier 2 告警通道 — v99 Phase 9 Task #32.4

用户铁律: "永远永远不要出现模拟牛逼, 实盘亏损的情况"
补齐断点: _monitor_config.json 中 actions: ["通知"] 未实现, 仅 logger.warning 输出

设计:
  - 复用 evolution_monitor._alerts 结构
  - 4级告警: INFO/WARN/CRITICAL/EMERGENCY
  - 多通道: console + log + alert_file (可扩展 webhook/email)
  - 告警去重: 同一 alert_key 5分钟内不重复

来源:
  - _monitor_config.json 4级告警阈值配置
  - evolution_monitor.py _alerts: List[Dict] 结构 (line 70, 170-205)
  - kpi_monitor.py ErrorFeedbackChannel + IterativeOptimizer
  - 用户铁律: "永远永远不要出现模拟牛逼, 实盘亏损的情况"
"""
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 4级告警阈值 (来自 _monitor_config.json)
ALERT_LEVELS: Dict[str, Dict] = {
    "INFO": {"severity": 1, "action": "记录"},
    "WARN": {"severity": 2, "action": "记录+通知"},
    "CRITICAL": {"severity": 3, "action": "记录+通知+减仓50%"},
    "EMERGENCY": {"severity": 4, "action": "记录+通知+全平仓+暂停策略"},
}

# 告警级别emoji
ALERT_EMOJI = {"INFO": "ℹ️", "WARN": "⚠️", "CRITICAL": "🔴", "EMERGENCY": "🚨"}


class Tier2AlertChannel:
    """Tier 2 告警通道

    4级告警系统, 补齐 _monitor_config.json 中 actions: ["通知"] 未实现的断点.
    多通道输出: console + log + alert_file (可扩展 webhook/email).
    告警去重: 同一 alert_key 5分钟内不重复.

    Attributes:
        alert_file: 告警持久化文件路径
        _alerts: 告警历史列表
        _dedup: 去重字典 {alert_key: last_trigger_time}
        _dedup_window: 去重窗口 (秒, 默认300=5分钟)
    """

    def __init__(self, alert_file: str = "tier2_alerts.json"):
        """初始化告警通道

        Args:
            alert_file: 告警持久化文件路径 (相对于当前工作目录)
        """
        self.alert_file = Path(alert_file)
        self._alerts: List[Dict] = []
        self._dedup: Dict[str, float] = {}  # {alert_key: last_trigger_time}
        self._dedup_window = 300  # 5分钟去重窗口

    def send_alert(
        self,
        level: str,
        category: str,
        message: str,
        metrics: Optional[Dict] = None,
        action_hint: str = "",
    ) -> bool:
        """发送告警

        Args:
            level: INFO/WARN/CRITICAL/EMERGENCY
            category: 告警类别 (kpi/risk/oos/rollback)
            message: 告警消息
            metrics: 相关指标数据
            action_hint: 建议动作

        Returns:
            bool: 是否成功发送 (False=被去重或级别非法)
        """
        if level not in ALERT_LEVELS:
            logger.warning("非法告警级别: %s (合法: %s)", level, list(ALERT_LEVELS.keys()))
            return False

        alert_key = f"{level}|{category}|{message[:50]}"
        now = time.time()

        # 去重检查: 同一 alert_key 5分钟内不重复
        if alert_key in self._dedup:
            if now - self._dedup[alert_key] < self._dedup_window:
                logger.debug("告警去重: %s (5分钟内已发送)", alert_key)
                return False

        self._dedup[alert_key] = now

        alert = {
            "timestamp": now,
            "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "level": level,
            "category": category,
            "message": message,
            "metrics": metrics or {},
            "action_hint": action_hint,
            "action_defined": ALERT_LEVELS[level]["action"],
        }

        self._alerts.append(alert)
        self._persist(alert)
        self._console_output(alert)
        self._log_output(alert)
        return True

    def _persist(self, alert: Dict) -> None:
        """持久化告警到文件

        保留最近1000条告警, 避免文件无限增长.
        修复: 文件存在但为空时, json.loads("") 会抛 JSONDecodeError, 需 strip() 后判空.
        """
        try:
            history: List[Dict] = []
            if self.alert_file.exists():
                content = self.alert_file.read_text(encoding="utf-8").strip()
                if content:  # 文件非空才解析 (空文件视为首次写入)
                    history = json.loads(content)
            history.append(alert)
            # 保留最近1000条
            if len(history) > 1000:
                history = history[-1000:]
            self.alert_file.write_text(
                json.dumps(history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("告警持久化失败: %s", e)

    def _console_output(self, alert: Dict) -> None:
        """控制台输出告警"""
        emoji = ALERT_EMOJI.get(alert["level"], "?")
        print(
            f"{emoji} [{alert['level']}] [{alert['category']}] {alert['message']}"
        )
        if alert["action_hint"]:
            print(f"   建议动作: {alert['action_hint']}")

    def _log_output(self, alert: Dict) -> None:
        """日志输出告警 (按级别路由到不同日志级别)"""
        level = alert["level"]
        msg = f"[{level}][{alert['category']}] {alert['message']} | action={alert['action_defined']}"
        if level == "INFO":
            logger.info(msg)
        elif level == "WARN":
            logger.warning(msg)
        elif level == "CRITICAL":
            logger.error(msg)
        elif level == "EMERGENCY":
            logger.critical(msg)

    def get_recent_alerts(self, n: int = 10) -> List[Dict]:
        """获取最近n条告警

        Args:
            n: 返回的告警数量

        Returns:
            最近n条告警列表
        """
        return self._alerts[-n:]

    def get_alerts_by_level(self, level: str) -> List[Dict]:
        """按级别获取告警

        Args:
            level: INFO/WARN/CRITICAL/EMERGENCY

        Returns:
            该级别的所有告警列表
        """
        return [a for a in self._alerts if a["level"] == level]

    def get_alert_summary(self) -> Dict[str, int]:
        """获取告警摘要 (按级别统计)

        Returns:
            {level: count} 字典
        """
        summary: Dict[str, int] = {level: 0 for level in ALERT_LEVELS}
        for alert in self._alerts:
            level = alert["level"]
            if level in summary:
                summary[level] += 1
        summary["total"] = len(self._alerts)
        return summary

    def clear_alerts(self) -> None:
        """清空告警历史 (不影响持久化文件)"""
        self._alerts.clear()
        self._dedup.clear()


# ============================================================
# 全局单例
# ============================================================

_global_alert_channel: Optional[Tier2AlertChannel] = None


def get_global_alert_channel() -> Tier2AlertChannel:
    """获取全局告警通道单例

    Returns:
        Tier2AlertChannel 全局单例
    """
    global _global_alert_channel
    if _global_alert_channel is None:
        _global_alert_channel = Tier2AlertChannel()
    return _global_alert_channel


def reset_global_alert_channel() -> None:
    """重置全局告警通道 (用于测试)"""
    global _global_alert_channel
    _global_alert_channel = None
