"""
输出保存工具 - 彻底解决 Trae CN 输出被截断的问题

用法:
    from utils.output_saver import print_and_save, clean_old_logs
    print_and_save("内容", "test_output")  # 同时打印和保存
    clean_old_logs(days=7)  # 定期清理
"""
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 配置
MAX_LOG_FILES = 100  # 最多保留100个日志文件
DEFAULT_KEEP_DAYS = 7  # 默认保留7天


def save_output(name: str, content: str, add_timestamp: bool = True) -> Path:
    """
    保存输出到文件，避免 Trae CN 截断
    
    参数:
        name: 文件名（不含后缀）
        content: 要保存的内容
        add_timestamp: 是否添加时间戳
    
    返回:
        保存的文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if add_timestamp else ""
    filename = f"{name}_{timestamp}.txt" if timestamp else f"{name}.txt"
    filepath = LOGS_DIR / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"✅ 输出已保存到: {filepath}")
    return filepath


def print_and_save(content: str, name: str = "output") -> Path:
    """
    同时打印和保存，最佳实践
    """
    print(content)
    filepath = save_output(name, content)
    
    # 每次保存后检查是否需要清理
    if len(list(LOGS_DIR.glob("*.txt"))) > MAX_LOG_FILES:
        clean_old_logs(quiet=True)
    
    return filepath


def clean_old_logs(days: int = DEFAULT_KEEP_DAYS, quiet: bool = False):
    """
    清理旧日志文件
    
    参数:
        days: 保留最近N天的日志
        quiet: 是否静默执行（不打印）
    """
    cutoff_time = time.time() - (days * 86400)
    deleted = 0
    files_to_delete = []
    
    # 收集所有日志文件
    for filepath in LOGS_DIR.glob("*.txt"):
        try:
            if filepath.stat().st_mtime < cutoff_time:
                files_to_delete.append(filepath)
        except Exception:
            pass
    
    # 如果还超数量，再按时间删除最旧的
    remaining = list(LOGS_DIR.glob("*.txt"))
    if len(remaining) > MAX_LOG_FILES:
        remaining.sort(key=lambda f: f.stat().st_mtime)
        files_to_delete.extend(remaining[:len(remaining) - MAX_LOG_FILES])
    
    # 去重
    files_to_delete = list(set(files_to_delete))
    
    # 删除
    for filepath in files_to_delete:
        try:
            filepath.unlink()
            deleted += 1
        except Exception:
            pass
    
    if not quiet:
        print(f"🧹 已清理 {deleted} 个旧日志文件，保留最近 {days} 天")


# 模块导入时自动检查清理
try:
    clean_old_logs(days=DEFAULT_KEEP_DAYS, quiet=True)
except Exception:
    pass
