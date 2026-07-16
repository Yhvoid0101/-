"""Phase 1 变更控制面包 — 智能体变更提案管理。

铁律：智能体只能提交提案，不能直接写生产配置或代码。
"""
from .change_plane import ChangeControlPlane, ChangeRequest

__all__ = ["ChangeControlPlane", "ChangeRequest"]
