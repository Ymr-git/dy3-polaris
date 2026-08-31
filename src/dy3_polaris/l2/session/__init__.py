"""L2 个性化会话子模块 — 学习会话生命周期管理.

导出:
- SessionManager: 个性化会话管理器 (创建/暂停/恢复/关闭 + 检查点 + 活跃会话查询)

会话状态机:
  created -> active -> paused -> active (可循环)
                    -> closed (终态)

检查点机制 (Claude Science): 检查点含 seq / ts / SHA-256 完整性哈希.
"""

from dy3_polaris.l2.session.session_manager import SessionManager

__all__ = [
    "SessionManager",
]
