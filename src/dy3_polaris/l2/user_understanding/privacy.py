"""隐私门 — 敏感信息过滤与脱敏 (硬性边界).

过滤规则:
- 身份/证件: 身份证号 (18 位), 学号不敏感但证件敏感
- 联系方式: 手机号 (11 位), 座机
- 健康/财务/家庭: 关键词表
- 脱敏: 手机号/身份证 打码
"""
from __future__ import annotations

import re
from typing import Any

_ID_CARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_LANDLINE_RE = re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)")
_BANK_CARD_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

_SENSITIVE_KEYWORDS = [
    "身份证", "护照", "银行卡", "信用卡", "密码", "余额", "转账", "汇款",
    "血压", "心率", "病历", "体检", "吃药", "处方", "怀孕", "抑郁症",
    "月薪", "年薪", "工资", "存款", "欠款", "房贷", "婚姻", "离婚",
    "家庭住址", "门牌号", "社保", "公积金",
]


class PrivacyGate:
    """敏感信息检查与脱敏."""

    def check(self, text: str) -> tuple[bool, str]:
        """检查文本是否含敏感信息.

        Returns:
            (is_safe, reason): is_safe=True 表示可采集; False 时 reason 说明原因.
        """
        if not text:
            return True, ""
        if _ID_CARD_RE.search(text):
            return False, "含敏感证件信息"
        if _PHONE_RE.search(text):
            return False, "含敏感联系方式"
        if _LANDLINE_RE.search(text):
            return False, "含敏感联系方式"
        if _BANK_CARD_RE.search(text):
            return False, "含敏感金融信息"
        for kw in _SENSITIVE_KEYWORDS:
            if kw in text:
                return False, f"含敏感关键词: {kw}"
        return True, ""

    def sanitize(self, text: str) -> str:
        """脱敏 (打码) 后返回, 供展示用途."""
        out = _ID_CARD_RE.sub(lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:], text)
        out = _PHONE_RE.sub(lambda m: "****" + m.group(0)[-4:], out)
        out = _BANK_CARD_RE.sub(lambda m: "****" + m.group(0)[-4:], out)
        return out

    def filter_signals(self, signals: list[Any]) -> list[Any]:
        """过滤含敏感内容的信号 payload (字符串字段)."""
        out: list[Any] = []
        for sig in signals:
            payload = sig.payload if hasattr(sig, "payload") else {}
            joined = " ".join(str(v) for v in payload.values())
            ok, _ = self.check(joined)
            if ok:
                out.append(sig)
        return out
