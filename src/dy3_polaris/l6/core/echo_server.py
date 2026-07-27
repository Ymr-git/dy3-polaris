"""Echo 验证 MCP Server.

用于验证 Dy3MCPServer 基类的核心功能：
- 工具注册与调用
- 限流中间件
- 溯源 KPA 链构建
- 健康检查

启动方式: python -m dy3_polaris.l6.core.echo_server
"""

from __future__ import annotations

from dy3_polaris.l6.core.models import (
    Dy3ToolAnnotations,
    LayerTag,
    ToolCategory,
    ToolRegistration,
)
from dy3_polaris.l6.core.server import Dy3MCPServer


# 创建 Server 实例
server = Dy3MCPServer(
    name="dy3-echo",
    layer=LayerTag.L6_PROTOCOL,
)


# ---- 注册工具 ----

BKT_SCHEMA = {
    "type": "object",
    "properties": {
        "learner_id": {"type": "string", "description": "学习者 ID"},
        "kp_id": {"type": "string", "description": "知识点 ID"},
        "response": {"type": "boolean", "description": "本次是否答对"},
    },
    "required": ["learner_id", "kp_id", "response"],
}


async def bkt_compute(learner_id: str, kp_id: str, response: bool) -> dict:
    """贝叶斯知识追踪 - 计算知识点掌握概率.

    BKT 四状态模型：P(L0)=0.15, P(T)=0.20, P(G)=0.25, P(S)=0.10
    返回更新后的 P(L_n) 和状态概率分布。
    """
    P_L0, P_T, P_G, P_S = 0.15, 0.20, 0.25, 0.10

    # 简化：stub 返回固定结果
    if response:
        p_know = (P_L0 + (1 - P_S) * 0.8) / (P_L0 + (1 - P_S) * 0.8 + (1 - P_L0) * P_G)
    else:
        p_know = (P_L0 * P_S) / (P_L0 * P_S + (1 - P_L0) * (1 - P_G))

    return {
        "learner_id": learner_id,
        "kp_id": kp_id,
        "p_know": round(p_know, 4),
        "p_state": {
            "know_correct": round(p_know * (1 - P_S), 4),
            "know_incorrect": round(p_know * P_S, 4),
            "dont_know_correct": round((1 - p_know) * P_G, 4),
            "dont_know_incorrect": round((1 - p_know) * (1 - P_G), 4),
        },
    }


server.register_dy3_tool(
    registration=ToolRegistration(
        name="bkt_compute",
        description="贝叶斯知识追踪计算，返回知识点掌握概率 P(L_n) 和四状态概率分布",
        input_schema=BKT_SCHEMA,
        annotations=Dy3ToolAnnotations(
            tags=["bkt", "personalization", "L2"],
            layer=LayerTag.L2_PERSONALIZATION,
            category=ToolCategory.INTERNAL,
            estimated_latency_ms=100,
            domain_scope=["DOM-A", "DOM-B", "DOM-C", "DOM-D"],
            rate_limit=200,
        ),
    ),
    handler=bkt_compute,
)


@server.tool()
async def echo(message: str) -> str:
    """回显输入消息，用于连接验证."""
    return f"Dy3+ Echo: {message}"


@server.tool()
async def health() -> dict:
    """健康检查端点."""
    return await server.health_check()


if __name__ == "__main__":
    server.run(transport="stdio")
