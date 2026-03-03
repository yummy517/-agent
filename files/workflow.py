"""
主工作流 - LangGraph 图构建
★ 核心特性：
  - 监管者模式 (Supervisor Pattern)
  - 异步并行 (Send API for parallel KB + Research)
  - 人工协作 (interrupt)
  - MemorySaver checkpointing
  - 全局迭代控制
"""
import asyncio
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Send

from graph.state import AgentState
from graph.agents import knowledge_base_agent, research_agent, review_agent, ppt_agent
from graph.supervisor import (
    supervisor_node,
    route_after_supervisor,
)
from schemas.models import AgentName, HumanFeedback
from monitoring.logger import logger


# ============================================================
# 并行研究节点 (Send API)
# ============================================================

async def parallel_research_node(state: AgentState) -> dict:
    """
    ★ 并行执行 KB检索 + 原料/竞品调研
    使用 asyncio.gather 实现真正的并发
    """
    logger.info("parallel_research_start", biomarker=state["biomarker_name"])

    # 并行执行两个 Agent
    kb_task = knowledge_base_agent(state)
    research_task = research_agent(state)

    kb_update, research_update = await asyncio.gather(
        kb_task, research_task, return_exceptions=True
    )

    # 合并结果，容错处理
    merged = {}

    if isinstance(kb_update, Exception):
        logger.error("kb_parallel_failed", error=str(kb_update))
        from schemas.models import ErrorRecord
        merged["errors"] = state.get("errors", []) + [
            ErrorRecord(agent="kb_agent", error_type=type(kb_update).__name__, message=str(kb_update))
        ]
    else:
        merged.update({k: v for k, v in kb_update.items() if v is not None})

    if isinstance(research_update, Exception):
        logger.error("research_parallel_failed", error=str(research_update))
        from schemas.models import ErrorRecord
        merged.setdefault("errors", state.get("errors", []))
        merged["errors"].append(
            ErrorRecord(agent="research_agent", error_type=type(research_update).__name__, message=str(research_update))
        )
    else:
        merged.update({k: v for k, v in research_update.items() if v is not None})

    # 合并 messages (避免覆盖)
    from langchain_core.messages import AIMessage
    kb_msgs = kb_update.get("messages", []) if not isinstance(kb_update, Exception) else []
    res_msgs = research_update.get("messages", []) if not isinstance(research_update, Exception) else []
    merged["messages"] = kb_msgs + res_msgs

    logger.info("parallel_research_complete",
                has_kb=merged.get("kb_result") is not None,
                has_research=merged.get("research_result") is not None)
    return merged


# ============================================================
# 人工审核节点 (interrupt)
# ============================================================

async def human_review_node(state: AgentState) -> dict:
    """
    ★ 人工协作检查点
    使用 LangGraph interrupt() 暂停图执行
    外部 API 通过 graph.update_state() 恢复
    """
    from langchain_core.messages import AIMessage

    checkpoint_data = state.get("human_checkpoint_data", {})
    review_result = state.get("review_result")

    # 构建给人工的审核摘要
    review_summary = {
        "biomarker": state["biomarker_name"],
        "review_score": review_result.overall_score if review_result else None,
        "review_decision": review_result.decision if review_result else None,
        "issues": review_result.issues if review_result else [],
        "suggestions": review_result.suggestions if review_result else [],
        "data_summary": checkpoint_data,
    }

    logger.info("human_review_interrupt", biomarker=state["biomarker_name"])

    # ★ interrupt() - 暂停图，等待人工输入
    # 外部调用 graph.update_state(config, {"human_feedback": {...}}) 来恢复
    human_input = interrupt({
        "action": "human_review_required",
        "task_id": state["task_id"],
        "review_summary": review_summary,
        "instructions": "请审核以下内容，通过则继续生成PPT，拒绝则重新调研。",
    })

    # 解析人工输入
    if isinstance(human_input, dict):
        feedback = HumanFeedback(
            approved=human_input.get("approved", True),
            comments=human_input.get("comments"),
            reviewer_name=human_input.get("reviewer_name"),
        )
    else:
        # 默认通过
        feedback = HumanFeedback(approved=True, comments="Auto-approved")

    logger.info(
        "human_review_completed",
        approved=feedback.approved,
        reviewer=feedback.reviewer_name,
    )

    return {
        "human_feedback": feedback,
        "awaiting_human": False,
        "messages": [AIMessage(
            content=f"[Human] {'✅ 通过' if feedback.approved else '❌ 拒绝'}: {feedback.comments or ''}",
            name="human"
        )],
    }


# ============================================================
# 构建主图
# ============================================================

def build_graph(checkpointer: MemorySaver = None) -> StateGraph:
    """
    构建完整的多智能体工作流图

    拓扑结构:
    START → supervisor → parallel_research ─┐
              ↑                              ↓
              └──── review_agent ←───────── (合并)
              ↓
          human_review
              ↓
          ppt_agent → END
    """
    builder = StateGraph(AgentState)

    # ── 注册节点 ──────────────────────────────────────────────
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("parallel_research", parallel_research_node)
    builder.add_node("review_agent", review_agent)
    builder.add_node("human_review", human_review_node)
    builder.add_node("ppt_agent", ppt_agent)

    # ── 入口 ──────────────────────────────────────────────────
    builder.add_edge(START, "supervisor")

    # ── Supervisor 条件路由 ───────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "parallel_research": "parallel_research",
            "review_agent": "review_agent",
            "human_review": "human_review",
            "ppt_agent": "ppt_agent",
            "__end__": END,
        }
    )

    # ── 子节点回到 Supervisor ─────────────────────────────────
    builder.add_edge("parallel_research", "supervisor")
    builder.add_edge("review_agent", "supervisor")
    builder.add_edge("human_review", "supervisor")
    builder.add_edge("ppt_agent", END)

    # ── 编译 ─────────────────────────────────────────────────
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer
    # interrupt_before human_review 节点前暂停
    compile_kwargs["interrupt_before"] = ["human_review"]

    graph = builder.compile(**compile_kwargs)
    logger.info("graph_compiled", nodes=list(builder.nodes.keys()))
    return graph


# ============================================================
# 图管理器 (单例)
# ============================================================

class GraphManager:
    """管理图实例和 checkpointer"""

    def __init__(self):
        self._checkpointer = MemorySaver()
        self._graph = build_graph(self._checkpointer)

    @property
    def graph(self):
        return self._graph

    @property
    def checkpointer(self):
        return self._checkpointer

    def get_config(self, thread_id: str) -> dict:
        """获取图执行配置"""
        return {
            "configurable": {
                "thread_id": thread_id,
            },
            "recursion_limit": 20,  # 防止无限循环
        }

    async def run(self, initial_state: dict, thread_id: str) -> dict:
        """异步运行图"""
        config = self.get_config(thread_id)
        final_state = None
        async for chunk in self._graph.astream(initial_state, config=config, stream_mode="updates"):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            logger.info("graph_step", node=node_name, status=node_output.get("status", ""))
            final_state = node_output
        return final_state

    async def resume_with_human_feedback(
        self,
        thread_id: str,
        feedback: dict,
    ) -> dict:
        """
        恢复被 interrupt 暂停的图
        注入人工反馈
        """
        config = self.get_config(thread_id)

        # 通过 Command 恢复 interrupt
        from langgraph.types import Command
        final_state = None
        async for chunk in self._graph.astream(
            Command(resume=feedback),
            config=config,
            stream_mode="updates",
        ):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            logger.info("graph_resumed", node=node_name)
            final_state = node_output

        return final_state

    def get_state(self, thread_id: str) -> Any:
        """获取当前图状态 (时间旅行)"""
        config = self.get_config(thread_id)
        return self._graph.get_state(config)

    def get_state_history(self, thread_id: str) -> list:
        """获取状态历史"""
        config = self.get_config(thread_id)
        return list(self._graph.get_state_history(config))


# 全局图管理器
graph_manager = GraphManager()
