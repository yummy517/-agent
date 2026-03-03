"""
监管者 (Supervisor) Agent
- 任务调度和路由决策
- 全局迭代计数控制 (≤3次)
- 结合 LLM 判断 + 规则路由
"""
from datetime import datetime
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from config.settings import settings
from graph.state import AgentState
from schemas.models import TaskStatus, ReviewDecision, AgentName
from monitoring.logger import logger


# ============================================================
# 路由类型
# ============================================================

SupervisorRoute = Literal[
    "knowledge_base_agent",
    "research_agent",
    "parallel_research",   # KB + Research 并行
    "review_agent",
    "human_review",
    "ppt_agent",
    "__end__",
]


# ============================================================
# 监管者核心逻辑
# ============================================================

async def supervisor_node(state: AgentState) -> dict:
    """
    监管者节点
    决定下一步执行哪个 Agent
    """
    iteration = state.get("iteration_count", 0)
    review_count = state.get("review_count", 0)
    kb_result = state.get("kb_result")
    research_result = state.get("research_result")
    review_result = state.get("review_result")
    ppt_result = state.get("ppt_result")
    errors = state.get("errors", [])
    biomarker = state["biomarker_name"]

    logger.info(
        "supervisor_routing",
        biomarker=biomarker,
        iteration=iteration,
        review_count=review_count,
        has_kb=kb_result is not None,
        has_research=research_result is not None,
        has_review=review_result is not None,
    )

    # ── 全局迭代上限检查 ──────────────────────────────────────
    if iteration >= settings.MAX_TOTAL_ITERATIONS:
        logger.warning(
            "max_iterations_reached",
            biomarker=biomarker,
            iteration=iteration,
        )
        return {
            "next_agent": AgentName.END.value,
            "status": TaskStatus.FAILED.value,
            "messages": [AIMessage(
                content=f"[Supervisor] ⚠️ 已达最大迭代次数({settings.MAX_TOTAL_ITERATIONS})，任务终止",
                name="supervisor"
            )],
            "updated_at": datetime.now().isoformat(),
        }

    # ── 路由决策 ──────────────────────────────────────────────

    # Step 1: 如果 KB 和 Research 都没有 → 触发并行收集
    if kb_result is None and research_result is None:
        return {
            "next_agent": "parallel_research",
            "iteration_count": iteration + 1,
            "status": TaskStatus.RUNNING.value,
            "messages": [AIMessage(
                content=f"[Supervisor] 第{iteration+1}轮 → 启动并行调研 (KB + Research)",
                name="supervisor"
            )],
            "updated_at": datetime.now().isoformat(),
        }

    # Step 2: 有数据但没审核 → 触发审核
    if kb_result is not None and research_result is not None and review_result is None:
        return {
            "next_agent": AgentName.REVIEW.value,
            "status": TaskStatus.RUNNING.value,
            "messages": [AIMessage(
                content="[Supervisor] 数据收集完成 → 进入审核",
                name="supervisor"
            )],
            "updated_at": datetime.now().isoformat(),
        }

    # Step 3: 审核不通过 → 重试
    if review_result is not None and review_result.decision == ReviewDecision.REJECTED:
        if review_count >= settings.MAX_REVIEW_RETRIES:
            logger.warning("max_review_retries_reached", biomarker=biomarker)
            # 超过重试次数 → 强制进入人工审核
            return {
                "next_agent": "human_review",
                "awaiting_human": True,
                "status": TaskStatus.AWAITING_HUMAN.value,
                "messages": [AIMessage(
                    content=f"[Supervisor] 审核已重试{review_count}次，需人工介入",
                    name="supervisor"
                )],
                "updated_at": datetime.now().isoformat(),
            }

        # 清除旧结果，触发重新生成
        return {
            "next_agent": "parallel_research",
            "kb_result": None,
            "research_result": None,
            "review_result": None,
            "iteration_count": iteration + 1,
            "status": TaskStatus.RUNNING.value,
            "messages": [AIMessage(
                content=(
                    f"[Supervisor] 审核不通过 (评分:{review_result.overall_score:.0%}) "
                    f"→ 第{iteration+1}轮重新生成"
                ),
                name="supervisor"
            )],
            "updated_at": datetime.now().isoformat(),
        }

    # Step 4: 审核通过 → 触发人工审核检查点
    if review_result is not None and review_result.decision == ReviewDecision.APPROVED:
        human_feedback = state.get("human_feedback")
        if human_feedback is None:
            # 首次通过 → 请求人工确认
            return {
                "next_agent": "human_review",
                "awaiting_human": True,
                "status": TaskStatus.AWAITING_HUMAN.value,
                "human_checkpoint_data": {
                    "kb_summary": kb_result.model_dump() if kb_result else {},
                    "research_summary": {
                        "raw_materials_count": len(research_result.raw_materials) if research_result else 0,
                        "competitors_count": len(research_result.competitors) if research_result else 0,
                        "market_overview": research_result.market_overview[:200] if research_result else "",
                    },
                    "review_score": review_result.overall_score,
                    "review_issues": review_result.issues,
                },
                "messages": [AIMessage(
                    content=f"[Supervisor] 审核通过 (评分:{review_result.overall_score:.0%}) → 等待人工确认",
                    name="supervisor"
                )],
                "updated_at": datetime.now().isoformat(),
            }

        # 人工已审核
        if not human_feedback.approved:
            # 人工拒绝 → 重新生成
            return {
                "next_agent": "parallel_research",
                "kb_result": None,
                "research_result": None,
                "review_result": None,
                "human_feedback": None,
                "iteration_count": iteration + 1,
                "status": TaskStatus.RUNNING.value,
                "messages": [AIMessage(
                    content=f"[Supervisor] 人工拒绝 → 第{iteration+1}轮重新生成",
                    name="supervisor"
                )],
                "updated_at": datetime.now().isoformat(),
            }

        # 人工通过 → 生成 PPT
        return {
            "next_agent": AgentName.PPT.value,
            "awaiting_human": False,
            "status": TaskStatus.RUNNING.value,
            "messages": [AIMessage(
                content="[Supervisor] 人工确认通过 → 生成PPT",
                name="supervisor"
            )],
            "updated_at": datetime.now().isoformat(),
        }

    # Step 5: PPT 已生成 → 完成
    if ppt_result is not None:
        return {
            "next_agent": AgentName.END.value,
            "status": TaskStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "messages": [AIMessage(
                content=f"[Supervisor] ✅ 任务完成！PPT: {ppt_result.file_name}",
                name="supervisor"
            )],
            "updated_at": datetime.now().isoformat(),
        }

    # 默认
    return {
        "next_agent": AgentName.END.value,
        "status": TaskStatus.FAILED.value,
        "messages": [AIMessage(content="[Supervisor] 未知状态，任务终止", name="supervisor")],
        "updated_at": datetime.now().isoformat(),
    }


# ============================================================
# 路由函数 (供 LangGraph conditional_edges 使用)
# ============================================================

def route_after_supervisor(state: AgentState) -> SupervisorRoute:
    """根据 supervisor 决策返回路由目标"""
    return state.get("next_agent", "__end__")


def route_after_review(state: AgentState) -> SupervisorRoute:
    """审核完成后的路由"""
    return "supervisor"


def route_after_parallel(state: AgentState) -> SupervisorRoute:
    """并行完成后的路由"""
    return "supervisor"


def route_after_human(state: AgentState) -> SupervisorRoute:
    """人工审核后的路由"""
    return "supervisor"
