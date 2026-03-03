"""
LangGraph 核心状态定义
使用 TypedDict + Annotated 实现可合并的状态
"""
from typing import TypedDict, Annotated, Optional, List, Dict, Any
from datetime import datetime

from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from schemas.models import (
    KnowledgeBaseResult,
    ResearchResult,
    ReviewResult,
    PPTResult,
    ErrorRecord,
    HumanFeedback,
    TaskStatus,
)


class AgentState(TypedDict):
    """
    主图状态 - 贯穿整个多智能体工作流
    所有字段都有默认值，避免 KeyError
    """
    # ── 基础信息 ──────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]  # 消息历史 (自动合并)
    task_id: str
    session_id: str
    biomarker_name: str
    task_description: Optional[str]

    # ── 流程控制 ──────────────────────────────────────────────
    current_agent: str                   # 当前执行的 Agent
    next_agent: str                      # 下一个路由目标
    iteration_count: int                 # 全局迭代次数 (≤3)
    review_count: int                    # 审核轮次 (≤3)
    status: str                          # TaskStatus 枚举值

    # ── 子任务结果 ────────────────────────────────────────────
    kb_result: Optional[KnowledgeBaseResult]       # 知识库检索结果
    research_result: Optional[ResearchResult]      # 调研结果
    review_result: Optional[ReviewResult]          # 审核结果
    ppt_result: Optional[PPTResult]                # PPT 生成结果

    # ── 人工协作 ──────────────────────────────────────────────
    awaiting_human: bool                  # 是否等待人工
    human_feedback: Optional[HumanFeedback]
    human_checkpoint_data: Optional[Dict[str, Any]]  # interrupt 暂存数据

    # ── 错误和重试 ────────────────────────────────────────────
    errors: List[ErrorRecord]             # 错误记录列表
    retry_counts: Dict[str, int]          # 每个 Agent 的重试次数 {agent_name: count}
    failed_agents: List[str]              # 已失败的 Agent

    # ── 时间戳 ────────────────────────────────────────────────
    created_at: str
    updated_at: str
    completed_at: Optional[str]


def create_initial_state(
    task_id: str,
    session_id: str,
    biomarker_name: str,
    task_description: Optional[str] = None,
) -> AgentState:
    """创建初始状态"""
    now = datetime.now().isoformat()
    return {
        "messages": [],
        "task_id": task_id,
        "session_id": session_id,
        "biomarker_name": biomarker_name,
        "task_description": task_description,
        "current_agent": "supervisor",
        "next_agent": "supervisor",
        "iteration_count": 0,
        "review_count": 0,
        "status": TaskStatus.PENDING.value,
        "kb_result": None,
        "research_result": None,
        "review_result": None,
        "ppt_result": None,
        "awaiting_human": False,
        "human_feedback": None,
        "human_checkpoint_data": None,
        "errors": [],
        "retry_counts": {},
        "failed_agents": [],
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    }
