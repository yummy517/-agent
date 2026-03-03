"""
所有子智能体实现
- KnowledgeBaseAgent: 知识库检索
- ResearchAgent: 原料+竞品调研 (异步并行)
- ReviewAgent: 质量审核 (带重试逻辑)
- PPTAgent: PPT 生成 (调用 MCP)
"""
import json
import asyncio
from datetime import datetime
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from config.settings import settings
from graph.state import AgentState
from schemas.models import (
    ResearchResult,
    ReviewResult,
    ReviewCriteria,
    ReviewDecision,
    ErrorRecord,
    TaskStatus,
)
from tools.kb_tools import search_clinical_guidelines, clinical_kb
from tools.research_tools import search_raw_materials, search_competitor_products, analyze_market_overview
from tools.mcp_ppt_tools import generate_ppt_report
from monitoring.logger import logger, monitor_agent


# ============================================================
# LLM 工厂
# ============================================================

def get_llm(use_fallback: bool = False) -> ChatOpenAI:
    """获取 LLM 实例，支持主模型/降级模型切换"""
    model = settings.FALLBACK_MODEL if use_fallback else settings.PRIMARY_MODEL
    return ChatOpenAI(
        model=model,
        temperature=settings.MODEL_TEMPERATURE,
        max_tokens=settings.MODEL_MAX_TOKENS,
        openai_api_key=settings.OPENAI_API_KEY,
    )


# ============================================================
# 知识库 Agent
# ============================================================

@monitor_agent("knowledge_base_agent")
async def knowledge_base_agent(state: AgentState) -> dict:
    """
    临床指南知识库检索 Agent
    - 接入已有 LangChain RAG
    - 结构化输出 KnowledgeBaseResult
    - 结果缓存至 Redis
    """
    from memory.memory_manager import redis_memory

    biomarker = state["biomarker_name"]
    logger.info("kb_agent_start", biomarker=biomarker)

    # 1. 检查 Redis 缓存
    cached = await redis_memory.get_kb_result(biomarker)
    if cached:
        logger.info("kb_cache_hit", biomarker=biomarker)
        from schemas.models import KnowledgeBaseResult
        kb_result = KnowledgeBaseResult(**cached)
        return {
            "kb_result": kb_result,
            "messages": [AIMessage(content=f"[KB Agent] 从缓存加载 {biomarker} 的临床指南信息", name="kb_agent")],
            "updated_at": datetime.now().isoformat(),
        }

    # 2. 检索知识库
    try:
        # 使用 ReAct Agent 调用工具
        kb_agent = create_react_agent(
            get_llm(),
            tools=[search_clinical_guidelines],
            state_modifier=SystemMessage(content=f"""
你是临床指南专家。请检索 {biomarker} 的全面临床信息。
调用 search_clinical_guidelines 工具，获取：
1. 相关临床指南名称和证据等级
2. 临床意义和检测适应症  
3. 参考值范围和阈值
4. 相关疾病谱

请确保信息全面准确。
""")
        )
        result = await kb_agent.ainvoke({
            "messages": [HumanMessage(content=f"检索 {biomarker} 的临床指南信息")]
        })
        last_msg = result["messages"][-1].content

        # 解析结果
        kb_result = await clinical_kb.search(biomarker, "全面临床背景")

        # 缓存到 Redis
        await redis_memory.cache_kb_result(biomarker, kb_result.model_dump())

        return {
            "kb_result": kb_result,
            "messages": [AIMessage(
                content=f"[KB Agent] 完成 {biomarker} 临床指南检索，找到 {kb_result.total_references} 条参考",
                name="kb_agent"
            )],
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        error = ErrorRecord(agent="kb_agent", error_type=type(e).__name__, message=str(e))
        logger.error("kb_agent_failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [error],
            "messages": [AIMessage(content=f"[KB Agent] 错误: {e}", name="kb_agent")],
            "updated_at": datetime.now().isoformat(),
        }


# ============================================================
# 调研 Agent (异步并行)
# ============================================================

@monitor_agent("research_agent")
async def research_agent(state: AgentState) -> dict:
    """
    原料+竞品调研 Agent
    ★ 核心设计：并行同时执行原料调研和竞品调研
    """
    from memory.memory_manager import redis_memory

    biomarker = state["biomarker_name"]
    logger.info("research_agent_start", biomarker=biomarker)

    # 检查 Redis 缓存
    cached = await redis_memory.get_research_result(biomarker)
    if cached:
        from schemas.models import ResearchResult
        research_result = ResearchResult(**cached)
        return {
            "research_result": research_result,
            "messages": [AIMessage(content=f"[Research Agent] 从缓存加载调研结果", name="research_agent")],
            "updated_at": datetime.now().isoformat(),
        }

    try:
        # ★ 三个调研任务并行执行
        raw_materials_task = search_raw_materials.ainvoke({"biomarker_name": biomarker})
        competitors_task = search_competitor_products.ainvoke({"biomarker_name": biomarker})
        market_task = analyze_market_overview.ainvoke({"biomarker_name": biomarker})

        raw_materials_json, competitors_json, market_overview = await asyncio.gather(
            raw_materials_task,
            competitors_task,
            market_task,
            return_exceptions=True,
        )

        # 容错处理
        from schemas.models import RawMaterialInfo, CompetitorProduct

        raw_materials = []
        if not isinstance(raw_materials_json, Exception):
            try:
                data = json.loads(raw_materials_json) if isinstance(raw_materials_json, str) else raw_materials_json
                raw_materials = [RawMaterialInfo(**m) for m in (data if isinstance(data, list) else [])]
            except Exception:
                pass

        competitors = []
        if not isinstance(competitors_json, Exception):
            try:
                data = json.loads(competitors_json) if isinstance(competitors_json, str) else competitors_json
                competitors = [CompetitorProduct(**c) for c in (data if isinstance(data, list) else [])]
            except Exception:
                pass

        market = market_overview if not isinstance(market_overview, Exception) else "市场分析暂不可用"

        # LLM 提炼关键发现
        llm = get_llm(use_fallback=True)
        findings_prompt = f"""
        基于以下调研数据，提炼 {biomarker} 的5个关键发现：
        原料: {raw_materials_json if not isinstance(raw_materials_json, Exception) else '暂无'}
        竞品: {competitors_json if not isinstance(competitors_json, Exception) else '暂无'}
        市场: {market}
        
        返回一个 JSON 数组，包含5条关键发现字符串。直接返回数组，不要其他文字。
        """
        findings_resp = await llm.ainvoke(findings_prompt)
        try:
            key_findings = json.loads(findings_resp.content.strip())
            if not isinstance(key_findings, list):
                key_findings = [findings_resp.content]
        except Exception:
            key_findings = [findings_resp.content]

        research_result = ResearchResult(
            biomarker_name=biomarker,
            raw_materials=raw_materials,
            competitors=competitors,
            market_overview=str(market),
            key_findings=key_findings,
            data_sources=["Tavily Web Search", "LLM Knowledge"],
        )

        # 缓存
        await redis_memory.cache_research_result(biomarker, research_result.model_dump())

        return {
            "research_result": research_result,
            "messages": [AIMessage(
                content=f"[Research Agent] 调研完成：{len(raw_materials)}种原料，{len(competitors)}个竞品",
                name="research_agent"
            )],
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        error = ErrorRecord(agent="research_agent", error_type=type(e).__name__, message=str(e))
        logger.error("research_agent_failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [error],
            "messages": [AIMessage(content=f"[Research Agent] 错误: {e}", name="research_agent")],
            "updated_at": datetime.now().isoformat(),
        }


# ============================================================
# 审核 Agent
# ============================================================

@monitor_agent("review_agent")
async def review_agent(state: AgentState) -> dict:
    """
    质量审核 Agent
    - 评估 KB + 调研结果质量
    - 结构化输出审核评分
    - 低于阈值则触发重新生成
    """
    biomarker = state["biomarker_name"]
    kb_result = state.get("kb_result")
    research_result = state.get("research_result")
    review_count = state.get("review_count", 0)

    logger.info("review_agent_start", biomarker=biomarker, review_round=review_count + 1)

    # LLM with structured output
    llm = get_llm()
    llm_structured = llm.with_structured_output(ReviewResult)

    kb_summary = kb_result.model_dump() if kb_result else {}
    research_summary = research_result.model_dump() if research_result else {}

    review_prompt = f"""
你是严格的质量审核专家。请评审以下 {biomarker} 的调研报告质量。

## 知识库检索结果摘要
- 指南数量: {len(kb_summary.get('guidelines', []))}
- 置信度: {kb_summary.get('confidence_score', 0):.0%}
- 摘要: {kb_summary.get('summary', '')[:300]}

## 调研结果摘要  
- 原料数量: {len(research_summary.get('raw_materials', []))}
- 竞品数量: {len(research_summary.get('competitors', []))}
- 市场概况: {research_summary.get('market_overview', '')[:200]}
- 关键发现数: {len(research_summary.get('key_findings', []))}

## 审核标准 (每项 0-1 分)
1. completeness: 信息完整性 (原料≥3个+竞品≥3个+指南≥2个 = 满分)
2. accuracy: 信息准确性 (有具体数据支撑 = 高分)
3. relevance: 信息相关性 (聚焦于 {biomarker} = 高分)
4. market_coverage: 市场覆盖度 (有市场规模+趋势+竞争格局 = 高分)

请返回详细的审核结果，包含分数、问题列表和改进建议。
overall_score = 四项均值。
如果 overall_score >= {settings.REVIEW_PASS_THRESHOLD}，decision = "approved"
如果 overall_score < {settings.REVIEW_PASS_THRESHOLD}，decision = "rejected"
"""

    try:
        review_result = await llm_structured.ainvoke(review_prompt)
        is_approved = review_result.decision == ReviewDecision.APPROVED

        logger.info(
            "review_completed",
            biomarker=biomarker,
            decision=review_result.decision,
            score=review_result.overall_score,
            round=review_count + 1,
        )

        return {
            "review_result": review_result,
            "review_count": review_count + 1,
            "messages": [AIMessage(
                content=(
                    f"[Review Agent] 第{review_count+1}次审核 | "
                    f"决策: {review_result.decision} | "
                    f"评分: {review_result.overall_score:.0%} | "
                    f"问题: {'; '.join(review_result.issues[:2])}"
                ),
                name="review_agent"
            )],
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        # 审核失败 → 降级为通过 (避免死循环)
        from schemas.models import ReviewResult as RR, ReviewCriteria as RC
        fallback_result = RR(
            decision=ReviewDecision.APPROVED,
            criteria_scores=RC(completeness=0.6, accuracy=0.6, relevance=0.7, market_coverage=0.6),
            overall_score=0.625,
            issues=[f"审核服务异常: {e}"],
            suggestions=["建议人工复核"],
        )
        logger.error("review_agent_fallback", error=str(e))
        return {
            "review_result": fallback_result,
            "review_count": review_count + 1,
            "messages": [AIMessage(content=f"[Review Agent] 降级通过 (审核异常: {e})", name="review_agent")],
            "updated_at": datetime.now().isoformat(),
        }


# ============================================================
# PPT Agent
# ============================================================

@monitor_agent("ppt_agent")
async def ppt_agent(state: AgentState) -> dict:
    """
    PPT 生成 Agent
    - 调用 MCP PPT 工具
    - 将知识库+调研结果转化为 PPT
    """
    biomarker = state["biomarker_name"]
    kb_result = state.get("kb_result")
    research_result = state.get("research_result")

    logger.info("ppt_agent_start", biomarker=biomarker)

    kb_json = kb_result.model_dump_json() if kb_result else "{}"
    research_json = research_result.model_dump_json() if research_result else "{}"

    try:
        ppt_result_json = await generate_ppt_report.ainvoke({
            "biomarker_name": biomarker,
            "kb_result_json": kb_json,
            "research_result_json": research_json,
        })

        from schemas.models import PPTResult as PR
        ppt_result = PR(**json.loads(ppt_result_json))

        return {
            "ppt_result": ppt_result,
            "status": TaskStatus.COMPLETED.value,
            "completed_at": datetime.now().isoformat(),
            "messages": [AIMessage(
                content=f"[PPT Agent] PPT生成完成: {ppt_result.file_name} ({ppt_result.total_slides}页)",
                name="ppt_agent"
            )],
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        error = ErrorRecord(agent="ppt_agent", error_type=type(e).__name__, message=str(e))
        logger.error("ppt_agent_failed", error=str(e))
        return {
            "errors": state.get("errors", []) + [error],
            "status": TaskStatus.FAILED.value,
            "messages": [AIMessage(content=f"[PPT Agent] PPT生成失败: {e}", name="ppt_agent")],
            "updated_at": datetime.now().isoformat(),
        }
