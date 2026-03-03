"""
调研工具 - 原料信息 + 竞品分析
使用 Tavily 网络搜索 + 结构化输出
"""
import json
import asyncio
from typing import List

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from config.settings import settings
from schemas.models import ResearchResult, RawMaterialInfo, CompetitorProduct
from monitoring.logger import logger


# ============================================================
# 搜索引擎工具
# ============================================================

def _get_search_tool():
    """获取搜索工具 (Tavily / 降级到模拟)"""
    if settings.TAVILY_API_KEY:
        return TavilySearchResults(
            max_results=settings.SEARCH_MAX_RESULTS,
            tavily_api_key=settings.TAVILY_API_KEY,
        )
    return None


# ============================================================
# 调研 Agent Tools
# ============================================================

@tool
async def search_raw_materials(biomarker_name: str) -> str:
    """
    搜索生物标志物的原料信息：抗体、抗原、酶等关键原料。
    
    Args:
        biomarker_name: 标志物名称
    
    Returns:
        原料信息列表 (JSON)
    """
    search = _get_search_tool()
    queries = [
        f"{biomarker_name} antibody raw material supplier specifications",
        f"{biomarker_name} immunoassay reagent antigen protein",
        f"{biomarker_name} 抗体原料 供应商 规格",
    ]

    all_results = []
    if search:
        # 并行搜索
        tasks = [asyncio.to_thread(search.invoke, q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if not isinstance(r, Exception) and r:
                all_results.extend(r if isinstance(r, list) else [r])

    # LLM 提取结构化信息
    llm = ChatOpenAI(
        model=settings.PRIMARY_MODEL,
        temperature=0.1,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    llm_structured = llm.with_structured_output(
        {"type": "array", "items": RawMaterialInfo.model_json_schema()}
    )

    extract_prompt = f"""
    基于以下搜索结果，提取 {biomarker_name} 的原料信息列表。
    如果搜索结果不足，基于行业知识补充合理信息。
    
    搜索结果：
    {json.dumps(all_results[:3], ensure_ascii=False, indent=2)}
    
    请提取3-5个主要原料供应商的信息，包括：
    - material_name: 原料名称
    - supplier: 供应商
    - purity: 纯度要求
    - specifications: 规格参数
    - estimated_cost: 估计成本
    - availability: 可获得性
    """
    try:
        materials = await llm.ainvoke(extract_prompt)
        return materials.content
    except Exception as e:
        logger.warning("raw_material_extraction_error", error=str(e))
        # Mock 降级
        mock = [
            RawMaterialInfo(
                material_name=f"{biomarker_name} 单克隆抗体",
                supplier="Abcam / R&D Systems",
                purity=">95%",
                specifications={"host": "Mouse", "clone": "待确认"},
                estimated_cost="$500-2000/mg",
                availability="现货",
            )
        ]
        return json.dumps([m.model_dump() for m in mock], ensure_ascii=False, indent=2)


@tool
async def search_competitor_products(biomarker_name: str) -> str:
    """
    搜索竞品信息：主要厂商产品、市场份额、优劣势分析。
    
    Args:
        biomarker_name: 标志物名称
    
    Returns:
        竞品信息列表 (JSON)
    """
    search = _get_search_tool()
    queries = [
        f"{biomarker_name} IVD kit manufacturer market share 2024",
        f"{biomarker_name} diagnostic test competitor comparison",
        f"{biomarker_name} 体外诊断 竞品 市场分析",
    ]

    all_results = []
    if search:
        tasks = [asyncio.to_thread(search.invoke, q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if not isinstance(r, Exception) and r:
                all_results.extend(r if isinstance(r, list) else [r])

    llm = ChatOpenAI(
        model=settings.PRIMARY_MODEL,
        temperature=0.1,
        openai_api_key=settings.OPENAI_API_KEY,
    )

    extract_prompt = f"""
    基于搜索结果，提取 {biomarker_name} 诊断试剂的主要竞品信息。
    请提取3-6个主要竞品产品，每个包含：
    - product_name: 产品名称
    - manufacturer: 厂商
    - market_share: 市场份额(估计)
    - key_features: 主要特性列表
    - price_range: 价格范围
    - regulatory_status: 注册状态(CE/FDA/NMPA)
    - strengths: 优势
    - weaknesses: 劣势
    
    搜索结果：{json.dumps(all_results[:3], ensure_ascii=False)[:2000]}
    
    请直接返回JSON数组，不要其他文字。
    """
    try:
        response = await llm.ainvoke(extract_prompt)
        return response.content
    except Exception as e:
        logger.warning("competitor_extraction_error", error=str(e))
        mock = [
            CompetitorProduct(
                product_name=f"{biomarker_name} ELISA Kit",
                manufacturer="Roche Diagnostics",
                market_share="~30%",
                key_features=["全自动", "高通量", "CV<5%"],
                price_range="$200-500/kit",
                regulatory_status="CE/FDA",
                strengths=["品牌知名度高", "技术成熟"],
                weaknesses=["价格较高", "需要专用仪器"],
            )
        ]
        return json.dumps([m.model_dump() for m in mock], ensure_ascii=False, indent=2)


@tool
async def analyze_market_overview(biomarker_name: str) -> str:
    """
    分析标志物相关市场概况：市场规模、增长趋势、竞争格局。
    
    Args:
        biomarker_name: 标志物名称
    
    Returns:
        市场分析摘要
    """
    search = _get_search_tool()
    query = f"{biomarker_name} diagnostics market size growth 2024 2025"
    
    results = []
    if search:
        try:
            results = await asyncio.to_thread(search.invoke, query)
        except Exception:
            pass

    llm = ChatOpenAI(
        model=settings.FALLBACK_MODEL,  # 市场分析用较便宜的模型
        temperature=0.2,
        openai_api_key=settings.OPENAI_API_KEY,
    )
    prompt = f"""
    基于以下信息，写一段200字左右的 {biomarker_name} 市场概况分析。
    包括：市场规模、主要增长驱动因素、市场竞争格局。
    
    参考信息：{json.dumps(results[:2], ensure_ascii=False)[:1500]}
    """
    response = await llm.ainvoke(prompt)
    return response.content
