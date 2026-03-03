"""
MCP PPT 工具集成
通过 MCP (Model Context Protocol) 调用 PPT 生成服务
"""
import json
import asyncio
from typing import Optional

import httpx
from langchain_core.tools import tool

from config.settings import settings
from schemas.models import PPTResult, PPTSlide
from monitoring.logger import logger


# ============================================================
# MCP 客户端
# ============================================================

class MCPPPTClient:
    """MCP PPT 工具客户端"""

    def __init__(self):
        self.base_url = settings.MCP_SERVER_URL
        self.timeout = httpx.Timeout(120.0)

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用 MCP 工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            }
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/mcp",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            if "error" in result:
                raise RuntimeError(f"MCP Error: {result['error']}")
            return result.get("result", {})

    async def create_ppt(self, ppt_config: dict) -> dict:
        """创建 PPT"""
        return await self.call_tool(settings.MCP_PPT_TOOL_NAME, ppt_config)

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False


mcp_client = MCPPPTClient()


# ============================================================
# PPT 内容构建器
# ============================================================

def build_ppt_config(
    biomarker_name: str,
    kb_result_json: str,
    research_result_json: str,
    output_path: str,
) -> dict:
    """
    将调研结果构建为 PPT 配置
    传给 MCP 工具
    """
    kb = json.loads(kb_result_json) if isinstance(kb_result_json, str) else kb_result_json
    research = json.loads(research_result_json) if isinstance(research_result_json, str) else research_result_json

    slides = [
        {
            "type": "title",
            "title": f"{biomarker_name} 标志物分析报告",
            "subtitle": "基于临床指南与市场调研",
            "layout": "title_slide",
        },
        {
            "type": "content",
            "title": "目录",
            "content": [
                "1. 标志物临床背景",
                "2. 临床指南证据",
                "3. 原料信息分析",
                "4. 竞品市场分析",
                "5. 市场机会总结",
                "6. 战略建议",
            ],
            "layout": "table_of_contents",
        },
        {
            "type": "content",
            "title": f"一、{biomarker_name} 临床背景",
            "content": [
                kb.get("summary", "临床背景信息"),
                f"参考指南数量：{kb.get('total_references', 0)}",
                f"信息置信度：{kb.get('confidence_score', 0):.0%}",
            ],
            "layout": "content_slide",
        },
        {
            "type": "table",
            "title": "二、临床指南证据汇总",
            "headers": ["指南名称", "版本", "证据等级", "相关疾病"],
            "rows": [
                [
                    g.get("guideline_name", "")[:30],
                    g.get("version", ""),
                    g.get("evidence_level", ""),
                    ", ".join(g.get("relevant_diseases", [])[:2]),
                ]
                for g in (kb.get("guidelines") or [])[:5]
            ],
            "layout": "table_slide",
        },
        {
            "type": "content",
            "title": "三、关键原料信息",
            "content": [
                f"• {m.get('material_name', '')}: {m.get('supplier', '')} | {m.get('purity', '')}"
                for m in (research.get("raw_materials") or [])[:5]
            ],
            "layout": "content_slide",
        },
        {
            "type": "table",
            "title": "四、竞品市场分析",
            "headers": ["产品名称", "厂商", "市场份额", "注册状态", "核心优势"],
            "rows": [
                [
                    c.get("product_name", "")[:25],
                    c.get("manufacturer", ""),
                    c.get("market_share", ""),
                    c.get("regulatory_status", ""),
                    (c.get("strengths") or [""])[0][:30],
                ]
                for c in (research.get("competitors") or [])[:5]
            ],
            "layout": "table_slide",
        },
        {
            "type": "content",
            "title": "五、市场机会",
            "content": [
                research.get("market_overview", "市场分析中..."),
            ] + (research.get("key_findings") or [])[:4],
            "layout": "content_slide",
        },
        {
            "type": "content",
            "title": "六、战略建议",
            "content": [
                "1. 差异化竞争：重点突破竞品薄弱环节",
                "2. 原料供应链：与优质供应商建立战略合作",
                "3. 注册策略：优先完成 NMPA 注册，兼顾 CE",
                "4. 市场切入：从核心医院场景切入，再扩展基层",
            ],
            "layout": "content_slide",
        },
        {
            "type": "closing",
            "title": "谢谢",
            "subtitle": "如有问题，欢迎交流",
            "layout": "closing_slide",
        },
    ]

    return {
        "title": f"{biomarker_name}标志物分析报告",
        "theme": "professional_blue",
        "output_path": output_path,
        "author": "Multi-Agent System",
        "slides": slides,
    }


# ============================================================
# LangChain Tool 封装
# ============================================================

@tool
async def generate_ppt_report(
    biomarker_name: str,
    kb_result_json: str,
    research_result_json: str,
) -> str:
    """
    调用 MCP PPT 工具生成分析报告 PPT。
    
    Args:
        biomarker_name: 标志物名称
        kb_result_json: 知识库检索结果 (JSON字符串)
        research_result_json: 调研结果 (JSON字符串)
    
    Returns:
        PPT生成结果 (JSON)
    """
    import os
    os.makedirs(settings.PPT_OUTPUT_DIR, exist_ok=True)
    safe_name = biomarker_name.replace(" ", "_").replace("/", "-")
    output_path = f"{settings.PPT_OUTPUT_DIR}/{safe_name}_report.pptx"

    ppt_config = build_ppt_config(biomarker_name, kb_result_json, research_result_json, output_path)

    # 检查 MCP 服务是否可用
    mcp_available = await mcp_client.health_check()

    if mcp_available:
        try:
            mcp_result = await mcp_client.create_ppt(ppt_config)
            result = PPTResult(
                file_path=mcp_result.get("file_path", output_path),
                file_name=mcp_result.get("file_name", f"{safe_name}_report.pptx"),
                total_slides=len(ppt_config["slides"]),
                slides_outline=[
                    PPTSlide(
                        slide_number=i + 1,
                        title=s.get("title", ""),
                        content=s.get("content", []),
                        slide_type=s.get("type", "content"),
                    )
                    for i, s in enumerate(ppt_config["slides"])
                ],
            )
            logger.info("ppt_created_via_mcp", file=result.file_path)
            return result.model_dump_json(indent=2)
        except Exception as e:
            logger.error("mcp_ppt_error", error=str(e))

    # 降级: 保存 PPT 配置 JSON (开发环境)
    logger.warning("ppt_fallback_mode", reason="MCP not available")
    fallback_path = output_path.replace(".pptx", "_config.json")
    with open(fallback_path, "w", encoding="utf-8") as f:
        json.dump(ppt_config, f, ensure_ascii=False, indent=2)

    result = PPTResult(
        file_path=fallback_path,
        file_name=f"{safe_name}_config.json",
        total_slides=len(ppt_config["slides"]),
        slides_outline=[
            PPTSlide(
                slide_number=i + 1,
                title=s.get("title", ""),
                content=s.get("content", []),
                slide_type=s.get("type", "content"),
            )
            for i, s in enumerate(ppt_config["slides"])
        ],
    )
    return result.model_dump_json(indent=2)
