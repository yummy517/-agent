"""
知识库工具 - 接入已有的 LangChain RAG 知识库
假设知识库已通过 LangChain 构建 (FAISS / Chroma / Pinecone 等向量存储)
此模块负责：接口适配 + 结构化输出 + 缓存
"""
import asyncio
from typing import Optional

from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

from config.settings import settings
from schemas.models import KnowledgeBaseResult, ClinicalGuidelineEntry
from monitoring.logger import logger


# ============================================================
# 知识库适配器 (接入已有 LangChain RAG)
# ============================================================

class ClinicalKnowledgeBase:
    """临床指南知识库适配器"""

    def __init__(self):
        self._vectorstore: Optional[VectorStore] = None
        self._qa_chain = None
        self._embeddings = None

    async def initialize(self):
        """初始化 - 加载已有的向量存储"""
        self._embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        try:
            # ★ 接入已有知识库 - 从磁盘加载已构建好的 FAISS 索引
            self._vectorstore = await asyncio.to_thread(
                FAISS.load_local,
                settings.VECTOR_STORE_PATH,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("kb_loaded", path=settings.VECTOR_STORE_PATH)
        except Exception as e:
            logger.warning("kb_load_failed", error=str(e), fallback="mock_mode")
            # 降级: 使用空的向量库 (开发/测试用)
            self._vectorstore = None

        # 构建 QA Chain
        if self._vectorstore:
            llm = ChatOpenAI(
                model=settings.PRIMARY_MODEL,
                temperature=0.1,
                openai_api_key=settings.OPENAI_API_KEY,
            )
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self._vectorstore.as_retriever(
                    search_kwargs={"k": settings.KB_RETRIEVAL_TOP_K}
                ),
                return_source_documents=True,
            )

    async def search(self, biomarker_name: str, query: str) -> KnowledgeBaseResult:
        """
        检索临床指南知识库
        返回结构化的 KnowledgeBaseResult
        """
        if self._qa_chain is None:
            return self._mock_result(biomarker_name)

        full_query = f"""
        请检索关于标志物 [{biomarker_name}] 的临床指南信息，包括：
        1. 相关临床指南名称和版本
        2. 该标志物的临床意义
        3. 推荐的检测方法和阈值
        4. 证据等级
        5. 相关疾病
        具体问题：{query}
        """

        try:
            response = await asyncio.to_thread(self._qa_chain.invoke, {"query": full_query})
            answer = response.get("result", "")
            source_docs = response.get("source_documents", [])

            # 解析来源文档生成结构化条目
            guidelines = []
            for i, doc in enumerate(source_docs[:5]):
                meta = doc.metadata
                guidelines.append(ClinicalGuidelineEntry(
                    guideline_name=meta.get("source", f"指南-{i+1}"),
                    version=meta.get("version", "未知"),
                    biomarker_context=doc.page_content[:500],
                    clinical_significance=meta.get("significance", "见原文"),
                    evidence_level=meta.get("evidence_level", "B"),
                    relevant_diseases=meta.get("diseases", ["待确认"]),
                    source_url=meta.get("url"),
                ))

            return KnowledgeBaseResult(
                biomarker_name=biomarker_name,
                guidelines=guidelines,
                summary=answer,
                total_references=len(source_docs),
                confidence_score=min(0.9, len(guidelines) * 0.18),
            )
        except Exception as e:
            logger.error("kb_search_error", error=str(e), biomarker=biomarker_name)
            return self._mock_result(biomarker_name)

    def _mock_result(self, biomarker_name: str) -> KnowledgeBaseResult:
        """降级 Mock 结果 (知识库不可用时)"""
        return KnowledgeBaseResult(
            biomarker_name=biomarker_name,
            guidelines=[
                ClinicalGuidelineEntry(
                    guideline_name=f"[MOCK] {biomarker_name}检测指南",
                    version="2024",
                    biomarker_context=f"{biomarker_name}是重要的临床生物标志物",
                    clinical_significance="用于疾病诊断和预后评估",
                    evidence_level="B",
                    relevant_diseases=["待确认"],
                )
            ],
            summary=f"[降级模式] {biomarker_name}的临床意义需进一步确认",
            confidence_score=0.3,
        )


# 全局知识库实例
clinical_kb = ClinicalKnowledgeBase()


# ============================================================
# LangChain Tool 封装
# ============================================================

@tool
async def search_clinical_guidelines(biomarker_name: str, specific_question: str = "") -> str:
    """
    搜索临床指南知识库，获取生物标志物的背景信息。
    
    Args:
        biomarker_name: 标志物名称 (如: PSA, HER2, CEA)
        specific_question: 具体问题 (可选)
    
    Returns:
        结构化的临床指南信息 (JSON格式)
    """
    query = specific_question or f"{biomarker_name}的临床意义、检测方法和证据等级"
    result = await clinical_kb.search(biomarker_name, query)
    return result.model_dump_json(indent=2)
