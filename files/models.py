"""
结构化数据模型 - 所有智能体的输入/输出严格类型化
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================
# 枚举类型
# ============================================================

class AgentName(str, Enum):
    SUPERVISOR = "supervisor"
    KNOWLEDGE_BASE = "knowledge_base_agent"
    RESEARCH = "research_agent"
    REVIEW = "review_agent"
    PPT = "ppt_agent"
    HUMAN = "human"
    END = "__end__"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"
    COMPLETED = "completed"


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_HUMAN_REVIEW = "needs_human_review"


# ============================================================
# 子任务结构化输出
# ============================================================

class ClinicalGuidelineEntry(BaseModel):
    """临床指南条目"""
    guideline_name: str = Field(description="指南名称")
    version: str = Field(description="版本号")
    biomarker_context: str = Field(description="标志物相关背景")
    clinical_significance: str = Field(description="临床意义")
    evidence_level: str = Field(description="证据等级 (A/B/C/D)")
    relevant_diseases: List[str] = Field(description="相关疾病列表")
    source_url: Optional[str] = Field(default=None, description="来源URL")


class KnowledgeBaseResult(BaseModel):
    """知识库检索结果"""
    biomarker_name: str
    guidelines: List[ClinicalGuidelineEntry] = Field(default_factory=list)
    summary: str = Field(description="综合摘要")
    total_references: int = Field(default=0)
    search_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)


class CompetitorProduct(BaseModel):
    """竞品信息"""
    product_name: str
    manufacturer: str
    market_share: Optional[str] = None
    key_features: List[str] = Field(default_factory=list)
    price_range: Optional[str] = None
    regulatory_status: Optional[str] = None
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)


class RawMaterialInfo(BaseModel):
    """原料信息"""
    material_name: str
    supplier: str
    purity: Optional[str] = None
    specifications: Dict[str, Any] = Field(default_factory=dict)
    estimated_cost: Optional[str] = None
    availability: str = Field(default="unknown")
    notes: Optional[str] = None


class ResearchResult(BaseModel):
    """调研结果"""
    biomarker_name: str
    raw_materials: List[RawMaterialInfo] = Field(default_factory=list)
    competitors: List[CompetitorProduct] = Field(default_factory=list)
    market_overview: str = Field(description="市场概况")
    key_findings: List[str] = Field(default_factory=list)
    research_timestamp: datetime = Field(default_factory=datetime.now)
    data_sources: List[str] = Field(default_factory=list)


class ReviewCriteria(BaseModel):
    """审核标准"""
    completeness: float = Field(ge=0.0, le=1.0, description="完整性评分")
    accuracy: float = Field(ge=0.0, le=1.0, description="准确性评分")
    relevance: float = Field(ge=0.0, le=1.0, description="相关性评分")
    market_coverage: float = Field(ge=0.0, le=1.0, description="市场覆盖度")


class ReviewResult(BaseModel):
    """审核结果"""
    decision: ReviewDecision
    criteria_scores: ReviewCriteria
    overall_score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list, description="发现的问题")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")
    review_timestamp: datetime = Field(default_factory=datetime.now)
    reviewer: str = Field(default="review_agent")


class PPTSlide(BaseModel):
    """PPT幻灯片"""
    slide_number: int
    title: str
    content: List[str]
    slide_type: str = Field(description="标题页/内容页/图表页/总结页")
    notes: Optional[str] = None


class PPTResult(BaseModel):
    """PPT生成结果"""
    file_path: str
    file_name: str
    total_slides: int
    slides_outline: List[PPTSlide]
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None


# ============================================================
# 主任务状态 (LangGraph State)
# ============================================================

class TaskRequest(BaseModel):
    """任务请求"""
    biomarker_name: str = Field(description="标志物名称")
    task_description: Optional[str] = Field(default=None)
    priority: str = Field(default="normal", description="优先级: low/normal/high")
    requester: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


class ErrorRecord(BaseModel):
    """错误记录"""
    agent: str
    error_type: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    retry_count: int = 0
    resolved: bool = False


class HumanFeedback(BaseModel):
    """人工反馈"""
    approved: bool
    comments: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None
    reviewer_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# ============================================================
# API 请求/响应
# ============================================================

class TaskCreateRequest(BaseModel):
    biomarker_name: str
    description: Optional[str] = None
    priority: str = "normal"
    requester: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    current_agent: Optional[str] = None
    iteration_count: int = 0
    message: Optional[str] = None
    result_summary: Optional[str] = None


class HumanFeedbackRequest(BaseModel):
    task_id: str
    approved: bool
    comments: Optional[str] = None
    reviewer_name: Optional[str] = None
