"""
系统配置 - 集中管理所有配置项
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """系统配置 (支持环境变量覆盖)"""

    # ── LLM 配置 ──────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    PRIMARY_MODEL: str = Field(default="gpt-4o", env="PRIMARY_MODEL")
    FALLBACK_MODEL: str = Field(default="gpt-4o-mini", env="FALLBACK_MODEL")  # 降级模型
    MODEL_TEMPERATURE: float = Field(default=0.1)
    MODEL_MAX_TOKENS: int = Field(default=4096)

    # ── LangSmith 监控 ─────────────────────────────────────────
    LANGCHAIN_TRACING_V2: str = Field(default="true", env="LANGCHAIN_TRACING_V2")
    LANGCHAIN_API_KEY: str = Field(default="", env="LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT: str = Field(default="biomarker-multi-agent", env="LANGCHAIN_PROJECT")
    LANGCHAIN_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT"
    )

    # ── Redis 配置 ──────────────────────────────────────────────
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_TTL_SECONDS: int = Field(default=86400 * 7)  # 7天

    # ── 知识库 (已有 LangChain RAG) ───────────────────────────
    VECTOR_STORE_PATH: str = Field(default="./data/vectorstore", env="VECTOR_STORE_PATH")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    KB_RETRIEVAL_TOP_K: int = Field(default=5)

    # ── MCP PPT 工具 ───────────────────────────────────────────
    MCP_SERVER_URL: str = Field(default="http://localhost:8001", env="MCP_SERVER_URL")
    MCP_PPT_TOOL_NAME: str = Field(default="create_ppt", env="MCP_PPT_TOOL_NAME")
    PPT_OUTPUT_DIR: str = Field(default="./output/ppt", env="PPT_OUTPUT_DIR")

    # ── FastAPI ────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    API_RELOAD: bool = Field(default=False)
    CORS_ORIGINS: list = Field(default=["*"])

    # ── 流程控制 ───────────────────────────────────────────────
    MAX_TOTAL_ITERATIONS: int = Field(default=3, description="全局最大循环次数(token控制)")
    MAX_REVIEW_RETRIES: int = Field(default=3, description="审核最大重试次数")
    MAX_AGENT_RETRIES: int = Field(default=3, description="单个Agent最大重试次数")
    REVIEW_PASS_THRESHOLD: float = Field(default=0.75, description="审核通过分数阈值")

    # ── 日志 ───────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", description="json / text")
    LOG_FILE: str = Field(default="./logs/agent.log")

    # ── 研究工具 ───────────────────────────────────────────────
    TAVILY_API_KEY: Optional[str] = Field(default=None, env="TAVILY_API_KEY")
    SEARCH_MAX_RESULTS: int = Field(default=5)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# 单例
settings = Settings()


def setup_langsmith():
    """初始化 LangSmith 追踪"""
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT


def setup_openai():
    """初始化 OpenAI 配置"""
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    if settings.OPENAI_BASE_URL != "https://api.openai.com/v1":
        os.environ["OPENAI_BASE_URL"] = settings.OPENAI_BASE_URL
