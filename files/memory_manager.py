"""
记忆系统
- MemorySaver: LangGraph 内置的临时会话记忆 (checkpointing)
- RedisMemory: Redis 持久化长期记忆
"""
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta

import redis.asyncio as aioredis
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings
from monitoring.logger import logger


# ============================================================
# 临时会话记忆 (LangGraph 内置 MemorySaver)
# ============================================================

def create_session_checkpointer() -> MemorySaver:
    """
    创建 LangGraph MemorySaver
    - 存储每个 thread_id 的图状态快照
    - 支持 interrupt/resume (Human-in-the-loop)
    - 支持时间旅行 (回放历史状态)
    """
    return MemorySaver()


# ============================================================
# Redis 长期记忆
# ============================================================

class RedisMemory:
    """
    Redis 持久化记忆层
    - 跨会话缓存知识库结果 (避免重复检索)
    - 缓存调研结果
    - 存储任务历史摘要
    - 存储用户偏好
    """

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self.ttl = settings.REDIS_TTL_SECONDS

    async def connect(self):
        """连接 Redis"""
        self._client = await aioredis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("redis_connected", url=settings.REDIS_URL)

    async def disconnect(self):
        if self._client:
            await self._client.aclose()

    def _make_key(self, namespace: str, identifier: str) -> str:
        """生成规范化的 Redis key"""
        safe_id = hashlib.md5(identifier.encode()).hexdigest()[:12]
        return f"biomarker:{namespace}:{safe_id}"

    # ── 知识库结果缓存 ────────────────────────────────────────

    async def cache_kb_result(self, biomarker_name: str, result: dict) -> None:
        key = self._make_key("kb", biomarker_name)
        payload = json.dumps({
            "biomarker_name": biomarker_name,
            "result": result,
            "cached_at": datetime.now().isoformat(),
        })
        await self._client.setex(key, self.ttl, payload)
        logger.info("kb_result_cached", biomarker=biomarker_name, key=key)

    async def get_kb_result(self, biomarker_name: str) -> Optional[dict]:
        key = self._make_key("kb", biomarker_name)
        raw = await self._client.get(key)
        if raw:
            data = json.loads(raw)
            logger.info("kb_cache_hit", biomarker=biomarker_name)
            return data["result"]
        return None

    # ── 调研结果缓存 ──────────────────────────────────────────

    async def cache_research_result(self, biomarker_name: str, result: dict) -> None:
        key = self._make_key("research", biomarker_name)
        payload = json.dumps({
            "result": result,
            "cached_at": datetime.now().isoformat(),
        })
        # 调研结果缓存时间较短 (1天)
        await self._client.setex(key, 86400, payload)

    async def get_research_result(self, biomarker_name: str) -> Optional[dict]:
        key = self._make_key("research", biomarker_name)
        raw = await self._client.get(key)
        return json.loads(raw)["result"] if raw else None

    # ── 任务历史 ──────────────────────────────────────────────

    async def save_task_summary(self, task_id: str, summary: dict) -> None:
        key = self._make_key("task", task_id)
        await self._client.setex(key, self.ttl, json.dumps(summary))

    async def get_task_summary(self, task_id: str) -> Optional[dict]:
        key = self._make_key("task", task_id)
        raw = await self._client.get(key)
        return json.loads(raw) if raw else None

    # ── 任务状态 (供 API 查询) ────────────────────────────────

    async def set_task_status(self, task_id: str, status: dict) -> None:
        key = f"biomarker:status:{task_id}"
        await self._client.setex(key, 3600, json.dumps(status))

    async def get_task_status(self, task_id: str) -> Optional[dict]:
        key = f"biomarker:status:{task_id}"
        raw = await self._client.get(key)
        return json.loads(raw) if raw else None

    # ── 等待人工审核队列 ──────────────────────────────────────

    async def push_human_review(self, task_id: str, context: dict) -> None:
        key = "biomarker:human_review_queue"
        payload = json.dumps({"task_id": task_id, "context": context,
                              "created_at": datetime.now().isoformat()})
        await self._client.lpush(key, payload)
        logger.info("human_review_queued", task_id=task_id)

    async def pop_human_review(self) -> Optional[dict]:
        key = "biomarker:human_review_queue"
        raw = await self._client.rpop(key)
        return json.loads(raw) if raw else None

    async def save_human_feedback(self, task_id: str, feedback: dict) -> None:
        key = f"biomarker:human_feedback:{task_id}"
        await self._client.setex(key, 3600, json.dumps(feedback))

    async def get_human_feedback(self, task_id: str) -> Optional[dict]:
        key = f"biomarker:human_feedback:{task_id}"
        raw = await self._client.get(key)
        return json.loads(raw) if raw else None

    # ── 健康检查 ──────────────────────────────────────────────

    async def ping(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False


# 全局 Redis 实例
redis_memory = RedisMemory()
