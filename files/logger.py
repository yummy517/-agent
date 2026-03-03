"""
日志 & LangSmith 监控
- structlog 结构化 JSON 日志
- LangSmith 追踪装饰器
- Prometheus 指标端点(可选)
"""
import time
import functools
import logging
import sys
from typing import Any, Callable, Optional
from datetime import datetime

import structlog
from langsmith import traceable
from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.callbacks import CallbackManagerForChainRun

from config.settings import settings


# ============================================================
# structlog 配置
# ============================================================

def setup_logging():
    """配置结构化日志"""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # 处理器链
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )

    # 文件日志
    import os
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    file_handler = logging.FileHandler(settings.LOG_FILE)
    file_handler.setLevel(log_level)
    logging.basicConfig(handlers=[file_handler], level=log_level)


# 全局 logger
logger = structlog.get_logger(__name__)


# ============================================================
# 监控指标 (使用字典简单实现，生产可替换为 Prometheus)
# ============================================================

_metrics: dict[str, Any] = {
    "task_total": 0,
    "task_success": 0,
    "task_failed": 0,
    "agent_calls": {},
    "agent_errors": {},
    "agent_latency_ms": {},
    "review_retries": 0,
    "human_interventions": 0,
}


def get_metrics() -> dict:
    """获取当前监控指标快照"""
    return {
        **_metrics,
        "success_rate": (
            _metrics["task_success"] / max(_metrics["task_total"], 1)
        ),
        "snapshot_time": datetime.now().isoformat(),
    }


def record_agent_call(agent_name: str, latency_ms: float, success: bool):
    """记录 Agent 调用指标"""
    _metrics["agent_calls"][agent_name] = _metrics["agent_calls"].get(agent_name, 0) + 1
    if not success:
        _metrics["agent_errors"][agent_name] = (
            _metrics["agent_errors"].get(agent_name, 0) + 1
        )
    # 运行均值
    prev_latency = _metrics["agent_latency_ms"].get(agent_name, 0.0)
    call_count = _metrics["agent_calls"][agent_name]
    _metrics["agent_latency_ms"][agent_name] = (
        (prev_latency * (call_count - 1) + latency_ms) / call_count
    )


# ============================================================
# 装饰器
# ============================================================

def monitor_agent(agent_name: str):
    """
    Agent 监控装饰器:
    - 自动计时
    - 记录成功/失败指标
    - structlog 上下文绑定
    - LangSmith 追踪
    """
    def decorator(func: Callable) -> Callable:
        @traceable(name=f"agent:{agent_name}", run_type="chain")
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            log = logger.bind(agent=agent_name, func=func.__name__)
            log.info("agent_start")
            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                record_agent_call(agent_name, elapsed, True)
                log.info("agent_success", latency_ms=round(elapsed, 2))
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                record_agent_call(agent_name, elapsed, False)
                log.error("agent_error", error=str(exc), latency_ms=round(elapsed, 2))
                raise

        @traceable(name=f"agent:{agent_name}", run_type="chain")
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            log = logger.bind(agent=agent_name, func=func.__name__)
            log.info("agent_start")
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                record_agent_call(agent_name, elapsed, True)
                log.info("agent_success", latency_ms=round(elapsed, 2))
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                record_agent_call(agent_name, elapsed, False)
                log.error("agent_error", error=str(exc), latency_ms=round(elapsed, 2))
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def retry_with_fallback(
    max_retries: int = 3,
    fallback: Optional[Callable] = None,
    exceptions: tuple = (Exception,),
    backoff_seconds: float = 1.0,
):
    """
    重试 + 降级装饰器
    - 按 max_retries 重试
    - 全部失败则调用 fallback(若提供)
    - 指数退避
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "retry_attempt",
                        func=func.__name__,
                        attempt=attempt,
                        max_retries=max_retries,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    if attempt < max_retries:
                        await asyncio.sleep(wait)

            # 全部重试失败
            logger.error(
                "all_retries_exhausted",
                func=func.__name__,
                max_retries=max_retries,
                error=str(last_exc),
            )
            if fallback:
                logger.info("using_fallback", func=func.__name__)
                return await fallback(*args, **kwargs)
            raise last_exc

        return async_wrapper
    return decorator
