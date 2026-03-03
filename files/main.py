"""
FastAPI 应用主入口
- 任务创建/查询/恢复
- 人工审核 Webhook
- 监控指标端点
- WebSocket 实时状态推送
- 健康检查
"""
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config.settings import settings, setup_langsmith, setup_openai
from graph.state import create_initial_state
from graph.workflow import graph_manager
from memory.memory_manager import redis_memory
from monitoring.logger import setup_logging, logger, get_metrics
from schemas.models import (
    TaskCreateRequest,
    TaskStatusResponse,
    HumanFeedbackRequest,
    TaskStatus,
)
from tools.kb_tools import clinical_kb


# ============================================================
# 应用生命周期
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭"""
    # 启动
    setup_logging()
    setup_langsmith()
    setup_openai()

    logger.info("app_starting", project=settings.LANGCHAIN_PROJECT)

    # 初始化服务
    await redis_memory.connect()
    await clinical_kb.initialize()

    # 创建 PPT 输出目录
    import os
    os.makedirs(settings.PPT_OUTPUT_DIR, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    logger.info("app_ready")
    yield

    # 关闭
    await redis_memory.disconnect()
    logger.info("app_shutdown")


# ============================================================
# FastAPI 实例
# ============================================================

app = FastAPI(
    title="Biomarker Multi-Agent System",
    description="基于 LangGraph 监管者模式的多智能体标志物分析系统",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 后台任务执行器
# ============================================================

# 活跃任务记录 (生产用 Redis 替代)
_active_tasks: Dict[str, dict] = {}
# WebSocket 连接池
_ws_connections: Dict[str, list] = {}


async def run_task_background(task_id: str, initial_state: dict, thread_id: str):
    """后台运行 Agent 任务"""
    try:
        await redis_memory.set_task_status(task_id, {
            "task_id": task_id,
            "status": TaskStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
        })

        config = graph_manager.get_config(thread_id)

        # 流式运行，推送 WebSocket 更新
        async for chunk in graph_manager.graph.astream(
            initial_state, config=config, stream_mode="updates"
        ):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            status = node_output.get("status", TaskStatus.RUNNING.value)

            # 更新 Redis 状态
            await redis_memory.set_task_status(task_id, {
                "task_id": task_id,
                "status": status,
                "current_agent": node_name,
                "iteration_count": node_output.get("iteration_count", 0),
                "updated_at": datetime.now().isoformat(),
            })

            # WebSocket 推送
            await broadcast_ws(task_id, {
                "event": "step_completed",
                "node": node_name,
                "status": status,
                "messages": [
                    m.content for m in node_output.get("messages", [])
                    if hasattr(m, "content")
                ],
            })

            # 等待人工审核
            if status == TaskStatus.AWAITING_HUMAN.value:
                checkpoint_data = node_output.get("human_checkpoint_data", {})
                await redis_memory.push_human_review(task_id, checkpoint_data)
                await broadcast_ws(task_id, {
                    "event": "awaiting_human_review",
                    "task_id": task_id,
                    "review_data": checkpoint_data,
                })
                logger.info("task_awaiting_human", task_id=task_id)
                # 任务会被 interrupt 暂停，此处循环退出
                break

        # 获取最终状态
        final_state = graph_manager.get_state(thread_id)
        if final_state and final_state.values:
            ppt_result = final_state.values.get("ppt_result")
            final_status = final_state.values.get("status", TaskStatus.COMPLETED.value)

            await redis_memory.set_task_status(task_id, {
                "task_id": task_id,
                "status": final_status,
                "ppt_file": ppt_result.file_path if ppt_result else None,
                "completed_at": datetime.now().isoformat(),
            })

            # 保存任务摘要到长期记忆
            await redis_memory.save_task_summary(task_id, {
                "biomarker": initial_state.get("biomarker_name"),
                "status": final_status,
                "ppt_file": ppt_result.file_path if ppt_result else None,
                "completed_at": datetime.now().isoformat(),
            })

    except Exception as e:
        logger.error("background_task_failed", task_id=task_id, error=str(e))
        await redis_memory.set_task_status(task_id, {
            "task_id": task_id,
            "status": TaskStatus.FAILED.value,
            "error": str(e),
        })
        await broadcast_ws(task_id, {"event": "task_failed", "error": str(e)})


async def broadcast_ws(task_id: str, message: dict):
    """广播 WebSocket 消息"""
    connections = _ws_connections.get(task_id, [])
    dead = []
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.remove(ws)


# ============================================================
# API 路由
# ============================================================

# ── 健康检查 ──────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """系统健康检查"""
    redis_ok = await redis_memory.ping()
    return {
        "status": "healthy" if redis_ok else "degraded",
        "redis": "connected" if redis_ok else "disconnected",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@app.get("/metrics", tags=["System"])
async def get_system_metrics():
    """获取系统监控指标"""
    return get_metrics()


# ── 任务管理 ──────────────────────────────────────────────────

@app.post("/api/v1/tasks", response_model=TaskStatusResponse, tags=["Tasks"])
async def create_task(
    request: TaskCreateRequest,
    background_tasks: BackgroundTasks,
):
    """
    创建新的标志物分析任务
    任务在后台异步执行，通过 /tasks/{id} 查询状态
    """
    task_id = str(uuid.uuid4())
    thread_id = f"thread_{task_id}"
    session_id = f"session_{task_id}"

    initial_state = create_initial_state(
        task_id=task_id,
        session_id=session_id,
        biomarker_name=request.biomarker_name,
        task_description=request.description,
    )
    # 添加初始 HumanMessage
    from langchain_core.messages import HumanMessage
    initial_state["messages"] = [
        HumanMessage(content=f"请对标志物 {request.biomarker_name} 进行全面分析，生成调研报告PPT")
    ]

    _active_tasks[task_id] = {"thread_id": thread_id, "request": request.model_dump()}

    # 后台执行
    background_tasks.add_task(run_task_background, task_id, initial_state, thread_id)

    logger.info("task_created", task_id=task_id, biomarker=request.biomarker_name)

    return TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"任务已创建，正在分析标志物: {request.biomarker_name}",
    )


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """查询任务状态"""
    status_data = await redis_memory.get_task_status(task_id)
    if not status_data:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

    return TaskStatusResponse(
        task_id=task_id,
        status=TaskStatus(status_data.get("status", "pending")),
        current_agent=status_data.get("current_agent"),
        iteration_count=status_data.get("iteration_count", 0),
        message=status_data.get("error") or status_data.get("ppt_file"),
    )


@app.get("/api/v1/tasks/{task_id}/state", tags=["Tasks"])
async def get_task_graph_state(task_id: str):
    """获取任务的完整图状态 (调试用)"""
    task_info = _active_tasks.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="任务不存在")

    thread_id = task_info["thread_id"]
    state = graph_manager.get_state(thread_id)
    if not state:
        raise HTTPException(status_code=404, detail="图状态不存在")

    return {
        "task_id": task_id,
        "values": {
            k: (v.model_dump() if hasattr(v, "model_dump") else str(v))
            for k, v in (state.values or {}).items()
            if k != "messages"
        },
        "next": list(state.next) if state.next else [],
        "created_at": state.created_at.isoformat() if hasattr(state, "created_at") else None,
    }


@app.get("/api/v1/tasks/{task_id}/history", tags=["Tasks"])
async def get_task_history(task_id: str):
    """获取任务状态历史 (时间旅行)"""
    task_info = _active_tasks.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="任务不存在")

    thread_id = task_info["thread_id"]
    history = graph_manager.get_state_history(thread_id)

    return {
        "task_id": task_id,
        "history_count": len(history),
        "checkpoints": [
            {
                "step": i,
                "next": list(h.next) if h.next else [],
                "status": (h.values or {}).get("status"),
                "iteration": (h.values or {}).get("iteration_count"),
            }
            for i, h in enumerate(history[-10:])  # 最近10个
        ],
    }


# ── 人工审核 ──────────────────────────────────────────────────

@app.post("/api/v1/tasks/{task_id}/review", tags=["Human Review"])
async def submit_human_review(
    task_id: str,
    feedback: HumanFeedbackRequest,
    background_tasks: BackgroundTasks,
):
    """
    提交人工审核结果
    - approved=True: 继续生成PPT
    - approved=False: 重新调研
    """
    task_info = _active_tasks.get(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="任务不存在")

    thread_id = task_info["thread_id"]

    # 保存反馈到 Redis
    await redis_memory.save_human_feedback(task_id, feedback.model_dump())

    # 恢复图执行
    human_input = {
        "approved": feedback.approved,
        "comments": feedback.comments,
        "reviewer_name": feedback.reviewer_name,
    }

    background_tasks.add_task(
        _resume_graph_background, task_id, thread_id, human_input
    )

    logger.info("human_review_submitted",
                task_id=task_id,
                approved=feedback.approved,
                reviewer=feedback.reviewer_name)

    return {
        "task_id": task_id,
        "message": "审核结果已提交，任务继续执行",
        "approved": feedback.approved,
    }


async def _resume_graph_background(task_id: str, thread_id: str, human_input: dict):
    """后台恢复图执行"""
    try:
        await graph_manager.resume_with_human_feedback(thread_id, human_input)
    except Exception as e:
        logger.error("graph_resume_failed", task_id=task_id, error=str(e))
        await redis_memory.set_task_status(task_id, {
            "task_id": task_id,
            "status": TaskStatus.FAILED.value,
            "error": str(e),
        })


@app.get("/api/v1/review/queue", tags=["Human Review"])
async def get_review_queue():
    """获取待审核任务队列"""
    items = []
    while True:
        item = await redis_memory.pop_human_review()
        if item is None:
            break
        items.append(item)
    return {"queue": items, "count": len(items)}


# ── WebSocket 实时推送 ────────────────────────────────────────

@app.websocket("/ws/tasks/{task_id}")
async def websocket_task_updates(websocket: WebSocket, task_id: str):
    """WebSocket 实时任务状态推送"""
    await websocket.accept()
    if task_id not in _ws_connections:
        _ws_connections[task_id] = []
    _ws_connections[task_id].append(websocket)

    try:
        # 发送当前状态
        status = await redis_memory.get_task_status(task_id)
        if status:
            await websocket.send_json({"event": "current_status", **status})

        # 保持连接
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        _ws_connections.get(task_id, []).remove(websocket)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
