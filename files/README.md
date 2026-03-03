# 🧬 Biomarker Multi-Agent System

> 基于 **LangGraph 监管者模式**的生物标志物分析多智能体系统  
> 集成 LangSmith 可视化监控 · Redis 持久记忆 · FastAPI 工程化部署

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Layer                           │
│  POST /tasks  GET /tasks/{id}  POST /tasks/{id}/review  WS     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    LangGraph Workflow                           │
│                                                                 │
│   START ──► SUPERVISOR ◄──────────────────────┐                │
│                  │                             │                │
│          ┌───────▼────────┐                   │                │
│          │ parallel_      │  (asyncio.gather)  │                │
│          │ research       │                   │                │
│          │  ├─ KB Agent   │──────────────────►│                │
│          │  └─ Research   │                   │                │
│          └───────┬────────┘                   │                │
│                  │                             │                │
│          ┌───────▼────────┐                   │                │
│          │ review_agent   │──(rejected)───────►│                │
│          └───────┬────────┘                   │                │
│                  │(approved)                   │                │
│          ┌───────▼────────┐                   │                │
│          │ human_review   │──(interrupt)────── │                │
│          │  [INTERRUPT]   │◄──(resume)──API    │                │
│          └───────┬────────┘                   │                │
│                  │                             │                │
│          ┌───────▼────────┐                   │                │
│          │   ppt_agent    │──(MCP call)────────►END            │
│          └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
          │                    │
┌─────────▼──────┐   ┌────────▼────────┐
│ MemorySaver    │   │  Redis Memory   │
│ (会话快照)      │   │  (持久化缓存)    │
└────────────────┘   └─────────────────┘
          │
┌─────────▼──────────────────────────────┐
│           LangSmith Tracing            │
│  每个 Agent 调用自动追踪 + 可视化       │
└────────────────────────────────────────┘
```

---

## ✨ 核心特性

| 特性 | 实现方案 | 说明 |
|------|----------|------|
| **监管者模式** | LangGraph Supervisor | 统一调度4个子Agent |
| **异步并行** | `asyncio.gather` | KB检索+调研同时执行，节省50%时间 |
| **结构化输出** | Pydantic + `with_structured_output` | 所有Agent输入/输出强类型 |
| **人工协作** | LangGraph `interrupt()` | 审核通过后人工确认再生成PPT |
| **容错重试** | `@retry_with_fallback` 装饰器 | 3次重试+降级策略 |
| **会话记忆** | LangGraph `MemorySaver` | 支持断点续传、时间旅行 |
| **持久记忆** | Redis + TTL | KB/调研结果跨会话缓存 |
| **迭代控制** | `MAX_TOTAL_ITERATIONS=3` | 防止无限循环，控制Token消耗 |
| **日志监控** | structlog JSON + LangSmith | 结构化日志 + 全链路追踪 |
| **PPT生成** | MCP Protocol 调用 | 标准化工具协议接入PPT服务 |

---

## 📁 项目结构

```
biomarker_agent_system/
├── api/
│   └── main.py              # FastAPI 应用（路由、WebSocket、后台任务）
├── config/
│   └── settings.py          # 配置管理（Pydantic Settings + .env）
├── graph/
│   ├── state.py             # LangGraph 状态定义（AgentState TypedDict）
│   ├── agents.py            # 4个子Agent实现
│   ├── supervisor.py        # 监管者路由逻辑
│   └── workflow.py          # 图构建、并行节点、interrupt节点
├── memory/
│   └── memory_manager.py    # MemorySaver + Redis 双层记忆
├── monitoring/
│   └── logger.py            # structlog + LangSmith + 指标
├── schemas/
│   └── models.py            # 所有 Pydantic 数据模型
├── tools/
│   ├── kb_tools.py          # 知识库工具（接入已有RAG）
│   ├── research_tools.py    # 调研工具（Tavily搜索）
│   └── mcp_ppt_tools.py     # MCP PPT 工具
├── .env.example             # 环境变量模板
├── docker-compose.yml       # 容器化部署
└── requirements.txt         # 依赖清单
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
cp .env.example .env
# 编辑 .env 填写 API Keys

pip install -r requirements.txt
```

### 2. 启动 Redis

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### 3. 启动应用

```bash
# 开发模式
uvicorn api.main:app --reload --port 8000

# 生产模式
docker-compose up -d
```

---

## 📡 API 使用示例

### 创建任务

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "biomarker_name": "PSA",
    "description": "前列腺特异性抗原全面分析",
    "priority": "high"
  }'

# 响应
{
  "task_id": "abc-123",
  "status": "pending",
  "message": "任务已创建，正在分析标志物: PSA"
}
```

### 查询任务状态

```bash
curl http://localhost:8000/api/v1/tasks/abc-123
```

### 提交人工审核

```bash
curl -X POST http://localhost:8000/api/v1/tasks/abc-123/review \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "abc-123",
    "approved": true,
    "comments": "数据完整，竞品分析深入，同意生成PPT",
    "reviewer_name": "张三"
  }'
```

### WebSocket 实时监听

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tasks/abc-123');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## 🔍 LangSmith 监控

配置 `LANGCHAIN_API_KEY` 后，所有 Agent 调用自动上报至 LangSmith：

- **追踪视图**: 查看完整的 Agent 调用链路
- **延迟分析**: 各节点耗时分布
- **Token 用量**: 每次运行的 Token 消耗
- **错误追踪**: 异常定位和重试记录

访问: https://smith.langchain.com → 项目 `biomarker-multi-agent`

---

## ⚙️ 知识库接入

系统假设已有 LangChain 构建的 FAISS 向量库：

```python
# 已有知识库构建代码（示例）
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("./data/vectorstore")
```

只需将向量库路径配置到 `VECTOR_STORE_PATH` 即可自动加载。

---

## 🔄 流程控制说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_TOTAL_ITERATIONS` | 3 | 全局最大循环次数（Token控制）|
| `MAX_REVIEW_RETRIES` | 3 | 审核最大重试次数 |
| `REVIEW_PASS_THRESHOLD` | 0.75 | 审核通过分数线 |

超过 `MAX_TOTAL_ITERATIONS` 次后任务自动标记为 `failed`，避免无限消耗Token。

---

## 🏥 降级策略

| 组件 | 主路径 | 降级路径 |
|------|--------|----------|
| 知识库 | FAISS RAG检索 | Mock结果（开发模式）|
| 搜索 | Tavily API | LLM知识库填充 |
| MCP PPT | MCP服务调用 | 生成JSON配置文件 |
| 审核LLM | GPT-4o | GPT-4o-mini |
| Redis | 正常缓存 | 跳过缓存直接执行 |
