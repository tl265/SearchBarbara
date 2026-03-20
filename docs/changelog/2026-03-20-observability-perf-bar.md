# SearchBarbara 可观测性体系建设 — 完整改动说明

> **日期**: 2026-03-20
> **分支**: `feature/angill-agent2skill-0317`（基于 `main@2bebcbd`）
> **作者**: Claude Code + wangjing

---

## 一、背景与目标

SearchBarbara 原先只有一个简单的 `agent_debug.log` 文件记录 agent 运行日志，存在以下问题：

- **日志分散**：所有 session 混在一个文件里，无法按 session 查看
- **格式不统一**：纯文本格式，无法程序化解析
- **无前端可视化**：排查问题只能登服务器看日志
- **无前端遥测**：不知道用户操作了什么、页面性能如何
- **无自观测**：dashboard 自身加载慢时没有手段定位瓶颈

**本次改动目标**：构建完整的可观测性体系 — 从日志基础设施、请求追踪、前端遥测到可视化 Dashboard，再到 Dashboard 自身的 Performance Bar 自观测。

---

## 二、改动全景

```
┌─────────────────────────────────────────────────────────────────┐
│                    可观测性体系架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: 日志基础设施 (infra/observability/)                    │
│  ├─ JSONFormatter + ConsoleFormatter (双格式输出)                │
│  ├─ setup_global_logger() → logs/searchbarbara.log (rotating)   │
│  ├─ SessionLogger → logs/sessions/{sid}.log (JSON Lines)        │
│  ├─ SpanContext (代码块计时上下文管理器)                           │
│  ├─ StageTracker (DFS 父子关系追踪)                              │
│  ├─ MetricsCollector (计数器 + timing 直方图)                    │
│  └─ cleanup_old_session_logs() (启动时清理 >30 天日志)           │
│                                                                 │
│  Layer 2: 请求追踪 (backend/server.py middleware)               │
│  ├─ tracing_middleware: X-Trace-ID 注入 + HTTP 请求日志          │
│  └─ auth_middleware: /observability 路由认证保护                  │
│                                                                 │
│  Layer 3: Agent 结构化日志 (agents/ + backend/orchestration/)    │
│  ├─ LLM.call() → session_logger.info("llm_call:stage", ...)    │
│  ├─ WebSearch.search() → session_logger.info("search:...", ...) │
│  └─ RunManager → 通过 _session_loggers 管理生命周期              │
│                                                                 │
│  Layer 4: 前端遥测 (frontend/static/telemetry.js)               │
│  ├─ 全局 fetch() 包装器: 注入 X-Trace-ID + 记录 API 耗时        │
│  ├─ SSE 事件间隔追踪                                             │
│  ├─ 页面渲染计时                                                 │
│  └─ 批量发送 → POST /api/telemetry                              │
│                                                                 │
│  Layer 5: Observability Dashboard (frontend/templates+static/)  │
│  ├─ Session 列表 (搜索/筛选/状态标记)                            │
│  ├─ 瀑布图 Timeline (分类过滤/层级折叠/详情展开)                 │
│  └─ Performance Bar (自观测: API/Parse/Render 耗时分解)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、修改文件清单

### 新增文件（7 个）

| 文件 | 行数 | 说明 |
|------|------|------|
| `infra/observability/cleanup.py` | 55 | 启动时清理过期 session 日志 |
| `frontend/static/observability.js` | 618 | Dashboard 前端逻辑 + Performance Bar |
| `frontend/static/observability.css` | 495 | Dashboard 样式 |
| `frontend/templates/observability.html` | 49 | Dashboard HTML 模板 |
| `frontend/static/telemetry.js` | 202 | 前端遥测 SDK（fetch 包装 + 批量上报） |

### 修改文件（6 个）

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `infra/observability/__init__.py` | 3→34 行 | 导出所有新模块符号 |
| `infra/observability/logging.py` | 34→276 行 | 全面重写：JSONFormatter / ConsoleFormatter / 全局 logger / SessionLogger |
| `infra/observability/metrics.py` | 1→81 行 | 新建 MetricsCollector（计数器 + timing 直方图 + P50/P95/P99） |
| `infra/observability/tracing.py` | 1→117 行 | 新建 generate_trace_id / SpanContext / StageTracker |
| `backend/server.py` | +375 行 | 新增 tracing middleware + telemetry API + observability dashboard 三组端点 + Server-Timing |
| `backend/orchestration/run_manager.py` | ~40 行改动 | 接入 SessionLogger + print→log 迁移 |
| `agents/deep_research/core.py` | ~50 行改动 | LLM 和 WebSearch 接入 session_logger 记录结构化调用日志 |
| `run_web.py` | 重写 | 接入 setup_global_logger + 启动时 cleanup |

---

## 四、各层详解

### 4.1 Layer 1：日志基础设施重建 — `infra/observability/`

#### 改动前

```
infra/observability/
├── __init__.py     → 只导出 setup_agent_logger
├── logging.py      → 34 行，只有一个 FileHandler 写 agent_debug.log
├── metrics.py      → 空占位
└── tracing.py      → 空占位
```

#### 改动后

**`logging.py`**（34→276 行，全面重写）

| 组件 | 说明 |
|------|------|
| `JSONFormatter` | 每条日志输出一个 JSON 对象，含 `ts`, `level`, `logger`, `message` + 任意 structured 字段 |
| `ConsoleFormatter` | 人可读格式，自动追加 `session_id`, `stage`, `node_id`, `duration_ms`, `tokens` 等上下文 |
| `setup_global_logger()` | 进程级 logger `searchbarbara`：文件 handler (JSON, 50MB×5 rotating) + 控制台 handler (text) |
| `SessionLogger` | 每个 session 独立 logger → `logs/sessions/{session_id}.log`（JSON Lines），自动携带 session_id / stage / node_id / trace_id 上下文 |
| `setup_agent_logger()` | 保留为兼容 shim，内部调用 `setup_global_logger()` |

**`tracing.py`**（1→117 行）

| 组件 | 说明 |
|------|------|
| `generate_trace_id()` | 16 字符十六进制随机 ID |
| `SpanContext` | 上下文管理器，自动计时 + 通过 SessionLogger 输出结构化日志 |
| `StageTracker` | 维护 DFS 栈，追踪 parent→child stage 关系 |

**`metrics.py`**（1→81 行）

| 组件 | 说明 |
|------|------|
| `MetricsCollector` | 线程安全的计数器 + timing 直方图，提供 `summary()` 返回 min/max/mean/P50/P95/P99 |

**`cleanup.py`**（新增 55 行）

- 启动时扫描 `logs/sessions/`，删除超过 30 天的日志文件
- 通过 `SESSION_LOG_MAX_AGE_DAYS` 环境变量可配

**`__init__.py`**（3→34 行）

- 统一导出所有新符号，外部一行 `from infra.observability import ...` 即可

---

### 4.2 Layer 2：请求追踪 — `backend/server.py` middleware

**新增 `tracing_middleware`**（25 行）

```python
@app.middleware("http")
async def tracing_middleware(request, call_next):
    trace_id = request.headers.get("x-trace-id") or generate_trace_id()
    request.state.trace_id = trace_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Trace-ID"] = trace_id
    # 记录结构化 HTTP 请求日志（跳过 /static/）
    ...
```

- 每个请求自动分配或继承 `X-Trace-ID`
- 响应头回传 trace_id，前端可关联
- 非静态资源请求自动记录 method / path / status / duration_ms

**`auth_middleware` 改动**（1 行）

```python
# before:
protected = path == "/" or path.startswith("/api/")
# after:
protected = path == "/" or path.startswith("/api/") or path == "/observability"
```

确保 `/observability` 页面也受认证保护。

---

### 4.3 Layer 3：Agent 结构化日志

**`agents/deep_research/core.py`**

- `LLM.__init__` 和 `WebSearch.__init__` 新增 `session_logger` 参数
- `LLM.call()` 成功后通过 session_logger 记录：
  - `llm_call:{stage}` — model / messages / response content / finish_reason / tokens / duration_ms
- `WebSearch.search()` 同理记录搜索调用详情

**`backend/orchestration/run_manager.py`**

- 新增 `_session_loggers` 字典管理每个 session 的 SessionLogger
- 新增 `get_session_logger(session_id)` 方法供 telemetry API 使用
- 所有 `print()` 调用迁移为 `log.debug()` / `log.error()`（约 10 处）

---

### 4.4 Layer 4：前端遥测 SDK — `frontend/static/telemetry.js`

全新文件（202 行），提供 `window.SBTelemetry` 全局对象：

| API | 说明 |
|-----|------|
| 全局 `fetch()` 替换 | 自动注入 `X-Trace-ID` header，记录每次 API 调用的 method / url / status / duration_ms |
| `trackSSE(eventSource)` | 追踪 SSE 事件间隔，每 5 条或间隔 >2s 时上报 |
| `trackRender(label)` | 返回 `{end()}` 对象，用于测量 DOM 渲染耗时 |
| `setSession(sessionId)` | 设置当前 session_id，后续上报自动携带 |
| `flush()` | 手动刷新缓冲区 |

缓冲 + 批量发送机制：
- 每 5 秒或缓冲满 50 条时 `POST /api/telemetry` 批量上报
- 页面隐藏/关闭时通过 `navigator.sendBeacon()` 最终刷新
- telemetry 自身的请求不会触发追踪（防递归）

**后端 Telemetry API**（`POST /api/telemetry`，约 30 行）：
- 接收批量事件，写入对应 session 日志 + 全局 server 日志

---

### 4.5 Layer 5：Observability Dashboard

#### 后端 API（`backend/server.py`，约 250 行新增）

| 端点 | 说明 |
|------|------|
| `GET /observability` | 渲染 Dashboard HTML 页面 |
| `GET /api/observability/sessions` | 列出所有 session 日志元数据（task / status / entry_count / file_size / modified_at） |
| `GET /api/observability/sessions/{sid}/trace` | 解析指定 session 日志，返回结构化 trace 数据 + Server-Timing header |

**trace API 核心逻辑**：
1. 读取 `logs/sessions/{sid}.log` 逐行解析 JSON
2. 计算每条事件的 offset_ms（相对于首条的时间偏移）
3. 分类：`llm_call` / `search` / `agent_event` / `frontend` / `lifecycle` / `error`
4. 聚合统计：LLM 调用数 / 搜索数 / token 总量 / 各类耗时
5. 返回 JSON + `Server-Timing: total;dur=X, file_read;dur=Y, parse;dur=Z`

#### 前端 Dashboard（3 个新文件）

**`observability.html`**（49 行）
- 顶栏（标题 + 返回按钮 + 用户名）
- 两列布局：左侧 session 列表 + 右侧 trace 主面板
- 底部 Performance Bar 容器

**`observability.css`**（495 行）

| 区域 | 要点 |
|------|------|
| 布局 | CSS Grid 两列，sidebar 320px + 主面板自适应 |
| Session 列表 | 搜索框 + 可点击卡片 + 状态标记 (completed/running/failed) |
| 瀑布图 | 固定表头 + 按 node_id 深度缩进 + 颜色编码的 duration bar |
| 分类过滤 | 按钮组 (All/LLM/Search/Agent/Lifecycle/Frontend/Error) + 颜色图例 |
| 详情面板 | 展开行显示完整 JSON + Copy 按钮 |
| Performance Bar | 固定底部 32px，monospace 字体，绿/黄/红三色编码 |
| 响应式 | 768px 以下单列布局 |

**`observability.js`**（618 行）

核心功能模块：

| 模块 | 函数 | 说明 |
|------|------|------|
| Session 列表 | `loadSessions()` / `renderSessionList()` | 加载 + 渲染 + 搜索过滤 |
| Trace 加载 | `selectSession(sid)` | 加载 trace 数据，running session 自动 3s 刷新 |
| 瀑布图渲染 | `renderTrace()` | summary 指标 + filter bar + timeline 表格 + 层级折叠 |
| 事件交互 | `bindTraceEvents()` | 行展开/折叠、分类过滤、文本搜索、JSON 复制 |
| **Performance Bar** | `recordPerf()` / `updatePerfBar()` | 采集各阶段耗时 + 渲染底部状态栏 + 可展开 history 表 |

**Performance Bar 计时链路**：

```
用户点击 session
  ├─ t0: selectSession() 开始
  ├─ t2: fetch 收到响应  →  API 耗时 = t2-t0
  │       ↑ 解析 Server-Timing header → 后端内部耗时
  ├─ t3: resp.json() 完成 →  Parse 耗时 = t3-t2
  ├─ t4: renderTrace() 完成 → Render 耗时 = t4-t3
  └─ 总耗时 = t4-t0
```

覆盖操作：`load_session` / `auto_refresh` / `filter` / `toggle_row` / `toggle_group`

颜色编码：<100ms 绿 / 100-500ms 黄 / >500ms 红

History 表：最近 20 条操作的完整耗时对比表，可展开/折叠。

---

### 4.6 `run_web.py` 改动

```python
# before:
print(f"[info] Agent debug log: logs/agent_debug.log")

# after:
log = setup_global_logger()
cleanup_old_session_logs()
log.info("Global log: logs/searchbarbara.log")
log.info("Session logs: logs/sessions/")
```

- 接入全局 logger 替代 print
- 启动时自动清理过期 session 日志

---

## 五、Code Review 中发现并修复的问题

| # | 问题 | 严重度 | 修复方案 |
|---|------|--------|---------|
| 1 | trace API 的 `response_size_bytes` 用 `json.dumps(entries)` 单独序列化再用 `JSONResponse` 二次序列化 → 大 session 双倍 CPU | 🟡 性能 | 手动序列化 body → 量大小 → 补回 → 用 `Response(content=body_bytes)` 直接返回 |
| 2 | `fmtKB(bytes)` 接受 bytes 并 `/1024`，调用处传 `kb * 1024` 先乘再除 → 绕圈 | 🟡 可读性 | `fmtKB` 改为直接接受 KB 值 |
| 3 | 文本搜索每次按键都 `recordPerf` → 快速打字刷屏 history | 🟡 体验 | 加 300ms debounce |

---

## 六、新增 API 端点汇总

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/observability` | ✅ | Dashboard 页面 |
| `GET` | `/api/observability/sessions` | ✅ | Session 列表元数据 |
| `GET` | `/api/observability/sessions/{sid}/trace` | ✅ | 单 session 结构化 trace + Server-Timing |
| `POST` | `/api/telemetry` | ✅ | 前端遥测批量上报 |

新增响应头：
- `X-Trace-ID` — 所有非静态请求
- `Server-Timing` — trace API（`total;dur=X, file_read;dur=Y, parse;dur=Z`）

---

## 七、日志文件结构

```
logs/
├── searchbarbara.log        ← 全局 JSON 日志 (rotating 50MB×5)
├── agent_debug.log          ← 保留兼容（shim 指向全局 logger）
└── sessions/
    ├── {session_id_1}.log   ← Session 1 的 JSON Lines 日志
    ├── {session_id_2}.log   ← Session 2 的 JSON Lines 日志
    └── ...                  ← 超过 30 天自动清理
```

单条 session 日志格式：
```json
{
  "ts": "2026-03-20T08:15:32.123+00:00",
  "level": "INFO",
  "logger": "searchbarbara.session.abc123",
  "message": "llm_call:synthesis",
  "session_id": "abc123",
  "stage": "synthesis",
  "node_id": "q1.q3",
  "trace_id": "a1b2c3d4e5f6g7h8",
  "duration_ms": 1823.45,
  "tokens": {"prompt_tokens": 342, "completion_tokens": 128, "total_tokens": 470},
  "input_data": {"model": "gpt-4.1", "messages": [...]},
  "output_data": {"content": "...", "finish_reason": "stop"}
}
```

---

## 八、验证方式

1. `python run_web.py` 启动后控制台应显示结构化日志格式
2. 运行一个 research session → `logs/sessions/{sid}.log` 出现 JSON Lines
3. 浏览器访问 `/observability` → 看到 session 列表
4. 点击 session → 瀑布图加载，底部 Performance Bar 显示耗时分解
5. DevTools Network → 任意 API 请求响应头含 `X-Trace-ID`
6. DevTools Network → trace API 响应头含 `Server-Timing`
7. 点击 filter / 展开行 → Performance Bar 实时更新
8. 点击 "History ▾" → 展开耗时历史对比表

---

## 九、后续迭代方向

| 方向 | 说明 | 优先级 |
|------|------|--------|
| 虚拟滚动 | 大 session (>500 events) 的 timeline 渲染优化 | 🔴 高 |
| 实时流 | 用 SSE 替代 3s 轮询实现 running session 的实时更新 | 🟡 中 |
| P95/P99 统计 | Performance Bar 显示百分位数 | 🟡 中 |
| 后端缓存 | trace API 对已完成 session 加内存缓存，Server-Timing 增加 cache_hit | 🟡 中 |
| 网络层拆分 | 用 `PerformanceObserver` 拆分 DNS/TCP/TTFB | 🟡 中 |
| 告警 | Total > 1s 时 bar 变红 + 抖动提示 | 🟢 低 |
| 导出 | History 表导出 CSV/JSON | 🟢 低 |

---

## 十、依赖与兼容性

- **无新 pip 依赖**：所有改动基于 Python 标准库 + FastAPI 内置
- **无新 npm 依赖**：前端纯 vanilla JS
- **向后兼容**：
  - `setup_agent_logger()` 保留为 shim，旧调用方不受影响
  - 新增的响应头 (`X-Trace-ID`, `Server-Timing`) 不影响现有客户端
  - `summary.response_size_bytes` 是新增字段，老前端忽略即可
- **环境变量**（均可选）：
  - `LOG_LEVEL` — 控制台日志级别（默认 INFO）
  - `SESSION_LOG_MAX_AGE_DAYS` — session 日志保留天数（默认 30）
