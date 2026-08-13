# Nawei 四模块语音项目

Nawei 对外保持 Gateway、ASR、LLM、TTS 四个业务板块。客户端统一访问 Gateway；Agent、Skills、Tools、Memory 和实际模型推理全部由 LLM 大板块管理。

```text
Client -> Gateway :8000
           |-> ASR :8001
           |-> LLM :8002
           |     |-> Agent / Skills / Tools
           |     |-> Redis / PostgreSQL Memory
           |     `-> DeepSeek / Stub Inference
           `-> TTS :19000
```

项目没有独立 Agent、Memory 或知识库服务。Redis 和 PostgreSQL 是 LLM 的私有基础设施；Gateway 不保存会话历史。项目不包含知识库或 Qdrant。

## 服务

| 服务 | 端口 | 职责 |
|---|---:|---|
| Gateway | 8000 | 公共 API、鉴权、并发控制、ASR/LLM/TTS 编排 |
| ASR | 8001 | SenseVoice 音频识别和情绪标签 |
| LLM | 8002 | Agent、Skills、Tools、Redis/PostgreSQL Memory、DeepSeek/Stub 推理与 SSE |
| TTS | 19000 | Qwen3-TTS、Nana 音色和流式音频 |

LLM 来源：`https://gitee.com/w3280306004/agent02/tree/master`，适配提交 `99fc166561aeea580b7beb0799d23865bef4829b`。

## 启动

1. 安装 Git LFS，并在克隆后执行 `git lfs pull` 下载 ASR/TTS 模型权重。
2. 从根目录 `.env.example` 创建本地 `.env` 并配置 LLM。
3. 运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
```

启动脚本会通过 Docker 启动 LLM 所需的 Redis/PostgreSQL，再启动四个业务进程。使用外部数据库时传入 `-SkipInfrastructure`。

4. 客户端连接 Gateway；接口说明位于 `http://127.0.0.1:8000/docs`。

状态和停止命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\status_project.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\stop_project.ps1
```

## 主要 API

| API | 链路 |
|---|---|
| `GET /health` | Gateway |
| `GET /api/v1/status` | Gateway 检查 ASR、LLM、TTS |
| `POST /api/v1/audio/transcriptions/base64` | Gateway -> ASR |
| `POST /api/v1/text/completions` | Gateway -> LLM |
| `POST /api/v1/text/speech-completions` | Gateway -> LLM -> TTS |
| `POST /api/v1/audio/speech` | Gateway -> TTS |
| `POST /api/v1/audio/speech/stream` | Gateway -> TTS SSE |
| `WS /api/v1/voice/realtime` | Gateway -> PCM realtime ASR -> LLM SSE -> optional TTS |

语音助手只开放实时语音入口。整段音频上传的 `/api/v1/voice/completions` 及其流式变体已移除；ASR、LLM 和 TTS 的独立 HTTP 接口仅用于文本交互、模块调用与诊断。

生产部署使用 [deploy/docker-compose.production.yml](deploy/docker-compose.production.yml)，其中 Redis/PostgreSQL 仅在内部网络供 LLM 使用。
