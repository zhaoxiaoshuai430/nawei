# Nawei Voice Lab

独立的实时语音展示前端，目录内不依赖 Node.js 或前端框架。

## 功能

- 浏览器麦克风实时采集和动态波形
- 无麦克风时上传 WAV、MP3、M4A、OGG 或 WebM 音频
- PCM16、16 kHz、单声道音频上传
- ASR 识别文本和语言展示
- LLM 回答流式展示
- TTS 语音分片顺序播放
- 语音采集、语音识别、智能推理和语音合成进程展示
- 阶段耗时、总耗时和实时事件日志
- Gateway、用户、说话人、语言和 JWT 令牌设置

## 启动

先从 `nawei` 目录启动后端：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_project.ps1
```

然后启动前端：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\app\start.ps1
```

浏览器访问 `http://127.0.0.1:5173`。

页面默认连接：

```text
ws://127.0.0.1:8000/api/v1/voice/realtime
```

点击右上角设置按钮可以修改连接地址。默认 `GATEWAY_AUTH_MODE=local` 时令牌留空；JWT 模式下需要填写有效令牌，并确保令牌用户与页面用户 ID 一致。

## 浏览器要求

推荐最新版 Chrome 或 Edge。麦克风 API 只允许在 `localhost` 或 HTTPS 页面使用，首次录音时需要允许麦克风权限。
