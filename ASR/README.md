# ASR Service

Standalone SenseVoice ASR service. In the integrated project, Gateway calls this service through the standard `ASRRequest` / `ASRResponse` JSON protocol.

## Runtime

| Item | Value |
|---|---|
| Path | `D:\daima\nawei\ASR` |
| Default port | `8001` |
| App entry | `src\main.py` |
| Model | `iic/SenseVoiceSmall` |
| Default device | `cuda:0` |
| Local model/cache path | `hf_download` |

## Start

```powershell
cd D:\daima\nawei\ASR
.\.venv\Scripts\python.exe src\main.py
```

Health check:

```text
GET http://127.0.0.1:8001/health
```

## Standard API

Gateway uses this endpoint:

```text
POST http://127.0.0.1:8001/api/v1/audio/transcriptions/base64
```

Alias:

```text
POST http://127.0.0.1:8001/api/v1/asr/transcriptions
```

Request body:

```json
{
  "request_id": "asr-1",
  "trace_id": "trace-1",
  "session_id": "session-1",
  "audio_base64": "UklGR...",
  "audio_url": null,
  "language": "auto",
  "return_timestamps": false,
  "format": "wav",
  "sample_rate": 16000,
  "channels": 1,
  "options": {
    "filename": "audio.wav",
    "use_itn": true
  }
}
```

Response body:

```json
{
  "request_id": "asr-1",
  "text": "你好呀。",
  "language": "zh",
  "confidence": null,
  "duration_ms": null,
  "segments": [],
  "asr": {
    "key": "audio.wav",
    "emotion": "SAD",
    "emotion_label": "悲伤",
    "emotion_emoji": "😔",
    "events": [],
    "emotion_hint": "情绪：悲伤",
    "raw_text": "<|zh|>...",
    "clean_text": "你好呀。"
  },
  "metadata": {
    "provider": "sensevoice",
    "model": "iic/SenseVoiceSmall"
  }
}
```

## Legacy Native APIs

These endpoints are kept for ASR debugging and old tools. They are not the Gateway standard path.

```text
POST /api/v1/asr
POST /api/v1/asr/base64
POST /api/v1/asr/stream
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ASR_HOST` | `0.0.0.0` | Bind address |
| `ASR_PORT` | `8001` | Service port |
| `ASR_DEVICE` | `cuda:0` | Inference device |
| `ASR_MODEL_ID` | `iic/SenseVoiceSmall` | SenseVoice model |

The service sets ModelScope and HuggingFace cache variables to project-local paths under `hf_download` so it does not write to the Windows user cache directory.
