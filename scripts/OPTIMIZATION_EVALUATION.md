# Speech and retrieval evaluation

Run commands from the repository root with the local services running.
Reports may contain transcripts and generated answers; keep them local and
remove personal information before sharing them.

## ASR

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_asr.py --phase baseline
.\.venv\Scripts\python.exe scripts/evaluate_asr.py --phase optimized
```

Both phases reuse the same generated WAV files in
`.runtime/asr-evaluation/audio`. They compare `auto` and `zh` language settings
with ITN enabled. They do not rewrite the recognizer output. CER uses Unicode
NFKC normalization, ignores punctuation/spacing, and preserves differences in
numbers, names, and negation. The bundled four synthetic samples are smoke
tests, not a microphone-accuracy benchmark.

For real recordings, use a JSON array as illustrated in
`tests/fixtures/asr_manifest.example.json`. Each entry needs a unique `id`, a
WAV `path` relative to the manifest, and the verbatim `reference` transcript:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_asr.py --manifest .runtime/asr-real/manifest.json --phase baseline --output .runtime/asr-real/before
.\.venv\Scripts\python.exe scripts/evaluate_asr.py --manifest .runtime/asr-real/manifest.json --phase optimized --output .runtime/asr-real/after
```

Record different speakers, distances, quiet/noisy rooms, names, numbers,
negations, and domain terms. Keep a held-out set when tuning. Compare paired
recordings with unchanged transcripts and separate cold/warm latency runs.

Validated complete mono PCM16 16 kHz WAV uploads now bypass FFmpeg conversion.
Other formats and malformed/truncated WAVs retain the conversion path. This
removes preprocessing overhead; it does not change the ASR model or guarantee
fewer recognition errors. No automatic text correction or language-default
change is applied without representative accuracy evidence.

## Robot first-audio latency

```powershell
.\.venv\Scripts\python.exe scripts/simulate_robot.py --rounds 3 --source-wav .runtime/asr-evaluation/audio/brief.wav --text '一加一等于几，请只回答结果。' --report .runtime/latency-evaluation/robot.json
```

`--source-wav` requires mono PCM16 audio and one text reference. It avoids
generating TTS input before the measured turn, which would otherwise warm TTS.
The simulator reports manual-mode timing from `listen/stop`; total time also
includes input transmission. A fresh Gateway is not a cold start of all models.

Gateway stderr emits `Robot turn timing` JSON, keyed by session and turn:

- `asr_after_stop_ms`: ASR completion after manual stop.
- `capacity_after_asr_ms`: waiting for a conversation slot after ASR.
- `llm_first_text_after_asr_ms`: first LLM text after ASR, including preparation.
- `tts_started_after_asr_ms`: TTS start event after ASR.
- `tts_first_chunk_after_asr_ms`: first TTS audio event after ASR.
- `llm_context_ms`, `llm_tools_ms`, `llm_memory_write_ms`: runtime stage timings.
- `total_from_listen_ms`: complete turn, including listening and response.

The first-chunk/start difference includes TTS service/transport overhead, not
just model compute. Audio arrival is not physical speaker playback. ASR-after-
stop is only present for manual mode. These structured timing lines omit raw
transcripts; simulator reports include them. For Knowledge timing and
model migration, see `services/knowledge/README.md`.

## Regression checks

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/check_project.ps1
.\.venv\Scripts\python.exe scripts/check_full_flow.py --output .runtime/optimization-final/core
.\.venv\Scripts\python.exe scripts/simulate_robot.py --rounds 20
```

Protocol tests use generated audio. ESP32 microphone, speaker, Wi-Fi,
public routing, and full model cold start require separate hardware tests.
