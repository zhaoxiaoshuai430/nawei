# Bundled runtime models

This directory contains the model files required by the ASR and TTS services.
The complete weights and split parts are stored locally. Validate them with:

```powershell
python .\scripts\restore_models.py
```

`scripts/restore_models.py` joins the split Liliya TTS weight when needed and
verifies its SHA-256 checksum before the services start. Keep the split parts
for container builds. Git history and the Git LFS object cache are not required
to run this local distribution.
# Additional native streaming ASR model

`asr/paraformer-online` contains the optional ParaformerStreaming weights for
realtime incremental recognition. It is separate from the original bundled
SenseVoice/FSMN manifest and does not replace either model.

From the project root, use the ASR interpreter:

```powershell
.\services\asr\.venv\Scripts\python.exe scripts/prepare_streaming_asr.py --download
```

Without `--download`, the script checks that required files exist and the model
configuration selects the native streaming architecture. File presence checks
are not a cryptographic audit; the ModelScope downloader verifies downloaded
files. Keep this directory when the Windows launcher uses Paraformer. Existing
`restore_models.py` checks continue to apply to the original bundled manifest.
