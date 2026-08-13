# Bundled runtime models

This directory contains the model files required by the ASR and TTS services.
Large weights are stored with Git LFS. After cloning, run:

```powershell
git lfs install
git lfs pull
python .\scripts\restore_models.py
```

`scripts/restore_models.py` joins the split Nana TTS weight and verifies its
SHA-256 checksum before the services start. Runtime caches, logs, sessions,
training outputs, and downloaded model caches are intentionally not tracked.
