@echo off
chcp 65001 >nul

cd /d "%~dp0"

set PYTHONUNBUFFERED=1
set ASR_PORT=8001
set ASR_HOST=0.0.0.0
if "%ASR_DEVICE%"=="" set ASR_DEVICE=cpu
if "%ASR_MODEL_ID%"=="" set ASR_MODEL_ID=iic/SenseVoiceSmall
set MODELSCOPE_CACHE=%CD%\hf_download\modelscope
set MODELSCOPE_HOME=%CD%\hf_download\modelscope
set HF_HOME=%CD%\hf_download\huggingface
set TRANSFORMERS_CACHE=%CD%\hf_download\huggingface\transformers
set HUGGINGFACE_HUB_CACHE=%CD%\hf_download\huggingface\hub
set HOME=%CD%\.home
set USERPROFILE=%CD%\.home

echo [ASR] Starting SenseVoice ASR Service on port %ASR_PORT% ...
.\.venv\Scripts\python.exe src\main.py
