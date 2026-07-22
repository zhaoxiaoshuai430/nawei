@echo off
chcp 65001 >nul
echo 启动 nana 独立 TTS 服务 (port 19000)

set "CONDA_ENV_PATH=%cd%\cuda_env"
set "CUDA_PATH=%CONDA_ENV_PATH%\Library"
set "PATH=%CUDA_PATH%\bin;%PATH%"

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set "PYTHON_PATH=%cd%\cuda_env\"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONEXECUTABLE=%PYTHON_PATH%python.exe"
set "PYTHON_EXECUTABLE=%PYTHON_PATH%python.exe"
set "SOX_PATH=%cd%\cuda_env\sox"
set "PATH=%PYTHON_PATH%;%PYTHON_PATH%Scripts;%SOX_PATH%;%PATH%"

REM 预加载 voice_lab/nana；浏览器打开 http://localhost:19000/demo
"%PYTHON_EXECUTABLE%" -m app.services.tts.server --model voice_lab/nana --port 19000
pause
