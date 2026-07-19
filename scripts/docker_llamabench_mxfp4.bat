@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set OUT=%~dp0llamabench_mxfp4.log
set MODELS=C:\Users\Admin\.lmstudio\models
set MODELREL=unsloth/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
set IMG=ghcr.io/ggml-org/llama.cpp:full-cuda

echo ==== DOCKER llama-bench MXFP4 %DATE% %TIME% ==== > "%OUT%"

echo --- Free VRAM: unload LM Studio model server --- >> "%OUT%"
taskkill /IM llama-server.exe /F >> "%OUT%" 2>&1
timeout /t 3 >nul

echo --- Run llama-bench MXFP4: batch 8192, ubatch 4096 vs 8192 --- >> "%OUT%"
docker run --rm --gpus all -v "%MODELS%:/models" --entrypoint /app/llama-bench %IMG% -m "/models/%MODELREL%" -ngl 99 -fa 1 -b 8192 -ub 4096,8192 -p 2048,4096,8192,16384 -n 128 -r 3 >> "%OUT%" 2>&1
echo EXIT_CODE=!ERRORLEVEL! >> "%OUT%"

echo ==== DONE ==== >> "%OUT%"
type "%OUT%"
echo.
echo Finished. You can close this window.
pause
