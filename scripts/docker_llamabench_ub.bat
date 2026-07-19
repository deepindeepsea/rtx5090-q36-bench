@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
set OUT=%~dp0llamabench_ub.log
set MODELS=C:\Users\Admin\.lmstudio\models
set MODELREL=lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf
set IMG=ghcr.io/ggml-org/llama.cpp:full-cuda

echo ==== DOCKER llama-bench ubatch sweep %DATE% %TIME% ==== > "%OUT%"

echo --- Free VRAM: unload LM Studio model server --- >> "%OUT%"
taskkill /IM llama-server.exe /F >> "%OUT%" 2>&1
timeout /t 3 >nul

echo --- Run llama-bench: batch 4096, ubatch 2048 vs 4096 --- >> "%OUT%"
docker run --rm --gpus all -v "%MODELS%:/models" --entrypoint /app/llama-bench %IMG% -m "/models/%MODELREL%" -ngl 99 -fa 1 -b 4096 -ub 2048,4096 -p 2048,4096,8192 -n 128 -r 3 >> "%OUT%" 2>&1
echo EXIT_CODE=!ERRORLEVEL! >> "%OUT%"

echo ==== DONE ==== >> "%OUT%"
type "%OUT%"
echo.
echo Finished. You can close this window.
pause
