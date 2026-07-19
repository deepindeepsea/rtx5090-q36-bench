@echo off
setlocal
cd /d "%~dp0"
set MODELS=C:\Users\Admin\.lmstudio\models
set LMBENCH=C:\Users\Admin\OneDrive\Documents\Claude\lmbench
set IMG=nvidia/cuda:13.1.2-devel-ubuntu24.04
set OUT=%LMBENCH%\q36_build.log

echo ==== q36 build+bench %DATE% %TIME% ==== > "%OUT%"

echo --- Free VRAM: unload LM Studio model server --- >> "%OUT%"
taskkill /IM llama-server.exe /F >> "%OUT%" 2>&1

echo --- docker run (CUDA 13.1.2 devel): build q36_bench and run --- >> "%OUT%"
docker run --rm --gpus all -v "%MODELS%:/models" -v "%LMBENCH%:/work" %IMG% bash -lc "sed -i 's/\r$//' /work/build_q36.sh && bash /work/build_q36.sh" >> "%OUT%" 2>&1

echo EXIT_CODE=%ERRORLEVEL% >> "%OUT%"
echo ==== DONE ==== >> "%OUT%"
type "%OUT%"
echo.
echo Finished. You can close this window.
pause
