@echo off
setlocal
cd /d "%~dp0"
set MODELS=C:\Users\Admin\.lmstudio\models
set LMBENCH=C:\Users\Admin\OneDrive\Documents\Claude\lmbench
set IMG=nvidia/cuda:13.1.2-devel-ubuntu24.04
set OUT=%LMBENCH%\q36dev_setup.log

echo ==== q36dev persistent container %DATE% %TIME% ==== > "%OUT%"
taskkill /IM llama-server.exe /F >> "%OUT%" 2>&1

echo --- remove old q36dev if present --- >> "%OUT%"
docker rm -f q36dev >> "%OUT%" 2>&1

echo --- start persistent container (sleep infinity) --- >> "%OUT%"
docker run -d --name q36dev --gpus all -v "%MODELS%:/models" -v "%LMBENCH%:/work" %IMG% sleep infinity >> "%OUT%" 2>&1

echo --- build q36 + run 90k experiment --- >> "%OUT%"
docker exec q36dev bash -lc "sed -i 's/\r$//' /work/setup_q36dev.sh && bash /work/setup_q36dev.sh" >> "%OUT%" 2>&1

echo EXIT_CODE=%ERRORLEVEL% >> "%OUT%"
echo ==== DONE ==== >> "%OUT%"
type "%OUT%"
echo.
echo Container 'q36dev' left running. Attach with:  docker exec -it q36dev bash
pause
