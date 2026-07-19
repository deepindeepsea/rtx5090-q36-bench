@echo off
REM Read-only environment probe for planning the llama.cpp / q36 benchmark setup.
cd /d "%~dp0"
set OUT=env_probe.log

echo ==== ENV PROBE %DATE% %TIME% ==== > %OUT%
echo. >> %OUT%

echo --- Windows GPU (nvidia-smi) --- >> %OUT%
nvidia-smi >> %OUT% 2>&1
echo. >> %OUT%

echo --- LM Studio GGUF models --- >> %OUT%
dir /s /b "%USERPROFILE%\.lmstudio\models\*.gguf" >> %OUT% 2>&1
dir /s /b "%USERPROFILE%\.cache\lm-studio\models\*.gguf" >> %OUT% 2>&1
echo. >> %OUT%

echo --- WSL distros --- >> %OUT%
wsl -l -v >> %OUT% 2>&1
echo. >> %OUT%

echo --- WSL GPU (nvidia-smi inside WSL) --- >> %OUT%
wsl nvidia-smi >> %OUT% 2>&1
echo. >> %OUT%

echo --- WSL CUDA compiler --- >> %OUT%
wsl bash -lc "which nvcc && nvcc --version" >> %OUT% 2>&1
echo. >> %OUT%

echo --- WSL build tools + distro --- >> %OUT%
wsl bash -lc "which git cmake gcc g++ make; echo ---; cat /etc/os-release 2>/dev/null | head -4" >> %OUT% 2>&1
echo. >> %OUT%

echo ==== DONE ==== >> %OUT%
type %OUT%
echo.
echo Probe complete. You can close this window.
pause
