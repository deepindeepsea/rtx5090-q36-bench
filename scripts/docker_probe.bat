@echo off
cd /d "%~dp0"
set OUT=docker_probe.log
echo ==== DOCKER PROBE %DATE% %TIME% ==== > %OUT%

echo --- docker version --- >> %OUT%
docker version >> %OUT% 2>&1
echo. >> %OUT%

echo --- docker info (look for daemon + GPU/runtime) --- >> %OUT%
docker info >> %OUT% 2>&1
echo. >> %OUT%

echo --- nvidia container runtime check --- >> %OUT%
docker info 2>nul | findstr /i "nvidia gpu runtime" >> %OUT% 2>&1
echo. >> %OUT%

echo ==== DONE ==== >> %OUT%
type %OUT%
echo.
echo Done. You can close this window.
pause
