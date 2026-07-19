@echo off
REM Launcher for the LM Studio prefill benchmark (PowerShell version).
REM PowerShell ships with Windows, so no Python install is needed.
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0lmstudio_bench.ps1" > lmstudio_bench_console.log 2>&1

echo BENCH_EXIT_CODE=%ERRORLEVEL% >> lmstudio_bench_console.log
echo ==== BENCHMARK FINISHED ==== >> lmstudio_bench_console.log
type lmstudio_bench_console.log
echo.
echo Done. You can close this window.
pause
