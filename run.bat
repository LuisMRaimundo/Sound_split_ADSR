@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Sound Split ADSR

if not exist "split_audio_segments.py" (
    echo Keep run.bat in the Sound Split ADSR folder ^(next to split_audio_segments.py^).
    pause
    exit /b 1
)

echo Starting Sound Split ADSR...

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py -3 "%~dp0split_audio_segments.py"
    goto :done
)

python "%~dp0split_audio_segments.py"

:done
set "RC=%ERRORLEVEL%"
if %RC% neq 0 (
    echo.
    echo Could not start. If libraries are missing, run once:
    echo   pip install -r requirements.txt
    echo.
    pause
)
exit /b %RC%
