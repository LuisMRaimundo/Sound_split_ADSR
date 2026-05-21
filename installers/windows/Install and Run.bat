@echo off
setlocal EnableDelayedExpansion
title Sound Split ADSR

cd /d "%~dp0\..\.."
set "ROOT=%CD%"

echo.
echo  Sound Split ADSR
echo  ================
echo.

where python >nul 2>&1
if %ERRORLEVEL%==0 (
    for /f "delims=" %%P in ('where python 2^>nul ^| findstr /i "WindowsApps"') do set "SKIP=1"
    if not defined SKIP (
        python "%ROOT%\installers\common\bootstrap.py" launch
        goto :done
    )
)

set "PY=%ROOT%\installers\runtime\windows\python\python.exe"
set "BOOT=%ROOT%\installers\common\bootstrap.py"

if not exist "%PY%" (
    echo First-time setup: downloading portable Python...
    where py >nul 2>&1
    if %ERRORLEVEL%==0 (
        py -3 "%BOOT%" launch
        goto :done
    )
    echo ERROR: Python is required for first-time setup on this PC.
    echo Install Python 3.10+ from https://www.python.org/downloads/ and run again.
    pause
    exit /b 1
)

"%PY%" "%BOOT%" launch

:done
if errorlevel 1 pause
