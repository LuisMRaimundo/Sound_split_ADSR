@echo off
setlocal EnableDelayedExpansion
title Sound Split ADSR

cd /d "%~dp0\..\.."
set "ROOT=%CD%"
set "BOOT=%ROOT%\installers\common\bootstrap.py"
set "PORTABLE=%ROOT%\installers\runtime\windows\python-full\python.exe"

echo.
echo  Sound Split ADSR
echo  ================
echo.

where python >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "SKIP="
    for /f "delims=" %%P in ('where python 2^>nul') do (
        echo %%P | findstr /i "WindowsApps" >nul && set "SKIP=1"
    )
    if not defined SKIP (
        python "%BOOT%" launch
        goto :done
    )
)

if not exist "%PORTABLE%" (
    echo First-time setup: downloading portable Python ^(includes Tkinter^)...
    where py >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        py -3 "%BOOT%" launch
        goto :done
    )
    echo ERROR: Python is required for first-time setup on this PC.
    echo Install Python 3.10+ from https://www.python.org/downloads/ and run again.
    pause
    exit /b 1
)

"%PORTABLE%" "%BOOT%" launch

:done
if errorlevel 1 pause
