@echo off
setlocal EnableExtensions
title Sound Split ADSR - Installer
cd /d "%~dp0\..\.." || (
  echo ERROR: Cannot find project root.
  pause
  exit /b 1
)

echo.
echo  *** USE THIS FILE FOR NORMAL INSTALL ***
echo.
echo  Sound Split ADSR
echo  ================
echo.
echo  GitHub: https://github.com/LuisMRaimundo/Sound_split_ADSR
echo.
echo  First run downloads portable Python with Tkinter and libraries.
echo  Do not close this window until finished.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Easy.ps1"
set ERR=%ERRORLEVEL%

echo.
if %ERR% NEQ 0 (
  echo Installation failed. See install.log in:
  echo   instalers\runtime\windows\install.log
) else (
  echo Done.
)
echo.
pause
exit /b %ERR%
