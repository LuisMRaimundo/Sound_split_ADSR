@echo off
cd /d "%~dp0"
call "%~dp0INSTALL.bat"
exit /b %ERRORLEVEL%
