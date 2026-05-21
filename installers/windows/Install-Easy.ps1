#Requires -Version 5.1
# Sound Split ADSR - Windows one-click installer

$ErrorActionPreference = 'Stop'
$InstallerRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$ProjectRoot = (Resolve-Path (Join-Path $InstallerRoot '..\..')).Path

. (Join-Path $InstallerRoot 'config.ps1')
. (Join-Path $InstallerRoot 'lib\InstallerHelpers.ps1')

$cfg = $script:SoundSplitConfig
$BootstrapPath = Join-Path $ProjectRoot ($cfg.BootstrapScript -replace '/', '\')
$script:InstallLogPath = Join-Path $ProjectRoot 'installers\runtime\windows\install.log'

Set-Location $ProjectRoot

Write-Host ''
Write-Host '================================================' -ForegroundColor Cyan
Write-Host "  $($cfg.AppName) - Installer" -ForegroundColor Cyan
Write-Host "  $($cfg.GitHubRepoUrl)" -ForegroundColor Cyan
Write-Host '================================================' -ForegroundColor Cyan
Write-Host "  Project: $ProjectRoot"
Write-Host ''

if (-not (Test-Path -LiteralPath $BootstrapPath)) {
    throw "Missing bootstrap.py at $BootstrapPath"
}

try {
    Write-InstallLog 'Starting setup (portable Python + dependencies + GUI)...'
    $exitCode = Start-BootstrapLaunch -ProjectRoot $ProjectRoot -BootstrapPath $BootstrapPath
    if ($exitCode -ne 0) {
        throw "Bootstrap exited with code $exitCode"
    }
    Write-Host ''
    Write-Host 'SUCCESS - Sound Split ADSR finished.' -ForegroundColor Green
    Write-Host "  Log: $script:InstallLogPath"
    exit 0
}
catch {
    Write-InstallLog $_.Exception.Message 'ERROR'
    if ($_.ScriptStackTrace) { Write-InstallLog $_.ScriptStackTrace 'ERROR' }
    Write-Host ''
    Write-Host 'INSTALLATION FAILED.' -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host "Log: $script:InstallLogPath"
    exit 1
}
