#Requires -Version 5.1

function Write-InstallLog {
    param([string]$Message, [string]$Level = 'INFO')
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    if ($script:InstallLogPath) {
        $dir = Split-Path $script:InstallLogPath -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        Add-Content -LiteralPath $script:InstallLogPath -Value $line -Encoding UTF8
    }
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        default { Write-Host $line }
    }
}

function Refresh-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-PythonVersionOk {
    param([string]$PythonExe, [string]$PyArg)
    try {
        if ($PyArg) {
            $out = & $PythonExe $PyArg -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        } else {
            $out = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        }
        if ($out -match '^3\.(\d+)$') {
            $minor = [int]$Matches[1]
            $cfg = $script:SoundSplitConfig
            return ($minor -ge $cfg.PythonMinMinor -and $minor -le $cfg.PythonMaxMinor)
        }
    } catch { }
    return $false
}

function Test-IsPythonPathCandidate {
    param([string]$PythonExe)
    if (-not $PythonExe) { return $false }
    $normalized = $PythonExe.Replace('/', '\')
    if ($normalized -match '\\WindowsApps\\') { return $false }
    if ($normalized -match '\\Microsoft\\WindowsApps\\') { return $false }
    if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
    try {
        if ((Get-Item -LiteralPath $PythonExe).Length -lt 1024) { return $false }
    } catch { return $false }
    return $true
}

function Get-KnownPythonCandidatePaths {
    $candidates = @()
    $localPython = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    foreach ($folder in @('Python312', 'Python311', 'Python310')) {
        $candidates += Join-Path $localPython "$folder\python.exe"
    }
    foreach ($root in @(${env:ProgramFiles}, ${env:ProgramFiles(x86)})) {
        if (-not $root) { continue }
        foreach ($folder in @('Python312', 'Python311', 'Python310')) {
            $candidates += Join-Path $root "$folder\python.exe"
        }
    }
    return $candidates | Where-Object { Test-IsPythonPathCandidate -PythonExe $_ }
}

function Resolve-PythonViaPyLauncher {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyCmd) { return $null }
    foreach ($tag in @('3.12', '3.11', '3.10')) {
        try {
            $exe = & $pyCmd.Source -$tag -c "import sys; print(sys.executable)" 2>$null
            $exe = ($exe | Select-Object -First 1).Trim()
            if ($exe -and (Test-IsPythonPathCandidate -PythonExe $exe) -and (Test-PythonVersionOk -PythonExe $exe)) {
                return @{ Exe = $exe; Arg = $null }
            }
        } catch { }
        if (Test-PythonVersionOk -PythonExe $pyCmd.Source -PyArg $tag) {
            return @{ Exe = $pyCmd.Source; Arg = $tag }
        }
    }
    return $null
}

function Find-ExistingPython {
    foreach ($exe in (Get-KnownPythonCandidatePaths)) {
        if (Test-PythonVersionOk -PythonExe $exe) {
            return @{ Exe = (Resolve-Path -LiteralPath $exe).Path; Arg = $null }
        }
    }
    $viaPy = Resolve-PythonViaPyLauncher
    if ($viaPy) { return $viaPy }
    foreach ($name in @('python3.12', 'python3.11', 'python3.10', 'python3', 'python')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Test-IsPythonPathCandidate -PythonExe $cmd.Source) -and (Test-PythonVersionOk -PythonExe $cmd.Source)) {
            return @{ Exe = $cmd.Source; Arg = $null }
        }
    }
    return $null
}

function Wait-ForPythonAfterInstall {
    param([int]$TimeoutSeconds = 90)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Refresh-SessionPath
        $py = Find-ExistingPython
        if ($py) { return $py }
        Start-Sleep -Seconds 2
    }
    return $null
}

function Test-PythonInstallerExitOk {
    param([int]$ExitCode)
    return ($ExitCode -eq 0 -or $ExitCode -eq 3010)
}

function Install-PythonIfNeeded {
    $cfg = $script:SoundSplitConfig
    Write-InstallLog "Python 3.10-3.12 not found. Installing Python $($cfg.PythonVersion)..."

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch { }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        foreach ($id in @('Python.Python.3.12', 'Python.Python.3.11', 'Python.Python.3.10')) {
            Write-InstallLog "Trying winget ($id, current user)..."
            $p = Start-Process -FilePath 'winget' -ArgumentList @(
                'install', '-e', '--id', $id,
                '--accept-package-agreements', '--accept-source-agreements',
                '--disable-interactivity',
                '--scope', 'user'
            ) -Wait -PassThru -NoNewWindow
            Write-InstallLog "winget exit code: $($p.ExitCode)"
            $py = Wait-ForPythonAfterInstall -TimeoutSeconds 45
            if ($py) {
                Write-InstallLog "Python available after winget: $($py.Exe)"
                return $py
            }
        }
        Write-InstallLog 'winget did not produce a usable Python; trying python.org installer.' 'WARN'
    } else {
        Write-InstallLog 'winget not available; using python.org installer.' 'WARN'
    }

    $installer = Join-Path $env:TEMP 'python-3.11.9-amd64.exe'
    Write-InstallLog 'Downloading Python installer from python.org...'
    try {
        Invoke-WebRequest -Uri $cfg.PythonInstallerUrl -OutFile $installer -UseBasicParsing
    } catch {
        throw "Could not download Python installer. Check Internet/firewall. $($_.Exception.Message)"
    }

    Write-InstallLog 'Running Python installer (per-user, adds to PATH)...'
    $installArgs = @(
        '/passive', 'InstallAllUsers=0', 'PrependPath=1',
        'Include_test=0', 'Include_pip=1', 'Include_launcher=1',
        'AssociateFiles=0', 'SimpleInstall=1'
    )
    $p = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    Write-InstallLog "python.org installer exit code: $($p.ExitCode)"

    if (-not (Test-PythonInstallerExitOk -ExitCode $p.ExitCode)) {
        throw "Python installer failed (exit $($p.ExitCode)). Install Python 3.10+ from https://www.python.org/downloads/ with Add to PATH, then run INSTALL.bat again."
    }

    $py = Wait-ForPythonAfterInstall -TimeoutSeconds 90
    if (-not $py) {
        throw "Python installed but not found yet. Install from https://www.python.org/downloads/, enable Add to PATH, then run INSTALL.bat again. Log: $script:InstallLogPath"
    }
    Write-InstallLog "Python installed: $($py.Exe)"
    return $py
}

function Invoke-PythonCommand {
    param(
        [hashtable]$Py,
        [string[]]$Args
    )
    if ($Py.Arg) {
        & $Py.Exe $Py.Arg @Args
    } else {
        & $Py.Exe @Args
    }
}

function Start-BootstrapLaunch {
    param(
        [string]$ProjectRoot,
        [string]$BootstrapPath
    )
    $cfg = $script:SoundSplitConfig
    $portable = Join-Path $ProjectRoot ($cfg.PortablePythonExe -replace '/', '\')

    if (Test-Path -LiteralPath $portable) {
        Write-InstallLog "Using portable runtime: $portable"
        & $portable $BootstrapPath launch
        return $LASTEXITCODE
    }

    $py = Find-ExistingPython
    if (-not $py) {
        $py = Install-PythonIfNeeded
    }
    if (-not $py) {
        throw 'Python 3.10+ is required for first-time setup.'
    }

    Write-InstallLog "Running bootstrap via: $($py.Exe)$(if ($py.Arg) { ' ' + $py.Arg })"
    Invoke-PythonCommand -Py $py -Args @($BootstrapPath, 'launch')
    return $LASTEXITCODE
}
