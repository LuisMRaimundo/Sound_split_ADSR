# Sound Split ADSR - Windows installer constants
$script:SoundSplitConfig = @{
    GitHubRepoUrl      = 'https://github.com/LuisMRaimundo/Sound_split_ADSR'
    AppName            = 'Sound Split ADSR'
    PythonVersion      = '3.11'
    PythonMinMinor     = 10
    PythonMaxMinor     = 12
    PythonInstallerUrl = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'
    BootstrapScript    = 'installers\common\bootstrap.py'
    PortablePythonExe  = 'installers\runtime\windows\python-full\python.exe'
}
