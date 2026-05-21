# Sound Split ADSR - Windows installation

**Repository:** https://github.com/LuisMRaimundo/Sound_split_ADSR

## Standard installation (no Python required)

1. Download a **fresh** ZIP from GitHub (**Code -> Download ZIP**) or clone the repo.
2. Open **`installers\windows`**.
3. Double-click **`INSTALL.bat`** or **`START-HERE.bat`** (same as **Install and Run.bat**).
4. Wait for setup to finish (first run: **10-25 minutes**, downloads portable Python with Tkinter).
5. The ADSR splitter GUI opens when ready.

**Do not** use an old ZIP saved before May 2026.

## Install log

`installers\runtime\windows\install.log`

## Developers (Python already installed)

Use **`run.bat`** at the project root.

## Troubleshooting

| Issue | Action |
|-------|--------|
| No window / closes instantly | Re-download from GitHub; run **`INSTALL.bat`**. Never use `>>>` in batch echo lines. |
| PowerShell parse error | Old copy with Unicode characters; download fresh from GitHub. |
| Python / Tkinter error | Delete `installers\runtime\` and run **`INSTALL.bat`** again. |
| Setup failed | Open `install.log`, check Internet/firewall. |
