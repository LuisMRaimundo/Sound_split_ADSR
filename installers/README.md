# Autonomous installers (no Python required)

These launchers install a **private copy** of Python and all app libraries on **first run**, then open the **Sound Split ADSR** desktop window. You do **not** need Python, pip, or conda on your computer.

**Requirements:** Internet on first run (~150–250 MB download). Disk space ~500 MB after install. Windows 10/11, macOS 11+, or a recent Linux (x86_64 or arm64).

---

## Windows 10 / 11

1. Open the project folder (or your ZIP after unpacking).
2. Double-click:

   **`installers\windows\Install and Run.bat`**

3. Wait for the first-time setup to finish (several minutes).
4. The ADSR splitter window opens. Keep the console window open while you use the app.

To stop: close the GUI window, then close the console or press **Ctrl+C**.

---

## macOS

1. In Terminal, make the launcher executable (once):

   ```bash
   chmod +x "installers/macos/Install and Run.command"
   chmod +x installers/macos/setup-runtime.sh
   ```

2. Double-click **`installers/macos/Install and Run.command`**
   (If macOS blocks it: **System Settings → Privacy & Security → Open Anyway**.)

Alternatively:

```bash
bash "installers/macos/Install and Run.command"
```

---

## Linux

1. Make the script executable (once):

   ```bash
   chmod +x installers/linux/install-and-run.sh installers/linux/setup-runtime.sh
   ```

2. Run:

   ```bash
   ./installers/linux/install-and-run.sh
   ```

---

## What gets installed?

| Location | Contents |
|----------|----------|
| `installers/runtime/` | Private Python + pip packages (not shared with the system) |
| Desktop | Tkinter GUI for batch ADSR splitting |

This folder is **gitignored**; each machine builds its own copy.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| “Setup failed” / download error | Check internet; proxy may block GitHub or python.org |
| GUI does not appear | Run from Terminal to see errors; ensure Tk is available (included in portable builds) |
| MP3 files fail to load | Install [ffmpeg](https://ffmpeg.org/) on your system PATH |
| Reinstall from scratch | Delete `installers/runtime/` and run the launcher again |

Diagnostics (if you have any Python 3.10+):

```bash
python installers/common/bootstrap.py doctor
```

---

## For maintainers

- **Windows** uses the official **embeddable** CPython zip from python.org (first run may use system `py` launcher if present).
- **macOS / Linux** use **python-build-standalone** install-only tarballs (no compiler, no root).
- After portable Python exists, `installers/common/bootstrap.py` runs `pip install -e .` and launches `split_audio_segments.py`.

Do not commit `installers/runtime/` to git (see `.gitignore`).

Modelled after [Interval-Homogeneity-Analyser](https://github.com/LuisMRaimundo/Interval-Homogeneity-Analyser) installers.
