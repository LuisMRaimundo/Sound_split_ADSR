#!/usr/bin/env python3
"""
Bootstrap portable Python + Sound Split ADSR dependencies, then launch the Tkinter GUI.

Used by installers/windows, installers/macos, installers/linux launchers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Embedded/portable Python may not put this folder on sys.path before imports run.
_COMMON_DIR = Path(__file__).resolve().parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from config import (
    APP_MODULE,
    PROJECT_ROOT,
    RUNTIME_DIR,
    STAMP_FILE,
    machine_key,
    pbs_download_url,
    platform_key,
    runtime_python_dir,
    runtime_python_exe,
)

GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
STAMP_VERSION = "2"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    _log("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env, check=True)


def _tkinter_available(py: Path) -> bool:
    try:
        subprocess.run(
            [str(py), "-c", "import tkinter"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _pip_available(py: Path) -> bool:
    try:
        subprocess.run(
            [str(py), "-m", "pip", "--version"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def _ensure_pip(py: Path, runtime_dir: Path) -> None:
    if _pip_available(py):
        return
    get_pip = runtime_dir / "get-pip.py"
    if not get_pip.is_file():
        _download(GET_PIP_URL, get_pip)
    _log("Installing pip into portable Python …")
    _run([str(py), str(get_pip)])
    if not _pip_available(py):
        raise RuntimeError("pip could not be installed in portable Python")


def _setup_pbs(platform_name: str) -> Path:
    py_exe = runtime_python_exe(platform_name)
    runtime_dir = runtime_python_dir(platform_name)

    if py_exe.is_file() and _tkinter_available(py_exe):
        if platform_name == "windows":
            _ensure_pip(py_exe, runtime_dir)
        return py_exe

    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    arch = machine_key()
    url = pbs_download_url(platform_name, arch)
    _log(f"Setting up portable Python for {platform_name} (includes Tkinter) …")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "python.tar.gz"
        _download(url, archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_path)
        extracted = next(p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("python"))
        runtime_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(runtime_dir))

    if not py_exe.is_file():
        raise RuntimeError(f"Portable Python not found after extract: {py_exe}")
    if not _tkinter_available(py_exe):
        raise RuntimeError(
            "Portable Python was installed but Tkinter is missing. "
            "Delete installers/runtime/ and run again."
        )
    _ensure_pip(py_exe, runtime_dir)
    return py_exe


def ensure_portable_python() -> Path:
    plat = platform_key()
    existing = runtime_python_exe(plat)
    if existing.is_file():
        if _tkinter_available(existing):
            if plat == "windows":
                _ensure_pip(existing, runtime_python_dir("windows"))
            return existing
        _log("Current portable Python cannot open the GUI (no Tkinter).")
        _log("Downloading a full Python runtime (one-time, ~50 MB) …")
        if STAMP_FILE.is_file():
            STAMP_FILE.unlink(missing_ok=True)

    _log(f"No bundled Python yet — setting up portable runtime for {plat} …")
    return _setup_pbs(plat)


def ensure_app_installed(py: Path) -> None:
    if STAMP_FILE.is_file():
        try:
            if STAMP_FILE.read_text(encoding="utf-8").strip() == _stamp_payload():
                return
        except OSError:
            pass

    _log("Installing Sound Split ADSR and dependencies (first run may take several minutes) …")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    _run(
        [str(py), "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
        env={**os.environ, "PIP_NO_WARN_SCRIPT_LOCATION": "1"},
    )
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(_stamp_payload(), encoding="utf-8")
    _log("Install complete.")


def _stamp_payload() -> str:
    plat = platform_key()
    pyproject = PROJECT_ROOT / "pyproject.toml"
    py_exe = runtime_python_exe(plat)
    return (
        f"v={STAMP_VERSION}\n"
        f"runtime={py_exe.resolve()}\n"
        f"root={PROJECT_ROOT.resolve()}\n"
        f"pyproject={pyproject.stat().st_mtime_ns if pyproject.is_file() else 0}\n"
    )


def launch_gui(py: Path) -> int:
    if not _tkinter_available(py):
        _log("ERROR: This Python build has no Tkinter — the GUI cannot start.")
        _log("Delete installers/runtime/ and run run.bat again.")
        return 1
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    _log("Starting Sound Split ADSR (Tkinter window).")
    _log("Close the window to exit.")
    return subprocess.call([str(py), str(APP_MODULE)], cwd=PROJECT_ROOT, env=env)


def cmd_setup(_: argparse.Namespace) -> int:
    py = ensure_portable_python()
    ensure_app_installed(py)
    _log(f"Ready. Python: {py}")
    return 0


def cmd_launch(_: argparse.Namespace) -> int:
    py = ensure_portable_python()
    ensure_app_installed(py)
    return launch_gui(py)


def cmd_doctor(_: argparse.Namespace) -> int:
    plat = platform_key()
    py = runtime_python_exe(plat)
    _log(f"Project root: {PROJECT_ROOT}")
    _log(f"Platform: {plat} / {machine_key()}")
    _log(f"Portable Python: {py} ({'found' if py.is_file() else 'missing'})")
    if py.is_file():
        _log(f"Tkinter: {'ok' if _tkinter_available(py) else 'MISSING'}")
        _log(f"pip: {'ok' if _pip_available(py) else 'missing'}")
    legacy = RUNTIME_DIR / "windows" / "python" / "python.exe"
    if legacy.is_file() and legacy != py:
        _log(f"Legacy embed Python (no Tkinter): {legacy} — safe to delete")
    _log(f"Install stamp: {STAMP_FILE} ({'ok' if STAMP_FILE.is_file() else 'missing'})")
    _log(f"App: {APP_MODULE} ({'found' if APP_MODULE.is_file() else 'missing'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sound Split ADSR bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="Download portable Python and install dependencies").set_defaults(func=cmd_setup)
    sub.add_parser("launch", help="Setup if needed, then start GUI").set_defaults(func=cmd_launch)
    sub.add_parser("doctor", help="Print installer diagnostics").set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as exc:
        _log(f"Command failed with exit code {exc.returncode}.")
        return exc.returncode or 1
    except Exception as exc:
        _log(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("launch")
    raise SystemExit(main())
