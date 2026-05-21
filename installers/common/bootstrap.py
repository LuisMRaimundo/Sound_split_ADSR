#!/usr/bin/env python3
"""
Bootstrap portable Python + Sound Split ADSR dependencies, then launch the Tkinter GUI.

Used by installers/windows, installers/macos, installers/linux launchers.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

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
    windows_embed_zip_url,
)

GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    _log("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env, check=True)


def _setup_windows_embed() -> Path:
    py_exe = runtime_python_exe("windows")
    if py_exe.is_file():
        return py_exe

    runtime_dir = runtime_python_dir("windows")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "python-embed.zip"
        _download(windows_embed_zip_url(), zip_path)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(runtime_dir)

        get_pip = runtime_dir / "get-pip.py"
        _download(GET_PIP_URL, get_pip)

        pth_files = list(runtime_dir.glob("python*._pth"))
        if pth_files:
            text = pth_files[0].read_text(encoding="utf-8")
            if "import site" not in text:
                lines = [ln for ln in text.splitlines() if ln.strip() != "#import site"]
                if not any(ln.strip() == "import site" for ln in lines):
                    lines.append("import site")
                pth_files[0].write_text("\n".join(lines) + "\n", encoding="utf-8")

        py_exe = runtime_python_exe("windows")
        _run([str(py_exe), str(get_pip)])
        return py_exe


def _setup_pbs(platform_name: str) -> Path:
    py_exe = runtime_python_exe(platform_name)
    if py_exe.is_file():
        return py_exe

    arch = machine_key()
    url = pbs_download_url(platform_name, arch)
    runtime_dir = runtime_python_dir(platform_name)
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "python.tar.gz"
        _download(url, archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_path)
        extracted = next(p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("python"))
        shutil.move(str(extracted), str(runtime_dir))

    if not py_exe.is_file():
        raise RuntimeError(f"Portable Python not found after extract: {py_exe}")
    return py_exe


def ensure_portable_python() -> Path:
    plat = platform_key()
    existing = runtime_python_exe(plat)
    if existing.is_file():
        return existing

    _log(f"No bundled Python yet — setting up portable runtime for {plat} …")
    if plat == "windows":
        return _setup_windows_embed()
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
    _run([str(py), "-m", "pip", "install", "-e", str(PROJECT_ROOT)])
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(_stamp_payload(), encoding="utf-8")
    _log("Install complete.")


def _stamp_payload() -> str:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    return f"v=1\nroot={PROJECT_ROOT.resolve()}\npyproject={pyproject.stat().st_mtime_ns if pyproject.is_file() else 0}\n"


def launch_gui(py: Path) -> int:
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
    _log(f"Project root: {PROJECT_ROOT}")
    _log(f"Platform: {platform_key()} / {machine_key()}")
    py = runtime_python_exe(platform_key())
    _log(f"Portable Python: {py} ({'found' if py.is_file() else 'missing'})")
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if len(sys.argv) == 1:
        sys.argv.append("launch")
    raise SystemExit(main())
