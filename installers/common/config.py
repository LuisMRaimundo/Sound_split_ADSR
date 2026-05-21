"""Paths and download URLs for autonomous installers (no system Python required)."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

PYTHON_VERSION = "3.11.9"
PBS_TAG = "20240415"

INSTALLERS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = INSTALLERS_DIR.parent
RUNTIME_DIR = INSTALLERS_DIR / "runtime"
STAMP_FILE = RUNTIME_DIR / ".install_ok"

APP_MODULE = PROJECT_ROOT / "split_audio_segments.py"


def platform_key() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported OS: {sys.platform}")


def machine_key() -> str:
    m = platform.machine().lower()
    if m in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if m in {"arm64", "aarch64"}:
        return "aarch64"
    raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}")


def pbs_artifact(platform_name: str, arch: str) -> str:
    triples = {
        ("windows", "x86_64"): "x86_64-pc-windows-msvc",
        ("macos", "x86_64"): "x86_64-apple-darwin",
        ("macos", "aarch64"): "aarch64-apple-darwin",
        ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    }
    triple = triples.get((platform_name, arch))
    if not triple:
        raise RuntimeError(f"No portable Python build for {platform_name} / {arch}")
    return f"cpython-{PYTHON_VERSION}+{PBS_TAG}-{triple}-install_only.tar.gz"


def pbs_download_url(platform_name: str, arch: str) -> str:
    name = pbs_artifact(platform_name, arch)
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PBS_TAG}/{name}"
    )


def runtime_python_dir(platform_name: str) -> Path:
    # Windows: full PBS install (includes Tkinter). Legacy embed used .../python/ (no Tkinter).
    if platform_name == "windows":
        return RUNTIME_DIR / platform_name / "python-full"
    return RUNTIME_DIR / platform_name / "python"


def runtime_python_exe(platform_name: str) -> Path:
    base = runtime_python_dir(platform_name)
    if platform_name == "windows":
        return base / "python.exe"
    return base / "bin" / "python3"
