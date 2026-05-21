#!/usr/bin/env bash
# Portable Python via python-build-standalone (no system Python required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNTIME="$ROOT/installers/runtime/macos/python"
PY="$RUNTIME/bin/python3"

if [[ -x "$PY" ]]; then
  exit 0
fi

arch="$(uname -m)"
case "$arch" in
  x86_64) PBS_ARCH="x86_64" ;;
  arm64|aarch64) PBS_ARCH="aarch64" ;;
  *)
    echo "Unsupported macOS architecture: $arch" >&2
    exit 1
    ;;
esac

VERSION="3.11.9"
PBS_TAG="20240415"
TRIPLE="${PBS_ARCH}-apple-darwin"
NAME="cpython-${VERSION}+${PBS_TAG}-${TRIPLE}-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${NAME}"

command -v curl >/dev/null 2>&1 || {
  echo "curl is required for first-time setup." >&2
  exit 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading portable Python ${VERSION} (one-time) ..."
curl -fsSL "$URL" -o "$TMP/python.tar.gz"
rm -rf "$RUNTIME"
mkdir -p "$(dirname "$RUNTIME")"
tar -xzf "$TMP/python.tar.gz" -C "$TMP"
mv "$TMP"/python "$RUNTIME"
chmod +x "$PY"
echo "Portable Python ready: $PY"
