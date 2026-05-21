#!/usr/bin/env bash
# Double-click in Finder (macOS). Installs portable Python on first run, then opens the app.
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
export ROOT

echo ""
echo " Sound Split ADSR"
echo " ================"
echo ""

bash "$ROOT/installers/macos/setup-runtime.sh"

PY="$ROOT/installers/runtime/macos/python/bin/python3"
BOOT="$ROOT/installers/common/bootstrap.py"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Portable Python setup failed." >&2
  read -r -p "Press Enter to close..."
  exit 1
fi

"$PY" "$BOOT" launch || {
  code=$?
  echo ""
  echo "The app exited with code $code."
  read -r -p "Press Enter to close..."
  exit "$code"
}
