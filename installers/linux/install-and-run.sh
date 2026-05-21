#!/usr/bin/env bash
# One-shot launcher for Linux. Make executable: chmod +x install-and-run.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

echo ""
echo " Sound Split ADSR"
echo " ================"
echo ""

bash "$ROOT/installers/linux/setup-runtime.sh"

PY="$ROOT/installers/runtime/linux/python/bin/python3"
BOOT="$ROOT/installers/common/bootstrap.py"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: Portable Python setup failed." >&2
  exit 1
fi

exec "$PY" "$BOOT" launch
