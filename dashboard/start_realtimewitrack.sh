#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/backend"
export DATA_SOURCE="${DATA_SOURCE:-mock}"
"$SCRIPT_DIR/../.venv/bin/python" -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$SCRIPT_DIR/frontend"
npm run dev -- --host 127.0.0.1 --port 5173
