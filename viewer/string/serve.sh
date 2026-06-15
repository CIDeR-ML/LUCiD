#!/bin/bash
# Serve the string viewer from the LUCiD repo root.
# Usage: bash viewer/string/serve.sh [PORT]
PORT="${1:-8766}"
LUCID_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "String viewer: http://localhost:${PORT}/viewer/string/viewer.html"
echo "Ctrl+C to stop"
cd "$LUCID_ROOT" && python3 -m http.server "$PORT" --bind 0.0.0.0
