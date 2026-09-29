#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install 'uv==0.9.7'
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime"
export UV_CACHE_DIR="$PWD/.cache/uv"
.bootstrap/bin/uv python install --no-bin 3.11.14
.bootstrap/bin/uv sync --locked --extra dev
