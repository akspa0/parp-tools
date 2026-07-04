#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

test -f "$ROOT_DIR/manifest.json"
test -f "$ROOT_DIR/src/harvester/v23/__init__.py"
test -f "$ROOT_DIR/src/harvester/v22_zarr_io.py"
test -f "$ROOT_DIR/scripts/train_v23_height.py"
test -f "$ROOT_DIR/scripts/infer_v23_height.py"

uv run python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path.cwd()
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("contains_game_client_files") is not False:
    raise SystemExit("manifest contains_game_client_files must be false")
from harvester.v23 import V23HeightPredictor  # noqa: F401
print("bundle verification passed")
PY
