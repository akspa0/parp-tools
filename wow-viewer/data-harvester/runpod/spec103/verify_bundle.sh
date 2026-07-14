#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

test -f "$ROOT_DIR/manifest.json"
test -f "$ROOT_DIR/src/harvester/spec103/v8_model.py"
test -f "$ROOT_DIR/src/harvester/spec103/v7_model.py"
test -f "$ROOT_DIR/scripts/train_spec103_v7.py"
test -f "$ROOT_DIR/data/curation/curation_manifest.parquet"

uv run python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path.cwd()
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("contains_game_client_files") is not False:
    raise SystemExit("manifest contains_game_client_files must be false")
store_dest = root / manifest["store"]["dest"]
if not store_dest.exists():
    raise SystemExit(f"bundled store missing: {store_dest}")

from harvester.spec103 import v7_model, v8_model, v7_inputs, v7_losses  # noqa: F401

import torch
m = v8_model.V8LeanUNet(use_wdl_global_trestle=True)
n_params = sum(p.numel() for p in m.parameters())
print(f"v8 params: {n_params:,}")
assert n_params < 10_000_000, "v8 param budget regressed"

print(f"bundled store: {manifest['store']['kept_rows']} tiles, fields={manifest['store']['fields_copied']}")
print("bundle verification passed")
PY

echo "[spec103] running bundled test suite..."
uv run pytest tests/spec103 -q
echo "[spec103] verify_bundle passed"
