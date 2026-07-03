#!/usr/bin/env python3
"""Bounded real-data proof for Spec 088 V22 enrichment from V18.

Runs end-to-end against a staged client for a single tile:
  1. Build a V18 store (if not already present) with --limit 1
  2. Run WowViewer.Tool.V22Enrich → enrichment stream
  3. Run build_v22_dataset.py build → V22 Zarr store
  4. Run inspect_v22_dataset.py summary → verify tile_count, model_count, tileset_count
  5. Save proof output JSON

Usage:
    uv run python specs/088-v22-enrichment-from-v18/scripts/proof_v22_bounded.py \\
        --build 3_3_5_12340 --map Azeroth --limit 1

Assumes:
- Staged client exists under output/tmp/wowarchive-clients/<build>
- V18 builder (build_v18_dataset.py) is available in data-harvester/scripts/
- V22Enrich C# tool is built
- build_v22_dataset.py is available in data-harvester/scripts/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
SPEC_DIR = Path(__file__).resolve().parent.parent
DATA_HARVESTER_DIR = SPEC_DIR.parent.parent / "data-harvester"
WOW_VIEWER_ROOT = DATA_HARVESTER_DIR.parent
WORKSPACE_ROOT = WOW_VIEWER_ROOT.parent
OUTPUT_PROOFS_DIR = WOW_VIEWER_ROOT / "output" / "proofs"
STAGED_CLIENT_DIR = WORKSPACE_ROOT / "output" / "tmp" / "wowarchive-clients"
V18_OUTPUT_DIR = WOW_VIEWER_ROOT / "output" / "datasets" / "v18"
V22_OUTPUT_DIR = WOW_VIEWER_ROOT / "output" / "datasets" / "v22"


def _resolve_path(path: Path) -> Path:
    """Resolve a path — try WORKSPACE_ROOT-based, WOW_VIEWER_ROOT-based,
    and as-is."""
    if path.exists():
        return path.resolve()
    # Try workspace root
    wp = (WORKSPACE_ROOT / str(path).lstrip(str(WORKSPACE_ROOT))).resolve()
    if wp.exists():
        return wp
    return path.resolve()


def _ensure_v18_store(build: str, map_name: str, limit: int) -> Path:
    """Build a V18 store if not already present."""
    store = V18_OUTPUT_DIR / f"{build}.zarr"
    if store.exists() and (store / "height_257").exists():
        return store

    if not store.parent.exists():
        store.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u",
        str(DATA_HARVESTER_DIR / "scripts" / "build_v18_dataset.py"),
        "build",
        "--build", build,
        "--maps", map_name,
        "--limit", str(limit),
        "--allow-zarr-write",
        "--output", str(store),
    ]
    print(f"[proof] Building V18 store: {store}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Failed to build V18 store (exit {result.returncode})")
    print(result.stdout)
    return store


def _run_enrich(
    v18_store: Path,
    client_root: Path,
    enrichment_output: Path,
    build_key: str,
) -> None:
    """Build the enrichment stream."""
    enrichment_output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        str(DATA_HARVESTER_DIR / "scripts" / "build_v22_dataset.py"),
        "enrich",
        "--v18-store", str(v18_store),
        "--client-root", str(client_root),
        "--enrichment-output", str(enrichment_output),
        "--build-key", build_key,
    ]
    print(f"[proof] Building enrichment stream: {enrichment_output}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 2):
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Enrichment failed (exit {result.returncode})")
    print(result.stdout)


def _run_build(v18_store: Path, enrichment_path: Path, output: Path) -> None:
    """Build the V22 Zarr store."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u",
        str(DATA_HARVESTER_DIR / "scripts" / "build_v22_dataset.py"),
        "build",
        "--v18-store", str(v18_store),
        "--enrichment", str(enrichment_path),
        "--output", str(output),
    ]
    print(f"[proof] Building V22 store: {output}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"V22 build failed (exit {result.returncode})")
    print(result.stdout)


def _run_inspect(store: Path) -> dict:
    """Inspect the V22 store and return its summary."""
    import zarr as z
    import zarr.storage as zs

    grp = z.open_group(zs.LocalStore(str(store), read_only=True), mode="r")
    model_count = int(grp["models/model_paths"].shape[0]) if "models" in grp else 0
    tileset_count = int(grp["tilesets/tileset_paths"].shape[0]) if "tilesets" in grp else 0

    result = {
        "path": str(store),
        "tile_count": int(grp.attrs.get("tile_count", 0)),
        "builds": list(grp.attrs.get("builds", [])),
        "root_arrays": sorted(grp.array_keys()),
        "model_count": model_count,
        "tileset_count": tileset_count,
        "proof_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spec 088 bounded real-data proof"
    )
    parser.add_argument("--build", default="3_3_5_12340",
                        help="Build key (e.g. 3_3_5_12340)")
    parser.add_argument("--map", default="Azeroth",
                        help="Map name (e.g. Azeroth)")
    parser.add_argument("--limit", type=int, default=1,
                        help="Max tiles to process")
    parser.add_argument("--client-root", type=Path, default=None,
                        help="Staged client root (auto-resolves if not specified)")
    args = parser.parse_args()

    # ── Resolve client root ───────────────────────────────────────
    client_root: Path
    if args.client_root is not None:
        client_root = args.client_root
    else:
        client_root = STAGED_CLIENT_DIR / args.build / "World of Warcraft"
        if not client_root.exists():
            client_root = STAGED_CLIENT_DIR / args.build
    client_root = client_root.resolve()

    if not client_root.exists():
        print(f"[FAIL] Client root not found: {client_root}", file=sys.stderr)
        print(f"  Stage the client first via WoWArchive staging workflow.", file=sys.stderr)
        return 1

    # ── Step 1: V18 store ─────────────────────────────────────────
    try:
        v18_store = _ensure_v18_store(args.build, args.map, args.limit)
    except RuntimeError as exc:
        print(f"[FAIL] V18 store: {exc}", file=sys.stderr)
        return 1

    # ── Step 2: Enrichment stream ─────────────────────────────────
    enrich_output = OUTPUT_PROOFS_DIR.parent / "tmp" / "v22_enrich" / f"{args.build}_{args.map}.bin"
    try:
        _run_enrich(v18_store, client_root, enrich_output, args.build)
    except RuntimeError as exc:
        print(f"[FAIL] Enrichment: {exc}", file=sys.stderr)
        return 1

    # ── Step 3: V22 build ─────────────────────────────────────────
    v22_output = V22_OUTPUT_DIR / f"{args.build}_{args.map}_proof.zarr"
    try:
        _run_build(v18_store, enrich_output, v22_output)
    except RuntimeError as exc:
        print(f"[FAIL] V22 build: {exc}", file=sys.stderr)
        return 1

    # ── Step 4: Inspect ────────────────────────────────────────────
    summary = _run_inspect(v22_output)

    # ── Step 5: Assertions ────────────────────────────────────────
    failures: list[str] = []
    if summary["tile_count"] < 1:
        failures.append(f"tile_count={summary['tile_count']} expected >= 1")
    if summary["model_count"] < 1:
        failures.append(f"model_count={summary['model_count']} expected >= 1")
    if summary["tileset_count"] < 1:
        failures.append(f"tileset_count={summary['tileset_count']} expected >= 1")

    summary["passed"] = len(failures) == 0
    summary["failures"] = failures

    # ── Step 6: Save ──────────────────────────────────────────────
    OUTPUT_PROOFS_DIR.mkdir(parents=True, exist_ok=True)
    proof_path = OUTPUT_PROOFS_DIR / f"v22_bounded_{args.build}_{args.map}.json"
    proof_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[proof] saved: {proof_path}")

    if failures:
        summary_str = " | ".join(failures)
        print(f"[PROOF FAILED] {summary_str}", file=sys.stderr)
        return 1

    print(
        f"[PROOF PASSED] {summary['tile_count']} tiles, "
        f"{summary['model_count']} models, {summary['tileset_count']} tilesets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())