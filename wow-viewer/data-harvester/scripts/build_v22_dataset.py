"""Build the V22 Zarr dataset from a V18 store + enrichment stream.

The C# ``WowViewer.Tool.V22Enrich`` reads a finished V18 store's placements,
decodes each unique M2 / WMO exactly once, and writes a stable-path-keyed
binary enrichment stream. The Python ``build`` subcommand reads the V18 store
+ the enrichment stream and writes the canonical V22 Zarr store.

Usage::

    # Step 1: build the enrichment stream
    uv run python scripts/build_v22_dataset.py enrich \\
        --v18-store ../output/datasets/v18/3_3_5_12340.zarr \\
        --client-root ../output/tmp/wowarchive-clients/3_3_5_12340 \\
        --enrichment-output ../output/tmp/v22_enrich/3_3_5_12340.bin \\
        --build-key 3_3_5_12340

    # Step 2: build the V22 Zarr store
    uv run python scripts/build_v22_dataset.py build \\
        --v18-store ../output/datasets/v18/3_3_5_12340.zarr \\
        --enrichment ../output/tmp/v22_enrich/3_3_5_12340.bin \\
        --output ../output/datasets/v22/3_3_5_12340.zarr

    # Inspect the store
    uv run python scripts/build_v22_dataset.py stats \\
        --store ../output/datasets/v22/3_3_5_12340.zarr

The ``enrich`` subcommand is a convenience wrapper that shells out to the
C# tool. The ``build`` subcommand is pure Python — it reads V18 Zarr arrays
directly and writes the V22 Zarr store.

V18 builders and trainers are not modified. V22 is a downstream consumer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA_HARVESTER_ROOT = Path(__file__).resolve().parent.parent
WOW_VIEWER_ROOT = DATA_HARVESTER_ROOT.parent
WORKSPACE_ROOT = WOW_VIEWER_ROOT.parent
sys.path.insert(0, str(DATA_HARVESTER_ROOT / "src"))

from harvester.v22_zarr_io import V22ZarrWriter  # noqa: E402


DEFAULT_V22_OUTPUT_ROOT = WOW_VIEWER_ROOT / "output" / "datasets" / "v22"
DEFAULT_ENRICH_TOOL_DIR = (
    WOW_VIEWER_ROOT / "tools" / "enrich" / "WowViewer.Tool.V22Enrich"
    / "bin" / "Debug" / "net10.0"
)
DEFAULT_ENRICH_DLL = DEFAULT_ENRICH_TOOL_DIR / "WowViewer.Tool.V22Enrich.dll"


def _resolve_cli_path(path: Path) -> Path:
    """Resolve a CLI path relative to workspace roots."""
    if path.is_absolute() and path.exists():
        return path

    candidates = [
        (Path.cwd() / path).resolve(),
        (DATA_HARVESTER_ROOT / path).resolve(),
        (WOW_VIEWER_ROOT / path).resolve(),
        (WORKSPACE_ROOT / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return (Path.cwd() / path).resolve()


def _run_enrich(
    v18_store: Path,
    client_root: Path,
    enrichment_output: Path,
    build_key: str,
    limit: int | None,
    enrich_tool: Path | None,
) -> None:
    """Shell out to WowViewer.Tool.V22Enrich to build the enrichment stream."""
    resolved_v18 = _resolve_cli_path(v18_store)
    resolved_client = _resolve_cli_path(client_root)

    enrichment_output.parent.mkdir(parents=True, exist_ok=True)

    if not resolved_v18.exists():
        raise RuntimeError(f"V18 store not found: {resolved_v18}")
    if not resolved_client.exists():
        raise RuntimeError(f"Client root not found: {resolved_client}")

    # Find the enrich tool
    if enrich_tool is not None:
        tool_path = _resolve_cli_path(enrich_tool)
    elif DEFAULT_ENRICH_DLL.exists():
        tool_path = DEFAULT_ENRICH_DLL
    else:
        raise RuntimeError(
            f"Enrich tool not found. Build it first:\n"
            f"  dotnet build {WOW_VIEWER_ROOT / 'tools' / 'enrich' / 'WowViewer.Tool.V22Enrich'} -c Debug"
        )

    cmd = [
        "dotnet", str(tool_path),
        "--v18-store", str(resolved_v18),
        "--client-root", str(resolved_client),
        "--output", str(enrichment_output),
        "--build-key", build_key,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode not in (0, 2):
        stderr = result.stderr or "(no stderr)"
        raise RuntimeError(f"V22Enrich failed with exit {result.returncode}: {stderr}")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="", flush=True)


def _build(v18_store: Path, enrichment: Path, output: Path) -> Path:
    """Read V18 store + enrichment stream, write V22 Zarr store."""
    resolved_v18 = _resolve_cli_path(v18_store)
    resolved_enrich = _resolve_cli_path(enrichment)

    if not resolved_v18.exists():
        raise RuntimeError(f"V18 store not found: {resolved_v18}")
    if not resolved_enrich.exists():
        raise RuntimeError(f"Enrichment stream not found: {resolved_enrich}")
    if resolved_enrich.stat().st_size < 8:
        # Empty enrichment stream — still produce a V22 store (no model/tileset entries)
        pass

    writer = V22ZarrWriter(output, overwrite=True)
    writer.add_from_v18(str(resolved_v18), str(resolved_enrich))
    return writer.finalize()


def _stats(store_path: Path) -> dict[str, object]:
    """Print a summary of an existing V22 Zarr store."""
    if not store_path.exists():
        return {"exists": False, "path": str(store_path)}

    import zarr
    import zarr.storage

    grp = zarr.open_group(zarr.storage.LocalStore(str(store_path), read_only=True), mode="r")
    model_count = int(grp["models/model_paths"].shape[0]) if "models" in grp else 0
    tileset_count = int(grp["tilesets/tileset_paths"].shape[0]) if "tilesets" in grp else 0
    return {
        "exists": True,
        "path": str(store_path),
        "tile_count": int(grp.attrs.get("tile_count", 0)),
        "builds": list(grp.attrs.get("builds", [])),
        "root_arrays": sorted(grp.array_keys()),
        "model_count": model_count,
        "tileset_count": tileset_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="V22 Zarr dataset builder (V18 + enrichment stream → V22 Zarr)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── enrich ─────────────────────────────────────────────────────
    enrich = sub.add_parser(
        "enrich",
        help="Run WowViewer.Tool.V22Enrich to build the enrichment stream",
    )
    enrich.add_argument("--v18-store", required=True, type=Path)
    enrich.add_argument("--client-root", required=True, type=Path)
    enrich.add_argument("--enrichment-output", required=True, type=Path)
    enrich.add_argument("--build-key", required=True)
    enrich.add_argument("--limit", type=int, default=None)
    enrich.add_argument("--enrich-tool", type=Path, default=None)

    # ── build ─────────────────────────────────────────────────────
    build = sub.add_parser(
        "build",
        help="Read V18 store + enrichment stream, write V22 Zarr store",
    )
    build.add_argument("--v18-store", required=True, type=Path)
    build.add_argument("--enrichment", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)

    # ── stats ─────────────────────────────────────────────────────
    stats = sub.add_parser("stats", help="Print a summary of an existing V22 Zarr store")
    stats.add_argument("--store", required=True, type=Path)

    args = parser.parse_args()

    try:
        if args.command == "enrich":
            _run_enrich(
                v18_store=args.v18_store,
                client_root=args.client_root,
                enrichment_output=args.enrichment_output,
                build_key=args.build_key,
                limit=args.limit,
                enrich_tool=args.enrich_tool,
            )
            print(f"wrote {args.enrichment_output}")
            return 0

        elif args.command == "build":
            out = _build(
                v18_store=args.v18_store,
                enrichment=args.enrichment,
                output=args.output,
            )
            print(f"wrote {out}")
            return 0

        elif args.command == "stats":
            result = _stats(args.store)
            print(json.dumps(result, indent=2, default=str))
            return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())