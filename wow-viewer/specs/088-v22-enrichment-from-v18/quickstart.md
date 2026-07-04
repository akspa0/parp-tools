# Quickstart: V22 Enrichment From V18

Operator guide for Spec 088. Two commands: a C# enrich tool and a Python build tool.

## Prerequisites

- A V18 Zarr store already built. The V22 build reads it as the substrate. Build one with:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v18_dataset.py build --build 3_3_5_12340 --limit 1
```

- A staged 3.3.5 client under `output/tmp/wowarchive-clients/3_3_5_12340` (or any of the three scoped builds: `0_5_3_3368`, `3_3_5_12340`, `4_0_0_11927`).
- The `WowViewer.Tool.V22Enrich` binary built. Build it with:

```powershell
dotnet build wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich -c Debug
```

## Step 1 — Build the Enrichment Stream

The C# tool reads the V18 store's `placements.parquet`, walks unique asset paths, decodes each M2 / WMO / BLP exactly once, and writes a stable-path-keyed binary stream.

### 1a. Via the Python wrapper (preferred)

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py enrich `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --client-root ../output/tmp/wowarchive-clients/3_3_5_12340 `
    --enrichment-output ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --build-key 3_3_5_12340 `
    --limit 1
```

### 1b. Via the C# tool directly (debug)

```powershell
dotnet wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich/bin/Debug/net10.0/WowViewer.Tool.V22Enrich.dll `
    --v18-store output/datasets/v18/3_3_5_12340.zarr `
    --client-root output/tmp/wowarchive-clients/3_3_5_12340 `
    --output output/tmp/v22_enrich/3_3_5_12340.bin `
    --build-key 3_3_5_12340 `
    --limit 1
```

## Step 2 — Build the V22 Zarr Store

The Python builder reads the V18 store + the enrichment stream and writes the V22 Zarr store.

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py build `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --enrichment ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --output ../output/datasets/v22/3_3_5_12340.zarr
```

The output store lives at `output/datasets/v22/3_3_5_12340.zarr/` with:
- 20 V18-derived root arrays
- 5 V22-patched signals (derived in pure Python)
- 4 V22 native placement arrays (mddf/modf + offsets + counts + ids)
- `models/` group with one entry per unique M2 / WMO path (`model_paths`, `model_kind`, `load_error`)
- `tilesets/` group with one entry per unique BLP path (`tileset_paths`, `texture_shape`, `load_error`)
- `mcly_tileset_ids` per tile (build-wide remap)
- `mddf_model_ids` / `modf_model_ids` per tile (resolved to `models/model_paths` indices)
- Audit sidecars (`finalization.json`, `index.parquet`, `placements.parquet`, `asset_inventory.parquet`)

The canonical build mode is `asset_payload_mode = "paths_only"`. That means the V22 store tracks stable asset ids, dimensions, and provenance, but does not treat embedded M2/WMO/BLP payload blobs as part of the canonical training contract.

## Step 3 — Inspect the V22 Store

```powershell
cd wow-viewer/data-harvester

# Store-level summary
uv run python scripts/inspect_v22_dataset.py summary `
    --store ../output/datasets/v22/3_3_5_12340.zarr

# Single-tile detail
uv run python scripts/inspect_v22_dataset.py tile `
    --store ../output/datasets/v22/3_3_5_12340.zarr `
    --tile-index 0 `
    --output-json ../output/tmp/v22_tile_0.json
```

The `summary` output reports `tile_count`, `builds`, `model_count`, `tileset_count`, and root array layout. The `tile` output reports per-array shape, dtype, nonzero count, and min/max/mean for one tile.

## One-Command Operator Surface

For routine use, the Python wrapper's `enrich` subcommand runs the C# tool as a subprocess and produces the enrichment stream; the `build` subcommand reads the stream. The two-step is explicit because operators often want to inspect the stream between steps (e.g. after a partial asset decode failure).

If you want a single command, run them sequentially in one shell:

```powershell
cd wow-viewer/data-harvester
uv run python scripts/build_v22_dataset.py enrich `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --client-root ../output/tmp/wowarchive-clients/3_3_5_12340 `
    --enrichment-output ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --build-key 3_3_5_12340 `
    --limit 1 `
&& uv run python scripts/build_v22_dataset.py build `
    --v18-store ../output/datasets/v18/3_3_5_12340.zarr `
    --enrichment ../output/tmp/v22_enrich/3_3_5_12340.bin `
    --output ../output/datasets/v22/3_3_5_12340.zarr
```

## Bounded Real-Data Proof

The Phase 9 proof script automates the three steps above and validates the result. Run it from the spec directory:

```powershell
cd wow-viewer
uv run python specs/088-v22-enrichment-from-v18/scripts/proof_v22_bounded.py `
    --build 3_3_5_12340 `
    --map Azeroth `
    --limit 1
```

The proof asserts:
- `tile_count == 1`
- `model_count > 0`
- `tileset_count > 0`
- All documented root arrays exist with correct shape and dtype
- The output JSON is saved to `output/proofs/v22_bounded_335_<timestamp>.json`

## Failure Modes and Recovery

| Failure | Cause | Recovery |
|---------|-------|----------|
| Tool not found | `WowViewer.Tool.V22Enrich` not built | `dotnet build wow-viewer/tools/enrich/WowViewer.Tool.V22Enrich -c Debug` |
| `Could not resolve client root` | Path missing nested `World of Warcraft` | Verify `output/tmp/wowarchive-clients/<build>/World of Warcraft/Data` exists |
| `load_error=1` for many assets | Staged client missing those assets | Re-stage from WoWArchive; some legacy assets are missing by design |
| `tile_count = 0` | V18 store empty or limit too small | Rerun V18 build with a higher `--limit` |
| Exit code 1 | Missing input file | Check the V18 store path, the enrichment stream path, and the client root |
| Exit code 2 | Partial failure | Inspect the enrichment stream for `load_error=1` entries; the V22 store will still be written for the assets that did decode |

## What V22 Does Not Replace

- V18 stores. V22 is built on top of V18. V18 trainers (`train_v18.py`, `train_v18_focus.py`) keep consuming V18 stores without changes.
- The V18 builder. V22 enrichment is a downstream consumer of V18. The V18 build path is unchanged.
- Spec 086 or 087. Both are marked superseded. Spec 088 is the canonical V22 design.
