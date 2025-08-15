# PM4FacesTool Restart Guide

Last updated: 2025-08-15 16:49 (local)

This guide helps you quickly regain context and continue work on the PM4 Faces/Object + Tile exporter after a chat reset.

## Mission
- Export coherent objects (grouped correctly) and per-tile OBJs from PM4 data.
- Avoid producing hundreds/thousands of tiny per-surface instance files.
- Always generate tiles alongside objects with diagnostics CSVs.

## Key Entry Points
- Export tool: `src/PM4FacesTool/Program.cs`
- OBJ writer: `src/PM4FacesTool/ObjWriter.cs`
- Scene model: `src/parpToolbox/Formats/PM4/Pm4Scene.cs`
- Region loaders:
  - Simple merge: `src/parpToolbox/Services/PM4/Pm4Adapter.Region.cs` (does NOT populate tile offset/counts)
  - Global loader: `src/parpToolbox/Services/PM4/Pm4GlobalTileLoader.cs` (builds tile offset/count metadata and provides `ToStandardScene()`)

## How to Run
- Single start tile:
  - `dotnet run --project src/PM4FacesTool --input <path/to/Some_00_00.pm4>`
- Batch by folder:
  - `dotnet run --project src/PM4FacesTool --input <dir/with/pm4s> --batch`
- Options:
  - Single-file input loads ONLY that tile. Use `--batch` to process all tiles with the same filename prefix.
  - `--group-by composite-instance|surface|groupkey|composite` (default: `composite-instance`)
  - `--legacy-parity` (OBJ parity mode)
  - `--project-local` (center vertices locally)
  - `--ck-min-tris <n>` minimum triangles per composite-instance component (filters tiny components from being written; still counted in CSVs)
  - `--ck-merge-components` retain per-object DSU components under the same CK24 (no monolithic merged OBJ)

Recommended run (object-focused with DSU components and filtering):

```
dotnet run --project src/PM4FacesTool -- \
  --input <path/to/Some_00_00.pm4> \
  --group-by composite-instance \
  --ck-merge-components \
  --ck-min-tris 1000
```

Output root is under `project_output/pm4faces_YYYYMMDD_HHMMSS/<tile_name>/`:
- `objects/CK_XXXXXX/*.obj` — object OBJs sharded by CK24 directory
- `tiles/*.obj` — per-tile OBJs using original PM4 tile basenames; tile X-flip is forced during export
- Diagnostics CSVs in the session root:
  - `surface_coverage.csv`, `ck_instances.csv`, `instance_members.csv`, `tile_coverage.csv`

## Current State (what works / issues)
- Composite-instance grouping implemented and default in `Program.cs`.
- `--ck-merge-components` retains per-object DSU components for each CK24 (no monolithic merged OBJ).
- `--ck-min-tris` implemented: filters tiny components from being written while still tracked in diagnostics.
- Tile export uses `Pm4GlobalTileLoader.LoadRegion()` + `ToStandardScene()` to populate tile offset/counts.
- Tile OBJ filenames preserved from original PM4 basenames; tile X-flip is forced (independent of legacy parity).

- Single-file inputs now load ONLY the specified tile; use `--batch` to wildcard by prefix and load a region.

- Issues:
  - Current tile export still slices raw tile index ranges; tiles are not yet composed from merged objects per tile. This breaks object coherence on tiles.
  - Significant face/surface loss remains; need instrumentation to identify skip reasons and OOB counters.

## Next Actions (checklist)
1. Refactor tile export to build each tile OBJ by composing triangles from merged object meshes per tile (not raw index slices).
2. Implement fast global-index → tileId interval map and per-triangle tile assignment during object assembly.
3. Add diagnostics:
   - `tile_object_coverage.csv` with per-tile per-object triangle counts and reason breakdowns
   - `skip_reasons.csv` with aggregated counts (IndexOutOfRange, VertexMapFail, Degenerate, etc.)
   - Optional `--diag-skips` flag to enable detailed logging
4. Re-run with `--ck-merge-components` and tuned `--ck-min-tris` to validate object count reduction and coverage.
5. Verify tile X-flip remains forced for tiles and object exports continue to respect `--legacy-parity`.

## Acceptance Criteria
- Tiles are assembled from coherent merged objects placed per tile; tile OBJs look correct and complete.
- Tile filenames preserved (original PM4 basenames). Tile X-flip is forced.
- Object count reduced significantly via `--ck-merge-components` + `--ck-min-tris`.
- Diagnostics populated and consistent: `surface_coverage.csv`, `ck_instances.csv`, `instance_members.csv`, `tile_coverage.csv`, plus `tile_object_coverage.csv`, `skip_reasons.csv`.
- No out-of-bounds vertex access in logs; MSCN remap applied as needed.

## Diagnostics & Logs
- `Program.cs` prints scene stats including `Tiles={scene.TileIndexOffsetByTileId.Count}`.
- `tile_coverage.csv` columns: `tile_id,start_index,index_count,faces_written,faces_skipped,obj_path`.
- Planned additional CSVs:
  - `tile_object_coverage.csv`: `tile_id,object_key,triangles_written,triangles_skipped,reason_counts_json`
  - `skip_reasons.csv`: `reason,total_skipped`
- If `TileIndexCountByTileId` missing for a tile, `ExportTiles()` logs and skips that tile.

## Troubleshooting
- Tiles not generated:
  - Ensure `ProcessOne()` constructs the scene via `Pm4GlobalTileLoader.ToStandardScene()`.
  - Check `TileIndexOffsetByTileId.Count` > 0.
- Too many tiny `inst_*` files:
  - Use `--ck-min-tris`.
  - Consider higher-level grouping (`--group-by composite` or `groupkey`) for diagnostics.
- Data loss or OOB indices:
  - Run using region/global load; verify MSCN remap paths. See `Pm4Adapter.Region.cs` and `Pm4GlobalTileLoader.ApplyMscnRemap()`.
 - Tiles not coherent / faces missing:
   - Expected until tile composition refactor lands. Validate object OBJs first; use planned diagnostics to isolate skip reasons.

## Glossary
- CK24: top 24 bits (`CompositeKey >> 8`), used as composite-instance grouping key.
- MSCN: exterior vertex pool; remap adjusts indices referencing this pool.
- TileId: linear id `Y*64 + X` parsed from filename suffix `_XX_YY`.
 - CK_XXXXXX directories: per-CK24 sharding of object outputs.

## Pointers to Related Docs
- `memory-bank/projectbrief.md` — scope and goals
- `memory-bank/productContext.md` — why and how users want to use exporter
- `memory-bank/activeContext.md` — latest work focus and decisions
- `memory-bank/progress.md` — current status and known issues
- `memory-bank/systemPatterns.md` — architecture and design patterns
- `migration-plan/overview/` — PM4 migration context and open issues

## Minimal Code Changes to Re-enable Tiles (implemented)
- In `Program.cs::ProcessOne()`:
  - Derive directory + filename prefix from `firstTilePath`.
  - If NOT `--batch`: `pattern = Path.GetFileName(firstTilePath)` (loads ONLY that tile)
  - If `--batch`: `pattern = prefix + "_*.pm4"` (loads region by prefix)
  - `var gs = Pm4GlobalTileLoader.LoadRegion(dir, pattern);`
  - `var scene = Pm4GlobalTileLoader.ToStandardScene(gs);`
  - Leave the rest of the export flow unchanged.
