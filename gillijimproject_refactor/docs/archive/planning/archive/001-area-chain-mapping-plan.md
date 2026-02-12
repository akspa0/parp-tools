# Plan: AreaID Chain Mapping via 0.6.0 and CSV-only Patch Integration

Date: 2025-09-17
Owner: AlphaWDTAnalysisTool / DBCTool.V2

## Why
- Subzones (e.g., Goldshire) collapse to parent zones (e.g., Elwynn Forest) when the mapping doesn’t carry authoritative parent relationships.
- Current single-hop 0.5.x → 3.3.5 matching lacks a reliable parent bridge, causing inconsistent subzone results.
- CSV-only patching (`--patch-only`) must be able to rely on precomputed mappings that already embed correct parent/zone structure.

## Goals
- Produce a chain mapping: 0.5.x → 0.6.0 → 3.3.5 that:
  - Preserves and enforces parent-child relationships.
  - Is map-locked (no cross-map results).
  - Distinguishes zone-level (ParentID == ID) vs. subzone-level entries.
- Emit chain crosswalk CSVs with columns sufficient for AlphaWDTAnalysisTool to patch per-MCNK precisely:
  - Includes: src_(mapId,mapName,areaNumber,parentNumber,name), mid060_*, tgt_(mapId_xwalk,mapName_xwalk,areaID,parentID,name), method, violations.
- Update AlphaWDTAnalysisTool to consume these CSVs in `--patch-only` mode and write:
  - exact subzone hit → subzone `tgt_areaID`
  - else zone fallback (optional) → zone `tgt_areaID`
  - else 0
- Provide a fast, stitched per-map visualization (single HTML + legend, names from CSV only).

## Non-Goals (for this iteration)
- Auto-fixing ambiguous names in DBCTool beyond enforced constraints.
- Cross-continent or cross-map mappings (explicitly disallowed).
- Runtime DBCD lookups while in `--patch-only` mode.

## Design Overview

### A) DBCTool.V2 – Chain Mapping
- Inputs:
  - 0.5.x AreaTable (src)
  - 0.6.0 AreaTable (mid)
  - 3.3.5 AreaTable (tgt)
- Process:
  1) Map 0.5.x → 0.6.0 with strict rules:
     - Zone match: ParentID == ID (top-level only); require `mapIdX` match when known.
     - Subzone match: chosen zone is fixed; candidate subzones must be children of that zone and on the same map.
  2) Map 0.6.0 → 3.3.5 with the same constraints.
  3) Compose the chain to produce final target: `tgt_areaID`, `tgt_parentID`, `tgt_name`, `tgt_mapId_xwalk`.
- Output CSV schema (global and per-map variants):
  - `src_mapId,src_mapName,src_areaNumber,src_parentNumber,src_name`
  - `mid060_mapId,mid060_mapName,mid060_areaID,mid060_parentID,mid060_name`
  - `tgt_mapId_xwalk,tgt_mapName_xwalk,tgt_areaID,tgt_parentID,tgt_name`
  - `method,violations`
- CLI additions:
  - `--chain-via-060`
  - `--mid-alias 0.6.0`
  - `--mid-dir <path-to-0.6.0-DBFilesClient>`
- Output file names (examples):
  - `Area_patch_crosswalk_via060_0.5.3_to_335.csv`
  - `Area_patch_crosswalk_via060_map0_0.5.3_to_335.csv`

### B) AlphaWDTAnalysisTool – CSV-only Patching
- `DbcPatchMapping`:
  - Parse chain CSVs and expose lookup by `(src_mapId, src_areaNumber)` → `(tgt_areaID, tgt_parentID, tgt_name)`.
- `--patch-only` write path:
  - exact subzone key: `zone<<16 | sub` → write `tgt_areaID`
  - optional zone fallback: `zone<<16 | 0` → write `tgt_areaID`
  - else 0
  - Add `--no-zone-fallback` to force exact-only behavior (missing subzones write 0).
- Visualization:
  - Single stitched HTML per map at `viz/maps/<Map>/index.html` (inline SVG + single legend from CSV names).
  - Suppress per-tile legends when stitched HTML is enabled.

## Enforcement Rules (hard constraints)
- Map-locked: suggested target’s map must equal `tgt_mapId_xwalk` (no cross-map).
- Zones: target zone must have `ParentID == ID`.
- Subzones: target subzone must be a child of the chosen zone and share the same map.
- If unsure or violation: mark as `violations` in DBCTool.V2 output and do not auto-map; AlphaWDTAnalysisTool treats as unmatched.

## Performance Plan
- AlphaWDTAnalysisTool:
  - Use RandomAccess streams for patching.
  - Pre-read MCIN offsets (avoid 256 seeks per ADT).
  - Skip writes if AreaID is unchanged.
  - Parallelize tiles when fixups/logging are disabled (`--mdp N`).

## Validation
- Regenerate chain CSVs for Azeroth (map 0) using 0.5.3 and 0.6.0 sources.
- Patch with `--patch-only` against the chain CSVs:
  - Verify Elwynn/Goldshire, Westfall/Sentinel Hill, and Duskwood/Darkshire subzones are distinct.
- Check stitched HTML at `viz/maps/Azeroth/index.html`.
- Inspect per-map mapping report (to be added) for:
  - exact hit vs zone fallback vs none
  - `tgt_areaID`, `tgt_parentID`, `tgt_name`
  - any anomalies

## CLI Examples
- DBCTool.V2 (generate chain CSVs):
```
dbctoolv2 --compare-areas \
  --src-alias 0.5.3 --src-dir ..\..\test_data\0.5.3\tree\DBFilesClient \
  --lk-dir ..\..\test_data\3.3.5\tree\DBFilesClient \
  --chain-via-060 --mid-alias 0.6.0 --mid-dir ..\..\test_data\0.6.0\tree\DBFilesClient \
  --out ..\..\DBCTool.V2\dbctool_outputs\session_chain\v2
```
- AlphaWDTAnalysisTool (CSV-only patching + stitched viz):
```
dotnet run --input ..\..\test_data\0.5.3\tree\World\Maps\Azeroth\Azeroth.wdt \
  --out 053-chain \
  --listfile ..\..\test_data\community-listfile-withcapitals.csv \
  --export-adt --export-dir 053-chain \
  --dbctool-patch-dir ..\..\DBCTool.V2\dbctool_outputs\session_chain\v2 \
  --patch-only --no-fixups --no-fallbacks --mdp 8 --viz-html
```
- Exact-only subzones (no zone fallback):
```
... --patch-only --no-zone-fallback
```

## Tasks
- DBCTool.V2
  - Add chain flags and load 0.6.0 AreaTable.
  - Implement 0.5.x→0.6.0 and 0.6.0→3.3.5 mapping with strict zone/sub rules.
  - Compose final crosswalk; emit global and per-map CSVs; log violations.
- AlphaWDTAnalysisTool
  - Extend `DbcPatchMapping` to parse chain CSVs (read `tgt_parentID`/`tgt_name`).
  - Add `--no-zone-fallback` and per-map mapping report CSV.
  - Ensure stitched viz uses CSV names only and suppress per-tile legends when enabled.

## Acceptance Criteria
- Goldshire renders as a distinct subzone (not Elwynn) in stitched HTML and in-game checks.
- Westfall’s subzones (e.g., Sentinel Hill) map to proper LK IDs.
- No cross-map or wrong-parent violations for sampled zones.
- `--patch-only` run produces no 0s where chain CSVs provide coverage.
- Patching throughput improved with `--mdp` and skip-unchanged writes.

## Risks & Mitigations
- Ambiguous names still possible: captured in `violations`, not auto-mapped.
- Partial 0.6.0 coverage: fall back to zone row or unmatched (0) depending on `--no-zone-fallback`.
- Performance regressions: keep parallelism opt-in via `--mdp`, maintain skip-unchanged.
