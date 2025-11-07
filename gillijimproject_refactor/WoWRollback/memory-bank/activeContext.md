# Active Context (2025-10-27)

## Current Focus
- CLI-first pipeline; GUI acts as a runner for Load → Prepare → Layers (no modal popups; overlay + inline logs).
- Energy-efficient preflight: skip work when cache outputs already exist (LK ADTs, tile_layers.csv, layers.json, areas.csv, DBCTool crosswalks).
- Presets: management in Settings; add “Load Preset” on the Load page.
- Adopt CsvHelper in GUI for robust CSV parsing (tolerant headers, 7/8-col variants).
- Gate Area Groups unless areas.csv is present and non-empty (show inline note otherwise).
- BYOD: never bundle copyrighted data; resolve from user-provided locations only.
- Global heatmap (build scope) with `heatmap_stats.json` persisted at build root.
- Layers scope control: Tile | Selection | Map; restore per‑tile lists deterministically.
- FDID pipeline: resolver + CSV enrichment (`fdid`, `asset_path`) + unresolved diagnostics.
- MCCV analyzer and overlay: detect “hidden by holes” and export per‑tile PNGs.
- CASC recompile routing: prefer LK client for WDT lookup; fall back to manual file picker.

## Decisions
- CLI-first with GUI runner and overlay; auto-tab navigation (Load→Build, Prepare→Layers); remove success/info modals.
- Keep CSV cache schema stable; normalize `tile_layers.csv` naming; GUI may fall back to `<map>_tile_layers.csv` and normalize.
- Feature gating: Area Groups UI only when areas.csv has data; do not synthesize from DBC alone.
- Energy efficiency: preflight/skip-if-exists for LK ADTs, crosswalks, tile layers, and layers.json.
- BYOD: tooling must not include copyrighted game data anywhere in repo/binaries.

## Hot Update (2025-11-07) – Alpha WDT monolithic pack: liquids & placements

### Current state
- StackOverflow fixed by removing stackalloc; pack completes.
- MDNM/MONM populated, but MDDF/MODF and MCRF effectively empty → no objects render.
- MCLQ count is low (~100) and most tiles are marked dont_render → liquids not visible in 0.5.3.
- Logging is not persisted; need tee-to-file to capture full runs.

### Root causes
- Using asset gating to build MDNM/MONM dropped referenced names; placements were skipped when global name index was missing.
- MH2O→MCLQ conversion writes payloads but tile flags/min/max/offsLiquid composition leads to empty/don't_render tiles.

### Decisions
- Build MDNM/MONM from the union of all referenced names (no culling). Use AssetGate only for reporting (kept/dropped), not for building the name tables.
- Never gate placements. Always resolve local indices to global MDNM/MONM indices and write all MDDF/MODF entries.
- Keep placement axis order X,Z,Y. Normalize names: convert `/`→`\` and `.m2`→`.mdx` for Alpha.
- Prefer MH2O-derived MCLQ when present. Set MCNK liquid flags and offsLiquid properly; compose per-tile flags (fishable/fatigue); only set dont_render when a subtile is truly absent.
- Add tee logging with `--log-file` and `--log-dir`; emit diagnostics CSVs.

### Next steps
- Fix placements: rebuild MDNM/MONM from all referenced names and map placements accordingly in both writer paths; emit per-chunk MCRF.
- Emit `objects_written.csv` with per-tile MDDF/MODF counts and sample rows; keep `dropped_assets.csv` and add kept assets report.
- Add per-tile verbose object counts in file-path `WriteMonolithic`.
- Instrument MH2O→MCLQ and write `mclq_summary.csv`; verify counts/flags/heights; export debug images if needed.
- Add CLI flags `--log-file`, `--log-dir`; wire tee logger.

## Hot Update (2025-10-30) – Build cache, discovery, recompile, layers roadmap

### What changed
- Build detection now used in GUI Data Preview/Load; cache path becomes `<cacheRoot>/<build>` instead of `unknown` when `.build.info` is present.
- Map discovery hardened:
  - Normalize flat outputs at cache root and version folders (`<map>_tile_layers.csv` → `<map>/tile_layers.csv`; same for `*_layers.json`).
  - Flatten accidental `<map>/<map>/tile_layers.csv` nests.
  - Do not WDT‑gate maps for CASC datasets (WDTs aren’t on disk); keep WDT gate for loose/install only.
- Recompile (GUI): for CASC datasets prefer LK Client folder for locating `<map>.wdt`; otherwise offer a file picker. Logs include attempted roots.
 - Recompile (GUI) logging: stream STDOUT/STDERR live to Build Log; log CWD and exact `dotnet` command; log exit code; persist full session to `<outRoot>/session.log`; show log path on completion/failure.
 - LK→Alpha textures: implemented MCAL 8‑bit→4‑bit packing (64×64 → 2048 bytes per layer), updated MCLY flags/offsets; emit MCAL as raw (no chunk header) per Alpha v18.

### In flight / next
- Global heatmap toggle (Local vs Global‑per‑build; epoch option next) with `heatmap_stats.json` persisted at build root.
- Layers scope control: Tile | Selection | Map; restore per‑tile lists deterministically.
- FDID pipeline: resolver (community listfile + snapshots), CSV enrichment (`fdid`, canonical path), unresolved diagnostics.
- MCCV analyzer and overlay: detect MCCV per MCNK, flag "hidden by holes", export PNGs; UI overlay toggle.

---
## Hot Update (2025-10-29) – CASC/DB2 + GUI + Asset Gate

### What works now
- CLI: `analyze-map-adts-casc` reads ADTs from CASC via `CascArchiveSource` with FDID listfile support.
- Discovery: If Map.dbc can’t be extracted, we read Map.db2 directly using DBCD over CASC; else fallback to WDT scan via listfile.
- GUI: Prepare routes by source
  - CASC → `analyze-map-adts-casc --all-maps`
  - Install → `analyze-map-adts-mpq --all-maps`
  - Loose → existing prepare‑layers
- Product auto‑detect from `.build.info`; no manual `--product` needed.

### Utilities for conversions
- New CLI:
  - `snapshot-listfile --client-path <dir> --alias <major.minor.patch.build> --out <json>` → JSON listfile snapshots (MPQ listfile + loose files).
  - `diff-listfiles --a <listA> --b <listB> --out <dir>` → added/removed/changed FDIDs.
  - `pack-monolithic-alpha-wdt --lk-wdt <file> --out <wdt> [--target-listfile <335>] [--strict-target-assets]` → MDNM/MONM gated by 3.3.5 listfile.
  - Aliases use full build strings.

### Known constraints
- CASC requires a listfile; discovery relies on FDID→path from the community listfile.
- Map list “Load” does not show CASC maps (they appear after Prepare).

### Next steps (in progress)
- Implement ListfileIndex/ListfileCatalog and AssetGate.
- Strict target assets: only reference assets present in a trusted 3.3.5 listfile during recompile/export; emit dropped_assets.csv.
- GUI: add Target Listfile picker and toggle for strict mode; pass to CLI.

## Completed (2025-10-27)
- Added loading overlay; wired ShowLoading/HideLoading around Load and Prepare.
- Load UX: prominent button, auto-switch to Build; Auto‑Prepare default ON.
- Prepare UX: auto-switch to Layers on success (no modal).
- Tiles: GUI fallback to `<map>_tile_layers.csv` when `tile_layers.csv` empty; copies to normalize; added load counts in logs.
- Map discovery: treat version folder (e.g., 0.5.3) as container; list actual map subfolders (no more version-as-map).

## Implementation Plan
- New WoWRollback.Pipeline service consumed by GUI/CLI.
- IAssetSource abstraction:
  - FileSystemSource (loose)
  - CascSource (WoWFormatLib/CascLib + listfile)
  - MpqSource (phase 2)
- Areas:
  - Prefer DBCTool crosswalks (compare/v2) to enrich per‑tile AreaIDs → areas.csv; fallback to LK ADT read; optional DBCD enrichment when available.
- Build detection: infer from paths or .build.info; allow override.

## Next Steps
- Integrate CsvHelper in GUI; refactor CSV reads (tile_layers.csv, areas.csv) to POCOs.
- Implement preflight cache checks and CLI skip-if-exists flags.
- Restructure tabs: Load (with Load Preset), Build, Layers, Settings (Presets management).
- Gate Area Groups UI; add inline note when areas.csv absent.
- WoWDataPlot: generate non-empty areas.csv via DBCTool crosswalks; keep LK ADT/DBC fallback.

## Risks
- CASC listfile coverage may vary; provide clear error messages and fallbacks.
- Alpha-only per‑tile AreaIDs may not be derivable; keep LK path canonical.

## History (2025-10-25) – Viewer Tools & Pipeline

## Recent Focus (2025-10-25)
**Run the Alpha→LK pipeline directly from the viewer UI**

We now have a CLI-first, strict non-pivot AreaID mapping pipeline that emits LK ADTs and a fresh `<Map>.wdt`. Next, we will expose this pipeline through the viewer with a Tools panel and a small backend API.

## What We Just Accomplished (2025-10-25)

### ✅ PROVEN: Core Rollback Works on Alpha 0.5.3!

**Successful Tests:**
```
Kalimdor 0.5.3 (951 ADT tiles)
  - Total Placements: 126,297
  - Kept (UID ≤ 78,000): 635
  - Buried (UID > 78,000): 125,662
  - Status: SUCCESS!

Azeroth 0.5.3
  - Multiple successful rollbacks
  - MD5 checksum generation confirmed
  - Status: SUCCESS!
```

**What Works:**
1. ✅ Load Alpha WDT files
2. ✅ Parse all ADT tiles (MHDR offsets from WDT MAIN chunk)
3. ✅ Extract MDDF/MODF chunks via `AdtAlpha`
4. ✅ Modify placement Z coordinates (bury at -5000.0)
5. ✅ Write modified data back to `wdtBytes`
6. ✅ Output modified WDT file + MD5 checksum
7. ✅ Selective hole clearing per MCNK using `MCRF` references (only if all referenced placements are buried)
8. ✅ Optional MCSH zeroing
9. ✅ LK export path via `--export-lk-adts` with `AdtAlpha.ToAdtLk(..., areaRemap)` and `AdtLk.ToFile(<dir>)`
10. ✅ Area mapping via CSV crosswalks with strict non-pivot decision order; pivot gated by `--chain-via-060`
11. ✅ LK WDT emitted and renamed to `<Map>.wdt` in LK output folder

### ✅ Technical Breakthroughs

**AdtAlpha Integration**
- AdtAlpha already parses MDDF and MODF chunks!
- Added `GetMddf()` and `GetModf()` accessors
- Added `GetMddfDataOffset()` and `GetModfDataOffset()` to locate chunks in file
- Stored `_adtFileOffset` in constructor for offset calculations

**Placement Format**
```
MDDF (M2 models) - 36 bytes per entry:
  +0x00: nameId (4 bytes)
  +0x04: uniqueId (4 bytes) ← FILTER BY THIS
  +0x08: position X (4 bytes)
  +0x0C: position Z (4 bytes) ← MODIFY THIS TO BURY
  +0x10: position Y (4 bytes)
  
MODF (WMO buildings) - 64 bytes per entry:
  +0x00: nameId (4 bytes)
  +0x04: uniqueId (4 bytes) ← FILTER BY THIS
  +0x08: position X (4 bytes)
  +0x0C: position Z (4 bytes) ← MODIFY THIS TO BURY
  +0x10: position Y (4 bytes)
  ... (rest of entry)
```

**File Structure**
```
WDT File (Alpha 0.5.3):
  MVER chunk
  MPHD chunk (flags)
  MAIN chunk (64x64 grid, offsets to ADT data)
  ... ADT data embedded inline ...
    ADT #0 @ offset XXXX
      MHDR (offsets to MDDF/MODF/etc)
      MDDF chunk
      MODF chunk
      MCNK chunks (256 per ADT)
    ADT #1 @ offset YYYY
    ...
```

## Architecture Decision: CLI-first with UI runner

**Problem**: WoWDataPlot was being built as a hybrid analysis+modification+visualization tool, which violates separation of concerns.

**Solution**: Keep the CLI as the primary entrypoint, expose its logic as services callable by the viewer backend, and run jobs with live logs.

```
AlphaWDTAnalysisTool/     (EXISTS - Analysis Phase)
  └─> Scans WDT/ADTs
  └─> Outputs CSVs with UniqueID data
  └─> Already has complete infrastructure!

WoWRollback.RollbackTool/  (NEW - Modification Phase)
  └─> Reads analysis CSVs
  └─> Modifies WDT files in-place
  └─> Manages terrain holes and shadows
  └─> Generates MD5 checksums

WoWDataPlot/               (REFOCUS - Visualization Phase)
  └─> Reads CSVs from analysis
  └─> Pre-generates overlay images
  └─> Lightweight HTML viewer
  └─> No modification, pure viz
```

## Current Implementation Status
- ✅ `WoWRollback.Cli` provides Alpha→LK, LK analysis, and viewer serving
- ✅ MCNK terrain hole management and MCSH zeroing in place
- ✅ AreaID mapping patched using crosswalks with strict guards
- ✅ LK WDT emitted alongside ADTs

## Next Steps

### Phase 1 (MVP): UI Tools panel + Alpha→LK job
1. Add Tools panel in viewer to configure globals:
   - UniqueID max, holes scope, preserve WMO holes, disable MCSH
   - Crosswalk: auto, strict (default), optional 0.6.0 pivot
   - Inputs: WDT path, DBC dirs/DBD dir, optional preset JSON
2. Backend API (`ViewerModule`):
   - POST `/api/build/alpha-to-lk` → returns job id
   - GET `/api/jobs/{id}/events` (SSE) → live logs and progress
3. Extract CLI logic into services (no shell): `AlphaToLkService`
4. On completion, offer “Open in viewer” of LK analysis results

### Phase 2: Per-tile overrides
- Allow per-tile `maxUniqueId` overrides; accept `uniqueId.perTile[]` in payload

### Phase 3: Asset taxonomy + filters
- Add `asset_type` to CSVs; add viewer filters and preset integration

### Phase 4: Analyze + Serve from UI
- POST `/api/analyze/lk-adts` and `/api/viewer/serve`; wire to existing modules

## Constraints & Guards
- Non-pivot mapping order with strict map guard; discard cross-map candidates
- Prefer loose DBCs; `--src-dbc-dir`/`--lk-dbc-dir` override client-path
- Always emit `<Map>.wdt` with LK ADTs

## Files Modified This Session (2025-10-21)
- `WoWDataPlot/Program.cs` - Added complete rollback command
- `AdtAlpha.cs` - Added `GetMddf()`, `GetModf()`, `Get*DataOffset()` methods
- `AdtAlpha.cs` - Added `_adtFileOffset` field to track position in file

## What Works Now
✅ Load Alpha 0.5.3 WDT files  
✅ Parse all embedded ADT tiles via offsets  
✅ Extract MDDF/MODF placements  
✅ Modify Z coordinates to bury objects  
✅ Write modified WDT back to disk  
✅ Generate MD5 checksums  
✅ TESTED on Kalimdor (951 tiles, 126K placements!)
