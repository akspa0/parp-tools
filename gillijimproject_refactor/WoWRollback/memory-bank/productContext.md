# Product Context - WoWRollback.RollbackTool

## Why This Exists

WoW map files contain object placements (M2 models and WMOs) tagged with **UniqueIDs**. These IDs increase monotonically as content is added during development. By analyzing UniqueID ranges and selectively "burying" objects above a threshold, we can create historical snapshots of the game world.

**Problem Solved**: Users want to see what WoW maps looked like at different stages of development without manually removing thousands of objects.

## How It Works

### Unified Alpha→LK Pipeline

#### Phase 1: Rollback (Patched Alpha WDT)
```
Alpha WDT + Threshold + Options → Modify placements → Fix terrain → Zero shadows → Write output + MD5
```
**Output**: Patched Alpha WDT with MDDF/MODF buried by UID, conservative hole clearing (per-MCNK all-buried rule), optional MCSH disabled, MD5 file.

#### Phase 2: AreaTable Mapping (CSV Crosswalks)
```
Use mapping to set `MCNK.AreaId` for LK ADTs.
Primary source: CSV crosswalks loaded via `--crosswalk-dir` or `--crosswalk-file`.
Fallbacks: `--area-remap-json` explicit mappings; write 0 when unmapped. `Map.dbc` is used only to resolve target map guards (no heuristics).
```
**Output**: Dictionary AlphaAreaId→LK AreaId, with zeros for unmapped

#### Phase 3: Export LK ADTs (Patched)
```
Patched Alpha WDT + AreaMap → Convert to LK ADTs (indices remapped, AreaId applied)
```
**Output**: LK ADTs written under lk_out/World/Maps/<map>/<map>_x_y.adt

### Symmetric LK→LK Patcher
```
LK ADTs dir + Threshold + Options → Bury placements → Selective hole clearing → Zero MCSH → Write to out
```
**Output**: Patched LK ADTs. Reverse LK→Alpha WDT conversion is considered later.

## User Experience Flow (CLI-first → UI-runner)

### Step 1: Alpha→LK (with crosswalks, non-pivot default)
```powershell
WoWRollback alpha-to-lk --input World/Maps/Kalimdor/Kalimdor.wdt --max-uniqueid 125000 \
  --fix-holes --disable-mcsh --out rollback_kl053 --export-lk-adts \
  --lk-out rollback_kl053/lk_adts/World/Maps/Kalimdor \
  --crosswalk-dir D:/crosswalks --lk-dbc-dir D:/lk_dbc
```
**Result**: Patched Alpha WDT + Patched LK ADTs with AreaIds and `<Map>.wdt` emitted.

```powershell
# Alternative: run via dotnet
dotnet run --project WoWRollback/WoWRollback.Cli -- \
  alpha-to-lk --input World/Maps/Kalimdor/Kalimdor.wdt --max-uniqueid 125000 \
  --fix-holes --disable-mcsh --out rollback_kl053 --export-lk-adts \
  --lk-out rollback_kl053/lk_adts/World/Maps/Kalimdor \
  --crosswalk-dir D:/crosswalks --lk-dbc-dir D:/lk_dbc
```
*Invariant*: LK export always emits `<Map>.wdt` alongside the ADTs in the LK output folder.

### Step 2: (Optional) LK patcher
```powershell
WoWRollback lk-to-alpha --lk-adts-dir World/Maps/Kalimdor --max-uniqueid 125000 \
  --fix-holes --disable-mcsh --bury-depth -5000 --out patched_lk_kl053
```
**Result**: Patched LK ADTs written to output dir.

### 2025-10-29: Versioned listfiles and asset gating
- Two listfiles used during conversions:
  - 3.3.5 target allowlist (path-only): authoritative set for outputs
  - Modern community listfile (FDID-aware): broad discovery and rename detection
- JSON snapshots: `snapshot-listfile --client-path <dir> --alias <major.minor.patch.build> --out <json>`
  - Entries: `{ path, fdid? }`, metadata includes alias (full build string)
- Asset Gate: recompile/pack pipelines reference only assets present in the 3.3.5 listfile
  - Names not present are dropped and reported in `dropped_assets.csv`
- New CLI utilities:
  - `snapshot-listfile`, `diff-listfiles`, `pack-monolithic-alpha-wdt` (supports `--target-listfile`, `--strict-target-assets`)
- Alias policy: full build strings derived from `.build.info`, DBD definitions, or path heuristics

### Analysis + Viewer
- Analyze LK ADTs to produce placements/terrain CSVs and viewer assets, then serve viewer.

### Presets
- Management lives in Settings (create/rename/delete; schema unchanged).
- Load Preset control lives on the Load page (applies before Prepare).
- BYOD: presets reference paths/aliases; no copyrighted assets included.

### UX Principles
- No modal popups; use overlay + inline banners for status/errors.
- Clear navigation: Load → Build (Prepare) → Layers; auto-switch on milestone.
- Scroll-safe layouts; window opens maximized by default.
- Energy-efficient preflight: reuse cache outputs and crosswalk CSVs when present; skip recomputation.

### Technical Approach

### Object Burial Strategy
- Modify Z coordinate in MDDF/MODF chunks
- Set Z = -5000.0 (deep underground, never rendered)
- Keeps object data intact, just moves it out of sight

### Terrain Hole Fix
- MCNK chunks have a `Holes` field (16 bits = 4x4 grid of 2x2 areas)
- Clear holes only if all referenced placements in chunk were buried (per-MCRF gating)
- Prevents revealing terrain where kept objects remain

### Shadow Removal (Optional)
- MCSH chunks contain baked shadow maps
- Can look weird when objects are removed
- Option to zero out MCSH data

## Why This Architecture?

### Pre-generated Overlays (Not On-the-Fly)
- **Performance**: No processing during viewing
- **Simplicity**: Pure HTML+JS, works anywhere
- **Reliability**: Generate once, view forever

### Separate Analysis and Modification
- **Safety**: Never modify original files during analysis
- **Flexibility**: Try different thresholds without re-scanning
- **Transparency**: See what will happen before doing it

### Reuse Existing Infrastructure (Library-first)
- **AlphaWDTAnalysisTool**: Already has complete ADT parsing
- **gillijimproject-csharp**: Proven WoW file format library
- **Don't Reinvent**: Build on what works

## Known Limitations
- Only works on extracted WDT files (not in MPQs... yet)
- Doesn't modify _obj0.adt or _obj1.adt files (terrain objects)
- MCNK spatial calculation assumes flat terrain (good enough for most cases)
- Pre-generation requires disk space (1-2 MB per map)

## Future Enhancements

- MPQ reading/writing support
- Batch processing (all maps at once)
- Diff mode (compare two rollback points)
- 3D preview (Three.js viewer with terrain mesh)
