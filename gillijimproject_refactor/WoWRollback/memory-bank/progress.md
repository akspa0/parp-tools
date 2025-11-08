# Progress - WoWRollback.RollbackTool (2025-11-08)

## Current Progress (concise)
- **Main outstanding issue: LK ADT positions → Alpha WDT writeout.**
- MPQ overlay precedence in place; CLI logs show counts/lists and summary.
- Plain patch support: `patch(.locale).MPQ` treated as numeric order 1.
- DBFilesClient precedence: locale patch MPQs searched before root patch MPQs, fixing `Map.dbc` selection.
- WDT tile presence fallback active when archive tiles=0; emits `tiles_missing.csv`.
- Tee logging available via `--log-dir` and `--log-file`.
- Alpha WDT monolithic pack builds; placements/liquids diagnostics emitted (tuning pending).

## TODOs (concise)
- Tests: `ArchiveLocator` order incl. plain patch; `MpqArchiveSource` DBC locale-first path.
- Logs: add plain-patch counts; optional verbose line for DBC source MPQ.
- Verify on maps (CataclysmCTF, development): overlay precedence and Map.dbc path.
- Placements: union MDNM/MONM; never gate placements; recompute MCRF; per‑tile MDDF/MODF counts.
- Liquids: instrument MH2O→MCLQ; write `mclq_summary.csv`; fix flags/heights; reduce `dont_render`.
- Textures: tileset resize option (256) with alpha preservation.

## Archive (historical)

Legacy details retained below; top sections are authoritative.

## Hot Update (2025-11-07) – Alpha WDT monolithic pack: liquids & placements

### What works
- StackOverflow eliminated (removed stackalloc); monolithic pack completes.
- MDNM/MONM names present; placement axis order corrected (X,Z,Y).
- `.m2`↔`.mdx` alias gating added; MCAL pack verbose logs in place.

### Known issues
- MDDF/MODF/MCRF effectively empty due to gating semantics → no objects render.
- MH2O→MCLQ conversion yields too few chunks; most tiles flagged `dont_render` → liquids invisible.
- Logs are not tee'd to a file → limited ability to compare runs.

### Next
- Build MDNM/MONM from union of referenced names (no culling). Do not gate placements; recompute MCRF.
- Emit `objects_written.csv` and kept assets CSV; keep `dropped_assets.csv`.
- Add per-tile MDDF/MODF counts to file-mode writer.
- Instrument MH2O→MCLQ and emit `mclq_summary.csv`; verify `offsLiquid` and MPHD liquid flags.
- Add `--log-file` and `--log-dir` flags; wire tee logger for persistent logs.

## Hot Update (2025-11-07) – MPQ overlay & WDT fallback

### What works
- MPQ overlay precedence implemented to mirror client: FS > root letter > locale letter > root numeric > locale numeric > base.
- CLI verbose now prints numeric and letter patch counts/lists (root vs locale) and a single overlay summary line.
- WDT-based tile presence fallback implemented when archive tiles=0; emits `tiles_missing.csv` with expected-but-missing ADTs.

### Next
- Validate logs on representative maps (e.g., CataclysmCTF, development) to confirm patch coverage.
- Proceed with tileset texture resizing path (256) gated by option; verify alpha preservation.

## ✅ Completed (2025-10-30)
- GUI: Fixed `MainWindow.cs` compile errors (`CS0136`, `CS0103`) in time range/global stats logic.
- GUI Recompile: Live STDOUT/STDERR streaming to Build Log; log CWD and full `dotnet` command; exit code; persist `<outRoot>/session.log`; show log path.
- GUI: Global heatmap stats integration for slider domain using `heatmap_stats.json` with per-map overrides.
- LkToAlpha: Implemented MCAL 8‑bit→4‑bit conversion (64×64 → 2048 bytes/layer), updated MCLY flags/offsets; emit Alpha MCAL raw (no chunk header).

## ✅ Completed (2025-10-29)
- CASC: Added `analyze-map-adts-casc` (CLI) using `CascArchiveSource` with FDID listfile support; listfile parsing accepts `;`, `,`, and tab delimiters; case-insensitive matching.
- Discovery: If `Map.dbc` is unavailable, read `Map.db2` directly via DBCD over CASC; if that fails, fall back to WDT scan using listfile enumeration.
- Product auto-detection: Extracted from `.build.info` (e.g., wow, wowt, wow_beta) when not explicitly provided.
- GUI Prepare routing: CASC → `analyze-map-adts-casc --all-maps`; Install → `analyze-map-adts-mpq --all-maps`; Loose → existing `prepare-layers`.
- GUI Load behavior: For CASC/Install, do not abort when no filesystem maps are found; proceed so Prepare can populate outputs.
- Listfile utilities: `snapshot-listfile` (JSON snapshots, full build aliases) and `diff-listfiles` (added/removed/changed FDIDs) implemented.
- Asset gating: `pack-monolithic-alpha-wdt` now supports `--target-listfile` and `--strict-target-assets`; writes `dropped_assets.csv`.
- Core services added: `ListfileIndex`, `ListfileCatalog`, `ListfileSnapshot`, `AssetGate` (under Core/Services/Assets).
 - Unified command: `alpha-to-lk` orchestrates rollback + AreaID mapping (CSV crosswalks) + LK ADT export.
 - CASC WDT handling: CASC datasets are not WDT‑gated; prefer LK client for `<map>.wdt` lookup; otherwise prompt via file picker in GUI.

## ✅ Completed (2025-10-27)
- GUI loading overlay added; wired around Load/Prepare.
- Load UX: prominent button; auto-switch to Build; Auto‑Prepare default ON.
- Prepare UX: auto-switch to Layers; removed success/info modals.
- Tiles: fallback to `<map>_tile_layers.csv` when `tile_layers.csv` empty; normalization + logs.
- Map discovery: version folder handled correctly; only real maps listed.

## ✅ Completed (2025-10-25)

### CLI-first pipeline hardened and documented

**Highlights:**
- Strict non-pivot AreaID mapping order enforced; 0.6.0 pivot opt-in via `--chain-via-060`.
- LK WDT is now emitted alongside LK ADTs and renamed to `<Map>.wdt`.
- End-to-end docs added: `docs/pipeline/alpha-to-lk-end-to-end.md`.
- Presets: schema docs and `presets/Westfall2001.json` sample added.
- Asset taxonomy and LK→Alpha status docs added.
- Main README refocused on `WoWRollback.Cli` (Orchestrator marked legacy).

**Verified Run (Azeroth 0.5.3, 75k, auto-crosswalks with loose DBCs):**
- ADTs exported (685/685), `<map>.wdt` written.
- AreaIds summary: present=175,360, mapped=152,736, patched=152,992, unmatched=22,624.

## ✅ Completed (2025-10-22)

### Core Rollback Functionality - WORKING AND TESTED!

**Milestone**: Successfully modified Alpha 0.5.3 WDT files and verified in-game compatibility!

#### Test Results
- ✅ **Kalimdor 0.5.3**: 951 ADT tiles, 126,297 placements, 125,662 buried
- ✅ **Azeroth 0.5.3**: Multiple successful tests
- ✅ **MD5 Checksum**: Auto-generation confirmed working
- ✅ **File Integrity**: Output WDTs valid and loadable

#### Implementation Details
1. **WDT Loading** - Load entire Alpha WDT into byte array
2. **ADT Parsing** - Parse each embedded ADT via `WdtAlpha.GetAdtOffsets In()`
3. **Chunk Access** - Use `AdtAlpha.GetMddf()` and `GetModf()` to access placement data
4. **Modification** - Modify Z coordinate at offset +12 in each entry
5. **Writeback** - Copy modified chunk data back to original byte array
6. **Output** - Write modified WDT + generate MD5 checksum
7. **Selective Hole Clearing** - Per-MCNK, clear `Holes` only if all referenced placements (via `MCRF`) were buried
8. **Shadow Removal** - Optional: zero Alpha MCSH payloads per MCNK offsets
9. **LK Export** - `--export-lk-adts` converts present tiles to LK ADTs and writes to `--lk-out` (or default path)
10. **Area Mapping Hook** - `--area-remap-json` supplies AlphaAreaId→LK AreaId mapping applied during export

#### Code Locations
- `WoWDataPlot/Program.cs` - Rollback command implementation (lines ~1980-2180)
- `AdtAlpha.cs` - Added accessor methods for chunks and file offsets

### New AdtAlpha Methods
```csharp
public Mddf GetMddf() => _mddf;
public Modf GetModf() => _modf;
public int GetMddfDataOffset() { ... }  // Calculate file offset
public int GetModfDataOffset() { ... }  // Calculate file offset
private readonly int _adtFileOffset;     // Store offset passed to constructor
```

### Chunk Format Discoveries
```
MDDF Entry (36 bytes):
  +0x00: nameId (int32)
  +0x04: uniqueId (int32) ← FILTER CRITERION
  +0x08: position X (float)
  +0x0C: position Z (float) ← MODIFY TO BURY
  +0x10: position Y (float)
  +0x14-0x23: rotation, scale, flags

MODF Entry (64 bytes):
  +0x00: nameId (int32)
  +0x04: uniqueId (int32) ← FILTER CRITERION
  +0x08: position X (float)
  +0x0C: position Z (float) ← MODIFY TO BURY
  +0x10: position Y (float)
  +0x14-0x3F: rotation, bbox, flags, etc
```

## ⏳ In Progress


## 🎯 Next Steps

### Phase 1: CLI Polish for Unified Pipeline
1. Ensure `alpha-to-lk` help text shows end-to-end examples and flags clearly
2. Verify logs reflect crosswalk usage, strict mapping guards, and `<Map>.wdt` emission

### Phase 2: AreaTable Auto-Mapper
1. Implement `AreaTableDbcReader` (IDs only) opened via `PrioritizedArchiveSource`/`MpqArchiveSource`
2. Build AlphaAreaId→LKAreaId: pass-through where present; else `--default-unmapped` (default 0)

### Phase 3: LK Patcher Command
1. Add `lk-to-alpha` (v1) to patch LK ADTs (bury/holes/mcsh) and write to `--out`
2. Validate counts and logs on Kalimdor

**Output Structure**:
```
overlays/
├── azeroth/
│   ├── uid_0-5000.png
│   ├── uid_0-10000.png
│   ├── uid_0-50000.png
│   └── overlay-index.json
└── kalimdor/
    ├── uid_0-5000.png
    └── ...
```

### Phase 5: Lightweight Viewer
**Goal**: HTML+JS slider UI for picking rollback threshold

**Features**:
- Slider snaps to pre-generated overlay thresholds
- Displays current UniqueID range
- Shows placement count (kept vs buried)
- Visual overlay updates in real-time
- Copy-to-clipboard rollback command

## 📊 Current Status

**Progress**: Core functionality complete (~60%), UX features pending (~40%)

```
✅ Core Rollback:                 100% (TESTED!)
✅ MD5 Generation:                100%
✅ MCNK Hole Management:          100% (MCRF-gated)
✅ MCSH Shadow Disabling:         100%
✅ LK ADT Export Path:            100%
⏳ AreaTable Auto-Mapper:          0%
⏳ Overlay Generation:             0%
⏳ Lightweight Viewer:             0%
```

## 🐛 Known Issues

- Without `--area-remap-json` (or future auto-mapper), LK ADTs may not display correct zone names

## ✨ Proven Capabilities

- [x] Load Alpha 0.5.3 WDT files (largest test: 951 tiles)
- [x] Parse embedded ADT data via offsets
- [x] Extract MDDF/MODF placement chunks
- [x] Modify placement Z coordinates
- [x] Write modified WDT back to disk
- [x] Generate MD5 checksums
- [x] Clear terrain holes (MCNK modification)
- [x] Disable baked shadows (MCSH modification)
- [ ] Pre-generate overlay images
- [ ] Lightweight HTML viewer

## 📁 Files Modified This Session

### New Files
None (modifications only)

### Modified Files
- `WoWRollback/WoWDataPlot/Program.cs`
  - Added `rollback` command (lines ~1980-2180)
  - Implemented WDT loading, parsing, modification, output
  
- `src/gillijimproject-csharp/WowFiles/Alpha/AdtAlpha.cs`
  - Added `GetMddf()` accessor
  - Added `GetModf()` accessor
  - Added `GetMddfDataOffset()` method
  - Added `GetModfDataOffset()` method
  - Added `_adtFileOffset` field

## 🎯 Success Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Alpha 0.5.3 Support | ✅ | ✅ | **ACHIEVED** |
| Large Map Support (900+ tiles) | ✅ | ✅ | **ACHIEVED** |
| MD5 Checksum | ✅ | ✅ | **ACHIEVED** |
| Terrain Hole Fixing | ✅ | ✅ | **ACHIEVED** |
| Overlay Pre-generation | ✅ | ⏳ | **PENDING** |
| Lightweight Viewer | ✅ | ⏳ | **PENDING** |
