# Progress - WoWRollback.RollbackTool

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

### Pipeline Integration
- Unifying Alpha→LK pipeline under a single command (`alpha-to-lk`): rollback + area map + export
- Adding LK ADT patcher command (`lk-to-alpha`, v1): bury/holes/mcsh on LK ADTs
- Implementing minimal LK `AreaTable.dbc` ID reader to auto-fill area mappings from MPQs

## 🎯 Next Steps

### Phase 1: Unified Pipeline Command
1. Add `alpha-to-lk` that composes rollback (bury + MCRF-gated hole clear + optional MCSH), area map (JSON or LK MPQs), and LK export
2. Update `PrintHelp()` and logs with examples

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
- [ ] Clear terrain holes (MCNK modification)
- [ ] Disable baked shadows (MCSH modification)
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
| Terrain Hole Fixing | ✅ | ⏳ | **PENDING** |
| Overlay Pre-generation | ✅ | ⏳ | **PENDING** |
| Lightweight Viewer | ✅ | ⏳ | **PENDING** |
