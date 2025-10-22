# Progress - WoWRollback.RollbackTool

## ✅ Completed (2025-10-21)

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

### Architecture Refactoring
- Moving rollback code from WoWDataPlot to new WoWRollback.RollbackTool project
- Separating concerns: Analysis vs Modification vs Visualization

## 🎯 Next Steps

### Phase 1: Project Structure (Next Session)
1. Create `WoWRollback.RollbackTool` CLI project
2. Extract rollback logic from `WoWDataPlot/Program.cs`
3. Commands: `analyze`, `generate-overlays`, `rollback`
4. Reference `gillijimproject-csharp` library

### Phase 2: MCNK Terrain Hole Management
**Goal**: Clear terrain holes where buried WMOs used to be

**Technical Approach**:
```
For each buried WMO placement:
  1. Get world coordinates (X, Y, Z)
  2. Calculate owning ADT tile
  3. Calculate MCNK index within tile (16x16 grid)
  4. Locate MCNK header in file (via MHDR offsets)
  5. Clear Holes field at offset +0x40 (set to 0x0000)
  6. Write modified header back
```

**Spatial Calculations**:
- ADT tile size: 533.33 yards square
- MCNK chunk size: 33.33 yards square  
- MCNK grid: 16x16 per ADT (256 chunks)
- Formula: `mcnkIndex = (chunkY * 16) + chunkX`

**MCNK Header Structure**:
- Offset +0x00: Flags (4 bytes)
- Offset +0x40: **Holes** (4 bytes) ← MODIFY THIS
- Holes field: 16 bits representing 4x4 grid of 2x2 hole areas

### Phase 3: MCSH Shadow Disabling (Optional)
**Goal**: Remove baked shadows that might look weird after object removal

**Approach**:
```
For each ADT with buried objects:
  1. Find all MCSH chunks (via MHDR offsets → MCNK headers → MCSH offsets)
  2. Zero out MCSH chunk data
  3. Update chunk size if needed
  4. Write back
```

### Phase 4: Overlay Generation
**Goal**: Pre-generate minimap images showing rollback thresholds

**Approach**:
```
For each significant UniqueID threshold (percentiles or every 1000):
  1. Read minimap BLP tiles
  2. Overlay placement markers (green=kept, red=buried)
  3. Save as PNG
  4. Generate manifest JSON
```

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
✅ Core Rollback:          100% (TESTED!)
✅ MD5 Generation:          100%
⏳ MCNK Hole Management:      0%
⏳ MCSH Shadow Disabling:     0%
⏳ Overlay Generation:        0%
⏳ Lightweight Viewer:        0%
⏳ Project Refactoring:       0%
```

## 🐛 Known Issues

None! Core functionality works flawlessly on Alpha 0.5.3 data.

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
