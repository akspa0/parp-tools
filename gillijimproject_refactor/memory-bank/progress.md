# Progress

## ✅ Working

### Input Parsers (Standardized)
- **Alpha WDT/ADT**: Monolithic format, MCLQ liquids, reversed FourCC handling
- **LK 3.3.5 ADT**: Split format (root + _obj0 + _tex0), MH2O liquids
- **WMO v14/v17**: Both directions implemented
- **M2/MDX**: Framework ready (needs testing)
- **BLP**: BlpResizer complete — 7956 tilesets processed from WoW 12.x

### Standalone Tools
- **BlpResizer**: ✅ Production-ready, CASC extraction works
- **AlphaWdtInspector**: ✅ Diagnostics CLI functional
- **DBCTool.V2**: ✅ Crosswalk CSV generation works

### Data Generation
- **WDL→ADT**: ✅ Generates terrain from WDL heights
- **MCCV Painting**: ✅ `MccvPainter.cs` generates vertex colors from minimap PNGs (interleaved layout fixed)
- **PM4 MODF Reconstruction**: ✅ 1101 entries in `pm4-adt-test12/modf_reconstruction/`

### PM4 Pipeline Components
- **`Pm4ModfReconstructor`**: ✅ Matches PM4 objects to WMO library, generates MODF entries
- **`Pm4WmoGeometryMatcher`**: ✅ Geometry-based WMO matching using principal extents
- **`wmo_library.json`**: ✅ 352 WMO entries with pre-computed geometry stats

## ⚠️ Partial / Broken

### AdtModfInjector - BROKEN
- **Problem**: Appends MWMO/MODF chunks to end of file
- **Result**: Corrupted ADTs that Noggit cannot read
- **Root cause**: ADT chunks must be in specific order with correct MHDR/MCIN offsets

### Warcraft.NET Terrain.Serialize() - BROKEN
- **Problem**: Corrupts MCNK data during parse→serialize roundtrip
- **Evidence**: MCNK loses ~2,048 bytes after roundtrip
- **Result**: Noggit crashes on load
- **DO NOT USE** for ADT serialization

### Split ADT Merging - ABANDONED
- Custom `AdtPatcher.MergeSplitAdt()` produces corrupted output
- **Decision**: Use WoWMuseum ADTs as base instead of merging split files

## 🔄 Next Steps: Chunk-Preserving ADT Patcher

1. **Create `MuseumAdtPatcher`** - Parse WoWMuseum ADT chunks as raw bytes
2. **Preserve MCNK exactly** - Store all 256 as raw bytes (keeps all subchunks)
3. **Modify only MWMO/MWID/MODF** - Append new WMO names and placements
4. **Rebuild with correct offsets** - Use WdlToAdtGenerator pattern for MHDR/MCIN
5. **Only patch tiles that need it** - Skip tiles without PM4 MODF entries

## Key Files

| File | Status |
|------|--------|
| `WoWRollback.PM4Module/Pm4AdtPatcher.cs` | ⚠️ Needs update to add WMO names |
| `WoWRollback.PM4Module/MccvPainter.cs` | ✅ Fixed interleaved vertex layout |
| `WoWRollback.PM4Module/Program.cs` | ✅ Has `inject-modf` command (needs fix) |
| `WoWRollback.Core/Services/PM4/AdtModfInjector.cs` | ❌ BROKEN - appends chunks incorrectly |
| `WoWRollback.Core/Services/PM4/Pm4ModfReconstructor.cs` | ✅ Works - generates MODF from PM4 |

## Data Inventory

| Data | Location | Count |
|------|----------|-------|
| PM4 files | `test_data/development/World/Maps/development/*.pm4` | 616 |
| Split Cata ADTs | `test_data/development/World/Maps/development/*.adt` | 466 root |
| WoWMuseum ADTs | `test_data/WoWMuseum/335-dev/World/Maps/development/*.adt` | 2303 |
| Minimap PNGs | `test_data/minimaps/development/*.png` | 2252 |
| MODF entries | `pm4-adt-test12/modf_reconstruction/modf_entries.csv` | 1101 |
| WMO names | `pm4-adt-test12/modf_reconstruction/mwmo_names.csv` | 352 |
