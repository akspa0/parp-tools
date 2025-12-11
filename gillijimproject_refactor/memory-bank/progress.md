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
- **WDL→ADT**: ✅ Generates terrain from WDL heights (1496 tiles with MCCV)
- **MCCV Painting**: ✅ `MccvPainter.cs` generates vertex colors from minimap PNGs
- **ADT Merger**: Merges split ADTs — works for tiles WITH `_tex0.adt`

## ⚠️ Partial

### Split ADT Merging
- **Works when all 3 files exist** (root + _obj0 + _tex0)
- **333 tiles have `_tex0.adt`** in source data
- **Some tiles missing `_tex0.adt`** → no texture data available to merge
- **Comparison with WoWMuseum reference** shows our merger produces correct structure

### Source Data Limitations
- Development map source files are from 2010 (Cataclysm split format)
- Not all tiles have complete split file sets
- Reference monolithic ADTs in `test_data/WoWMuseum/335-dev/` may have been assembled from multiple sources

## 🔄 Next Steps

1. **Use Warcraft.NET library** for proper split→monolithic conversion
2. **Validate tiles with complete data** against reference files
3. **Accept missing texture data** for incomplete tiles, or find alternate sources

## Reference Libraries

| Library | Path | Purpose |
|---------|------|---------|
| **MapUpconverter** | `lib/MapUpconverter/` | WotLK→Legion/BfA conversion (reverse our direction) |
| **Warcraft.NET** | `lib/Warcraft.NET/` | ADT chunk definitions, `Wotlk.Terrain` class |
| **WoWFormatLib** | `lib/wow.tools.local/WoWFormatLib/` | Additional format utilities |

## Key Files

| File | Status |
|------|--------|
| `WoWRollback.PM4Module/AdtPatcher.cs` | ✅ Correct FourCC, single merge implementation |
| `WoWRollback.PM4Module/MccvPainter.cs` | ✅ NEW - Minimap→MCCV conversion |
| `WoWRollback.PM4Module/WdlToAdtProgram.cs` | ✅ Updated with `--minimap` support |
| `WoWRollback.PM4Module/WdlToAdtTest.cs` | ✅ Updated to accept MCCV data |
