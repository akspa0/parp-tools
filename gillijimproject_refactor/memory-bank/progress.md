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
- **vlm-export**: ✅ Extracts ADT/WDT to JSON dataset + Stitched Atlases
- **train_local.py**: ✅ Unsloth Qwen2-VL training script (Windows compatible)
- **export_gguf.py**: ✅ Manual GGUF export (Merge -> Convert -> Quantize)
- **train_tiny_regressor.py**: ✅ Tiny ViT Image-to-Height training complete
- **terrain_librarian.py**: ✅ Canonical geometry/alpha prefab detection
- **MinimapBakeService.cs**: 🚧 C# Super-Resolution baker (Build Errors)

### Data Generation
- **WDL→ADT**: ✅ Generates terrain from WDL heights
- **MCCV Painting**: ✅ `MccvPainter.cs` generates vertex colors from minimap PNGs
- **PM4 MODF Reconstruction**: ✅ 1101 entries in `pm4-adt-test12/modf_reconstruction/`
- **VLM Datasets**: ✅ Azeroth v10 (685 tiles), Kalidar v1 (56 tiles), Razorfen v1 (6 tiles)
- **V8 Binary Export**: ✅ `.bin` format implemented with Heights/Normals/Shadows/Alpha.
- **Split ADT Support**: ✅ `_tex0` / `_obj0` reading implemented for Cata support.

## ⚠️ Partial / Broken

### LK/Cata ADT Processing - PARTIALLY BROKEN (Jan 19, 2026)
- **Minimap Tile Resolution**: ✅ FIXED - TRS parsing column order was reversed
- **Normal Maps**: ❌ BROKEN - Generating incorrect data for 3.0.1 ADTs
- **Heightmaps**: ❌ BROKEN - Values appear corrupted/incorrect for 3.0.1 ADTs
- **Root cause**: Likely MCVT/MCNR offset or format differences between Alpha and LK

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

## Current Status Summary

| V7 Inference | 🔧 Refining | Adding smoothing, Z-scaling, and downscaling |
| V8 Spec | ✅ Complete | Transitioning to `reconstruction` branch |
| V8 Training | ✅ Initial Run | 0.5.3 Azeroth (685 tiles), best loss 0.3178 |
| Multi-Version ADT | 🔧 WIP | 0.5.3 ✅, 3.x ⚠️ (minimap OK, heightmaps broken), 4.x untested |
| Native Resolution | ✅ Set | 145×145 (native ADT) for V8 accuracy |
| Digital Archeology | 🚀 Initiated | Reconstructing lost data from minimap/WDL/PM4 |
| Minimap TRS | ✅ Fixed | Jan 19 - Column order and coordinate padding corrected |

## Key Files

| File | Status |
|------|--------|
| `WoWRollback.PM4Module/AdtPatcher.cs` | ✅ Single source of truth for merging |
| `WoWRollback.PM4Module/MccvPainter.cs` | ✅ Minimap→MCCV conversion |
| `regenerate_heightmaps_global.py` | ✅ Dual-mode heightmap generator |
| `VlmDatasetExporter.cs` | ✅ Fixed GenerateHeightmap |
| `HeightmapBakeService.cs` | ✅ Updated to use Alpha MCVT format |
