# Progress

## ✅ Working

### Input Parsers
- **0.5.3 Alpha WDT/ADT**: ✅ Monolithic format, sequential MCNK, works 100%
- **WMO v14/v17**: ✅ Both directions implemented
- **BLP**: ✅ BlpResizer complete

### Standalone Tools
- **vlm-export**: ✅ Works for Alpha ADT
- **BlpResizer**: ✅ Production-ready
- **DBCTool.V2**: ✅ Crosswalk CSV generation

### Data Generation
- **VLM Datasets (Alpha)**: ✅ Azeroth v10 (685 tiles)
- **V8 Binary Export**: ✅ `.bin` format implemented

## ⚠️ Partial / In Progress

### LK 3.3.5 / Cata 4.0.0 ADT Processing

| Component | Status | Notes |
|-----------|--------|-------|
| Minimap TRS | ✅ FIXED | Column order + coordinate padding |
| JSON height_min/max | ✅ FIXED | MCIN-based parsing working |
| JSON heights[] array | ✅ FIXED | 256 chunks populated |
| Heightmap PNG | 🔧 FIX APPLIED | Removed posZ addition - untested |
| Alpha masks | ⚠️ NEEDS TESTING | May need MCAL parsing fix |
| Normal maps | ⚠️ NEEDS TESTING | Requires VlmChunkLayers.Normals |

**Root Causes Fixed (Jan 19-20, 2026):**
1. MCNK chunks accessed via **MCIN offsets**, not linear scanning
2. Heights stored with `posZ` addition - should be raw MCVT values

## ❌ Broken

### AdtModfInjector
- **Problem**: Appends MWMO/MODF chunks to end of file
- **Result**: Corrupted ADTs that Noggit cannot read
- **DO NOT USE**

### Warcraft.NET Terrain.Serialize()
- **Problem**: Corrupts MCNK data during parse→serialize roundtrip
- **DO NOT USE** for ADT serialization

## Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| 0.5.3 Alpha Export | ✅ Working | Heightmaps, masks, everything correct |
| LK/Cata JSON Data | ✅ Working | Heights correctly extracted via MCIN |
| LK/Cata Image Gen | 🔧 Fix Applied | Awaiting test (file lock) |
| V8 Training | ✅ Initial Run | 0.5.3 Azeroth, best loss 0.3178 |

## Key Technical Insight

**Why 0.5.3 works but LK+ doesn't:**

0.5.3 Alpha ADTs have MCNK chunks **sequentially** after header chunks. Linear byte scanning finds them.

LK/Cata ADTs store MCNK at **MCIN-specified offsets** scattered through the file. Linear scanning finds MVER→MHDR→MCIN→MTEX→...→end, completely missing MCNK chunks.

**Solution:** Parse MCIN chunk (256 × 16-byte entries containing offset+size), then jump to each offset to parse MCNK.
