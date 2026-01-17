# VLM Dataset Exporter - Technical Reference

## Overview

The VLM Dataset Exporter extracts terrain data from WoW ADT files for machine learning training. It supports both Alpha 0.5.3 format (embedded ADTs in WDT) and modern 3.3.5+ format (separate ADT files).

---

## Command Reference

### Single Map Export
```bash
dotnet run -- vlm-export --client "H:\053-client" --map Azeroth --out "./datasets"
```

### Batch Export (All Standard Maps)
```bash
dotnet run -- vlm-export --client "H:\053-client" --batch-all --out "./datasets"
```

**Batch exports:** Azeroth, Kalimdor, EasternKingdoms, Northrend, Outland, Pandaria, BrokenIsles, KulTiras

---

## Data Extraction Pipeline

### Phase 1: Tile Discovery
1. Load WDT file
2. Parse MAIN chunk for tile existence flags
3. Load WDL file for low-resolution height data
4. Detect format (Alpha vs Modern) based on file size

### Phase 2: Per-Tile Extraction
For each existing tile:

| Chunk | Data Extracted | Use |
|-------|----------------|-----|
| MCVT | 145 height values per chunk | Ground truth heightmap |
| MCNR | 145 normal vectors per chunk | Normal map generation |
| MCSH | 64x64 shadow bitmap | Shadow maps |
| MCAL | Alpha layer data | Texture blending |
| MCLY | Layer flags + texture IDs | Texture identification |
| MCLQ | Liquid heights + flags | Water data |
| MTEX | Texture path list | Tileset reference |

### Phase 3: Image Generation
| Output | Resolution | Method |
|--------|------------|--------|
| Heightmap | 512x512 | Barycentric interpolation from 145 points |
| Normal Map | 512x512 | Barycentric interpolation from MCNR |
| Minimap | 512x512 | BLP conversion + NearestNeighbor upscale |
| Water Mask | 512x512 | Stitched from MCLQ flags |

### Phase 4: Post-Processing
1. Stitch per-chunk shadows into tile-level image
2. Stitch per-chunk alpha masks
3. Generate liquid masks and height maps
4. Update JSON with stitched paths

---

## JSON Schema

Each tile produces a JSON file with this structure:

```json
{
  "image": "images/Azeroth_32_48.png",
  "terrain_data": {
    "adt_tile": "Azeroth_32_48",
    "heightmap": "images/Azeroth_32_48_heightmap.png",
    "heightmap_local": "images/Azeroth_32_48_heightmap.png",
    "heightmap_global": "images/Azeroth_32_48_heightmap_global.png",
    "normalmap": "images/Azeroth_32_48_normal.png",
    "height_min": 23.5,
    "height_max": 156.2,
    "height_global_min": -1000.0,
    "height_global_max": 3000.0,
    "wdl_heights": {
      "outer_17": [/* 289 values */],
      "inner_16": [/* 256 values */]
    },
    "liquid_mask": "liquids/Azeroth_32_48_liq_mask.png",
    "liquid_min": 0.0,
    "liquid_max": 45.0,
    "textures": ["Tileset/Grass.blp", "Tileset/Dirt.blp"],
    "objects": [/* M2/WMO placements */]
  }
}
```

---

## Key Functions

### `ExtractFromAdtAlpha`
Handles Alpha 0.5.3 format ADTs embedded in WDT files.

### `ExtractFromLkAdt`
Handles 3.3.5+ format separate ADT files from MPQ archives.

### `GenerateHeightmap`
Renders 512x512 heightmap using barycentric interpolation.

### `GenerateNormalmap`
Renders 512x512 normal map from MCNR data.

### `RenderHeightmapImage` / `RenderNormalmapImage`
Core rendering with proper edge clamping to prevent seams.

### `ConvertBlpToPng`
Converts BLP textures/minimaps with 512x512 NearestNeighbor upscaling.

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| No normal maps | `GenerateNormalmap` not called | Fixed in latest version |
| 256x256 outputs | Wrong size parameter | Check `RenderHeightmapImage` uses 512 |
| Black tile edges | Interpolation overflow | Apply `Math.Clamp(0,1)` to local coords |
| Missing water | No MCLQ chunk | Alpha 0.5.3 may lack water data |
| Blurry minimaps | Bicubic upscaling | Use NearestNeighbor for crisp edges |
