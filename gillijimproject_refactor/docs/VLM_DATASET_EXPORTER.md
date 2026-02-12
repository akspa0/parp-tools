# VLM Dataset Exporter - Reference

## Quick Start

### Batch Export All Maps
```bash
cd src/WoWMapConverter/WoWMapConverter.Cli
dotnet run -- vlm-export --client "H:\053-client\" --batch-all --out "J:\vlm-datasets"
```

### Single Map Export
```bash
dotnet run -- vlm-export --client "H:\053-client\" --map Azeroth --out "J:\vlm-datasets"
```

---

## Output Structure
```
J:\vlm-datasets\053_Azeroth_v30\
├── dataset/                    # JSON metadata per tile
│   └── Azeroth_32_48.json
├── images/                     # Visual data
│   ├── Azeroth_32_48.png       # Minimap (512x512, NearestNeighbor upscale)
│   ├── Azeroth_32_48_normal.png # Normal map from MCNR
│   ├── Azeroth_32_48_heightmap.png      # Local heightmap
│   └── Azeroth_32_48_heightmap_global.png # Global heightmap
├── liquids/                    # Water data
│   ├── Azeroth_32_48_liq_mask.png
│   └── Azeroth_32_48_liq_height.png
├── shadows/                    # Per-chunk shadows
└── masks/                      # Per-layer alpha masks
```

---

## JSON Schema (per tile)
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
    "wdl_heights": { "outer_17": [...], "inner_16": [...] },
    "liquid_mask": "liquids/Azeroth_32_48_liq_mask.png",
    "objects": [
      {
        "name": "Inn",
        "name_id": 42,
        "x": 234.5, "y": 178.2, "z": 45.0,
        "rot_x": 0, "rot_y": 0, "rot_z": 1.57,
        "scale": 1.0,
        "category": "wmo",
        "bounds_min": [-10.5, -8.2, 0],
        "bounds_max": [10.5, 8.2, 15.3]
      }
    ]
  }
}
```

---

## Verifying Export Success

### Check for Required Files
After export, verify each tile has:
- [ ] `images/{tile}.png` - Minimap
- [ ] `images/{tile}_normal.png` - Normal map (NEW in V7.1)
- [ ] `images/{tile}_heightmap.png` - Height ground truth
- [ ] `dataset/{tile}.json` - Metadata with `objects` array

### Sample Check Command
```bash
# Count normal maps (should match tile count)
ls images/*_normal.png | wc -l

# Check object bounds in JSON
cat dataset/Azeroth_32_48.json | jq '.terrain_data.objects[0].bounds_min'
```

---

## Data Flow

1. **MCVT** → 145 heights/chunk → 512x512 heightmap
2. **MCNR** → 145 normals/chunk → 512x512 normal map
3. **MCLQ/MH2O** → Liquid data → water mask PNG
4. **MDDF/MODF** → Object placements → `objects[]` with bounds
5. **MDX/WMO files** → Bounding boxes → `bounds_min`/`bounds_max`
