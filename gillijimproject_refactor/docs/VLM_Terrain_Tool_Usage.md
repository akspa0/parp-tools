# VLM Terrain Data Export Tool

This tool exports terrain data from WoW client archives for Vision-Language Model (VLM) training. It generates a rich dataset correlating visual minimaps, geometric terrain meshes, and "decompiled" texture distribution masks.

## Usage

The tool is integrated into the `WoWRollback` CLI.

### Command

```bash
dotnet run --project WoWRollback/WoWRollback.Cli/WoWRollback.Cli.csproj -- vlm-export --client-path <WOW_CLIENT_PATH> --map <MAP_NAME> --out <OUTPUT_DIR>
```

### Arguments

- `--client-path`: Path to the WoW client root directory (containing `Data` folder). Supports MPQ (1.12.1, 3.3.5) and CASC (modern) formats (via `ArchiveLocator`).
- `--map`: The internal name of the map to export (e.g., `Azeroth`, `Kalimdor`, `Development`).
- `--out`: Directory where the exported dataset will be saved.
- `--listfile`: (Optional) Path to a community listfile (CSV) to resolve FileDataIDs to filenames.

### 3. Usage Examples

#### Standard Export (Retail/Classic)
```bash
dotnet run --project WoWRollback/WoWRollback.Cli -- vlm-export \
  --client-path "C:/WoW/Retail" \
  --map "Azeroth" \
  --out "./output_vlm" \
  --listfile "J:/wowDev/parp-tools/test_data/community-listfile-withcapitals.csv"
```

#### Alpha 0.5.3 Export
The tool automatically detects monolithic WDT files (e.g., `Azeroth.wdt`) used in Alpha clients.
- Point `--client-path` to your Alpha client folder.
- If no `.adt` files are found, the tool parses the `.wdt` to extract terrain data.

#### Flat File Input
You can point `--client-path` to any custom folder containing loose files. The tool handles non-MPQ/CASC directories natively.

### Output Structure

The output directory will organize the data for Unsloth ingestion as follows:

```
<OUTPUT_DIR>/
├── images/
│   ├── <map>_<x>_<y>.png              # Minimap tile (Visual context)
│   └── ...
├── masks/
│   ├── <map>_<x>_<y>_c<idx>_l<id>.png # Alpha masks (Visual texture distribution)
│   └── ...
├── shadows/
│   ├── <map>_<x>_<y>_c<idx>.png       # Shadow maps (Geometric lighting)
│   └── ...
├── liquids/
│   ├── <map>_<x>_<y>_liq_height.png   # Liquid heightmap (stitched)
│   ├── <map>_<x>_<y>_liq_mask.png     # Liquid existence mask (stitched)
│   └── ...
├── tilesets/
│   ├── <texture_name>.png             # Exported tileset textures
│   └── ...
├── stitched/
│   ├── <map>_<x>_<y>_shadow.png       # Stitched full-tile shadow map
│   └── ...
├── depths/                            # (Optional) DepthAnything3 depth maps
│   ├── <map>_<x>_<y>_depth.png
│   └── ...
├── dataset/
│   ├── <map>_<x>_<y>.json             # Rich metadata for Unsloth training
│   └── ...
└── texture_database.json              # List of all unique texture paths found
```

### JSON Dataset Schema (Unsloth Ready)

Each `dataset/*.json` file contains a training sample linking visual, geometric, and semantic data:

```json
{
  "image": "images/Azeroth_32_48.png",
  "terrain_data": {
    "adt_tile": "Azeroth_32_48",
    "obj_content": "v 1.0 2.0 3.0 ...",   // Raw OBJ mesh string
    "mtl_content": "newmtl ...",         // Raw MTL material string
    "layer_masks": [                     // Paths to PNG alpha masks
      "masks/Azeroth_32_48_c0_l1.png",
      "masks/Azeroth_32_48_c0_l2.png"
    ],
    "alpha_maps": "...",                 // Base64 raw MCAL (Binary fallback)
    "shadow_map": "...",                 // Base64 raw MCSH
    "textures": [                        // Texture filenames
      "Tileset\\Expansion05\\Tundra\\TundraRock.blp",
      "Tileset\\Expansion05\\Tundra\\TundraGrass.blp"
    ],
    "layers": [                          // Logical layer definitions
      { "texture_id": 1, "flags": 256, "alpha_offset": 4096, "effect_id": 0 }
    ],
    "objects": [                         // Object placements (M2/WMO)
      { "name": "Tree01", "x": 100.0, "y": 200.0, "z": 50.0, "rot_x": 0, "category": "m2" }
    ],
    "wdl_heights": [10, 12, 11, ...],    // (Optional) 17x17 Low-res global heightmap
    "height_min": 100.0,
    "height_max": 250.0
  }
}
```

## VLM Integration Strategy

This dataset is designed to teach a VLM to "read" terrain:
1.  **Visual Input**: The model sees the **Minimap** (`image`).
2.  **Geometric Ground Truth**: The **Mesh** (`obj_content`) provides the physical shape corresponding to that visual.
3.  **Textural Ground Truth**: The **Alpha Masks** (`layer_masks`) teach the model *where* specific textures (like grass, rock, snow) are painted, effectively "decompiling" the artist's brush strokes from the final rendered terrain.

By training on this triplet (Map Image -> Mesh + Texture Masks), the model learns to infer 3D structure and material composition solely from 2D map images.
