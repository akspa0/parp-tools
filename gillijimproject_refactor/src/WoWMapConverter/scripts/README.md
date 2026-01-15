# WoW AI Terrain Tools

This directory contains a suite of tools for analyzing WoW ADT terrain data, training AI models to reconstruct 3D geometry from 2D images, and regenerating minimap textures from terrain data.

---

## Quick Start: Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    0.5.3 CLIENT → DATASET → OUTPUTS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────────┐│
│  │ 0.5.3 Client │ ──► │ VLM Dataset  │ ──► │ Outputs:                     ││
│  │ (MPQ files)  │     │ (JSON+Images)│     │  • Rebaked Minimaps          ││
│  └──────────────┘     └──────────────┘     │  • Stitched World Maps       ││
│         │                    │             │  • AI Training Data          ││
│         │                    │             │  • 3D Mesh Reconstruction    ││
│         ▼                    ▼             └──────────────────────────────┘│
│    vlm-export           vlm-bake                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Extract VLM Dataset from 0.5.3 Client

```bash
cd src/WoWMapConverter/WoWMapConverter.Cli

# Export a single map (e.g., Kalidar)
dotnet run -- vlm-export --client "H:\053-client\" --map Kalidar --out "test_data/vlm-datasets/053_Kalidar"

# Export Azeroth (Eastern Kingdoms)
dotnet run -- vlm-export --client "H:\053-client\" --map Azeroth --out "test_data/vlm-datasets/053_azeroth"

# Export Kalimdor
dotnet run -- vlm-export --client "H:\053-client\" --map Kalimdor --out "test_data/vlm-datasets/053_kalimdor"
```

**What gets exported:**
| Data | Format | Description |
|------|--------|-------------|
| Heights | JSON (145 floats × 256 chunks) | Full ADT vertex heights (MCVT) |
| Normals | JSON (435 bytes × 256 chunks) | Vertex normals (MCNR) |
| Shadows | PNG (64×64 per chunk) | Shadow maps (MCSH) |
| Alpha Masks | PNG (64×64 per layer) | Texture blend masks (MCAL) |
| Textures | List of paths | Referenced textures (MTEX) |
| Minimaps | PNG (256×256) | Original minimap tiles |
| Stitched | PNG/WebP (1024×1024) | Composited full-tile images |

### Step 2: Rebake Minimaps from Dataset

```bash
# Bake all tiles in a dataset
dotnet run -- vlm-bake -d "test_data/vlm-datasets/053_Kalidar"

# Export individual layers for AI training
dotnet run -- vlm-bake -d "test_data/vlm-datasets/053_Kalidar" --export-layers

# Bake without shadows
dotnet run -- vlm-bake -d "test_data/vlm-datasets/053_Kalidar" --no-shadows
```

### Step 3: Train AI Models (Optional)

```bash
cd src/WoWMapConverter/scripts

# Train height+normals regression model (V3 - recommended)
python train_height_regressor_v3.py
```

### Step 4: Generate 3D Meshes from Minimaps

```bash
# From trained model
python img2mesh_v3.py path/to/minimap_tile.png

# Output: terrain.obj with 37,120 vertices
```

---

## Dataset Structure

After running `vlm-export`, you get:

```
vlm-datasets/053_MapName/
├── MapName_X_Y.json          # Per-tile terrain data
├── stitched/                 # Composited minimap tiles
│   ├── MapName_X_Y.png       # 1024×1024 stitched minimap
│   └── ...
├── shadows/                  # Per-chunk shadow maps
│   ├── MapName_X_Y_c0.png    # Chunk 0 shadow (64×64)
│   └── ...
├── alphas/                   # Per-layer alpha masks
│   ├── MapName_X_Y_c0_l0.png # Chunk 0, Layer 0 alpha
│   └── ...
├── liquids/                  # Liquid height/mask maps
└── baked/                    # Regenerated minimaps (after vlm-bake)
    ├── MapName_X_Y.png       # Final composite
    └── MapName_X_Y_layers/   # Individual layers (if --export-layers)
```

### JSON Tile Format

Each `MapName_X_Y.json` contains:

```json
{
  "terrain_data": {
    "adt_tile": "MapName_X_Y",
    "heights": [
      {"idx": 0, "h": [145 floats...]},
      {"idx": 1, "h": [145 floats...]},
      ...  // 256 chunks total
    ],
    "chunk_layers": [
      {
        "idx": 0,
        "normals": [435 signed bytes...],  // 145 vertices × 3 components
        "layers": [
          {"tex_id": 0, "texture_path": "path/to/texture.blp", "flags": 0}
        ]
      }
    ],
    "textures": ["texture1.blp", "texture2.blp", ...],
    "height_min": -100.0,
    "height_max": 500.0
  }
}
```

---

## 1. Topography & Analysis

### `terrain_librarian.py`
**Purpose**: Analyzes the dataset to find and catalog recurring "Prefab" patterns.

**Usage**:
```bash
python terrain_librarian.py
```

**Output**: 
- `prefab_library.json`: Database of unique chunks (hashed by geometry).
- `prefab_instances.json`: Map of all chunk locations.
- `Prefab_Zoo_X_Y.json`: Reconstructed ADT tiles visualizing the unique prefabs.

---

## 2. AI Training Pipeline (Image-to-Mesh)

### A. Data Generation: `generate_height_regression_dataset.py`
**Purpose**: Prepares the training dataset by pairing 64x64 Minimap crops with their corresponding 3D Height Arrays (145 floats).

**Usage**:
```bash
python generate_height_regression_dataset.py
```

**Output**: `height_regression.jsonl` (Training data).

### B. Training: `train_tiny_regressor.py` (V1 - Deprecated)
**Purpose**: Original ViT regressor with per-vertex normalization.
**Status**: ⚠️ Produces spiky/WDL-like output due to normalization issues.

### C. Training: `train_height_regressor_v2.py` (V2 - Chunk-Level - Deprecated)
**Purpose**: Improved ViT regressor with fixes for smooth terrain output.

**Key Improvements over V1**:
| Issue in V1 | Fix in V2 |
|-------------|-----------|
| Per-vertex normalization | Global normalization (preserves height relationships) |
| No smoothness constraint | Smoothness loss (encourages coherent surfaces) |
| Single dataset | Multi-dataset support (Kalidar, Azeroth, Kalimdor) |
| 5 epochs, high LR | 20 epochs, cosine annealing scheduler |
| No augmentation | Horizontal flip augmentation |

**Usage**:
```bash
python train_height_regressor_v2.py
```

**Output**: Model saved to `j:\vlm_output\wow_height_regressor_v2`.

### D. Training: `train_height_regressor_v3.py` (V3 - Full ADT Resolution - Recommended)
**Purpose**: U-Net model that predicts full ADT terrain geometry from minimap images.

**Key Improvements over V2**:
| Issue in V2 | Fix in V3 |
|-------------|-----------|
| Chunk-level input (64×64) | Full tile input (256×256 native minimap) |
| Heights only (145 values) | Heights + Normals (37,120 + 111,360 values) |
| ViT architecture | U-Net CNN for spatial coherence |
| No 2D smoothness loss | 2D smoothness loss respecting WoW's chunk grid |
| No gradient matching | Gradient matching preserving terrain slopes |
| No boundary continuity | **Boundary continuity loss** for seamless chunk edges |
| 3-channel input | **5-channel input** (RGB + Shadow + Alpha) |
| No position context | **Chunk position embedding** for absolute height context |

**Model Architecture**:
```
Input:  256×256×5 (RGB + Shadow + Alpha)
        + Chunk Positions [256, 3] → Position Embedding [64]
        ↓
     U-Net Encoder (5 levels: 256→128→64→32→16→8)
        ↓
     Bottleneck (1024 channels @ 8×8)
        ↓
     Decoder (back to 16×16 = 256 chunk positions)
        + Position Embedding injection
        ↓
Output: Heights [256, 145] = 37,120 vertex heights
        Normals [256, 145, 3] = 111,360 normal components
```

**Loss Function**:
```
loss = MSE(heights) + 0.1 * smoothness + 0.1 * gradient + 0.5 * boundary + 0.2 * MSE(normals)
```
- **boundary**: Enforces adjacent chunks share edge heights (critical for seamless terrain)
- **holes_mask**: Excludes hole regions from loss calculation

**Usage**:
```bash
# Basic training
python train_height_regressor_v3.py

# Resume from checkpoint
python train_height_regressor_v3.py --resume checkpoint_epoch50.pt

# Custom epochs and batch size
python train_height_regressor_v3.py --epochs 200 --batch-size 4

# All options
python train_height_regressor_v3.py --resume best_model.pt --epochs 150 --batch-size 2 --lr 5e-5 --output ./my_model
```

**CLI Options**:
| Option | Description |
|--------|-------------|
| `--resume`, `-r` | Resume from checkpoint file |
| `--epochs`, `-e` | Number of epochs (default: 100) |
| `--batch-size`, `-b` | Batch size (default: 2) |
| `--lr` | Learning rate (default: 1e-4) |
| `--output`, `-o` | Output directory |

**Checkpoints**:
- `best_model.pt` - Saved whenever validation loss improves
- `checkpoint_epoch{N}.pt` - Saved every 10 epochs
- `emergency_checkpoint.pt` - Saved on Ctrl+C or error (resume with `--resume`)

**Output**: Model saved to `j:\vlm_output\wow_height_regressor_v3`.

**Requirements for V3 training:**
- Native 256×256 minimap images (or stitched tiles that get resized)
- Complete tile JSON with all 256 chunks of height data
- Normals data in `chunk_layers[].normals` (optional but recommended)
- Shadow maps in `stitched/<tile>_shadow.png` (optional, improves accuracy)
- Alpha masks in `stitched/<tile>_alpha_l0.png` (optional, improves accuracy)

---

## 3. Inference: Image-to-Mesh

### `img2mesh.py` (V1 - for V1 model)
**Purpose**: Inference for V1 model. Produces spiky output.

### `img2mesh_v2.py` (V2 - Chunk-Level)
**Purpose**: Inference for V2 model with improved height prediction. Includes post-processing smoothing.

**Usage**:
```bash
# V2 model (chunk-level)
python img2mesh_v2.py path/to/minimap_chunk.png

# With smoothing (default: 1 iteration)
python img2mesh_v2.py path/to/minimap_chunk.png --smooth 2 --sigma 1.0

# Custom output directory
python img2mesh.py path/to/minimap_chunk.png -o J:\custom\output\folder

# Explicit output file
python img2mesh.py path/to/minimap_chunk.png J:\specific\path\output.obj
```

**Options**:
| Option | Description |
|--------|-------------|
| `--smooth N` | Number of smoothing iterations (default: 0) |
| `--smooth-method` | `gaussian`, `laplacian`, `bilateral`, `median` (default: gaussian) |
| `--sigma S` | Gaussian/bilateral sigma (default: 1.0) |
| `--output-dir`, `-o` | Output directory (default: `test_output/rebaked_minimaps`) |
| `--model` | Custom model path |

**Smoothing Methods**:
- **gaussian**: Standard Gaussian blur, good general-purpose smoothing
- **laplacian**: Preserves more terrain detail (cliffs, ridges)
- **bilateral**: Edge-preserving smoothing (requires scikit-image)
- **median**: Good for removing outlier spikes

**Output**: 
- `<input_name>.obj`: 3D Geometry with WoW chunk vertex layout (145 vertices)
- `<input_name>.mtl`: Material file referencing the input image as texture

### `img2mesh_v3.py` (V3 - Full ADT Resolution - Recommended)
**Purpose**: Inference for V3 model. Outputs full ADT heightmap (37,120 vertices) from a single minimap tile, plus a VLM-compatible dataset JSON for minimap regeneration.

**Usage**:
```bash
# Full tile inference - creates output folder with all files
python img2mesh_v3.py path/to/minimap_tile.png

# With smoothing
python img2mesh_v3.py path/to/minimap_tile.png --smooth

# Custom output directory
python img2mesh_v3.py path/to/minimap_tile.png --output ./my_output

# JSON only (no OBJ mesh)
python img2mesh_v3.py path/to/minimap_tile.png --json-only
```

**Options**:
| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output directory (default: `<input>_output/`) |
| `--smooth` | Apply Gaussian smoothing to mesh |
| `--model` | Custom model directory |
| `--json-only` | Only output JSON and images, skip OBJ generation |

**Output Folder Structure**:
```
<tile_name>_output/
├── <tile_name>.json           # VLM dataset JSON (for minimap regeneration)
├── <tile_name>_minimap.png    # Copy of input minimap
├── <tile_name>_heightmap.png  # 16-bit grayscale heightmap (144×144)
├── <tile_name>_normalmap.png  # RGB normal map (144×144)
└── <tile_name>.obj            # 3D mesh (37,120 vertices)
```

**VLM Dataset JSON Format**:
The output JSON is compatible with `MinimapBakeService` for minimap regeneration:
```json
{
  "image": "tile_name_minimap.png",
  "terrain_data": {
    "adt_tile": "tile_name",
    "heights": [{"idx": 0, "h": [145 floats]}, ...],  // 256 chunks
    "chunk_positions": [x,y,z, ...],  // 256×3 floats
    "holes": [0, 0, ...],  // 256 bitmasks
    "chunk_layers": [{"idx": 0, "normals": [...], ...}, ...],
    "height_min": float,
    "height_max": float
  }
}
```

---

## 4. Minimap Regeneration (C# CLI)

The `WoWMapConverter.Cli` tool includes a `vlm-bake` command for regenerating minimap textures from VLM dataset exports.

### Basic Usage
```bash
# Bake all tiles in a dataset
dotnet run -- vlm-bake -d "path/to/vlm_dataset"

# Bake a specific tile
dotnet run -- vlm-bake -d "path/to/vlm_dataset" -i "path/to/Tile_X_Y.json"

# Export individual layers for debugging
dotnet run -- vlm-bake -d "path/to/vlm_dataset" --export-layers
```

### Options
| Option | Description |
|--------|-------------|
| `-d`, `--dataset` | VLM dataset directory (required) |
| `-i`, `--input` | Specific JSON tile to process |
| `-o`, `--output` | Output PNG path |
| `--export-layers`, `-l` | Export individual texture layers as separate PNGs |
| `--shadows` | Enable shadow overlay (default) |
| `--no-shadows` | Disable shadow overlay |
| `--debake` | Remove shadows from existing minimaps |
| `--shadow-intensity` | Shadow intensity 0.0-1.0 (default: 0.5) |

### Texture Blending Algorithm
The minimap baker uses WoW's weighted blend algorithm (from `adt.fragment.shader`):
- **Layer 0 weight** = `1.0 - sum(layer1..N alphas)`
- **Layer N weight** = `alpha[N]`
- **Final color** = `sum(layer[i].rgb * weight[i]) / sum(weights)`

This ensures proper terrain texture compositing matching the game engine.

### Output Structure
```
vlm_dataset/
├── baked/                              # Regenerated minimap tiles
│   ├── Map_X_Y.png                    # Final composite minimap (4096x4096)
│   └── Map_X_Y_layers/                # Individual layers (if --export-layers)
│       ├── raw/                       # Raw textures only (no blending)
│       │   ├── Map_X_Y_layer0_raw.png
│       │   ├── Map_X_Y_layer1_raw.png
│       │   └── ...
│       ├── weighted/                  # Texture * WoW weight (alpha = weight)
│       │   ├── Map_X_Y_layer0_weighted.png
│       │   ├── Map_X_Y_layer1_weighted.png
│       │   └── ...
│       └── cumulative/                # Progressive blend up to layer N
│           ├── Map_X_Y_layer0_cumulative.png
│           ├── Map_X_Y_layer1_cumulative.png
│           └── ...
```

### Layer Export Types (for ViT Training)
| Type | Description | Use Case |
|------|-------------|----------|
| **raw** | Original texture, no alpha applied | Texture classification training |
| **weighted** | RGB × weight, Alpha = weight value | Learn per-layer contribution |
| **cumulative** | Progressive composite (layers 0..N) | Learn blending progression |

---

## 5. Full Map Stitching

The VLM export automatically generates stitched world maps from individual tiles.

### Automatic Outputs

During `vlm-export`, the following stitched images are generated:

```
vlm-datasets/053_MapName/
└── stitched/
    ├── MapName_full_minimap.webp       # Complete world map (all tiles)
    ├── MapName_full_minimap_75pct.webp # 75% scale version
    ├── MapName_full_minimap_50pct.webp # 50% scale version
    ├── MapName_full_shadow.webp        # Stitched shadow maps
    ├── MapName_full_alpha_l0.webp      # Stitched alpha layer 0
    ├── MapName_full_alpha_l1.webp      # Stitched alpha layer 1
    └── ...
```

### Smart Scaling

Large maps automatically scale down to fit within practical limits:
- **Maximum dimension**: 16384 pixels
- **Scale factors**: Rounded to 25%, 33%, 50%, or 75%
- **Format**: WebP at 99% quality (handles large images better than PNG)

Example output for a 18×38 tile map:
```
Auto-scaling: 18x38 tiles at 1024px would be 18432x38912
  -> Scaling to 33% (338px tiles) = 6084x12844
```

### Per-Tile Stitching

Each ADT tile also gets a stitched 1024×1024 image combining all 256 chunks:
- **Shadows**: `MapName_X_Y_shadow.png` (64×64 chunks → 1024×1024)
- **Alphas**: `MapName_X_Y_alpha_l0.png` per layer

---

## 6. Lighting Reference (from Ghidra Analysis)

WoW's default light settings (from `CGxLight` constructor in WoWClient.exe):

| Property | Default Value |
|----------|---------------|
| **Direction** | `(0.0, 0.0, 1.0)` - Z-up |
| **Directional Color** | `RGB(255, 255, 255)` - pure white |
| **Ambient Color** | `RGB(0, 0, 0)` - black |
| **Ambient Intensity** | `1.0` |
| **Directional Intensity** | `1.0` |

Actual terrain lighting is data-driven from `Light.dbc` entries, which define time-of-day color transitions for:
- Direct light color
- Ambient color
- Sky colors (6 bands)
- Cloud colors
- Water colors
- Fog parameters
- Shadow opacity

---

## 7. WoW Terrain Data Reference

### ADT Structure (Alpha 0.5.3)

Each ADT file contains 256 MCNK chunks arranged in a 16×16 grid.

| Chunk | Size | Description |
|-------|------|-------------|
| **MCVT** | 580 bytes | 145 floats - vertex heights |
| **MCNR** | 448 bytes | 145×3 signed bytes - vertex normals + 13 padding |
| **MCSH** | 512 bytes | 64×64 bits - shadow map (8 bytes × 64 rows) |
| **MCAL** | variable | Alpha maps for texture blending |
| **MCLY** | 16 bytes/layer | Texture layer definitions |

### Vertex Layout (145 per chunk)

WoW uses an interleaved 9+8 pattern:
```
Row 0: 9 outer vertices (corners)
Row 1: 8 inner vertices (centers)
Row 2: 9 outer vertices
...
Row 16: 9 outer vertices

Total: 9×9 outer (81) + 8×8 inner (64) = 145 vertices
```

### Height Data

- **MCVT**: 145 floats, relative to chunk base height
- **Range**: Typically -500 to +2000 in world units
- **Resolution**: ~2.08 yards per vertex (533.33 / 256 yards per chunk)

### Normal Data

- **MCNR**: 145×3 signed bytes (X, Y, Z components)
- **Range**: -127 to +127, normalized to unit vectors
- **Alpha format**: Non-interleaved (81 outer, then 64 inner)
- **LK+ format**: Interleaved (9-8-9-8 pattern)

### Shadow Data

- **MCSH**: 64×64 bitmap (512 bytes = 4096 bits)
- **1 bit per pixel**: 0 = lit, 1 = shadowed
- **Baked from**: Sun position + terrain geometry

---

## Dependencies

**Python** (for AI tools):
```bash
pip install torch transformers pillow numpy scipy scikit-image tqdm
```

**C#** (for minimap baking):
- .NET 9.0
- SixLabors.ImageSharp
