# WoW AI Terrain Tools

This directory contains a suite of tools for analyzing WoW ADT terrain data, training AI models to reconstruct 3D geometry from 2D images, and regenerating minimap textures from terrain data.

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
**Purpose**: Improved ViT regressor with fixes for smooth terrain output.

**Key Improvements over V2**:
| Issue in V2 | Fix in V3 |
|-------------|-----------|
| Chunk-level input | Full tile input (1024×1024 downsampled minimap) |
| Chunk-level output | Full tile output (256 chunks × 145 heights = 37,120 vertices) |
| No spatial coherence | U-Net architecture for spatial coherence |
| No 2D smoothness loss | 2D smoothness loss respecting WoW's chunk grid structure |
| No gradient matching | Gradient matching preserving terrain slopes |

**Usage**:
```bash
python train_height_regressor_v3.py
```

**Output**: Model saved to `j:\vlm_output\wow_height_regressor_v3`.

**Requirements for V3 training:**
- Stitched minimap images (4096×4096 tiles)
- Complete tile JSON with all 256 chunks of height data.

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
**Purpose**: Inference for V3 model. Outputs full ADT heightmap (37,120 vertices) from a single minimap tile.

**Usage**:
```bash
# Full tile inference (recommended)
python img2mesh_v3.py path/to/minimap_tile.png

# With smoothing
python img2mesh_v3.py path/to/minimap_tile.png --smooth

# Custom output
python img2mesh_v3.py path/to/minimap_tile.png --output terrain.obj
```

**Options**:
| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output OBJ file path |
| `--smooth` | Apply Gaussian smoothing |
| `--model` | Custom model directory |

**Output**: 
- `<input_name>.obj`: Full ADT mesh with 256 chunks × 145 vertices = 37,120 vertices

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

## 5. Lighting Reference (from Ghidra Analysis)

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

## Dependencies

**Python** (for AI tools):
```bash
pip install torch transformers pillow numpy scipy scikit-image
```

**C#** (for minimap baking):
- .NET 9.0
- SixLabors.ImageSharp
