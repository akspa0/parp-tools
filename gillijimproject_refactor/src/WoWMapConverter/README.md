# WoWMapConverter v3

**Complete bidirectional WoW map/asset conversion toolkit** supporting all versions from Alpha 0.5.3 through modern retail, with integrated DBC crosswalks, PM4 reconstruction, and 3D viewer.

## Supported Versions

| Version | ADT | WMO | Models | Status |
|---------|-----|-----|--------|--------|
| **Alpha 0.5.3** | Monolithic WDT | v14 (mono) | MDX | ✅ Verified (Ghidra) |
| **Classic 1.x** | v18 | v17 | M2 | ✅ Full support |
| **TBC 2.x** | v18 | v17 | M2 | ✅ Full support |
| **WotLK 3.x** | v18 + MH2O | v17 | M2 | ✅ Full support |
| **Cata 4.x** | Split (_tex0/_obj0) | v17 | M2 | 🔧 In progress |
| **MoP-Legion** | Split + _lod | v17+ | M2/M3 | 🔧 Planned |
| **BfA-DF** | Split + MAID | v17+ | M2/M3 | 🔧 Planned |

## Architecture

```
WoWMapConverter/
├── WoWMapConverter.Core/           # Core library
│   ├── Formats/
│   │   ├── Alpha/                  # Alpha WDT/ADT/WMO v14/MDX
│   │   ├── Classic/                # v18 ADT (monolithic)
│   │   ├── Cataclysm/              # Split ADT (_tex0/_obj0)
│   │   ├── Modern/                 # Legion+ formats (_lod, MAID)
│   │   ├── Wmo/                    # WMO v14-v17+
│   │   ├── Models/                 # MDX, M2, M3
│   │   └── Shared/                 # Common chunks
│   ├── Converters/
│   │   ├── AdtConverter.cs         # Universal ADT converter
│   │   ├── WmoConverter.cs         # WMO v14 ↔ v17+
│   │   ├── ModelConverter.cs       # MDX ↔ M2 ↔ M3
│   │   └── Pipeline.cs             # Full asset pipeline
│   ├── Dbc/
│   │   ├── DbcReader.cs            # DBC/DB2 parser
│   │   ├── AreaIdMapper.cs         # AreaTable crosswalk
│   │   └── MapCrosswalk.cs         # Map.dbc crosswalk
│   ├── Pm4/                        # PM4 pathfinding mesh
│   │   ├── Pm4Reader.cs            # PM4 parser
│   │   ├── Pm4Matcher.cs           # WMO geometry matching
│   │   └── ModfReconstructor.cs    # Reconstruct placements
│   └── Services/
│       ├── ListfileService.cs      # Asset path resolution
│       └── CascReader.cs           # Modern CASC archive support
│
├── WoWMapConverter.Cli/            # Command-line interface
│
├── WoWMapConverter.Gui/            # GUI application (Avalonia)
│   ├── Views/                      # XAML views
│   ├── ViewModels/                 # MVVM view models
│   └── 3DViewer/                   # WebGPU/Three.js viewer
│
└── WoWMapConverter.Tests/          # Unit tests
```

## Features

### Map Conversion (Bidirectional)
- **Alpha ↔ LK**: Monolithic WDT ↔ Split ADT files with **strict Ghidra-verified compliance** (Fixed offsets, 15KB chunk limits)
- **LK ↔ Cata+**: Handle split file format changes
- **AreaID Crosswalk**: Integrated mapping across all versions
- **Coordinate Transform**: Y-up (Alpha) ↔ Z-up (LK+)

### Asset Conversion
- **WMO v14 ↔ v17+**: Monolithic ↔ Split format
- **MDX ↔ M2**: Alpha models ↔ Modern M2
- **M3 Support**: New model format (Legion+)
- **BLP Handling**: Resize/convert textures

### PM4 Pipeline (from WoWRollback)
- **Geometry Extraction**: Parse PM4 pathfinding meshes
- **WMO Matching**: PCA-based fingerprint matching
- **MODF Reconstruction**: Generate placement data
- **Noggit Integration**: Output Noggit-ready projects

### DBC/DB2 Integration
- **Built-in parsing**: No external tools needed
- **All versions**: DBC (Classic-WotLK) and DB2 (Cata+)
- **Crosswalks**: AreaTable, Map, AreaTrigger, etc.

### VLM Dataset Export (Native)
Export ADT terrain data for Vision-Language Model training. Bidirectional: JSON ↔ ADT.

**Export:**
```bash
dotnet run --project WoWMapConverter.Cli -- vlm-export \
  --client /path/to/alpha/Data \
  --map development \
  --out ./vlm_dataset \
  --limit 10
```

**Decode (round-trip):**
```bash
dotnet run --project WoWMapConverter.Cli -- vlm-decode \
  --input ./vlm_dataset/dataset/development_31_31.json \
  --output ./reconstructed.adt
```

**Output Structure:**
```
vlm_dataset/
├── images/           # Minimap PNGs
├── shadows/          # MCSH per-chunk (64×64) and stitched (1024×1024)
├── masks/            # MCAL layer alphas (per chapter and stitched)
├── liquids/          # MH2O/MCLQ stitched heights and masks
├── tilesets/         # Unique tileset textures (PNG)
├── stitched/         # Stitched shadow and alpha maps (1024×1024)
├── depths/           # Depth maps (requires setup)
├── dataset/          # Structured JSON
└── texture_database.json
```

**Data Exported:**
- Heights, positions, holes, normals
- Shadow maps (MCSH), alpha masks (MCAL)
- Texture layers (MCLY), liquids (MH2O/MCLQ)
- Object placements (MDDF/MODF)
- Compatible with DepthAnything3 for depth map correlation

**DepthAnything3 Setup (Optional):**
To enable depth map generation, you must set up the `da3` Conda environment:
```powershell
# Windows (PowerShell)
cd src/WoWMapConverter/WoWMapConverter.Core/VLM/DepthAnything3
./setup_da3.ps1
```
Then use the `--depth` flag when running the export command.
If you see `EnvironmentNameNotFound`, ensure you have run the setup script successfully.

### Minimap Regeneration (vlm-bake)

Regenerate high-resolution minimap tiles from VLM dataset JSON files using WoW's weighted blend algorithm.

**Basic Usage:**
```bash
dotnet run --project WoWMapConverter.Cli -- vlm-bake -d ./vlm_dataset
```

**Options:**
| Option | Description |
|--------|-------------|
| `-d, --dataset` | Path to VLM dataset root (required) |
| `-t, --tile` | Specific tile to bake (e.g., `development_31_31`) |
| `-o, --output` | Output directory (default: `<dataset>/baked`) |
| `--shadows` | Apply shadow mask overlay |
| `--export-layers` | Export individual texture layers |

**With Layer Export (for ViT training):**
```bash
dotnet run --project WoWMapConverter.Cli -- vlm-bake -d ./vlm_dataset --export-layers
```

**Output Structure:**
```
vlm_dataset/
├── baked/
│   ├── Map_X_Y_composite_noshadow.png   # Final composite without shadows
│   ├── Map_X_Y_composite_shadowed.png   # Final composite with shadows applied
│   └── Map_X_Y_layers/                  # Per-layer exports (ground truth data)
│       ├── raw/                         # Raw textures (no blending)
│       ├── weighted/                    # Texture × weight (alpha = weight)
│       ├── cumulative/                  # Progressive blend up to layer N (no shadow)
│       ├── shadowed/                    # Progressive blend + shadows applied
│       └── shadow_masks/                # Full tile shadow mask
```

**Layer Export Types (for ViT/ML Training):**
| Type | Description | Use Case |
|------|-------------|----------|
| **raw** | Original texture, no alpha | Texture classification |
| **weighted** | RGB × weight, Alpha = weight | Per-layer contribution |
| **cumulative** | Progressive composite (layers 0..N) | Blending progression |
| **shadowed** | Cumulative + shadow overlay | Final appearance learning |
| **shadow_masks** | Full tile shadow mask | Shadow prediction |

**Blending Algorithm:**
Uses WoW's weighted blend from `adt.fragment.shader`:
- Layer 0 weight = `1.0 - sum(layer1..N alphas)`
- Layer N weight = `alpha[N]`
- Final color = `sum(layer[i].rgb × weight[i])`

**Shadow Convention:**
- White (255) = transparent/no shadow
- Black (0) = opaque/full shadow

## Usage

```bash
# Single map conversion
dotnet run --project WoWMapConverter.Cli -- convert World/Maps/Azeroth/Azeroth.wdt -o ./output

# With Listfile (AreaID crosswalk is automated)
dotnet run --project WoWMapConverter.Cli -- convert map.wdt \
  --listfile community-listfile.csv

# Batch conversion
dotnet run --project WoWMapConverter.Cli -- batch --input-dir ./World/Maps -o ./output
```

## Migration from v2

This library consolidates:
- `src/gillijimproject-csharp/` - Original Alpha→LK converter
- `AlphaWDTAnalysisTool/` - Analysis and AreaID patching

The v2 code remains available for reference but new development should use v3.

## Dependencies

- .NET 9.0
- No external NuGet packages required for core library

## Building

```bash
dotnet build src/WoWMapConverter/WoWMapConverter.Core
dotnet build src/WoWMapConverter/WoWMapConverter.Cli
```
