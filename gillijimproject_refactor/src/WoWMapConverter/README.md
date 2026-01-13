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
├── shadows/          # MCSH per-chunk (64×64)
├── masks/            # MCAL layer alphas
├── liquids/          # MH2O/MCLQ data
├── dataset/          # Structured JSON
└── texture_database.json
```

**Data Exported:**
- Heights (145 × 256 chunks), positions, holes
- Shadow maps (MCSH), alpha masks (MCAL)
- Texture layers (MCLY), liquids (MH2O/MCLQ)
- Object placements (MDDF/MODF)
- Compatible with DepthAnything3 for depth map correlation

**DepthAnything3 Setup (Optional):**
```bash
# Windows (run from Anaconda Prompt)
cd src/WoWMapConverter/WoWMapConverter.Core/VLM/DepthAnything3
setup_da3.bat

# Linux/WSL
./setup_da3.sh
```
Then use `--depth` flag to enable depth map generation.

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
