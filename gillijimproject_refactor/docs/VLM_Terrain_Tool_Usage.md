# VLM Terrain Data Export Tool

This tool exports terrain data from WoW client archives for Vision-Language Model (VLM) training.
It generates a dataset containing:
- **Minimap Images**: PNG format.
- **Metadata**: JSON format describing textures, objects, and terrain height ranges.
- **Terrain Meshes**: OBJ/MTL format representing the terrain geometry.

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

### Output Structure

The output directory will contain:

```
<OUTPUT_DIR>/
├── images/
│   ├── <map>_<x>_<y>.png       # Minimap tile
│   └── ...
├── meshes/
│   ├── <map>_<x>_<y>.obj       # Terrain mesh
│   ├── <map>_<x>_<y>.mtl       # Material file
│   └── ...
└── metadata/
    ├── <map>_<x>_<y>.json      # Metadata (includes "mesh_path")
    └── ...
```

### Example

To export the **Development** map from a 3.3.5 client to `C:\VLM_Dataset`:

```bash
dotnet run --project WoWRollback/WoWRollback.Cli/WoWRollback.Cli.csproj -- vlm-export --client-path "C:\Games\WoW335" --map Development --out "C:\VLM_Dataset"
```

## Features

- **Terrain Mesh Export**: Converts ADT heightmaps into OBJ files.
  - Handles **holes** (low-res) correctly using WotLK/ADTPreFabTool logic.
  - Generates UVs matching the minimap tile.
- **Metadata Integration**: The generated `metadata/*.json` files include a `mesh_path` field pointing to the corresponding OBJ file.
- **Minimap Resolution**: Automatically resolves minimap textures using `Md5Translate` logic if available, or standard path conventions.
