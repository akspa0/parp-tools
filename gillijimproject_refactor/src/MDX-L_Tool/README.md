# MDX-L_Tool

**MDX-L_Tool** is a digital archaeology utility designed to parse, analyze, and convert 3D models from the **World of Warcraft Alpha 0.5.3** client. It focuses on preserving early Blizzard assets by bridging the gap between legacy formats and modern tools.


## Usage

The tool is a command-line utility. All outputs are saved to the specified output path, and the tool will automatically create any missing directories.

### Show Model Info
Displays metadata and sub-components of an MDX file.
```powershell
dotnet run -- info path/to/model.mdx
```

### Convert to MDL with Texture Export
Exports the binary model to a text-based MDL file and converts all referenced textures to PNG. Use `--game-path` to specify a WoW Alpha/Beta client directory for texture resolution from MPQs.

```powershell
dotnet run -- convert path/to/model.mdx mdx-l_outputs/model.mdl --game-path "H:\053-client"
```

### Convert to OBJ + MTL
Exports the model geometry and materials to Wavefront OBJ format. This is ideal for importing Alpha assets into modern 3D software.

```powershell
dotnet run -- convert path/to/model.mdx mdx-l_outputs/model.obj --game-path "H:\053-client"
```

### Batch Convert
Process an entire directory of models at once. Supports wildcards.

```powershell
# Convert all MDX files in a folder to OBJ
dotnet run -- batch "H:\053-client\Data\*.mdx" mdx-l_outputs/ --target obj --game-path "H:\053-client"
```

The tool will:
1. Parse the MDX geometry and metadata (using robust Alpha 0.5.3 logic).
2. Search for referenced `.blp` files (local first, then MPQ).
3. Convert BLPs to PNG and save them alongside the model file.
4. Export the geometry to `.obj` with proper material assignments.

## Features

- **Alpha 0.5.3 Support**: Native parsing of MDX v1300. Correctly handles:
    - `GEOS` sub-chunk structure (`Tag` -> `Count` -> `Data`).
    - `ReplaceableId: 11` (Creature Skin) -> Model Name mapping.
    - 1-byte `PTYP` and direct `UVAS` streams.
- **Batch Processing**: Convert thousands of models in a single run.
- **MDX to MDL**: Export to human-readable text format.
- **MDX to OBJ + MTL**: Export to industry-standard 3D format.
- **Auto Texture Export**: Resolves and converts BLP textures to PNG.

## Supported Chunks
- `VERS`: Format version validation.
- `MODL`: Model metadata and global bounds.
- `SEQS`: Animation sequences and intervals.
- `MTLS`/`TEXS`: Materials and textures.
- `GEOS`: Geometry, normals, and UV layouts (including 0.5.3 direct UVAS).
- `BONE`/`PIVT`: Bone hierarchies and joint pivots.
- `ATCH`: Attachment points (e.g., weapon mounts).
- `CAMS`: Camera definitions and targets.
- `LITE`/`HELP`: Light sources and helper nodes.

## Technical Details

The tool is built on research performed via **Ghidra** decompilation of the original `BinToModelData` and `ReadBinModelGlobals` functions in the Alpha client. It correctly implements the specific 0.5.3 quirks:
- Radius-first MODL bounds.
- Count-prefixed MTLS headers.
- Size-prefixed GEOS record skipping.
- 1-byte `PTYP` primitive types (fixing geometry misalignment).
- Direct `UVAS` coordinate streams.

## Roadmap
- [x] **Phase 1**: Core MDX Reader & MDL Writer.
- [x] **Phase 1.5**: Texture Export & MPQ Resolution.
- [ ] **Phase 2**: M2 Format Support (WotLK 3.3.5 / v264).
- [ ] **Phase 3**: Alpha-to-Retail Model Conversion.

## License
Part of the **parp-tools** suite for WoW history preservation.
