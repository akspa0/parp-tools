# MDX-L_Tool

**MDX-L_Tool** is a digital archaeology utility designed to parse, analyze, and convert 3D models from the **World of Warcraft Alpha 0.5.3** client. It focuses on preserving early Blizzard assets by bridging the gap between legacy formats and modern tools.

## Features

- **Alpha 0.5.3 Support**: Native parsing of the specific MDX (version 1300) format used in the 2003 WoW Alpha.
- **MDX to MDL Conversion**: Export binary models to human-readable Warcraft III MDL text format.
- **Automatic Texture Export**: Resolves BLP textures from game archives or local folders and converts them to PNG.
- **Structural Analysis**: Detailed CLI output for model metadata, geosets, bones, and animation sequences.
- **Pure C# MPQ Reader**: Internal service to extract assets directly from early WoW MPQ archives.
- **Robust Parsing**: Advanced chunk-skipping logic to handle malformed or unknown data without hanging.

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

The tool will:
1. Parse the MDX geometry and metadata.
2. Search for referenced `.blp` files in the model's directory.
3. If not found, search the game MPQs (if `--game-path` is provided).
4. Convert BLPs to PNG and save them alongside the MDL.
5. Update the MDL's texture references to point to the new `.png` files.

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
