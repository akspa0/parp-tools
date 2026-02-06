# MDX-L_Tool

**MDX-L_Tool** is a digital archaeology utility designed to parse, analyze, and convert 3D models from the **World of Warcraft Alpha 0.5.3** client. It focuses on preserving early Blizzard assets by bridging the gap between legacy formats and modern tools.

## Features

- **Alpha 0.5.3 Support**: Native parsing of the specific MDX (version 1300) format used in the 2003 WoW Alpha.
- **MDX to MDL Conversion**: Export binary models to human-readable Warcraft III MDL text format.
- **Structural Analysis**: Detailed CLI output for model metadata, geosets, bones, and animation sequences.
- **Robust Parsing**: Advanced chunk-skipping logic to handle malformed or unknown data without hanging.

## Usage

The tool is a command-line utility.

### Show Model Info
Displays metadata and sub-components of an MDX file.
```powershell
dotnet run -- info path/to/model.mdx
```

### Convert to MDL
Exports the binary model to a text-based MDL file.
```powershell
dotnet run -- convert path/to/model.mdx path/to/output.mdl
```

## Supported Chunks
- `VERS`: Format version validation.
- `MODL`: Model metadata and global bounds.
- `SEQS`: Animation sequences and intervals.
- `MTLS`/`TEXS`: Materials and textures.
- `GEOS`: Geometry, normals, and UV layouts.
- `BONE`/`PIVT`: Bone hierarchies and joint pivots.
- `ATCH`: Attachment points (e.g., weapon mounts).
- `CAMS`: Camera definitions and targets.
- `LITE`/`HELP`: Light sources and helper nodes.

## Technical Details

The tool is built on research performed via **Ghidra** decompilation of the original `BinToModelData` and `ReadBinModelGlobals` functions in the Alpha client. It correctly implements the specific 0.5.3 quirks:
- Radius-first MODL bounds.
- Count-prefixed MTLS headers.
- Size-prefixed GEOS record skipping.
- 140-byte fixed-size SEQS records.

## Roadmap
- [x] **Phase 1**: Core MDX Reader & MDL Writer (Complete).
- [ ] **Phase 2**: M2 Format Support (WotLK 3.3.5 / v264).
- [ ] **Phase 3**: Alpha-to-Retail Model Conversion.

## License
Part of the **parp-tools** suite for WoW history preservation.
