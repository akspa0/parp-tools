# MDL Writer Deep Dive Analysis

This document details the analysis of the "Hidden" MDL (Warcraft III Model) Writer found in the WoW Alpha 0.5.3 client.

## Overview
The client contains a complete library for serializing in-memory model data back to the disk-based `.mdl` (text) and potentially `.mdx` (binary) formats. This functionality is technically "Dead Code" (unreachable) in the retail build but constitutes a fully functional Model Converter/Exporter.

## Entry Points
| Function | Address | Description |
|:---|:---|:---|
| `MDLFileWrite` | `0078b300` | High-level API to write an MDL file. |
| `MDLFileBinaryWrite` | `0078bea0` | Binary (MDX?) writer variant. |
| `IWriteMdlFile` | `0078b370` | Core serialization logic. |

## Input Data Structures
The primary goal of this analysis is to determine **what input** the writer expects. Based on the function signatures and traversal, we will map the in-memory structures (CModel, CGeoset, etc.) that feed this writer.

---

## Detailed Analysis

### 1. Function Logic: `IWriteMdlFile`
Address: `0078b5d0`
This function serves as the orchestrator.
1.  **Format Detection**: Calls `DiscoverFileType`. If the extension is `.mdl`, it selects Text Mode. If `.mdx` (presumably), Binary Mode.
2.  **Dispatch**:
    - **Text Mode**: Allocates a `TSGrowableArray<char>` buffer and calls `ModelDataToText` (`0078b590`).
    - **Binary Mode**: Allocates a `CMsgBuffer` and calls `ModelDataToBin`.
3.  **File System**: Uses `IWriteFile` (Engine abstraction) to save the buffer to disk.

### 2. Input Structure: `MDLDATA` and `MDLGEOSET`
The writer operates on a high-level `MDLDATA` structure, which contains arrays of sub-components (Geosets, Bones, etc.).
Based on `IWriteGeosetSection` (`007a9680`), we can reconstruct the `MDLGEOSET` layout:

**`MDLGEOSET` Structure Layout:**
| Offset | Field Type | Description |
|:---|:---|:---|
| `0x00` | `TSGrowableArray<C3Vector>` | **Vertices**. (Size: 20 bytes). |
| `0x14` | `TSGrowableArray<C3Vector>` | **Normals**. |
| `0x28` | `TSGrowableArray<...>` | **Texture Coordinate Sets**. |
| `0x3C` | `TSGrowableArray<uint8>` | **Vertex Group Indices**. |
| `0x50` | `TSGrowableArray<Primitives>` | **Faces/Triangles**. |
| ... | | |
| `0xE0` | `CMdlBounds` | **Extent/Bounds**. |
| `0x110` | `int` | **MaterialID**. |
| `0x114` | `uint` | **Flags** (e.g., Unselectable). |

**Dependencies**:
- **NTempest**: The structure uses `NTempest::C3Vector` for geometry, confirming the `Tempest` math library usage.
- **TSGrowableArray**: A specific Storm/Template collection with a VTable (Size `0x14` / 20 bytes: `vptr`, `data`, `size`, `capacity`, `chunk`).

### 3. Serialization Tokens
The text writer uses `MDL::TokenText(id)` to retrieve format strings. Examples:
- `0x10a`: "Geoset"
- `0x1d8`: "Vertices"
- `0x17b`: "Normals"
- `0x1b1`: "MaterialID"

### 4. Implementation Details
The `ModelDataToText` function triggers a cascade of handlers via `CallTextWriteHandlers` (`0078c300`), which iterates a function pointer table at `008b2580`. There are 22 handlers, corresponding to the MDL sections (Version, Model, Sequences, Textures, Materials, etc.).

## Reconstruction Success (February 2026)
The theoretical analysis of the 0.5.3 model writer has been successfully validated through the implementation of **MDX-L_Tool**. The "Hidden" logic was successfully ported to C# and verified against original Alpha assets.

### Verified Structural Details
Development revealed several critical nuances in the 0.5.3 MDX format that differ from research based on later versions:

1.  **MODL Chunk Order**: Unlike retail WC3/WoW where Bounds come first, the 0.5.3 format places the `BoundsRadius` (float) *before* the `Min` and `Max` extent vectors. This was confirmed by matching the exact 373-byte (0x175) chunk size requirement found in Ghidra.
2.  **GEOS Record Format**: Geosets in 0.5.3 are not fixed-size. They use a custom sub-chunk system (VRTX, NRMS, TVTX, etc.) where each sub-chunk has a count prefix. Robust parsing requires skipping unknown tags using local size offsets.
3.  **MTLS and LAYS**: Materials in 0.5.3 do *not* use a `LAYS` tag to prefix their layer list. They contain a direct layer count followed by size-prefixed layer records.
4.  **SEQS Format**: Sequences use a fixed 140-byte (0x8C) record size.

### Implementation Reference
The core logic has been ported to:
- [MdxFile.cs](file:///j:/wowDev/parp-tools/gillijimproject_refactor/src/MDX-L_Tool/Formats/Mdx/MdxFile.cs): The binary parser based on `BinToModelData`.
- [MdlWriter.cs](file:///j:/wowDev/parp-tools/gillijimproject_refactor/src/MDX-L_Tool/Formats/Mdx/MdlWriter.cs): The text serializer based on `ModelDataToText`.

## Conclusion
The successful reactivation of this pipeline allows for the perfect preservation of Alpha 0.5.3 assets by providing a gateway to modern model editing suites.
