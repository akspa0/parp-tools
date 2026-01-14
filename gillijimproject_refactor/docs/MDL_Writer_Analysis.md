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

## Reactivation Strategy
To use this writer:
1.  **Locate `MDLDATA`**: Find where the engine stores the loaded model. It is likely the `CModel` class or easily convertible to `MDLDATA`.
2.  **Injection**: Create a DLL that calls `MDLFileWrite`.
3.  **Input**: Pass the pointer to the in-memory `MDLDATA` object and a target path.
4.  **Result**: The game will dump a perfectly valid `.mdl` file, which can be opened in text editors or converted to modern formats.

## Conclusion
The `MDL Writer` is a robust, production-ready serialization pipeline left dormant in the binary. It expects a structured C++ object (`MDLDATA`) populated with `NTempest` vectors and standard primitive arrays. Its presence confirms the Alpha client's lineage from the Warcraft III internal tooling suite.
