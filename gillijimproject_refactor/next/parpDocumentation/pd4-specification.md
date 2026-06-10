# PD4 File Format Specification

This document provides a comprehensive specification of the PD4 file format, reverse-engineered from the parpToolbox codebase. PD4 files are a single-WMO scoped variant of PM4, used for phasing/mesh/scene data tied to individual WMO structures. Like PM4, PD4 is structured as an IFF-style chunked file with signatures such as MVER, MSHD, MSLK, MSUR, etc. The key difference is the addition of an MCRC chunk (checksum placeholder, typically 0). PD4 inherits the PM4 chunk set but is parsed via dedicated classes like PD4File, which extends PM4File in some implementations.

The specification follows the same format as [pm4-specification.md](pm4-specification.md), with additions for PD4-specific elements. Chunks are optional but follow a typical order. For shared chunks (MSLK, MSUR, etc.), refer to the PM4 spec.

## Overall File Structure
PD4 files start with:
1. MVER (version, 4 bytes)
2. MCRC (checksum, 4 bytes; always 0 in observed files)
3. MSHD (header, 32 bytes)
4. MSLK (linkage entries, variable)
5. MSPI (path indices, variable)
6. MSPV (path vertices, variable)
7. MSVT (mesh vertices, variable; often absent)
8. MSVI (mesh indices, variable)
9. MSUR (surface definitions, variable)
10. MSCN (exterior vertices, variable)
11. ... (other PM4 chunks: MSRN, MPRL, MPRR; optional destructibles: MDBH, MDOS, MDSF)

Total size: ~100KB-10MB. PD4 is a PM4 superset, parsed via PD4File for WMO-like phasing. Validation: Inherits PM4 checks (e.g., MSVI indices bound to MSVT).

### C Struct for Chunk Header (Common to All)
```c
// IFF-style chunk header (8 bytes)
struct ChunkHeader {
    char signature[4];  // e.g., "MCRC" (reversed on disk)
    uint32_t size;      // Payload size (excluding header)
};
```

## PD4-Specific Chunk: MCRC (4 bytes)
**Purpose**: Checksum placeholder for the PD4 file (single uint32, always 0 in all observed samples). Likely a legacy or reserved field for integrity validation.

**Plain English**: MCRC provides file verification but is unused (value=0). Code reads it as a simple uint32; exporters ignore it. Present in all PD4 files post-MVER.

**C Struct**:
```c
struct MCRC {
    uint32_t checksum;  // Always 0
};
```

**Usage in Code**: `MCRCChunk.Load` reads the uint32; `PD4File.MCRC` property. In PM4FacesTool/PM4NextExporter, skipped during parsing but logged in diagnostics if non-zero.

## Shared Chunks (Inherited from PM4)
PD4 uses the same chunks as PM4 for geometry and linkage:
- **MSHD**: Header (32 bytes, 8 uint32 unknowns).
- **MSLK**: Linkage (20 bytes/entry; object metadata, tile crossing via LinkId=0xFFFFYYXX).
- **MSUR**: Surfaces (32 bytes/entry; GroupKey=0 for M2 props, non-zero for walkable; references MSVI indices).
- **MSCN**: Exterior vertices (12 bytes/vertex, Vector3 XYZ).
- **MSVI**: Mesh indices (4 bytes/index, uint into MSVT).
- **MSPV/MSPI**: Path vertices/indices (12/4 bytes; for non-geometry paths via MSLK).
- **MSVT**: Mesh vertices (12 bytes/vertex; often absent, use MSPV; transform: offset-X/Y, Z per PD4.md).
- **Others**: MSRN (normals, optional), MPRL/MPRR (positions/references, TBD), MDBH/MDOS/MDSF (destructibles, via MSUR.MdosIndex).

For detailed structs and usage, see [pm4-specification.md](pm4-specification.md). PD4-specific notes:
- Transformations: MSVT uses PD4 offset (17066.666f for X/Y); Z unscaled (per PD4FileTests).
- Parsing: `PD4File.FromFile(path)` loads via PD4ChunkFactory; inherits PM4 validation.

## Data Arrangement and Usage in Priority Tools
PD4 forms a scene graph like PM4: MSLK nodes link MSUR meshes (MSVI→MSVT) and paths (MSPI→MSPV); MSCN bounds. Scoped to single WMO, no cross-tile LinkId usage.

- **PM4FacesTool**: Treats PD4 as PM4 (via PD4File : PM4File); groups MSUR via DSU on shared verts; exports OBJ/GLTF with transforms. Handles MCRC implicitly.
- **PM4NextExporter**: `--assembly` strategies (e.g., composite-hierarchy) apply to PD4; `--export-mscn-obj` for exterior verts. CLI: `pm4next-export file.pd4 [options]`.
- **parpDataHarvester**: Batch ingestion via `Pm4BatchProcessor.Process` (PD4 as PM4 superset); CSV/JSON output includes MCRC=0.

This spec enables unified PM4/PD4 parsers; validate with PD4FileTests (e.g., chunk presence, MSVT transforms).