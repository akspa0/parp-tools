# PM4 and PD4 File Formats Documentation

## Overview
PM4 (Player Model) and PD4 (Player Data) are file formats used in World of Warcraft for server-side supplementary files to ADTs and WMOs respectively. These formats contain information used by the server and are not shipped to the client.

## File Structure
Both PM4 and PD4 formats follow the standard chunk-based structure used in many World of Warcraft file formats:

1. **Header**: Contains a magic identifier and version information
2. **Chunks**: A series of data chunks, each with a 4-character identifier, size, and data block
3. **Optional Padding**: Some chunks may include padding for alignment

## PM4 Format Chunks
| Chunk | Name | Description |
|-------|------|-------------|
| `MVER` | Version | Version information (48 in version 6.0.1.18297, 6.0.1.18443) |
| `MSHD` | Header | Header data |
| `MSPV` | MSP Vertices | Array of MSP (Mesh Shape Point) vertices |
| `MSPI` | MSP Indices | Array of indices into MSPV |
| `MSCN` | Normal Vectors | Array of normal vectors |
| `MSLK` | Links | Link definitions using MSPI indices |
| `MSVT` | Vertices | Array of vertices with special YXZ ordering |
| `MSVI` | Vertex Indices | Array of indices into MSVT |
| `MSUR` | Surface Definitions | Surface definitions using MSVI indices |
| `MPRL` | Position Data | Array of position data |
| `MPRR` | Reference Data | Array of reference data |
| `MDBH` | Destructible Building Header | Container for MDBF and MDBI subchunks |
| `MDBF` | Destructible Building Filename | Filenames for destructible buildings |
| `MDBI` | Destructible Building Index | Indices for destructible buildings |
| `MDOS` | Object Data | Array of object data |
| `MDSF` | Structure Data | Array of structure data |

## PD4 Format Chunks
| Chunk | Name | Description |
|-------|------|-------------|
| `MVER` | Version | Version information (48 in version 6.0.1.18297, 6.0.1.18443) |
| `MCRC` | CRC | CRC data |
| `MSHD` | Header | Header data |
| `MSPV` | MSP Vertices | Array of MSP (Mesh Shape Point) vertices |
| `MSPI` | MSP Indices | Array of indices into MSPV |
| `MSCN` | Normal Vectors | Array of normal vectors |
| `MSLK` | Links | Link definitions using MSPI indices |
| `MSVT` | Vertices | Array of vertices with special YXZ ordering |
| `MSVI` | Vertex Indices | Array of indices into MSVT |
| `MSUR` | Surface Definitions | Surface definitions using MSVI indices |

## Implementation Status
| Format | Chunk | Status | File |
|--------|-------|--------|------|
| PM4 | MVER - Version | ✅ Implemented | chunks/PM4/M001_MVER.md |
| PM4 | MSPV - MSP Vertices | ✅ Implemented | chunks/PM4/M002_MSPV.md |
| PM4 | MSPI - MSP Indices | ✅ Implemented | chunks/PM4/M003_MSPI.md |
| PM4 | MSLK - Links | ✅ Implemented | chunks/PM4/M004_MSLK.md |
| PM4 | MSVT - Vertices | ✅ Implemented | chunks/PM4/M005_MSVT.md |
| PM4 | MSVI - Vertex Indices | ✅ Implemented | chunks/PM4/M006_MSVI.md |
| PM4 | MSUR - Surface Definitions | ✅ Implemented | chunks/PM4/M007_MSUR.md |
| PM4 | MSCN - Normal Vectors | ✅ Implemented | chunks/PM4/M008_MSCN.md |
| PM4 | MSHD - Header | ✅ Implemented | chunks/PM4/M009_MSHD.md |
| PM4 | MPRL - Position Data | ✅ Implemented | chunks/PM4/M010_MPRL.md |
| PM4 | MPRR - Reference Data | ✅ Implemented | chunks/PM4/M011_MPRR.md |
| PM4 | MDBH - Destructible Building Header | ✅ Implemented | chunks/PM4/M012_MDBH.md |
| PM4 | MDOS - Object Data | ✅ Implemented | chunks/PM4/M013_MDOS.md |
| PM4 | MDSF - Structure Data | ✅ Implemented | chunks/PM4/M014_MDSF.md |
| PD4 | MVER - Version | ✅ Same as PM4 | chunks/PM4/M001_MVER.md |
| PD4 | MSPV - MSP Vertices | ✅ Same as PM4 | chunks/PM4/M002_MSPV.md |
| PD4 | MSPI - MSP Indices | ✅ Same as PM4 | chunks/PM4/M003_MSPI.md |
| PD4 | MSLK - Links | ✅ Same as PM4 | chunks/PM4/M004_MSLK.md |
| PD4 | MSVT - Vertices | ✅ Same as PM4 | chunks/PM4/M005_MSVT.md |
| PD4 | MSVI - Vertex Indices | ✅ Same as PM4 | chunks/PM4/M006_MSVI.md |
| PD4 | MSUR - Surface Definitions | ✅ Same as PM4 | chunks/PM4/M007_MSUR.md |
| PD4 | MSCN - Normal Vectors | ✅ Same as PM4 | chunks/PM4/M008_MSCN.md |
| PD4 | MSHD - Header | ✅ Same as PM4 | chunks/PM4/M009_MSHD.md |
| PD4 | MCRC - CRC | ✅ Implemented | chunks/PD4/P001_MCRC.md |

## Relationship to Other Formats
PM4 and PD4 formats relate to other World of Warcraft file formats:

1. **PM4**: Server-side supplementary files to ADTs, containing additional information for server-side operations not needed by the client.

2. **PD4**: Server-side supplementary files to WMOs, similarly containing information exclusively used by the server.

## Key Features
1. **Coordinate Systems**: Special coordinate system with YXZ ordering in MSVT, requiring specific transformation formulas for proper positioning.

2. **Vertex and Index Structure**: Two separate vertex-index systems (MSPV/MSPI and MSVT/MSVI) for different geometric representations.

3. **Range Reference Pattern**: MSLK and MSUR chunks reference ranges of indices rather than individual vertices.

4. **Destructible Building System**: MDBH chunk contains subchunks for destructible building information.

5. **Position Reference System**: MPRL and MPRR chunks form a position and reference system similar to the vertex-index pattern.

6. **Object and Structure Data**: MDOS and MDSF chunks contain data related to objects and structures in the game world.

7. **File Integrity Verification**: PD4 format includes a MCRC chunk for CRC-based integrity checking.

## Next Steps
1. Create a unified parsing framework for both PM4 and PD4 formats
2. Implement the full file parser with chunk registry
3. Develop a sample application to view and edit PM4/PD4 files
4. Write unit tests to ensure correct parsing and writing of all chunks
5. Create documentation for the overall file parsing system

## Implementation Considerations
1. **Coordinate Transformation**: Implement proper coordinate transformation for MSVT vertices.
2. **Index Validation**: Ensure indices in MSPI, MSVI, and MPRR are within bounds of their respective referenced arrays.
3. **Packed Values**: Handle packed values like the 24-bit signed + 8-bit unsigned in MSLK properly.
4. **Common Parsing Framework**: Implement common parsing code for chunks shared between PM4 and PD4.
5. **Surface Construction**: Handle the surface definitions from MSUR correctly, particularly the way they reference MSVI index ranges.
6. **Geometry System Interaction**: Understand how the two geometric systems (MSPV/MSPI and MSVT/MSVI) relate and interact with each other. 
7. **Normal Vector Integration**: Associate normal vectors from MSCN with the appropriate geometric elements.
8. **Position Reference System**: Handle the position reference system in MPRL/MPRR correctly.
9. **Embedded Chunks**: Implement proper parsing for embedded chunks in the MDBH structure.
10. **CRC Calculation**: Calculate CRC values correctly in PD4 files, excluding the MCRC chunk itself from the calculation.
11. **Error Handling**: Implement robust error handling for corrupted or invalid files.
12. **Version Compatibility**: Ensure the parser can handle potential future versions of the formats. 