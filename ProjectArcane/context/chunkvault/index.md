# World Format Documentation Index

## Format Overview

| Format | Status | Main Chunks | Subchunks | Notes |
|--------|--------|-------------|-----------|-------|
| ADT v18 | ✅ Complete | 18/18 (100%) | 23/23 (100%) | Complete documentation |
| ADT v22 | ✅ Complete | 16/16 (100%) | 0/0 (N/A) | Complete documentation |
| ADT v23 | ✅ Complete | 8/8 (100%) | 4/4 (100%) | Complete documentation |
| WDT | 📝 Planned | 0/6 (0%) | 0/0 (N/A) | Planned for future |
| WDL | 📝 Planned | 0/4 (0%) | 0/0 (N/A) | Planned for future |
| WMO | 🔄 In Progress | 10/17 (59%) | 4/14 (29%) | Documentation in progress |
| M2 | 📝 Planned | 0/20 (0%) | 0/0 (N/A) | Planned for future |
| DBC/DB2/ADB | 📝 Planned | 0/3 (0%) | 0/0 (N/A) | Database formats planned |

## Recent Updates

- **2025-03-24**: Documented three additional WMO group file chunks (MOGP, MOPY, MOVI)
- **2023-06-25**: Began documentation of WMO format, completed 10 main root chunks
- **2023-06-20**: Completed documentation of all ADT v23 format chunks (12/12)
- **2023-06-15**: Completed documentation of ADT v22 format
- **2023-06-10**: Completed documentation of ADT v18 format
- **2023-06-01**: Created ChunkVault structure and initial format organization

## Implementation Progress

| Format | Documented | Implemented | Notes |
|--------|------------|-------------|-------|
| ADT v18 | 41/41 (100%) | 25/41 (61%) | Core implementation complete |
| ADT v22 | 16/16 (100%) | 0/16 (0%) | Planned for Q3 2023 |
| ADT v23 | 12/12 (100%) | 0/12 (0%) | Planned for Q3 2023 |
| WDT | 0/6 (0%) | 0/6 (0%) | Planned for Q4 2023 |
| WDL | 0/4 (0%) | 0/4 (0%) | Planned for Q4 2023 |
| WMO | 21/31 (68%) | 0/31 (0%) | Documentation in progress |
| M2 | 0/20 (0%) | 0/20 (0%) | Planned for Q1 2024 |
| DBC/DB2/ADB | 0/3 (0%) | 0/3 (0%) | Planned for Q1 2024 |

## Documentation Guidelines

Each chunk should include:

### Identification
- **Chunk ID**: Four-character identifier
- **Type**: Main chunk, subchunk, or special role
- **Source**: Original format documentation reference

### Structure Definition
- C# structure definition with field offsets and types
- Complete field definitions including arrays, nested types, and sizes
- Comments for each field explaining its purpose

### Properties Table
- Complete list of fields with types and detailed descriptions
- Flag definitions where applicable
- Enumerations of possible values

### Dependencies
- List of related chunks that this chunk depends on
- Description of how the dependencies work (references, counts, etc.)

### Implementation Notes
- Special handling requirements
- Version differences
- Endianness considerations
- Alignment requirements
- Size constraints

### Usage Context
- How the chunk is used in the game client
- Purpose and significance of the data
- Relationship to game mechanics or visuals

## Next Steps

1. Document remaining WMO group file chunks
2. Implement parsers for WMO root chunks
3. Implement parsers for WMO group chunks
4. Continue implementation of ADT v18 format
5. Implement parsers for ADT v22 and v23 formats
6. Add serialization support for all documented formats
7. Begin documentation of WDT format
8. Begin documentation of WDL format
9. Develop terrain rendering utilities
10. Create diagrams showing relationships between different formats
11. Document M2 model format

## Contributors

- **Arcane Preservation Team** - Documentation and implementation

## Documentation Structure

Each chunk is documented using the following structure:

1. **Identification**
   - Type: Format and chunk type
   - Source: Reference to original documentation

2. **Structure Definition**
   - C# struct representation
   - Detailed field descriptions

3. **Properties Table**
   - Offset, name, type, and description for each field
   - Special values and flags documented

4. **Dependencies**
   - Relationships with other chunks
   - Order constraints

5. **Implementation Notes**
   - Special handling requirements
   - Version differences
   - Edge cases

6. **Implementation Example**
   - C# code for reading/writing the chunk
   - Helper methods

7. **Validation Requirements**
   - Consistency checks
   - Value constraints

8. **Usage Context**
   - How the chunk is used in the game
   - Practical applications

## Common Chunks Across Formats

Several chunks appear in multiple formats with similar structure:

| Chunk ID | Formats | Description |
|----------|---------|-------------|
| MVER | ADT, WDT, WDL, WMO, M2 | Version information |
| MHDR/MOHD | ADT, WMO | Header information |
| MCNK/ACNK/MOGP | ADT, WMO | Main container for geometry |
| MCVT/ACVT/MOVT | ADT, WMO | Vertex positions |
| MCNR/ANRM/MONR | ADT, WMO | Vertex normals |
| MTEX/ATEX/MOTX | ADT, WMO | Texture filenames | 