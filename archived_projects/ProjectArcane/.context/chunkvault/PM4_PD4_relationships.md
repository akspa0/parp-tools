# PM4/PD4 Chunk Relationship Map

## Direct Dependencies

This graph shows how chunks directly depend on or reference each other:

```
MVER  <----  MSHD
              
MSPV  <----  MSPI  <----  MSLK
              
MSVT  <----  MSVI  <----  MSUR
              
MDBH  ---+---  MDBF
         |
         +---  MDBI

MPRL  <----  MPRR
```

## Detailed Relationships

### Core Structure
- **MVER**: No dependencies, just specifies file version
- **MSHD**: Contains header information, depends on MVER

### MSP System (Point System)
- **MSPV**: Contains MSP vertices (C3Vectori array)
- **MSPI**: Contains indices into MSPV for referencing vertices
- **MSLK**: References entries in MSPI via MSPI_first_index and MSPI_index_count

### MSV System (Vertex System)
- **MSVT**: Contains vertex data (C3Vectori array) with special YXZ ordering
- **MSVI**: Contains indices into MSVT for referencing vertices
- **MSUR**: References entries in MSVI via MSVI_first_index

### Destructible Building System
- **MDBH**: Contains metadata about destructible buildings
  - **MDBF**: Contains filenames for destructible buildings
  - **MDBI**: Contains indices for destructible buildings

### Position and Reference System
- **MPRL**: Contains position and reference data
- **MPRR**: References entries in MPRL

### Additional Chunks
- **MSCN**: Contains standalone vector data, not directly related to MSPV/MSLK
- **MDOS**: Object data with no direct references to other chunks
- **MDSF**: Structure data with no direct references to other chunks
- **MCRC**: (PD4 only) Contains CRC data, no references to other chunks

## Index Relationships

### MSP References
- **MSPI**: Contains indices into MSPV array
  - Documentation states: `uint32_t msp_indices[]; // index into #MSPV`
- **MSLK.MSPI_first_index**: References starting index in MSPI array
- **MSLK.MSPI_index_count**: Specifies how many MSPI entries to use

### MSV References
- **MSVI**: Contains indices into MSVT array
  - Documentation states: `uint32_t msv_indices[]; // index into #MSVT`
- **MSUR.MSVI_first_index**: References starting index in MSVI array
- **MSUR._0x01**: Specifies count of indices in MSVI

### Building References
- **MDBH**: Contains MDBF and MDBI as embedded chunks
  - Structure: `struct { CHUNK index; CHUNK filename[3]; } m_destructible_building_header[count];`

## Common Patterns

1. **Vertex/Index Pattern**:
   - MSPV (vertices) + MSPI (indices)
   - MSVT (vertices) + MSVI (indices)
   - This pattern allows efficient reuse of vertex data in multiple geometric constructs

2. **Index Range Pattern**:
   - MSLK references a range in MSPI via first_index + count
   - MSUR references a range in MSVI via first_index + count
   - This allows multiple logical objects to share the same index array

3. **Subchunk Pattern**:
   - MDBH contains MDBF and MDBI as subchunks
   - This enables hierarchical data organization

4. **Coordinate System Pattern**:
   - MSVT uses a special YXZ ordering (rather than XYZ)
   - Requires transformation: 
     - `worldPos.y = 17066.666 - position.y;`
     - `worldPos.x = 17066.666 - position.x;`
     - `worldPos.z = position.z / 36.0f;`

## Implementation Considerations

When implementing parsing for these chunks, we need to be careful about:

1. **Order of parsing**: Parse dependency chunks before dependent chunks
   - MSPV before MSPI before MSLK
   - MSVT before MSVI before MSUR

2. **Index validation**: 
   - Ensure indices in MSPI are within bounds of MSPV array
   - Ensure indices in MSVI are within bounds of MSVT array
   - Validate MSLK.MSPI_first_index + MSLK.MSPI_index_count <= MSPI.length
   - Validate MSUR.MSVI_first_index + MSUR._0x01 <= MSVI.length

3. **Coordinate transformation**:
   - MSVT vertex coordinates require special transformation
   - Account for the YXZ ordering and scaling factors

4. **Embedded chunk handling**:
   - MDBH contains embedded chunks that need their own parsing logic
   - Maintain proper offset tracking when parsing embedded chunks

5. **PD4 vs PM4 similarities**:
   - Many chunks (MVER, MSHD, MSPV, MSPI, MSCN, MSLK, MSVT, MSVI, MSUR) are shared between formats
   - Some chunks are format-specific (MPRL, MPRR, MDBH, MDOS, MDSF for PM4; MCRC for PD4)

## Implementation Strategies

1. **Common Parsing Framework**:
   - Create a shared parsing system for chunks common to both formats
   - Implement format-specific extensions for unique chunks

2. **Geometry Processing**:
   - Implement vertex and index buffer creation for efficient rendering
   - Support the coordinate system transformation required by MSVT

3. **Reference Resolution**:
   - Two-phase parsing approach: first load all chunks, then resolve references
   - Create helper methods for index range lookups

4. **Embedded Chunk Handling**:
   - Implement recursive chunk parsing for MDBH and its subchunks
   - Maintain proper scoping and context during parsing

5. **Coordinate System Handling**:
   - Implement utility functions for the MSVT coordinate transformations
   - Document the world coordinate system relationship 