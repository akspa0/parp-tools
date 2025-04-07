# ADT Chunk Relationship Map

## Direct Dependencies

This graph shows how chunks directly depend on or reference each other:

```
MVER  <----  MHDR  <----+  MCIN
              |         |
              +-------->+  MTEX/MDID/MHID
              |         |
              +-------->+  MMDX  <----  MMID  <----  MDDF
              |         |
              +-------->+  MWMO  <----  MWID  <----  MODF
              |         |
              +-------->+  MFBO
              |         |
              +-------->+  MH2O
              |         |
              +-------->+  MTXF
              |         |
              +-------->+  MTXP  <-+
                           |       |
MCNK  -------->  MCLY  ---+       |
                |                 |
                +---------------->+
```

## Detailed Relationships

### Core Structure
- **MVER**: No dependencies, just specifies file version
- **MHDR**: Contains offsets to all other root chunks, depends on MVER

### Texture System
- **MTEX**: Contains texture filenames, referenced by MHDR
- **MDID**: (Legion+) Contains texture file IDs, replaces MTEX in newer versions
- **MNID**: (Legion+) Contains normal texture file IDs, paired with MDID
- **MSID**: (Legion+) Contains specular texture file IDs, paired with MDID
- **MLID**: (Legion+) Contains height texture file IDs, paired with MDID
- **MTXF**: Contains extended texture flags, one per texture in MTEX/MDID
- **MTXP**: Contains texture parameters, one per texture in MTEX/MDID

### Model System
- **MMDX**: Contains M2 model filenames
- **MMID**: Contains offsets into MMDX (indexes for model filenames)
- **MDDF**: References entries in MMID to place M2 models in the world

### WMO (World Model Object) System
- **MWMO**: Contains WMO filenames
- **MWID**: Contains offsets into MWMO (indexes for WMO filenames)
- **MODF**: References entries in MWID to place WMO models in the world
- **MWDR**: (Shadowlands+) Contains WMO doodad references using FileDataIDs
- **MWDS**: (Shadowlands+) Contains WMO doodad sets for grouped rendering

### Map Chunks and Terrain
- **MCIN**: Contains offsets to MCNK chunks
- **MCNK**: Contains terrain data and references to textures in MTEX/MDID
  - **MCVT**: Height map vertices for the chunk
  - **MCNR**: Normal vectors for lighting calculations
  - **MCLY**: Texture layer definitions

### Blend Mesh System (MoP+)
- **MBMH**: Blend mesh headers
- **MBBB**: Blend mesh bounding boxes
- **MBNV**: Blend mesh normal vectors
- **MBMI**: Blend mesh indices

### Additional Data
- **MFBO**: Flight boundaries for the map tile
- **MH2O**: Liquid/water data
- **MLDB**: (BfA+) LOD blend configurations

## Index Relationships

### Texture References
- **MCLY** (subchunk of MCNK): References textures by index in MTEX or MDID
- **MCLY** may reference texture parameters in MTXP for advanced rendering

### Model References
- **MDDF.nameId**: References an entry (by index) in MMID, which points to a filename in MMDX
- If MDDF has the flag `mddf_entry_is_filedata_id` set, nameId is a direct file ID instead
- **MCRF** (subchunk of MCNK): References models in MDDF via index for local doodad placement

### WMO References
- **MODF.nameId**: References an entry (by index) in MWID, which points to a filename in MWMO
- If MODF has the flag `IsFileId` set (0x2000), nameId is a direct file ID instead

## Common Patterns

1. **String Tables + Offsets Pattern**:
   - MMDX (strings) + MMID (offsets into MMDX)
   - MWMO (strings) + MWID (offsets into MWMO)
   - This pattern allows efficient lookup of string data

2. **Placement Pattern**:
   - MDDF (places M2 models) references MMID/MMDX
   - MODF (places WMO models) references MWID/MWMO
   - Both contain position, rotation, scale, and flags

3. **Version Evolution Pattern**:
   - MTEX (older) → MDID/MNID/MSID/MLID (Legion+)
   - String paths → File IDs
   - Additional metadata added in newer expansions (MTXF, MTXP)

4. **Layer System Pattern**:
   - MCLY contains multiple texture layers per MCNK
   - Each layer references a texture and contains alpha blending information
   - Layers are rendered in order from bottom to top with alpha blending

5. **Coordinate System Pattern**:
   - Global coordinates for ADT placement in the world
   - Local coordinates within each MCNK chunk
   - Transformation between coordinate systems required when placing objects

## Implementation Considerations

When implementing parsing for these chunks, we need to be careful about:

1. **Order of parsing**: Parse dependency chunks before dependent chunks
2. **Index validation**: Ensure indices are within bounds when referencing other chunks
3. **Version checks**: Some chunks are only present in specific game versions
4. **Split files vs. Monolithic files**: 
   - All ADT files use version 18 (MVER=18), regardless of expansion or file organization
   - In Cataclysm and later, ADT data is distributed across multiple files (still v18):
     - Root file (.adt): MVER, MHDR, MCIN, MH2O
     - Texture file (_tex0.adt): MTEX/MDID/MNID/MSID/MLID, MTXF, MTXP
     - Object file (_obj0.adt): MMDX, MMID, MWMO, MWID, MDDF, MODF, MWDR, MWDS
   - Pre-Cataclysm ADTs use a single file containing all chunks (also v18)
   - The v22/v23 formats only appeared in Cataclysm beta and were not used in final release

## Implementation Strategies

1. **Lazy Loading**: Load only the headers initially, then load chunk data as needed
   - Reduces memory footprint for large terrain files
   - Allows selective processing of only needed chunks

2. **Reference Resolution**:
   - Two-phase parsing: first load all chunks, then resolve references
   - Use dictionary lookups for efficient reference resolution
   - Create helper methods for common reference patterns (string table lookups)

3. **Version-Specific Handling**:
   - Use interface-based approach with expansion-specific implementations
   - Abstract factory pattern to create appropriate chunk parsers based on expansion
   - Fallback mechanisms for backward compatibility

4. **MCNK Processing Optimization**:
   - MCNKs are independent and can be processed in parallel
   - Batch processing of similar subchunks across multiple MCNKs
   - Height map calculations can be optimized with vector operations

5. **Cross-Format Integration**:
   - ADT + WDT: For full world map understanding
   - ADT + M2: For proper model placement
   - ADT + WMO: For proper structure placement
   - Consider building a unified world representation layer above format parsers 

# Format Relationship Maps

## ADT Format (Terrain)

### Direct Dependencies
```
MVER  <----  MHDR  <----+  MCIN
              |         |
              +-------->+  MTEX/MDID/MHID
              |         |
              +-------->+  MMDX  <----  MMID  <----  MDDF
              |         |
              +-------->+  MWMO  <----  MWID  <----  MODF
              |         |
              +-------->+  MFBO
              |         |
              +-------->+  MH2O
              |         |
              +-------->+  MTXF
              |         |
              +-------->+  MTXP  <-+
                           |       |
MCNK  -------->  MCLY  ---+       |
                |                 |
                +---------------->+
```

### MCNK Subchunks
```
MCNK  <----+  MCVT (Heights)
           |
           +--  MCNR (Normals)
           |
           +--  MCLY (Layers)  ---->  MTEX/MDID
           |
           +--  MCRF (Refs)  ----->  MDDF/MODF
           |
           +--  MCSH (Shadows)
           |
           +--  MCAL (Alpha)
           |
           +--  MCLQ (Liquid)
           |
           +--  MCSE (Sound)
```

## PM4 Format (Phased Model)

### Core Dependencies
```
MVER (48) <---- MSHD <----+  MSPV (Vertices)  <----  MSPI (Indices)
                |         |
                +-------->+  MSUR (Unknown)  <----  MSLK (Links)
                |         |
                +-------->+  MSVT (Vertices)  <----  MSVI (Indices)
                |         |
                +-------->+  MSCN (Unknown)
                |         |
                +-------->+  MPRL (Unknown)  <----  MPRR (Unknown)
                |         |
                +-------->+  MDBH (Unknown)  <----  MDOS (Unknown)
                |         |                         |
                |         |                         +----  MDSF (Unknown)
```

### Geometry System
```
MSPV (Vertices)  <----+  MSPI (Indices)
                      |
                      +--  MSLK (Links)
                      |
MSVT (Vertices)  <----+  MSVI (Indices)
```

## Common Patterns Between Formats

1. **Header + Dependencies**
   - ADT: MVER -> MHDR -> Content Chunks
   - WMO: MVER -> MOHD -> Content Chunks
   - M2: MD20/MD21 -> Content Chunks
   - PM4: MVER -> MSHD -> Content Chunks

2. **Geometry Organization**
   - ADT: MCNK contains MCVT (heights) + MCNR (normals)
   - WMO: MOGP contains MOVT (vertices) + MONR (normals)
   - M2: Vertices + Normals in main chunk
   - PM4: MSPV/MSVT for vertices, indices in MSPI/MSVI

3. **Reference Systems**
   - ADT: MMID/MWID for model/WMO references
   - WMO: MODD/MODN for doodad references
   - M2: Bone/texture references in main chunk
   - PM4: MSLK for vertex link references

4. **Asset Management**
   - ADT: MTEX/MDID for textures
   - WMO: MOTX for textures
   - M2: Texture references in main chunk
   - PM4: No direct texture references observed

## Implementation Considerations

1. **Loading Order**
   - ADT: MVER -> MHDR -> Referenced Chunks -> MCNK Subchunks
   - WMO: MVER -> MOHD -> Group Chunks -> Subchunks
   - M2: MD20/MD21 -> Bones -> Vertices -> Animations
   - PM4: MVER -> MSHD -> Vertex Data -> Link Data

2. **Validation Requirements**
   - ADT: Terrain consistency, texture references, model placement
   - WMO: Group consistency, portal validity, doodad placement
   - M2: Bone hierarchy, animation ranges, texture coordinates
   - PM4: Vertex indices, link validity, size constraints

3. **Version Handling**
   - ADT: Pre/Post-Cataclysm differences, FileDataID transitions
   - WMO: Legacy (v14) vs Modern (v17) formats
   - M2: Multiple format versions with different features
   - PM4: Single version (48) observed

4. **Memory Management**
   - ADT: Chunk-based loading, terrain streaming
   - WMO: Group-based loading, portal culling
   - M2: Animation state management, bone transforms
   - PM4: Vertex/index buffer management