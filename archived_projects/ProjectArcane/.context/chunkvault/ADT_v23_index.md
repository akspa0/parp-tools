# ADT v23 Format Documentation

## Historical Significance
The ADT v23 format was an experimental terrain format developed during the Cataclysm beta period. Unlike v22, which was also experimental but shipped on certain pre-release builds, v23 was never used in any publicly available version of the game. It represents Blizzard's continued exploration of terrain format improvements, particularly focused on performance optimizations, memory usage reductions, and rendering enhancements. Though never used in a retail release, the format provides valuable insights into the technical directions being explored during this development period.

## Main Chunks

| ID | Name | Description | Status |
|----|------|-------------|--------|
| C001 | AHDR | ADT Header - Defines overall size and structure | ✅ |
| C002 | AVTX | Vertex Height Data - Global height information for terrain | ✅ |
| C003 | ANRM | Normal Map Data - Global normal vectors for lighting | ✅ |
| C004 | ATEX | Texture List - Registry of texture filenames | ✅ |
| C005 | ADOO | Doodad/Object List - Registry of model filenames | ✅ |
| C006 | ACNK | Chunk Data - Container for terrain chunk information | ✅ |
| C007 | AFBO | Flight Box Data - Defines flyable boundaries | ✅ |
| C008 | ACVT | Vertex Shading Data - Global vertex color information | ✅ |

## ACNK Subchunks

| ID | Name | Description | Status |
|----|------|-------------|--------|
| S001 | ALYR | Alpha Layer - Texture layer definition | ✅ |
| S002 | AMAP | Alpha Map - Texture blending data | ✅ |
| S003 | ASHD | Shadow Map - Shadow data for terrain | ✅ |
| S004 | ACDO | Chunk Doodad/Object - Model placements in chunk | ✅ |

## Format Relationships
- Descends from ADT v22, continuing experimentation with format changes
- Never used in any retail version of the game
- Some concepts later influenced ADT formats used in retail (v18+)

## Key Additions from ADT v22
- AFBO and ACVT chunks for better flight boundaries and vertex coloration
- Global storage of vertex-related data (heights, normals, colors)
- More efficient referencing system for models and textures
- Enhanced texture layer system with more detailed control
- Terrain data organization structured around chunk-specific subchunks

## Implementation Notes
1. This format continues the experimental nature of v22 but pushes further toward global data storage for certain elements
2. The global vertex data storage (AVTX, ANRM, ACVT) was designed to reduce memory duplication
3. The subchunk approach organizes data more clearly by terrain chunk
4. The format generally favors memory efficiency over file size efficiency
5. Parsing requires careful handling of the nested chunk structure

## Key Design Experiments in v23

1. **Global Vertex Data Storage**
   - Centralization of vertex heights (AVTX), normals (ANRM), and colors (ACVT)
   - Potential memory savings through reduced duplication of shared vertices
   - Improved data locality for rendering operations

2. **Unified Model System**
   - Combines doodads (M2) and objects (WMO) into a single reference system
   - Localized model placement within chunk-specific ACDO subchunks
   - Enhanced model instance properties including non-uniform scaling

3. **Memory-Optimized Structures**
   - Compressed reference systems for textures and models
   - More efficient storage of flight boundaries (shorts instead of floats)
   - Potential reduction in overall memory footprint

4. **Enhanced Chunk Headers**
   - Expanded ACNK headers with additional properties
   - Better support for special terrain features
   - More detailed control over chunk rendering behavior

5. **Terrain Data Organization**
   - Clearer separation between global data and local chunk-specific data
   - Potentially more efficient memory access patterns for rendering
   - More structured approach to subchunk organization

## Documentation Status
- Main chunks: 8/8 documented (100%)
- Subchunks: 4/4 documented (100%)
- Total: 12/12 (100%) 