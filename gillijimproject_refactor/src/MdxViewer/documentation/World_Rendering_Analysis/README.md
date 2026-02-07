# World Rendering Analysis Documentation

This documentation covers the World of Warcraft Alpha 0.5.3 world rendering system, including terrain, collision detection, player movement, mesh traversal, liquid rendering, detail doodads, frustum culling, terrain lighting, texture layering, and chunk rendering.

## Core Systems

### Terrain and Environment

1. **[01_Terrain_System.md](01_Terrain_System.md)** - Complete terrain system documentation
   - Terrain hierarchy (World → Continent → Zone → Map Area → Map Chunk → Terrain Cell)
   - Map Area and Map Chunk structures
   - Area of interest calculation
   - Terrain rendering and chunk rendering
   - C# implementation examples

2. **[02_Collision_Detection.md](02_Collision_Detection.md)** - Collision detection system
   - Three-phase hierarchical approach (broad/medium/narrow)
   - Terrain intersection with DDA algorithm
   - Subchunk intersection and facet retrieval
   - Triangle retrieval and C# implementation

3. **[03_Player_Movement.md](03_Player_Movement.md)** - Player movement system
   - Event-driven movement system
   - 20 movement event types documented
   - Movement processing algorithm
   - Collision handling and C# implementation

4. **[04_Mesh_Traversal.md](04_Mesh_Traversal.md)** - Mesh traversal methods
   - Terrain height query methods
   - Facet and triangle queries
   - Walkable surface determination
   - C# mesh traversal implementation

5. **[05_Liquid_Rendering.md](05_Liquid_Rendering.md)** - Liquid rendering system
   - Liquid types (water, ocean, magma, slime)
   - Particle effects for water and magma
   - Liquid status query and texture management
   - C# liquid renderer implementation

### Rendering Systems

6. **[06_Detail_Doodads.md](06_Detail_Doodads.md)** - Detail doodad system
   - Detail doodad creation and rendering
   - Density control and console commands
   - C# detail doodad manager implementation

7. **[07_Frustum_Culling.md](07_Frustum_Culling.md)** - Frustum culling system
   - Frustum plane extraction from view-projection matrix
   - Point, sphere, and AABB frustum tests
   - Chunk, WMO, and doodad frustum culling
   - C# frustum culling implementation

8. **[08_Terrain_Lighting.md](08_Terrain_Lighting.md)** - Terrain lighting system
   - Day/night cycle
   - Sun position calculation
   - Light, ambient, and fog color calculation
   - C# terrain lighting implementation

9. **[09_Texture_Layering.md](09_Texture_Layering.md)** - Texture layering system
   - Multi-layer terrain rendering
   - Blend modes (opaque, alpha, add, modulate, modulate2X)
   - Alpha maps and texture coordinates
   - C# texture layering implementation

10. **[10_Chunk_Rendering.md](10_Chunk_Rendering.md)** - Chunk rendering system
    - Chunk rendering pipeline
    - Opaque and transparent geometry rendering
    - Liquid and detail doodad rendering
    - C# chunk rendering implementation

## Key Constants

### Terrain Constants

- `CHUNK_SCALE` = 1/533.3333 (0.001875)
- `CHUNK_OFFSET` = 266.6667
- `MAX_CHUNK` = 1023
- `CHUNK_SIZE` = 533.3333
- `CELL_SIZE` = 66.6667
- `NUM_CELLS` = 8
- `NUM_VERTICES` = 9

### Movement Constants

- `WALK_SPEED` = 3.5f
- `RUN_SPEED` = 7.0f
- `SWIM_SPEED` = 3.5f
- `JUMP_VELOCITY` = 8.0f
- `GRAVITY` = 9.8f

### Liquid Constants

- `LIQUID_WATER` = 0x0
- `LIQUID_OCEAN` = 0x1
- `LIQUID_MAGMA` = 0x2
- `LIQUID_SLIME` = 0x3
- `LIQUID_NONE` = 0xf

### Detail Doodad Constants

- `DETAIL_DOODAD_DISTANCE` = 100.0f
- `MAX_DETAIL_DOODADS` = 64

## Ghidra Analysis

All analysis is based on Ghidra analysis of the original World of Warcraft Alpha 0.5.3 client. Key function addresses are provided for each system.

## Implementation Notes

The C# implementation examples provided in each document are designed to be compatible with modern OpenGL and can be integrated into the MdxViewer project.

## References

- [`CWorldScene::OnWorldRender`](0x0066a3e0) (0x0066a3e0) - Main world rendering function
- [`CWorldScene::UpdateFrustum`](0x0066a460) (0x0066a460) - Update frustum planes
- [`CWorldScene::PrepareRenderLiquid`](0x0066a590) (0x0066a590) - Prepare liquid rendering
- [`CWorld::UpdateDayNightCycle`](0x0066a5c0) (0x0066a5c0) - Update day/night cycle
- [`CMap::QueryLiquidStatus`](0x00664e70) (0x00664e70) - Query liquid status
- [`CMap::QueryTerrainHeight`](0x00664e50) (0x00664e50) - Query terrain height
- [`CMap::QueryTerrainFacet`](0x00664e60) (0x00664e60) - Query terrain facet
- [`CMap::QueryTerrainTriangle`](0x00664e80) (0x00664e80) - Query terrain triangle
- [`CMapChunk::Render`](0x006a6d80) (0x006a6d80) - Render chunk
- [`CMapChunk::CalcLighting`](0x006a6d30) (0x006a6d30) - Calculate lighting for chunk
- [`CMapChunk::CreateDetailDoodads`](0x006a6cf0) (0x006a6cf0) - Create detail doodads for chunk
- [`CDetailDoodadInst::CDetailDoodadInst`](0x006a2580) (0x006a2580) - Detail doodad instance constructor
