# WoW Alpha 0.5.3 Terrain System

## Overview

The WoW Alpha 0.5.3 terrain system uses a hierarchical chunk-based architecture to efficiently manage large outdoor environments.

## Terrain Hierarchy

```
World
└── Continent
    └── Zone
        └── Map Area (16x16 chunks)
            └── Map Chunk (8x8 cells, 9x9 vertices)
                └── Terrain Cell (2 triangles)
```

## Map Area Structure

**Address:** [`CMapArea::CMapArea`](0x006aa880) (0x006aa880)

### Key Fields

```c
struct CMapArea {
    // Chunk management
    TSExplicitList<CMapBaseObjLink, 8> chunkLinkList;  // List of chunks
    
    // Position information
    C2Vector mIndex;                                    // Map index
    C2Vector cOffset;                                   // Chunk offset
    CiRect localRect;                                    // Local rectangle
    
    // Texture management
    TSFixedArray<HTEXTURE, 96> texIdTable;             // Texture ID table (96 entries)
    int texCount;                                        // Texture count
    
    // Object management
    TSGrowableArray<SMDoodadDef> doodadDefList;         // Doodad definitions
    TSGrowableArray<SMMapObjDef> mapObjDefList;         // Map object definitions
    
    // Async loading
    CAsyncObject* asyncObject;                            // Async loading object
    
    // Chunk data
    SMChunkInfo chunkInfo[256];                          // 256 chunk info entries
    CMapChunk* chunkTable[256];                          // 256 chunk pointers
};
```

### Chunk Info Structure

```c
struct SMChunkInfo {
    uint offset;           // Offset in file
    uint size;             // Size in bytes
    uint asyncId;          // Async loading ID
};
```

## Map Chunk Structure

**Address:** [`CMapChunk::CMapChunk`](0x00698510) (0x00698510)

### Key Fields

```c
struct CMapChunk {
    // Object links
    TSExplicitList<CMapBaseObjLink, 8> doodadDefLinkList;  // Doodad definitions
    TSExplicitList<CMapBaseObjLink, 8> mapObjDefLinkList;  // Map object definitions
    TSExplicitList<CMapBaseObjLink, 8> entityLinkList;      // Entity definitions
    TSExplicitList<CMapBaseObjLink, 8> lightLinkList;       // Light definitions
    TSList<CMapSoundEmitter> soundEmitterList;              // Sound emitters
    
    // Geometry data
    C3Vector normalList[145];                              // 145 normals (9x9 grid)
    C3Vector vertexList[145];                              // 145 vertices (9x9 grid)
    C4Plane planeList[256];                                // 256 planes (8x8 cells * 2 triangles)
    
    // Textures
    CChunkTex* shadowTexture;                               // Shadow texture
    CChunkTex* shaderTexture;                               // Shader texture
    
    // Detail doodads
    CDetailDoodadInst* detailDoodadInst;                    // Detail doodad instances
    
    // Liquids
    CChunkLiquid* liquids[4];                              // 4 liquid objects
    
    // Rendering
    int nLayers;                                           // Number of texture layers
    CGxBuf* gxBuf;                                       // Graphics buffer
    CAsyncObject* asyncObject;                              // Async loading object
    
    // Position
    ushort holes;                                          // Hole mask
    C2Vector aIndex;                                       // Area index
    C2Vector sOffset;                                      // Subchunk offset
    C2Vector cOffset;                                      // Chunk offset
    CRndSeed rSeed;                                        // Random seed
};
```

### Terrain Grid

Each map chunk contains a **9x9 vertex grid** (145 vertices total):

```
Vertices: 9x9 = 145
Cells: 8x8 = 64
Triangles: 64 * 2 = 128
```

Each cell contains **2 triangles** (forming a quad), resulting in **128 triangles per chunk**.

### Hole Mask

The hole mask is a 16-bit value where each bit represents a 2x2 cell group:

```
Bit 0:  Cells (0,0), (0,1), (1,0), (1,1)
Bit 1:  Cells (2,0), (2,1), (3,0), (3,1)
...
Bit 15: Cells (14,14), (14,15), (15,14), (15,15)
```

If a bit is set, those cells are "holes" and have no terrain geometry.

## Area of Interest

**Address:** [`CWorld::PrepareAreaOfInterest`](0x00665310) (0x00665310)

### Purpose

Calculate which chunks are visible based on camera position.

### Constants

```c
// Chunk coordinate system
const float CHUNK_SCALE = 1.0f / 533.3333f;  // 1/533.3333
const float CHUNK_OFFSET = 533.3333f / 2.0f;  // 266.6667

// Chunk dimensions
const int CHUNKS_PER_AREA = 16;           // 16x16 chunks per area
const int CELLS_PER_CHUNK = 8;            // 8x8 cells per chunk
const int VERTICES_PER_CHUNK_SIDE = 9;    // 9x9 vertices per chunk

// Valid chunk range
const int MIN_CHUNK = 0;
const int MAX_CHUNK = 1023;              // 0x3ff

// AOI sizes
const float GROUP_AOI_SIZE = 100.0f;     // 100 units for doodads
const float OBJECT_AOI_SIZE = 500.0f;    // 500 units for game objects
```

### Algorithm

1. Convert camera position to chunk coordinates
2. Calculate chunk rectangle based on AOI size
3. Clamp to valid chunk range (0-1023)
4. Set up area rectangle for rendering
5. Set up group AOI (for doodads)
6. Set up object AOI (for game objects)

### Pseudocode

```c
void PrepareAreaOfInterest(C3Vector* cameraPos, C3Vector* cameraTarget) {
    // Convert camera position to chunk coordinates
    int chunkX = (int)ROUND(-(cameraPos->y - CHUNK_OFFSET) * CHUNK_SCALE - CHUNK_OFFSET);
    int chunkY = (int)ROUND(-(cameraPos->x - CHUNK_OFFSET) * CHUNK_SCALE - CHUNK_OFFSET);
    
    // Calculate chunk rectangle
    chunkRectHi.minX = chunkX - chunkAoiSize.x;
    chunkRectHi.maxX = chunkX + chunkAoiSize.x;
    chunkRectHi.minY = chunkY - chunkAoiSize.y;
    chunkRectHi.maxY = chunkY + chunkAoiSize.y;
    
    // Clamp to valid range
    chunkRectHi.minX = MAX(0, chunkRectHi.minX);
    chunkRectHi.maxX = MIN(1023, chunkRectHi.maxX);
    chunkRectHi.minY = MAX(0, chunkRectHi.minY);
    chunkRectHi.maxY = MIN(1023, chunkRectHi.maxY);
    
    // Expand by 1 chunk for safety
    gbChunkRect.minX = chunkRectHi.minX - 1;
    gbChunkRect.maxX = chunkRectHi.maxX + 1;
    gbChunkRect.minY = chunkRectHi.minY - 1;
    gbChunkRect.maxY = chunkRectHi.maxY + 1;
    
    // Clamp again
    gbChunkRect.minX = MAX(0, gbChunkRect.minX);
    gbChunkRect.maxX = MIN(1023, gbChunkRect.maxX);
    gbChunkRect.minY = MAX(0, gbChunkRect.minY);
    gbChunkRect.maxY = MIN(1023, gbChunkRect.maxY);
    
    // Calculate area rectangle (divide by 16)
    areaRect.minX = gbChunkRect.minX >> 4;
    areaRect.minY = gbChunkRect.minY >> 4;
    areaRect.maxX = gbChunkRect.maxX >> 4;
    areaRect.maxY = gbChunkRect.maxY >> 4;
    
    // Set up group AOI (for doodads)
    groupAoi.min.x = cameraPos->x - GROUP_AOI_SIZE;
    groupAoi.min.y = cameraPos->y - GROUP_AOI_SIZE;
    groupAoi.min.z = cameraPos->z - GROUP_AOI_SIZE;
    groupAoi.max.x = cameraPos->x + GROUP_AOI_SIZE;
    groupAoi.max.y = cameraPos->y + GROUP_AOI_SIZE;
    groupAoi.max.z = cameraPos->z + GROUP_AOI_SIZE;
    
    // Set up object AOI (for game objects)
    objectAoi.min.x = cameraPos->x - farClip;
    objectAoi.min.y = cameraPos->y - farClip;
    objectAoi.min.z = cameraPos->z - farClip;
    objectAoi.max.x = cameraPos->x + farClip;
    objectAoi.max.y = cameraPos->y + farClip;
    objectAoi.max.z = cameraPos->z + farClip;
}
```

## Terrain Rendering

**Address:** [`CWorldScene::PrepareRender`](0x0066a740) (0x0066a740)

### Purpose

Prepare world scene for rendering.

### Key Operations

1. Set camera position and target
2. Calculate camera vector
3. Setup view matrix
4. Setup projection matrix
5. Setup viewport
6. Calculate model-view-projection matrix
7. Calculate frustum corners
8. Calculate frustum bounds
9. Prepare liquid rendering

### Pseudocode

```c
void PrepareRender(C3Vector* cameraPos, C3Vector* cameraTarget) {
    // Set camera position and target
    camPos = *cameraPos;
    camTarg = *cameraTarget;
    
    // Calculate camera vector
    camVec.x = cameraTarget->x - cameraPos->x;
    camVec.y = cameraTarget->y - cameraPos->y;
    camVec.z = cameraTarget->z - cameraPos->z;
    
    // Normalize camera vector
    float length = SQRT(camVec.x * camVec.x + camVec.y * camVec.y + camVec.z * camVec.z);
    float invLength = 1.0f / length;
    camVec.x *= invLength;
    camVec.y *= invLength;
    camVec.z *= invLength;
    
    // Setup camera plane (XY plane)
    camPlaneXY.n.x = camVec.x;
    camPlaneXY.n.y = camVec.y;
    camPlaneXY.n.z = 0.0f;
    camPlaneXY.d = -(camPlaneXY.n.x * cameraPos->x + camPlaneXY.n.y * cameraPos->y);
    
    // Setup matrices
    GxXformView(&mv);
    GxXformProjection(&mp);
    GxXformViewport(&vpMinPos.x, &vpMaxPos.x, &vpMinPos.y, &vpMaxPos.y, &vpMinPos.z, &vpMaxPos.z);
    
    // Calculate MVP matrix
    mvp = mv * mp;
    
    // Calculate frustum corners
    GxuXformCalcFrustumCorners(&mv, &mp, &camFrustumCorners);
    
    // Calculate frustum bounds
    camFrustumBounds = Bounding(&camFrustumCorners, 8);
    
    // Prepare liquid rendering
    PrepareRenderLiquid();
}
```

## Chunk Rendering

**Address:** [`CWorldScene::RenderChunks`](0x0066de50) (0x0066de50)

### Purpose

Render visible terrain chunks.

### Key Operations

1. Iterate through visible chunks from sort table
2. For each chunk:
   - Set up world transform matrix
   - Select lights for the chunk
   - Render the chunk
   - Create detail doodads if within distance
3. Sort chunks by distance for proper rendering order

### Transform Matrix

```c
// World transform matrix for chunk
C44Matrix cMat;
cMat.a0 = 1.0f;  cMat.a1 = 0.0f;  cMat.a2 = 0.0f;  cMat.a3 = 0.0f;
cMat.b0 = 0.0f;  cMat.b1 = 1.0f;  cMat.b2 = 0.0f;  cMat.b3 = 0.0f;
cMat.c0 = 0.0f;  cMat.c1 = 0.0f;  cMat.c2 = 1.0f;  cMat.c3 = 0.0f;
cMat.d0 = chunkX - camPos.x;
cMat.d1 = chunkY - camPos.y;
cMat.d2 = chunkZ - camPos.z;
cMat.d3 = 1.0f;
```

## Implementation Guidelines

### C# Terrain Chunk Structure

```csharp
public class TerrainChunk
{
    public C2Vector Index { get; set; }
    public C2Vector Offset { get; set; }
    public C3Vector[] Vertices { get; set; }  // 145 vertices (9x9)
    public C3Vector[] Normals { get; set; }   // 145 normals
    public C4Plane[] Planes { get; set; }     // 256 planes (8x8 * 2)
    public ushort Holes { get; set; }
    public int NumLayers { get; set; }
    public TextureId[] Textures { get; set; }
    public List<DoodadDef> Doodads { get; set; }
    public List<MapObjDef> MapObjects { get; set; }
    
    public bool IsHole(int cellX, int cellY)
    {
        int maskIndex = (cellX >> 1) + (cellY >> 1) * 4;
        return ((HoleMask[maskIndex] & Holes) != 0);
    }
    
    public float GetHeight(float localX, float localY)
    {
        // Convert to cell coordinates
        int cellX = (int)(localX * 8.0f);
        int cellY = (int)(localY * 8.0f);
        
        // Clamp to valid range
        cellX = Math.Clamp(cellX, 0, 7);
        cellY = Math.Clamp(cellY, 0, 7);
        
        // Check for holes
        if (IsHole(cellX, cellY))
        {
            return 0.0f;
        }
        
        // Get triangle vertices
        int vertexIndex = cellY * 17 + cellX;
        C3Vector v0 = Vertices[vertexIndex];
        C3Vector v1 = Vertices[vertexIndex + 1];
        C3Vector v2 = Vertices[vertexIndex + 17];
        C3Vector v3 = Vertices[vertexIndex + 18];
        
        // Calculate local position within cell
        float fx = (localX * 8.0f) - cellX;
        float fy = (localY * 8.0f) - cellY;
        
        // Interpolate height based on triangle
        if (fx + fy < 1.0f)
        {
            // First triangle
            return v0.z + (v1.z - v0.z) * fx + (v2.z - v0.z) * fy;
        }
        else
        {
            // Second triangle
            return v3.z + (v2.z - v3.z) * (1.0f - fx) + (v1.z - v3.z) * (1.0f - fy);
        }
    }
}
```

### C# Area of Interest Management

```csharp
public class TerrainManager
{
    private const float CHUNK_SCALE = 1.0f / 533.3333f;
    private const float CHUNK_OFFSET = 533.3333f / 2.0f;
    private const int MAX_CHUNK = 1023;
    
    private Dictionary<(int x, int y), TerrainChunk> loadedChunks;
    private CiRect currentAoI;
    
    public void UpdateAreaOfInterest(C3Vector cameraPos)
    {
        // Convert camera position to chunk coordinates
        int chunkX = (int)Math.Round(-(cameraPos.Y - CHUNK_OFFSET) * CHUNK_SCALE - CHUNK_OFFSET);
        int chunkY = (int)Math.Round(-(cameraPos.X - CHUNK_OFFSET) * CHUNK_SCALE - CHUNK_OFFSET);
        
        // Calculate chunk rectangle
        int aoiSize = 10;  // 10 chunks in each direction
        CiRect newAoI;
        newAoI.MinX = Math.Max(0, chunkX - aoiSize);
        newAoI.MaxX = Math.Min(MAX_CHUNK, chunkX + aoiSize);
        newAoI.MinY = Math.Max(0, chunkY - aoiSize);
        newAoI.MaxY = Math.Min(MAX_CHUNK, chunkY + aoiSize);
        
        // Load new chunks
        for (int y = newAoI.MinY; y <= newAoI.MaxY; y++)
        {
            for (int x = newAoI.MinX; x <= newAoI.MaxX; x++)
            {
                if (!loadedChunks.ContainsKey((x, y)))
                {
                    LoadChunk(x, y);
                }
            }
        }
        
        // Unload old chunks
        foreach (var key in loadedChunks.Keys.ToList())
        {
            if (key.x < newAoI.MinX || key.x > newAoI.MaxX ||
                key.y < newAoI.MinY || key.y > newAoI.MaxY)
            {
                UnloadChunk(key.x, key.y);
            }
        }
        
        currentAoI = newAoI;
    }
    
    private void LoadChunk(int chunkX, int chunkY)
    {
        // Load chunk from file
        TerrainChunk chunk = LoadChunkFromFile(chunkX, chunkY);
        loadedChunks[(chunkX, chunkY)] = chunk;
    }
    
    private void UnloadChunk(int chunkX, int chunkY)
    {
        loadedChunks.Remove((chunkX, chunkY));
    }
}
```

## References

- [`CMapArea::CMapArea`](0x006aa880) (0x006aa880) - Map area constructor
- [`CMapChunk::CMapChunk`](0x00698510) (0x00698510) - Map chunk constructor
- [`CWorld::PrepareAreaOfInterest`](0x00665310) (0x00665310) - Calculate area of interest
- [`CWorldScene::PrepareRender`](0x0066a740) (0x0066a740) - Prepare world scene for rendering
- [`CWorldScene::RenderChunks`](0x0066de50) (0x0066de50) - Render visible terrain chunks
