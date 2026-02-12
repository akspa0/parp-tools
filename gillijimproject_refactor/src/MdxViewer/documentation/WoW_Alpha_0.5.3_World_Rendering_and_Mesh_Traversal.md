# WoW Alpha 0.5.3 World Rendering and Mesh Traversal Analysis

## Executive Summary

This document provides a comprehensive analysis of how the WoW Alpha 0.5.3 engine displays the game world and allows the player to traverse over the mesh, based on Ghidra analysis of the original client.

## Table of Contents

1. [World Rendering Architecture](#world-rendering-architecture)
2. [Terrain System](#terrain-system)
3. [Collision Detection](#collision-detection)
4. [Player Movement](#player-movement)
5. [Mesh Traversal](#mesh-traversal)
6. [Implementation Guidelines](#implementation-guidelines)

---

## World Rendering Architecture

### Overview

The WoW Alpha 0.5.3 world rendering system is built around a hierarchical chunk-based architecture that efficiently manages large outdoor environments. The world is divided into:

1. **Continents** (top-level)
2. **Zones** (within continents)
3. **Map Areas** (16x16 chunks)
4. **Map Chunks** (8x8 cells, 9x9 vertices)
5. **Terrain Cells** (individual triangles)

### Rendering Pipeline

The world rendering pipeline follows this sequence:

```
RenderWorld()
├── Save view/projection matrices
├── OnWorldUpdate()
│   ├── Update camera
│   ├── Setup world projection
│   ├── Place camera in scene
│   ├── Calculate frustum planes
│   ├── Update day/night info
│   ├── Update player alpha
│   ├── Update units
│   ├── CWorld::PrepareUpdate()
│   │   ├── PrepareAreaOfInterest()
│   │   ├── CWorldScene::PrepareRender()
│   │   └── CMap::PrepareUpdate()
│   ├── Update particle emitters
│   ├── Update ribbon emitters
│   └── Update spell visuals
├── OnWorldRender()
│   ├── Render world (WMOs, terrain, etc.)
│   ├── Render opaque models
│   ├── Render transparent models
│   └── Render collision info
├── Render player nameplates
└── Restore view/projection matrices
```

### Key Functions

#### `OnWorldRender` @ 0x0066a3e0 / 0x004f3440

**Address:** 0x0066a3e0 (Scene Level) / 0x004f3440 (Frame Level)

**Purpose:** Main world rendering entry point. Executes the multi-pass world rendering pipeline including Opaque, Alpha, and Post-process steps.

#### `RenderWorld` @ 0x004f2d00

**Pseudocode:**
```c
void RenderWorld(CGWorldFrame* frame) {
    C44Matrix saved_view = identity;
    C44Matrix saved_proj = identity;
    
    GxXformProjection(&saved_proj);
    GxXformView(&saved_view);
    
    OnWorldUpdate(frame);
    OnWorldRender(frame);
    PlayerNameRenderWorldText();
    
    GxXformSetProjection(&saved_proj);
    GxXformSetView(&saved_view);
}
```

#### [`OnWorldUpdate`](0x004f2e30) (0x004f2e30)

**Address:** 0x004f2e30

**Purpose:** Update world state before rendering

**Key Operations:**
1. Update camera position and orientation
2. Setup world projection matrix
3. Place camera in scene
4. Calculate frustum planes
5. Update day/night lighting
6. Update player alpha
7. Update units
8. Prepare world update
9. Update particle/ribbon emitters
10. Update spell visuals

---

## Terrain System

### Terrain Hierarchy

The terrain system uses a hierarchical chunk structure:

```
World
└── Continent
    └── Zone
        └── Map Area (16x16 chunks)
            └── Map Chunk (8x8 cells, 9x9 vertices)
                └── Terrain Cell (2 triangles)
```

### Map Area Structure

**Address:** [`CMapArea::CMapArea`](0x006aa880) (0x006aa880)

**Key Fields:**
```c
struct CMapArea {
    TSExplicitList<CMapBaseObjLink, 8> chunkLinkList;  // List of chunks
    C2Vector mIndex;                                    // Map index
    C2Vector cOffset;                                   // Chunk offset
    CiRect localRect;                                    // Local rectangle
    TSFixedArray<HTEXTURE, 96> texIdTable;             // Texture ID table (96 entries)
    TSGrowableArray<SMDoodadDef> doodadDefList;         // Doodad definitions
    TSGrowableArray<SMMapObjDef> mapObjDefList;         // Map object definitions
    int texCount;                                        // Texture count
    CAsyncObject* asyncObject;                            // Async loading object
    SMChunkInfo chunkInfo[256];                          // 256 chunk info entries
    CMapChunk* chunkTable[256];                          // 256 chunk pointers
};
```

### Map Chunk Structure

**Address:** [`CMapChunk::CMapChunk`](0x00698510) (0x00698510)

**Key Fields:**
```c
struct CMapChunk {
    TSExplicitList<CMapBaseObjLink, 8> doodadDefLinkList;  // Doodad definitions
    TSExplicitList<CMapBaseObjLink, 8> mapObjDefLinkList;  // Map object definitions
    TSExplicitList<CMapBaseObjLink, 8> entityLinkList;      // Entity definitions
    TSExplicitList<CMapBaseObjLink, 8> lightLinkList;       // Light definitions
    TSList<CMapSoundEmitter> soundEmitterList;              // Sound emitters
    C3Vector normalList[145];                              // 145 normals
    C3Vector vertexList[145];                              // 145 vertices (9x9 grid)
    C4Plane planeList[256];                                // 256 planes (8x8 cells * 2 triangles)
    CChunkTex* shadowTexture;                               // Shadow texture
    CChunkTex* shaderTexture;                               // Shader texture
    CDetailDoodadInst* detailDoodadInst;                    // Detail doodad instances
    CChunkLiquid* liquids[4];                              // 4 liquid objects
    int nLayers;                                           // Number of texture layers
    CGxBuf* gxBuf;                                       // Graphics buffer
    CAsyncObject* asyncObject;                              // Async loading object
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

### Area of Interest

**Address:** [`CWorld::PrepareAreaOfInterest`](0x00665310) (0x00665310)

**Purpose:** Calculate which chunks are visible based on camera position

**Key Operations:**
1. Convert camera position to chunk coordinates
2. Calculate chunk rectangle based on AOI size
3. Clamp to valid chunk range (0-1023)
4. Set up area rectangle for rendering
5. Set up group AOI (for doodads)
6. Set up object AOI (for game objects)

**Constants:**
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
const float FAR_CLIP_DEFAULT = 500.0f;   // Standard far clip for objects
```

**Pseudocode:**
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

### Terrain Rendering

**Address:** [`CWorldScene::PrepareRender`](0x0066a740) (0x0066a740)

**Purpose:** Prepare world scene for rendering

**Key Operations:**
1. Set camera position and target
2. Calculate camera vector
3. Setup view matrix
4. Setup projection matrix
5. Setup viewport
6. Calculate model-view-projection matrix
7. Calculate frustum corners
8. Calculate frustum bounds
9. Prepare liquid rendering

**Pseudocode:**
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

---

## Collision Detection

### Collision System Overview

The WoW Alpha 0.5.3 collision system uses a hierarchical approach:

1. **Broad Phase:** Chunk-level culling
2. **Medium Phase:** Subchunk-level culling
3. **Narrow Phase:** Triangle-level intersection

### Terrain Intersection

**Address:** [`CMap::VectorIntersectTerrain`](0x00679690) (0x00679690)

**Purpose:** Test ray intersection with terrain

**Parameters:**
- `start`: Ray start position
- `end`: Ray end position
- `t`: Output intersection parameter (0-1)
- `flags`: Collision flags
- `hitChunk`: Output chunk that was hit

**Key Operations:**
1. Convert ray to chunk coordinates
2. Calculate chunk rectangle
3. Determine traversal direction (X or Y dominant)
4. Traverse chunks using DDA algorithm
5. Test intersection with subchunks
6. Return closest intersection

**Pseudocode:**
```c
bool VectorIntersectTerrain(C3Vector* start, C3Vector* end, float* t, uint flags, CMapChunk** hitChunk) {
    // Convert to chunk coordinates
    C2Vector v0;
    v0.x = -(start->y - CHUNK_OFFSET);
    v0.y = -(start->x - CHUNK_OFFSET);
    
    C2Vector v1;
    v1.x = -(end->y - CHUNK_OFFSET);
    v1.y = -(end->x - CHUNK_OFFSET);
    
    // Calculate chunk rectangle
    CiRect sRect;
    sRect.minX = (int)ROUND(CHUNK_SCALE * v1.x - CHUNK_OFFSET);
    sRect.maxY = (int)ROUND(CHUNK_SCALE * v1.y - CHUNK_OFFSET);
    sRect.minY = (int)ROUND(CHUNK_SCALE * v0.x - CHUNK_OFFSET);
    sRect.maxX = (int)ROUND(CHUNK_SCALE * v0.y - CHUNK_OFFSET);
    
    // Determine traversal direction
    float dx = ABS(v1.x - v0.x);
    float dy = ABS(v1.y - v0.y);
    
    scCollideCnt = 0;
    
    if (dx < EPSILON && sRect.minX != sRect.maxX) {
        if (dy < EPSILON && sRect.minY != sRect.maxY) {
            // Both X and Y change
            C3Vector localStart;
            localStart.z = 0.0f;
            localStart.x = v1.x;
            localStart.y = v1.y;
            
            C3Vector localEnd;
            localEnd.x = v0.x;
            localEnd.y = v0.y;
            localEnd.z = 0.0f;
            
            if (dx <= dy) {
                VectorIntersectDY(&localEnd, &localStart, &sRect);
            } else {
                VectorIntersectDX(&localEnd, &localStart, &sRect);
            }
        } else {
            // Only X changes
            VectorIntersectSX(&sRect);
        }
    } else {
        // Only Y changes
        VectorIntersectSY(&sRect);
    }
    
    // Test intersection with subchunks
    return VectorIntersectSubchunks(start, end, t, flags, hitChunk);
}
```

### Subchunk Intersection

**Address:** [`CMap::VectorIntersectSubchunks`](0x0067a7b0) (0x0067a7b0)

**Purpose:** Test ray intersection with terrain subchunks

**Key Operations:**
1. Iterate through subchunks in collision list
2. For each subchunk:
   - Check if subchunk is in same area
   - Get chunk from area table
   - Test intersection with doodads (if flag set)
   - Test intersection with game objects (if flag set)
   - Test intersection with terrain triangles
3. Return closest intersection

**Pseudocode:**
```c
bool VectorIntersectSubchunks(C3Vector* start, C3Vector* end, float* t, uint flags, CMapChunk** hitChunk) {
    float hitT = *t;
    CMapChunk* hitChunkPtr = NULL;
    
    uint* scPtr = DAT_00e60e20;  // Subchunk collision list
    uint scCnt = scCollideCnt;
    
    uint bMaskY = scPtr[0] & 0x2000;
    uint bMaskX = scPtr[1] & 0x2000;
    
    while (scCnt > 0) {
        uint scX = *scPtr;
        uint scY = scPtr[1];
        scPtr += 2;
        scCnt -= 2;
        
        // Check for end marker
        if (scX > 0x2000 || scY > 0x2000) {
            return false;
        }
        
        // Check if subchunk is in same area
        if ((scX & 0x1ff8) != bMaskX || (scY & 0x1ff8) != bMaskY) {
            // Get chunk from area table
            CMapArea* area = areaTable[(scY >> 7 & 0x3f) * 64 + (scX >> 7 & 0x3f)];
            if (area == NULL) {
                return false;
            }
            
            CMapChunk* chunk = area->chunkTable[(scY >> 3 & 0xf) * 16 + (scX >> 3 & 0xf)];
            if (chunk == NULL) {
                return false;
            }
            
            bMaskY = scY & 0x1ff8;
            bMaskX = scX & 0x1ff8;
            
            // Test intersection with doodads
            if (flags & 0xf) {
                float doodadHitT = 1.0f;
                bool hit = VectorIntersectDoodadDefLinkList(&chunk->doodadDefLinkList, start, end, &doodadHitT, flags);
                if (hit && doodadHitT < hitT) {
                    hitT = doodadHitT;
                }
                
                // Test intersection with game objects
                hit = VectorIntersectGameObjLinkList(&chunk->entityLinkList, start, end, &doodadHitT, flags);
                if (hit && doodadHitT < hitT) {
                    hitT = doodadHitT;
                }
            }
        }
        
        // Test intersection with terrain triangles
        uint cellX = scX & 7;
        uint cellY = scY & 7;
        
        // Check for holes
        if ((g_holeMask[(cellX >> 1) + (cellY >> 1) * 4] & chunk->holes) == 0) {
            // Transform ray to chunk space
            float localStartX = start->x - chunk->field_0x64;
            float localStartY = start->y - chunk->field_0x68;
            float localStartZ = start->z - chunk->field_0x6c;
            
            float localEndX = end->x - chunk->field_0x64;
            float localEndY = end->y - chunk->field_0x68;
            float localEndZ = end->z - chunk->field_0x6c;
            
            // Get vertex index
            uint vertexIndex = cellY * 17 + cellX;
            
            // Test intersection with triangles
            C4Plane* plane = &chunk->planeList[(cellX + cellY * 8) * 4];
            
            for (int i = 0; i < 4; i++) {
                // Calculate plane distances
                float ip0 = localStartX * plane->n.x + localStartY * plane->n.y + localStartZ * plane->n.z + plane->d;
                float ip1 = localEndX * plane->n.x + localEndY * plane->n.y + localEndZ * plane->n.z + plane->d;
                
                // Check if ray intersects plane
                if ((ip0 <= 0.0f || ip1 <= 0.0f) && (ip0 < 0.0f || ip1 < 0.0f)) {
                    float t = ip0 / (ip0 - ip1);
                    if (t >= 0.0f && t <= 1.0f) {
                        // Calculate intersection point
                        C3Vector ip;
                        ip.x = (localEndX - localStartX) * t + localStartX;
                        ip.y = (localEndY - localStartY) * t + localStartY;
                        ip.z = (localEndZ - localStartZ) * t + localStartZ;
                        
                        // Test intersection with triangle
                        bool hit = VectorIntersectTri(&ip, 
                            &chunk->vertexList[vertexIndex + 9],
                            &chunk->vertexList[vertexIndex + triangleIndices[i * 2]],
                            &chunk->vertexList[vertexIndex + triangleIndices[i * 2 + 1]],
                            plane);
                        
                        if (hit && t < hitT) {
                            hitT = t;
                            hitChunkPtr = chunk;
                        }
                    }
                }
                
                plane++;
            }
        }
    }
    
    if (hitT < *t) {
        *t = hitT;
        if (hitChunk != NULL) {
            *hitChunk = hitChunkPtr;
        }
        return true;
    }
    
    return false;
}
```

### Facet Retrieval

**Address:** [`CMap::GetFacetTerrain`](0x0067bbd0) (0x0067bbd0)

**Purpose:** Get terrain facet (plane) at a given position

**Parameters:**
- `segment`: Line segment to test
- `t`: Output intersection parameter
- `plane`: Output facet plane
- `flags`: Collision flags

**Key Operations:**
1. Convert segment to chunk coordinates
2. Calculate chunk rectangle
3. Determine traversal direction
4. Traverse chunks using DDA algorithm
5. Test intersection with subchunks
6. Return facet plane

**Pseudocode:**
```c
bool GetFacetTerrain(C3Segment* segment, float* t, C4Plane* plane, uint flags) {
    // Convert to chunk coordinates
    C2Vector v0;
    v0.x = -(segment->start.y - CHUNK_OFFSET);
    v0.y = -(segment->start.x - CHUNK_OFFSET);
    
    C2Vector v1;
    v1.x = -(segment->end.y - CHUNK_OFFSET);
    v1.y = -(segment->end.x - CHUNK_OFFSET);
    
    // Calculate chunk rectangle
    CiRect sRect;
    sRect.minX = (int)ROUND(CHUNK_SCALE * v1.x - CHUNK_OFFSET);
    sRect.maxY = (int)ROUND(CHUNK_SCALE * v1.y - CHUNK_OFFSET);
    sRect.minY = (int)ROUND(CHUNK_SCALE * v0.x - CHUNK_OFFSET);
    sRect.maxX = (int)ROUND(CHUNK_SCALE * v0.y - CHUNK_OFFSET);
    
    // Determine traversal direction
    float dx = ABS(v1.x - v0.x);
    float dy = ABS(v1.y - v0.y);
    
    scCollideCnt = 0;
    
    if (dx < EPSILON && sRect.minX != sRect.maxX) {
        if (dy < EPSILON && sRect.minY != sRect.maxY) {
            // Both X and Y change
            C3Vector localEnd;
            localEnd.z = 0.0f;
            localEnd.x = v1.x;
            localEnd.y = v1.y;
            
            C3Vector localStart;
            localStart.x = v0.x;
            localStart.y = v0.y;
            localStart.z = 0.0f;
            
            if (dx <= dy) {
                VectorIntersectDY(&localStart, &localEnd, &sRect);
            } else {
                VectorIntersectDX(&localStart, &localEnd, &sRect);
            }
        } else {
            // Only X changes
            VectorIntersectSX(&sRect);
        }
    } else {
        // Only Y changes
        VectorIntersectSY(&sRect);
    }
    
    // Calculate new segment end
    float nt = *t;
    C3Vector dir = segment->end - segment->start;
    C3Vector newEnd = segment->start + dir * nt;
    
    // Create new segment
    C3Segment newSegment;
    newSegment.start = segment->start;
    newSegment.end = newEnd;
    
    // Test intersection with subchunks
    bool hit = GetFacetSubchunks(&newSegment, &nt, plane, flags);
    
    if (hit) {
        *t = nt * *t;
    }
    
    return hit;
}
```

### Triangle Retrieval

**Address:** [`CMap::GetTrisTerrain`](0x0067c3c0) (0x0067c3c0)

**Purpose:** Get terrain triangles within a bounding box

**Parameters:**
- `box`: Bounding box to test
- `triData`: Output triangle data
- `flags`: Collision flags

**Key Operations:**
1. Convert bounding box to chunk coordinates
2. Calculate chunk rectangle
3. Iterate through chunks in rectangle
4. For each chunk:
   - Check for holes
   - Get triangles within bounding box
   - Add to triangle data
5. Return true if any triangles found

**Pseudocode:**
```c
bool GetTrisTerrain(CAaBox* box, CWTriData* triData, uint flags) {
    bool got = false;
    
    // Convert to chunk coordinates
    float minX = -(box->max.x - CHUNK_OFFSET);
    float minY = -(box->max.y - CHUNK_OFFSET);
    float maxX = -(box->min.x - CHUNK_OFFSET);
    float maxY = box->min.y - CHUNK_OFFSET;
    
    // Validate range
    if (minY < 0.0f || minX < 0.0f) {
        Error("Invalid location");
    }
    if (maxX < 533.3333f) {
        Error("Invalid location");
    }
    
    // Calculate chunk rectangle
    CiRect sRect;
    sRect.minX = (int)ROUND(CHUNK_SCALE * -maxY - CHUNK_OFFSET);
    sRect.maxY = (int)ROUND(CHUNK_SCALE * maxX - CHUNK_OFFSET);
    sRect.minY = (int)ROUND(CHUNK_SCALE * minY - CHUNK_OFFSET);
    sRect.maxX = (int)ROUND(CHUNK_SCALE * minX - CHUNK_OFFSET);
    
    // Iterate through chunks
    int chunkMinY = sRect.minY >> 3;
    int chunkMinX = sRect.minX >> 3;
    int chunkMaxY = sRect.maxY >> 3;
    int chunkMaxX = sRect.maxX >> 3;
    
    for (int chunkY = chunkMinY; chunkY <= chunkMaxY; chunkY++) {
        for (int chunkX = chunkMinX; chunkX <= chunkMaxX; chunkX++) {
            bool hit = GetTrisChunk(chunkX, chunkY, &sRect, box, triData, flags);
            got = got || hit;
        }
    }
    
    return got;
}
```

---

## Player Movement

### Movement System Overview

The WoW Alpha 0.5.3 movement system is event-driven, with movement commands being queued and processed each frame.

### Movement Events

**Address:** [`CMovement::UpdatePlayerMovement`](0x004c4d90) (0x004c4d90)

**Purpose:** Process player movement events

**Movement Event Types:**
```c
enum MovementEventType {
    MOVE_START_FORWARD = 0,      // Start moving forward
    MOVE_START_BACKWARD = 1,     // Start moving backward
    MOVE_STOP = 2,               // Stop moving
    MOVE_START_STRAFE_LEFT = 3,  // Start strafing left
    MOVE_START_STRAFE_RIGHT = 4, // Start strafing right
    MOVE_STOP_STRAFE = 5,       // Stop strafing
    MOVE_START_FALLING = 6,      // Start falling
    MOVE_JUMP = 7,               // Jump
    MOVE_START_TURN_LEFT = 8,     // Start turning left
    MOVE_START_TURN_RIGHT = 9,    // Start turning right
    MOVE_STOP_TURN = 10,         // Stop turning
    MOVE_START_PITCH_UP = 11,    // Start pitching up
    MOVE_START_PITCH_DOWN = 12,  // Start pitching down
    MOVE_STOP_PITCH = 13,        // Stop pitching
    MOVE_SET_RUN = 14,          // Set run mode
    MOVE_SET_WALK = 15,         // Set walk mode
    MOVE_SET_FACING = 16,        // Set facing direction
    MOVE_SET_PITCH = 17,         // Set pitch
    MOVE_START_SWIM = 18,        // Start swimming
    MOVE_STOP_SWIM = 19,         // Stop swimming
};
```

**Pseudocode:**
```c
int UpdatePlayerMovement(CMovement* movement, ulong currentTime) {
    void* globals = MovementGetGlobals();
    
    // Process movement events
    while (globals->moveEventCount > 0) {
        TSLink<CPlayerMoveEvent>* event = globals->moveEventList;
        
        // Check if event is in the past
        if ((int)(currentTime - event->timestamp) < 0) {
            break;
        }
        
        // Remove event from list
        RemoveMoveEvent(event);
        
        // Process event based on type
        switch (event->type) {
            case MOVE_START_FORWARD:
                StartMove(movement, currentTime, true);
                break;
            case MOVE_START_BACKWARD:
                StartMove(movement, currentTime, false);
                break;
            case MOVE_STOP:
                StopMove(movement, currentTime);
                break;
            case MOVE_START_STRAFE_LEFT:
                StartStrafe(movement, currentTime, true);
                break;
            case MOVE_START_STRAFE_RIGHT:
                StartStrafe(movement, currentTime, false);
                break;
            case MOVE_STOP_STRAFE:
                StopStrafe(movement, currentTime);
                break;
            case MOVE_START_FALLING:
                StartFalling(movement, currentTime);
                OnCollideFalling(movement->object, currentTime);
                break;
            case MOVE_JUMP:
                Jump(movement, currentTime);
                break;
            case MOVE_START_TURN_LEFT:
                StartTurn(movement, currentTime, true);
                break;
            case MOVE_START_TURN_RIGHT:
                StartTurn(movement, currentTime, false);
                break;
            case MOVE_STOP_TURN:
                StopTurn(movement, currentTime);
                break;
            case MOVE_START_PITCH_UP:
                StartPitch(movement, currentTime, true);
                break;
            case MOVE_START_PITCH_DOWN:
                StartPitch(movement, currentTime, false);
                break;
            case MOVE_STOP_PITCH:
                StopPitch(movement, currentTime);
                break;
            case MOVE_SET_RUN:
                SetRunMode(movement, currentTime, true);
                break;
            case MOVE_SET_WALK:
                SetRunMode(movement, currentTime, false);
                break;
            case MOVE_SET_FACING:
                SetFacing(movement, currentTime, event->facing);
                break;
            case MOVE_SET_PITCH:
                SetPitch(movement, currentTime, event->pitch);
                break;
            case MOVE_START_SWIM:
                StartSwimLocal(movement, currentTime);
                break;
            case MOVE_STOP_SWIM:
                StopSwimLocal(movement, currentTime);
                break;
        }
        
        // Free event
        ObjectFree(event->object);
    }
    
    // Check if there are any active movements
    if (globals->moveEventCount == 0 && (movement->flags & 0x40ff) == 0) {
        return 0;
    }
    
    return 1;
}
```

### Collision Handling

**Collision Functions:**
- [`OnCollideFalling`](0x005f3540) (0x005f3540) - Handle collision while falling
- [`OnCollideFallLand`](0x005f34f0) (0x005f34f0) - Handle landing after fall
- [`OnCollideRedirected`](0x005f33b0) (0x005f33b0) - Handle redirected collision
- [`OnCollideStuck`](0x005f3410) (0x005f3410) - Handle stuck collision

---

## Mesh Traversal

### Terrain Height Query

The engine provides several methods to query terrain height at a given position:

1. **Ray Intersection:** Cast a ray from above and find intersection with terrain
2. **Facet Retrieval:** Get the terrain facet (plane) at a position
3. **Triangle Retrieval:** Get all triangles within a bounding box

### Ray Casting

**Purpose:** Find terrain height at a given position

**Method:**
1. Cast a ray from (x, y, z + height) to (x, y, z - height)
2. Find intersection with terrain
3. Return intersection point

**Pseudocode:**
```c
float GetTerrainHeight(float x, float y, float startHeight, float endHeight) {
    C3Vector start;
    start.x = x;
    start.y = y;
    start.z = startHeight;
    
    C3Vector end;
    end.x = x;
    end.y = y;
    end.z = endHeight;
    
    float t = 1.0f;
    CMapChunk* hitChunk = NULL;
    
    bool hit = CMap::VectorIntersectTerrain(&start, &end, &t, 0, &hitChunk);
    
    if (hit) {
        return start.z + (end.z - start.z) * t;
    }
    
    return endHeight;
}
```

### Facet Query

**Purpose:** Get terrain facet (plane) at a given position

**Method:**
1. Cast a ray from above to below
2. Find intersection with terrain
3. Return facet plane

**Pseudocode:**
```c
bool GetTerrainFacet(float x, float y, float startHeight, float endHeight, C4Plane* plane) {
    C3Segment segment;
    segment.start.x = x;
    segment.start.y = y;
    segment.start.z = startHeight;
    
    segment.end.x = x;
    segment.end.y = y;
    segment.end.z = endHeight;
    
    float t = 1.0f;
    
    return CMap::GetFacetTerrain(&segment, &t, plane, 0);
}
```

### Triangle Query

**Purpose:** Get all terrain triangles within a bounding box

**Method:**
1. Create bounding box around position
2. Query terrain for triangles
3. Return triangle data

**Pseudocode:**
```c
bool GetTerrainTriangles(float x, float y, float size, CWTriData* triData) {
    CAaBox box;
    box.min.x = x - size;
    box.min.y = y - size;
    box.min.z = -1000.0f;
    
    box.max.x = x + size;
    box.max.y = y + size;
    box.max.z = 1000.0f;
    
    return CMap::GetTrisTerrain(&box, triData, 0);
}
```

---

## Implementation Guidelines

### Terrain Rendering Implementation

#### 1. Chunk Loading

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

#### 2. Area of Interest Management

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

#### 3. Collision Detection

```csharp
public class TerrainCollision
{
    public bool RayCast(C3Vector start, C3Vector end, out float t, out C3Vector point, out C4Plane plane)
    {
        t = 1.0f;
        point = end;
        plane = default;
        
        // Convert to chunk coordinates
        int startChunkX = WorldToChunkX(start.X, start.Y);
        int startChunkY = WorldToChunkY(start.X, start.Y);
        int endChunkX = WorldToChunkX(end.X, end.Y);
        int endChunkY = WorldToChunkY(end.X, end.Y);
        
        // Traverse chunks using DDA algorithm
        int dx = Math.Abs(endChunkX - startChunkX);
        int dy = Math.Abs(endChunkY - startChunkY);
        int sx = startChunkX < endChunkX ? 1 : -1;
        int sy = startChunkY < endChunkY ? 1 : -1;
        
        int err = dx - dy;
        int chunkX = startChunkX;
        int chunkY = startChunkY;
        
        while (true)
        {
            // Check if chunk is loaded
            if (terrainManager.TryGetChunk(chunkX, chunkY, out TerrainChunk chunk))
            {
                // Test intersection with chunk
                if (RayCastChunk(chunk, start, end, out float chunkT, out C3Vector chunkPoint, out C4Plane chunkPlane))
                {
                    if (chunkT < t)
                    {
                        t = chunkT;
                        point = chunkPoint;
                        plane = chunkPlane;
                    }
                }
            }
            
            // Check if we've reached the end
            if (chunkX == endChunkX && chunkY == endChunkY)
            {
                break;
            }
            
            // Move to next chunk
            int e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                chunkX += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                chunkY += sy;
            }
        }
        
        return t < 1.0f;
    }
    
    private bool RayCastChunk(TerrainChunk chunk, C3Vector start, C3Vector end, out float t, out C3Vector point, out C4Plane plane)
    {
        t = 1.0f;
        point = end;
        plane = default;
        
        // Transform to chunk space
        C3Vector localStart = start - chunk.Position;
        C3Vector localEnd = end - chunk.Position;
        
        // Iterate through cells
        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                // Check for holes
                if (chunk.IsHole(cellX, cellY))
                {
                    continue;
                }
                
                // Get triangle vertices
                int vertexIndex = cellY * 17 + cellX;
                C3Vector v0 = chunk.Vertices[vertexIndex];
                C3Vector v1 = chunk.Vertices[vertexIndex + 1];
                C3Vector v2 = chunk.Vertices[vertexIndex + 17];
                C3Vector v3 = chunk.Vertices[vertexIndex + 18];
                
                // Test intersection with first triangle
                if (RayTriangleIntersect(localStart, localEnd, v0, v1, v2, out float triT))
                {
                    if (triT < t)
                    {
                        t = triT;
                        point = localStart + (localEnd - localStart) * triT;
                        plane = CalculatePlane(v0, v1, v2);
                    }
                }
                
                // Test intersection with second triangle
                if (RayTriangleIntersect(localStart, localEnd, v1, v3, v2, out triT))
                {
                    if (triT < t)
                    {
                        t = triT;
                        point = localStart + (localEnd - localStart) * triT;
                        plane = CalculatePlane(v1, v3, v2);
                    }
                }
            }
        }
        
        if (t < 1.0f)
        {
            point = point + chunk.Position;
            return true;
        }
        
        return false;
    }
    
    private bool RayTriangleIntersect(C3Vector origin, C3Vector direction, C3Vector v0, C3Vector v1, C3Vector v2, out float t)
    {
        // Möller–Trumbore intersection algorithm
        C3Vector edge1 = v1 - v0;
        C3Vector edge2 = v2 - v0;
        C3Vector h = Vector3.Cross(direction, edge2);
        float a = Vector3.Dot(edge1, h);
        
        if (Math.Abs(a) < float.Epsilon)
        {
            t = 1.0f;
            return false;
        }
        
        float f = 1.0f / a;
        C3Vector s = origin - v0;
        float u = f * Vector3.Dot(s, h);
        
        if (u < 0.0f || u > 1.0f)
        {
            t = 1.0f;
            return false;
        }
        
        C3Vector q = Vector3.Cross(s, edge1);
        float v = f * Vector3.Dot(direction, q);
        
        if (v < 0.0f || u + v > 1.0f)
        {
            t = 1.0f;
            return false;
        }
        
        t = f * Vector3.Dot(edge2, q);
        return t >= 0.0f && t <= 1.0f;
    }
}
```

#### 4. Player Movement

```csharp
public class PlayerMovement
{
    private C3Vector position;
    private float facing;
    private float pitch;
    private bool isMoving;
    private bool isStrafing;
    private bool isFalling;
    private bool isJumping;
    private bool isRunning;
    private float velocity;
    private float strafeVelocity;
    private float verticalVelocity;
    
    public void Update(float deltaTime, TerrainCollision collision)
    {
        // Calculate movement direction
        C3Vector moveDirection = new C3Vector();
        
        if (isMoving)
        {
            moveDirection.X = (float)Math.Sin(facing);
            moveDirection.Y = (float)Math.Cos(facing);
        }
        
        if (isStrafing)
        {
            moveDirection.X += (float)Math.Sin(facing + Math.PI / 2);
            moveDirection.Y += (float)Math.Cos(facing + Math.PI / 2);
        }
        
        // Normalize movement direction
        if (moveDirection.Length() > 0)
        {
            moveDirection = Vector3.Normalize(moveDirection);
        }
        
        // Calculate speed
        float speed = isRunning ? 7.0f : 3.5f;
        float moveSpeed = speed * deltaTime;
        
        // Calculate new position
        C3Vector newPosition = position;
        newPosition.X += moveDirection.X * moveSpeed;
        newPosition.Y += moveDirection.Y * moveSpeed;
        
        // Apply gravity if falling
        if (isFalling)
        {
            verticalVelocity -= 9.8f * deltaTime;
            newPosition.Z += verticalVelocity * deltaTime;
        }
        
        // Apply jump velocity
        if (isJumping)
        {
            verticalVelocity = 8.0f;
            isJumping = false;
            isFalling = true;
        }
        
        // Check for terrain collision
        C3Vector groundPosition = newPosition;
        groundPosition.Z = position.Z + 100.0f;
        
        if (collision.RayCast(groundPosition, new C3Vector(newPosition.X, newPosition.Y, newPosition.Z - 200.0f), 
            out float t, out C3Vector point, out C4Plane plane))
        {
            // Check if we're below the ground
            if (newPosition.Z < point.Z)
            {
                // Snap to ground
                newPosition.Z = point.Z;
                
                // Stop falling
                if (isFalling)
                {
                    isFalling = false;
                    verticalVelocity = 0.0f;
                }
            }
        }
        else
        {
            // No ground below, start falling
            if (!isFalling)
            {
                isFalling = true;
                verticalVelocity = 0.0f;
            }
        }
        
        // Update position
        position = newPosition;
    }
    
    public void StartMove(bool forward)
    {
        isMoving = true;
        velocity = forward ? 1.0f : -1.0f;
    }
    
    public void StopMove()
    {
        isMoving = false;
        velocity = 0.0f;
    }
    
    public void StartStrafe(bool left)
    {
        isStrafing = true;
        strafeVelocity = left ? 1.0f : -1.0f;
    }
    
    public void StopStrafe()
    {
        isStrafing = false;
        strafeVelocity = 0.0f;
    }
    
    public void Jump()
    {
        if (!isFalling)
        {
            isJumping = true;
        }
    }
    
    public void SetFacing(float newFacing)
    {
        facing = newFacing;
    }
    
    public void SetPitch(float newPitch)
    {
        pitch = newPitch;
    }
    
    public void SetRunMode(bool run)
    {
        isRunning = run;
    }
}
```

---

## Conclusion

The WoW Alpha 0.5.3 world rendering and mesh traversal system is a sophisticated hierarchical chunk-based architecture that efficiently manages large outdoor environments. Key features include:

1. **Hierarchical Chunk System:** World divided into continents, zones, areas, chunks, and cells
2. **Efficient Culling:** Area of interest system only loads visible chunks
3. **Fast Collision:** Hierarchical collision detection with broad, medium, and narrow phases
4. **Smooth Movement:** Event-driven movement system with terrain following
5. **Flexible Rendering:** Support for multiple texture layers, doodads, and game objects

This architecture provides excellent performance and scalability, allowing the game to render large, detailed worlds while maintaining smooth frame rates.

---

## References

- [`RenderWorld`](0x004f2d00) (0x004f2d00) - Main world rendering entry point
- [`OnWorldUpdate`](0x004f2e30) (0x004f2e30) - Update world state before rendering
- [`CWorld::PrepareUpdate`](0x00663180) (0x00663180) - Prepare world for update
- [`CWorld::PrepareAreaOfInterest`](0x00665310) (0x00665310) - Calculate area of interest
- [`CWorldScene::PrepareRender`](0x0066a740) (0x0066a740) - Prepare world scene for rendering
- [`CMap::VectorIntersectTerrain`](0x00679690) (0x00679690) - Test ray intersection with terrain
- [`CMap::VectorIntersectSubchunks`](0x0067a7b0) (0x0067a7b0) - Test ray intersection with subchunks
- [`CMap::GetFacetTerrain`](0x0067bbd0) (0x0067bbd0) - Get terrain facet at position
- [`CMap::GetTrisTerrain`](0x0067c3c0) (0x0067c3c0) - Get terrain triangles in bounding box
- [`CMapChunk::CMapChunk`](0x00698510) (0x00698510) - Map chunk constructor
- [`CMapArea::CMapArea`](0x006aa880) (0x006aa880) - Map area constructor
- [`CMovement::UpdatePlayerMovement`](0x004c4d90) (0x004c4d90) - Process player movement events
