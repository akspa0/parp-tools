# WoW Alpha 0.5.3 Collision Detection

## Overview

The WoW Alpha 0.5.3 collision system uses a hierarchical approach with three phases: broad phase (chunk-level culling), medium phase (subchunk-level culling), and narrow phase (triangle-level intersection).

## Collision Hierarchy

```
World
└── Chunk Level (Broad Phase)
    └── Subchunk Level (Medium Phase)
        └── Triangle Level (Narrow Phase)
```

## Terrain Intersection

**Address:** [`CMap::VectorIntersectTerrain`](0x00679690) (0x00679690)

### Purpose

Test ray intersection with terrain.

### Parameters

```c
bool VectorIntersectTerrain(
    C3Vector* start,      // Ray start position
    C3Vector* end,        // Ray end position
    float* t,             // Output intersection parameter (0-1)
    uint flags,            // Collision flags
    CMapChunk** hitChunk   // Output chunk that was hit
);
```

### Algorithm

1. Convert ray to chunk coordinates
2. Calculate chunk rectangle
3. Determine traversal direction (X or Y dominant)
4. Traverse chunks using DDA algorithm
5. Test intersection with subchunks
6. Return closest intersection

### Pseudocode

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

## Subchunk Intersection

**Address:** [`CMap::VectorIntersectSubchunks`](0x0067a7b0) (0x0067a7b0)

### Purpose

Test ray intersection with terrain subchunks.

### Key Operations

1. Iterate through subchunks in collision list
2. For each subchunk:
   - Check if subchunk is in same area
   - Get chunk from area table
   - Test intersection with doodads (if flag set)
   - Test intersection with game objects (if flag set)
   - Test intersection with terrain triangles
3. Return closest intersection

### Pseudocode

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
                    float triT = ip0 / (ip0 - ip1);
                    if (triT >= 0.0f && triT <= 1.0f) {
                        // Calculate intersection point
                        C3Vector ip;
                        ip.x = (localEndX - localStartX) * triT + localStartX;
                        ip.y = (localEndY - localStartY) * triT + localStartY;
                        ip.z = (localEndZ - localStartZ) * triT + localStartZ;
                        
                        // Test intersection with triangle
                        bool hit = VectorIntersectTri(&ip, 
                            &chunk->vertexList[vertexIndex + 9],
                            &chunk->vertexList[vertexIndex + triangleIndices[i * 2]],
                            &chunk->vertexList[vertexIndex + triangleIndices[i * 2 + 1]],
                            plane);
                        
                        if (hit && triT < hitT) {
                            hitT = triT;
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

## Facet Retrieval

**Address:** [`CMap::GetFacetTerrain`](0x0067bbd0) (0x0067bbd0)

### Purpose

Get terrain facet (plane) at a given position.

### Parameters

```c
bool GetFacetTerrain(
    C3Segment* segment,  // Line segment to test
    float* t,            // Output intersection parameter
    C4Plane* plane,       // Output facet plane
    uint flags            // Collision flags
);
```

### Algorithm

1. Convert segment to chunk coordinates
2. Calculate chunk rectangle
3. Determine traversal direction
4. Traverse chunks using DDA algorithm
5. Test intersection with subchunks
6. Return facet plane

### Pseudocode

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

## Triangle Retrieval

**Address:** [`CMap::GetTrisTerrain`](0x0067c3c0) (0x0067c3c0)

### Purpose

Get all terrain triangles within a bounding box.

### Parameters

```c
bool GetTrisTerrain(
    CAaBox* box,      // Bounding box to test
    CWTriData* triData, // Output triangle data
    uint flags           // Collision flags
);
```

### Algorithm

1. Convert bounding box to chunk coordinates
2. Calculate chunk rectangle
3. Iterate through chunks in rectangle
4. For each chunk:
   - Check for holes
   - Get triangles within bounding box
   - Add to triangle data
5. Return true if any triangles found

### Pseudocode

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

## Implementation Guidelines

### C# Collision Detection

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
    
    private C4Plane CalculatePlane(C3Vector v0, C3Vector v1, C3Vector v2)
    {
        C3Vector normal = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
        float d = -Vector3.Dot(normal, v0);
        return new C4Plane(normal, d);
    }
}
```

## References

- [`CMap::VectorIntersectTerrain`](0x00679690) (0x00679690) - Test ray intersection with terrain
- [`CMap::VectorIntersectSubchunks`](0x0067a7b0) (0x0067a7b0) - Test ray intersection with subchunks
- [`CMap::GetFacetTerrain`](0x0067bbd0) (0x0067bbd0) - Get terrain facet at position
- [`CMap::GetTrisTerrain`](0x0067c3c0) (0x0067c3c0) - Get terrain triangles in bounding box
