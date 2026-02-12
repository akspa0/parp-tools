# WoW Alpha 0.5.3 Mesh Traversal

## Overview

The WoW Alpha 0.5.3 mesh traversal system provides several methods for querying terrain height and surface information.

## Terrain Height Query

### Purpose

Find terrain height at a given position.

### Method

Cast a ray from above and find intersection with terrain.

### Pseudocode

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

### Usage

```csharp
// Get terrain height at player position
float groundHeight = GetTerrainHeight(playerPosition.X, playerPosition.Y, 100.0f, -200.0f);

// Snap player to ground
playerPosition.Z = groundHeight;
```

## Facet Query

### Purpose

Get terrain facet (plane) at a given position.

### Method

Cast a ray from above to below and return the facet plane.

### Pseudocode

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

### Usage

```csharp
// Get terrain facet at player position
C4Plane groundPlane;
bool hit = GetTerrainFacet(playerPosition.X, playerPosition.Y, 100.0f, -200.0f, out groundPlane);

if (hit) {
    // Use ground plane for surface normal
    C3Vector groundNormal = new C3Vector(groundPlane.n.X, groundPlane.n.Y, groundPlane.n.Z);
    
    // Align player to surface
    playerFacing = AlignToSurface(playerFacing, groundNormal);
}
```

## Triangle Query

### Purpose

Get all terrain triangles within a bounding box.

### Method

Create a bounding box around position and query terrain for triangles.

### Pseudocode

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

### Usage

```csharp
// Get terrain triangles around player position
CWTriData triData;
bool found = GetTerrainTriangles(playerPosition.X, playerPosition.Y, 10.0f, out triData);

if (found) {
    // Use triangles for detailed collision
    for (int i = 0; i < triData.nBatches; i++) {
        RenderBatch(triData.batches[i]);
    }
}
```

## Walkable Surface Determination

### Purpose

Determine if a surface is walkable.

### Method

Check terrain height and surface normal to determine if surface is walkable.

### Pseudocode

```c
bool IsSurfaceWalkable(float x, float y, float z) {
    // Get terrain height at position
    float groundHeight = GetTerrainHeight(x, y, z + 100.0f, z - 200.0f);
    
    // Check if position is close to ground
    if (Math.Abs(z - groundHeight) < 0.5f) {
        // Get terrain facet
        C4Plane groundPlane;
        bool hit = GetTerrainFacet(x, y, z + 100.0f, z - 200.0f, out groundPlane);
        
        if (hit) {
            // Check surface normal
            C3Vector normal = new C3Vector(groundPlane.n.X, groundPlane.n.Y, groundPlane.n.Z);
            
            // Surface is walkable if normal is mostly up
            if (normal.Z > 0.7f) {
                return true;
            }
        }
    }
    
    return false;
}
```

### Usage

```csharp
// Check if player can walk on surface
bool canWalk = IsSurfaceWalkable(playerPosition.X, playerPosition.Y, playerPosition.Z);

if (canWalk) {
    // Allow movement
    playerMovement.StartMove(true);
} else {
    // Prevent movement
    playerMovement.StopMove();
}
```

## Implementation Guidelines

### C# Mesh Traversal

```csharp
public class MeshTraversal
{
    private TerrainCollision collision;
    
    public float GetTerrainHeight(float x, float y)
    {
        // Cast ray from above to below
        C3Vector start = new C3Vector(x, y, 100.0f);
        C3Vector end = new C3Vector(x, y, -200.0f);
        
        float t;
        C3Vector point;
        C4Plane plane;
        
        if (collision.RayCast(start, end, out t, out point, out plane))
        {
            return start.Z + (end.Z - start.Z) * t;
        }
        
        return end.Z;
    }
    
    public bool GetTerrainFacet(float x, float y, float z, out C4Plane plane)
    {
        // Cast ray from above to below
        C3Vector start = new C3Vector(x, y, z + 100.0f);
        C3Vector end = new C3Vector(x, y, z - 200.0f);
        
        float t;
        C3Vector point;
        
        if (collision.RayCast(start, end, out t, out point, out plane))
        {
            return true;
        }
        
        plane = default;
        return false;
    }
    
    public bool GetTerrainTriangles(float x, float y, float size, out CWTriData triData)
    {
        // Create bounding box
        CAaBox box = new CAaBox();
        box.Min.X = x - size;
        box.Min.Y = y - size;
        box.Min.Z = -1000.0f;
        
        box.Max.X = x + size;
        box.Max.Y = y + size;
        box.Max.Z = 1000.0f;
        
        // Query terrain for triangles
        return collision.GetTriangles(box, out triData);
    }
    
    public bool IsSurfaceWalkable(float x, float y, float z)
    {
        // Get terrain height at position
        float groundHeight = GetTerrainHeight(x, y);
        
        // Check if position is close to ground
        if (Math.Abs(z - groundHeight) < 0.5f)
        {
            // Get terrain facet
            C4Plane groundPlane;
            if (GetTerrainFacet(x, y, z, out groundPlane))
            {
                // Check surface normal
                C3Vector normal = new C3Vector(groundPlane.n.X, groundPlane.n.Y, groundPlane.n.Z);
                
                // Surface is walkable if normal is mostly up
                if (normal.Z > 0.7f)
                {
                    return true;
                }
            }
        }
        
        return false;
    }
}
```

## References

- [`CMap::VectorIntersectTerrain`](0x00679690) (0x00679690) - Test ray intersection with terrain
- [`CMap::GetFacetTerrain`](0x0067bbd0) (0x0067bbd0) - Get terrain facet at position
- [`CMap::GetTrisTerrain`](0x0067c3c0) (0x0067c3c0) - Get terrain triangles in bounding box
