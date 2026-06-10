# WoW Alpha 0.5.3 Frustum Culling

## Overview

The WoW Alpha 0.5.3 frustum culling system uses view-projection matrix plane extraction to determine which objects are visible within the camera's view frustum.

## Frustum Planes

### Purpose

Extract the 6 frustum planes from the view-projection matrix for visibility testing.

### Frustum Plane Order

```
0: Left plane
1: Right plane
2: Top plane
3: Bottom plane
4: Near plane
5: Far plane
```

### Plane Extraction Algorithm

```c
void UpdateFrustum(C44Matrix* viewProjMatrix, FrustumPlanes* frustumPlanes) {
    // Left plane: row4 + row1
    frustumPlanes[0].a = viewProjMatrix.m41 + viewProjMatrix.m11;
    frustumPlanes[0].b = viewProjMatrix.m42 + viewProjMatrix.m12;
    frustumPlanes[0].c = viewProjMatrix.m43 + viewProjMatrix.m13;
    frustumPlanes[0].d = viewProjMatrix.m44 + viewProjMatrix.m14;
    
    // Right plane: row4 - row1
    frustumPlanes[1].a = viewProjMatrix.m41 - viewProjMatrix.m11;
    frustumPlanes[1].b = viewProjMatrix.m42 - viewProjMatrix.m12;
    frustumPlanes[1].c = viewProjMatrix.m43 - viewProjMatrix.m13;
    frustumPlanes[1].d = viewProjMatrix.m44 - viewProjMatrix.m14;
    
    // Top plane: row4 - row2
    frustumPlanes[2].a = viewProjMatrix.m41 - viewProjMatrix.m21;
    frustumPlanes[2].b = viewProjMatrix.m42 - viewProjMatrix.m22;
    frustumPlanes[2].c = viewProjMatrix.m43 - viewProjMatrix.m23;
    frustumPlanes[2].d = viewProjMatrix.m44 - viewProjMatrix.m24;
    
    // Bottom plane: row4 + row2
    frustumPlanes[3].a = viewProjMatrix.m41 + viewProjMatrix.m21;
    frustumPlanes[3].b = viewProjMatrix.m42 + viewProjMatrix.m22;
    frustumPlanes[3].c = viewProjMatrix.m43 + viewProjMatrix.m23;
    frustumPlanes[3].d = viewProjMatrix.m44 + viewProjMatrix.m24;
    
    // Near plane: row4 + row3
    frustumPlanes[4].a = viewProjMatrix.m41 + viewProjMatrix.m31;
    frustumPlanes[4].b = viewProjMatrix.m42 + viewProjMatrix.m32;
    frustumPlanes[4].c = viewProjMatrix.m43 + viewProjMatrix.m33;
    frustumPlanes[4].d = viewProjMatrix.m44 + viewProjMatrix.m34;
    
    // Far plane: row4 - row3
    frustumPlanes[5].a = viewProjMatrix.m41 - viewProjMatrix.m31;
    frustumPlanes[5].b = viewProjMatrix.m42 - viewProjMatrix.m32;
    frustumPlanes[5].c = viewProjMatrix.m43 - viewProjMatrix.m33;
    frustumPlanes[5].d = viewProjMatrix.m44 - viewProjMatrix.m34;
    
    // Normalize planes
    for (int i = 0; i < 6; i++) {
        float length = sqrt(frustumPlanes[i].a * frustumPlanes[i].a + 
                           frustumPlanes[i].b * frustumPlanes[i].b + 
                           frustumPlanes[i].c * frustumPlanes[i].c);
        frustumPlanes[i].a /= length;
        frustumPlanes[i].b /= length;
        frustumPlanes[i].c /= length;
        frustumPlanes[i].d /= length;
    }
}
```

## Point Frustum Test

### Purpose

Test if a point is within the view frustum.

### Algorithm

```c
bool IsPointInFrustum(C3Vector* point, FrustumPlanes* frustumPlanes) {
    for (int i = 0; i < 6; i++) {
        float distance = frustumPlanes[i].a * point->x +
                       frustumPlanes[i].b * point->y +
                       frustumPlanes[i].c * point->z +
                       frustumPlanes[i].d;
        if (distance < 0.0f) {
            return false;
        }
    }
    return true;
}
```

## Sphere Frustum Test

### Purpose

Test if a sphere is within or intersecting the view frustum.

### Algorithm

```c
bool IsSphereInFrustum(C3Vector* center, float radius, FrustumPlanes* frustumPlanes) {
    for (int i = 0; i < 6; i++) {
        float distance = frustumPlanes[i].a * center->x +
                       frustumPlanes[i].b * center->y +
                       frustumPlanes[i].c * center->z +
                       frustumPlanes[i].d;
        if (distance < -radius) {
            return false;
        }
    }
    return true;
}
```

## AABB Frustum Test

### Purpose

Test if an axis-aligned bounding box is within the view frustum.

### Algorithm

```c
bool IsAABBInFrustum(CAaBox* box, FrustumPlanes* frustumPlanes) {
    C3Vector corners[8];
    
    // Get box corners
    corners[0] = box->min;
    corners[1] = C3Vector(box->max.x, box->min.y, box->min.z);
    corners[2] = C3Vector(box->min.x, box->max.y, box->min.z);
    corners[3] = C3Vector(box->max.x, box->max.y, box->min.z);
    corners[4] = C3Vector(box->min.x, box->min.y, box->max.z);
    corners[5] = C3Vector(box->max.x, box->min.y, box->max.z);
    corners[6] = C3Vector(box->min.x, box->max.y, box->max.z);
    corners[7] = box->max;
    
    // Test each corner against each plane
    for (int i = 0; i < 6; i++) {
        int inside = 0;
        for (int j = 0; j < 8; j++) {
            float distance = frustumPlanes[i].a * corners[j].x +
                           frustumPlanes[i].b * corners[j].y +
                           frustumPlanes[i].c * corners[j].z +
                           frustumPlanes[i].d;
            if (distance >= 0.0f) {
                inside++;
            }
        }
        if (inside == 0) {
            return false;
        }
    }
    return true;
}
```

## Chunk Frustum Test

### Purpose

Test if a terrain chunk is visible within the view frustum.

### Algorithm

```c
bool IsChunkInFrustum(CMapChunk* chunk, FrustumPlanes* frustumPlanes) {
    // Get chunk bounding box
    CAaBox chunkBox;
    chunkBox.min.x = chunk->position.x - CHUNK_SIZE / 2.0f;
    chunkBox.min.y = chunk->position.y - CHUNK_SIZE / 2.0f;
    chunkBox.min.z = chunk->minHeight;
    chunkBox.max.x = chunk->position.x + CHUNK_SIZE / 2.0f;
    chunkBox.max.y = chunk->position.y + CHUNK_SIZE / 2.0f;
    chunkBox.max.z = chunk->maxHeight;
    
    return IsAABBInFrustum(&chunkBox, frustumPlanes);
}
```

## WMO Frustum Test

### Purpose

Test if a World Map Object (WMO) is visible within the view frustum.

### Algorithm

```c
bool IsWMOInFrustum(CMapObjDef* wmo, FrustumPlanes* frustumPlanes) {
    // Get WMO bounding box in world space
    CAaBox wmoBox;
    wmoBox.min = wmo->boundingBox.min + wmo->position;
    wmoBox.max = wmo->boundingBox.max + wmo->position;
    
    return IsAABBInFrustum(&wmoBox, frustumPlanes);
}
```

## Doodad Frustum Test

### Purpose

Test if a doodad is visible within the view frustum.

### Algorithm

```c
bool IsDoodadInFrustum(CDoodadDef* doodad, FrustumPlanes* frustumPlanes) {
    // Get doodad bounding sphere
    C3Vector center = doodad->position;
    float radius = doodad->boundingRadius;
    
    return IsSphereInFrustum(&center, radius, frustumPlanes);
}
```

## Implementation Guidelines

### C# Frustum Culling

```csharp
public class Frustum
{
    public Plane[] Planes { get; set; } = new Plane[6];
    
    public void UpdateFrustum(Matrix4x4 viewProjMatrix)
    {
        // Left plane: row4 + row1
        Planes[0] = new Plane(
            viewProjMatrix.M41 + viewProjMatrix.M11,
            viewProjMatrix.M42 + viewProjMatrix.M12,
            viewProjMatrix.M43 + viewProjMatrix.M13,
            viewProjMatrix.M44 + viewProjMatrix.M14);
        
        // Right plane: row4 - row1
        Planes[1] = new Plane(
            viewProjMatrix.M41 - viewProjMatrix.M11,
            viewProjMatrix.M42 - viewProjMatrix.M12,
            viewProjMatrix.M43 - viewProjMatrix.M13,
            viewProjMatrix.M44 - viewProjMatrix.M14);
        
        // Top plane: row4 - row2
        Planes[2] = new Plane(
            viewProjMatrix.M41 - viewProjMatrix.M21,
            viewProjMatrix.M42 - viewProjMatrix.M22,
            viewProjMatrix.M43 - viewProjMatrix.M23,
            viewProjMatrix.M44 - viewProjMatrix.M24);
        
        // Bottom plane: row4 + row2
        Planes[3] = new Plane(
            viewProjMatrix.M41 + viewProjMatrix.M21,
            viewProjMatrix.M42 + viewProjMatrix.M22,
            viewProjMatrix.M43 + viewProjMatrix.M23,
            viewProjMatrix.M44 + viewProjMatrix.M24);
        
        // Near plane: row4 + row3
        Planes[4] = new Plane(
            viewProjMatrix.M41 + viewProjMatrix.M31,
            viewProjMatrix.M42 + viewProjMatrix.M32,
            viewProjMatrix.M43 + viewProjMatrix.M33,
            viewProjMatrix.M44 + viewProjMatrix.M34);
        
        // Far plane: row4 - row3
        Planes[5] = new Plane(
            viewProjMatrix.M41 - viewProjMatrix.M31,
            viewProjMatrix.M42 - viewProjMatrix.M32,
            viewProjMatrix.M43 - viewProjMatrix.M33,
            viewProjMatrix.M44 - viewProjMatrix.M34);
        
        // Normalize planes
        for (int i = 0; i < 6; i++)
        {
            float length = (float)Math.Sqrt(
                Planes[i].Normal.X * Planes[i].Normal.X +
                Planes[i].Normal.Y * Planes[i].Normal.Y +
                Planes[i].Normal.Z * Planes[i].Normal.Z);
            Planes[i].Normal /= length;
            Planes[i].D /= length;
        }
    }
    
    public bool ContainsPoint(Vector3 point)
    {
        for (int i = 0; i < 6; i++)
        {
            float distance = Vector3.Dot(Planes[i].Normal, point) + Planes[i].D;
            if (distance < 0.0f)
            {
                return false;
            }
        }
        return true;
    }
    
    public bool IntersectsSphere(Vector3 center, float radius)
    {
        for (int i = 0; i < 6; i++)
        {
            float distance = Vector3.Dot(Planes[i].Normal, center) + Planes[i].D;
            if (distance < -radius)
            {
                return false;
            }
        }
        return true;
    }
    
    public bool IntersectsAABB(BoundingBox box)
    {
        Vector3[] corners = box.GetCorners();
        for (int i = 0; i < 6; i++)
        {
            int inside = 0;
            for (int j = 0; j < 8; j++)
            {
                float distance = Vector3.Dot(Planes[i].Normal, corners[j]) + Planes[i].D;
                if (distance >= 0.0f)
                {
                    inside++;
                }
            }
            if (inside == 0)
            {
                return false;
            }
        }
        return true;
    }
}

public class Plane
{
    public Vector3 Normal { get; set; }
    public float D { get; set; }
    
    public Plane(float a, float b, float c, float d)
    {
        Normal = new Vector3(a, b, c);
        D = d;
    }
}
```

## References

- [`CWorldScene::UpdateFrustum`](0x0066a460) (0x0066a460) - Update frustum planes
- [`CWorldScene::IsInFrustum`](0x0066a4a0) (0x0066a4a0) - Test if point is in frustum
- [`CWorldScene::IsChunkInFrustum`](0x0066a4e0) (0x0066a4e0) - Test if chunk is in frustum
- [`CWorldScene::IsWMOInFrustum`](0x0066a520) (0x0066a520) - Test if WMO is in frustum
- [`CWorldScene::IsDoodadInFrustum`](0x0066a560) (0x0066a560) - Test if doodad is in frustum
