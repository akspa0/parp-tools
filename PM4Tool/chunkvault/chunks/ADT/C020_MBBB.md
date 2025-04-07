# MBBB (Map Blend Bounding Box)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MBBB chunk was introduced in Mists of Pandaria (MoP) and contains bounding box information for blend meshes. Each MBBB entry corresponds to an MBMH entry with the same index and defines the spatial boundaries of the blend mesh used for culling and visibility determination.

## Structure

```csharp
public struct MBBB
{
    public uint MapObjectID;    // Unique ID for the object (repeated from MBMH)
    public CAaBox Bounding;     // Axis-aligned bounding box
}

public struct CAaBox
{
    public Vector3 Min;         // Minimum point (lower bounds)
    public Vector3 Max;         // Maximum point (upper bounds)
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| MapObjectID | uint32 | Unique identifier for the map object, repeated from the corresponding MBMH chunk |
| Bounding | CAaBox | Axis-aligned bounding box that contains the entire blend mesh |

## Dependencies

- **MBMH (C019)** - Each MBBB entry corresponds to an MBMH entry with the same index
- **MBNV (C021)** - The vertices referenced by MBMH are contained within this bounding box

## Implementation Notes

- The MapObjectID is repeated from the MBMH chunk for unknown reasons, possibly for validation or to allow independent loading
- Bounding boxes are used for spatial culling during rendering to quickly determine if a blend mesh is visible
- The axis-aligned bounding box (AABB) is defined by minimum and maximum points in 3D space
- Blend mesh bounding boxes are found in both the root ADT file and LOD files
- These bounding boxes are essential for efficient rendering, especially in areas with many blend meshes

## Implementation Example

```csharp
public class BlendMeshCulling
{
    private Dictionary<uint, BoundingBox> _objectBoundingBoxes = new Dictionary<uint, BoundingBox>();
    
    public void LoadBoundingBoxes(List<MBBB> boundingBoxes)
    {
        foreach (var box in boundingBoxes)
        {
            _objectBoundingBoxes[box.MapObjectID] = new BoundingBox(
                new Vector3(box.Bounding.Min.X, box.Bounding.Min.Y, box.Bounding.Min.Z),
                new Vector3(box.Bounding.Max.X, box.Bounding.Max.Y, box.Bounding.Max.Z)
            );
        }
    }
    
    public bool IsVisible(uint objectId, Camera camera)
    {
        if (!_objectBoundingBoxes.TryGetValue(objectId, out var boundingBox))
        {
            return false;
        }
        
        // Check if the bounding box is visible to the camera
        return camera.Frustum.Intersects(boundingBox);
    }
    
    public float GetDistanceSquared(uint objectId, Vector3 position)
    {
        if (!_objectBoundingBoxes.TryGetValue(objectId, out var boundingBox))
        {
            return float.MaxValue;
        }
        
        // Calculate the distance from the position to the closest point on the bounding box
        return Vector3.DistanceSquared(position, boundingBox.ClosestPoint(position));
    }
}
```

## Usage Context

The MBBB chunk works in tandem with the MBMH chunk to provide efficient rendering and culling of blend meshes in World of Warcraft. These blend meshes are used to create smooth transitions between terrain and WMO objects.

Bounding boxes are a fundamental optimization technique in 3D graphics, allowing the rendering engine to quickly determine whether a complex mesh might be visible to the camera. This is particularly important for blend meshes, which are typically only visible when the player is relatively close to the transition area between terrain and structures.

By providing tight-fitting bounding boxes for each blend mesh, the game engine can make rapid decisions about which blend meshes need to be rendered, significantly improving performance in areas with many WMO objects and their associated blend meshes. 