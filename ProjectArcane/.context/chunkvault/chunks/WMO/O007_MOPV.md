# O007: MOPV

## Type
WMO Root Chunk

## Source
WMO.md

## Description
The MOPV (Map Object Portal Vertices) chunk contains a list of 3D coordinates that define the vertices of portals in the WMO. Portals are used for visibility determination between connected groups, allowing the engine to efficiently render only what is potentially visible from the player's current position. Each portal is defined as a convex polygon made up of vertices from this array.

## Structure
```csharp
struct MOPV
{
    C3Vector[] vertices;  // Array of 3D vertices
};

struct C3Vector
{
    /*0x00*/ float x;
    /*0x04*/ float y;
    /*0x08*/ float z;
};
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| vertices | C3Vector[] | Array of 3D vectors defining portal vertices |

## C3Vector Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | x | float | X coordinate in model space |
| 0x04 | y | float | Y coordinate in model space |
| 0x08 | z | float | Z coordinate in model space |

## Dependencies
- MOHD: The nPortals field indicates how many portals reference these vertices
- MOPT: References vertices from this chunk to define portal polygons

## Implementation Notes
- Each vertex is a 3D point in model space (12 bytes: x, y, z as floats)
- Portal vertices are shared between multiple portals for memory efficiency
- The MOPT chunk references vertices from this array to define complete portal polygons
- The total number of vertices in this chunk depends on the complexity of all portals in the WMO
- All coordinates are in the WMO's model space
- Vertices are typically arranged in counter-clockwise order when referenced by portals
- The chunk size must be a multiple of 12 (size of C3Vector)

## Implementation Example
```csharp
public class MOPV : IChunk
{
    public List<Vector3> Vertices { get; private set; }
    
    public MOPV()
    {
        Vertices = new List<Vector3>();
    }
    
    public void Parse(BinaryReader reader, long size)
    {
        // Calculate how many vertices we expect
        int vertexCount = (int)(size / 12); // Each vertex is 12 bytes (3 floats)
        
        Vertices.Clear();
        
        for (int i = 0; i < vertexCount; i++)
        {
            float x = reader.ReadSingle();
            float y = reader.ReadSingle();
            float z = reader.ReadSingle();
            
            Vertices.Add(new Vector3(x, y, z));
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (Vector3 vertex in Vertices)
        {
            writer.Write(vertex.X);
            writer.Write(vertex.Y);
            writer.Write(vertex.Z);
        }
    }
    
    public void AddVertex(float x, float y, float z)
    {
        Vertices.Add(new Vector3(x, y, z));
    }
    
    public void AddVertex(Vector3 vertex)
    {
        Vertices.Add(vertex);
    }
    
    public Vector3 GetVertex(int index)
    {
        if (index >= 0 && index < Vertices.Count)
        {
            return Vertices[index];
        }
        
        throw new IndexOutOfRangeException($"Vertex index {index} is out of range. Valid range: 0-{Vertices.Count - 1}");
    }
}
```

## Validation Requirements
- The chunk size must be a multiple of 12 (size of C3Vector)
- All vertices should be within the overall bounding box of the WMO
- Vertices referenced by portals in MOPT should exist within this chunk
- The number of vertices should be sufficient for all portal definitions

## Usage Context
The MOPV chunk provides the geometric definition for portals in the WMO. Portals serve several critical functions in the rendering pipeline:

1. **Visibility Determination**: The engine uses portals to determine which groups are potentially visible from the player's position
2. **Culling Optimization**: By only rendering what can be seen through portals, performance is greatly improved
3. **Group Connectivity**: Portals define logical connections between groups, indicating how they relate to each other spatially
4. **Occlusion Boundaries**: Portals mark transitions between occluded and visible areas

When rendering a WMO:
1. The engine starts with the group containing the camera
2. It checks which portals are visible from the camera position
3. For each visible portal, it adds the connected group to the render queue
4. This process continues recursively until all potentially visible groups are identified
5. Only groups determined to be potentially visible are rendered

Portals are typically placed at doorways, windows, and other openings between rooms or sections of a building. The portal vertices define the exact shape of these openings, allowing the engine to perform precise visibility tests. This system is especially important for complex interior spaces where traditional frustum culling alone would be inefficient. 