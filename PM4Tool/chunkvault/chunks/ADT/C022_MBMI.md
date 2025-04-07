# MBMI (Map Blend Mesh Indices)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MBMI chunk was introduced in Mists of Pandaria (MoP) and contains index data for blend meshes. These indices reference vertices stored in the MBNV chunk and define the triangles that make up blend meshes, which create smooth transitions between terrain and WMO objects.

## Structure

```csharp
public struct MBMI
{
    // Each value is an index into the MBNV vertex array
    public ushort[] Indices;  // Size determined by count in corresponding MBMH entry
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| Indices | ushort[] | Array of indices that reference vertices in the MBNV chunk |

## Dependencies

- **MBMH (C019)** - Contains header information including the count and start offset for MBMI entries
- **MBNV (C021)** - Contains the vertex data that MBMI indices reference
- **MCBB (S010)** - Contains blend batches that use indices from MBMI

## Implementation Notes

- Indices are stored as 16-bit unsigned integers (ushort) to save space
- Indices are grouped into triangles (3 indices per triangle)
- Each MBMH entry specifies the count and start offset for accessing a specific range of MBMI values
- The index values are relative to the MBNV array, not absolute memory positions
- Triangles follow a counter-clockwise winding order for front-face determination

## Implementation Example

```csharp
public class BlendMeshBuilder
{
    private List<MBNV> vertices;
    private List<ushort> indices;
    
    public BlendMeshBuilder(MBMH header, List<MBNV> allVertices, List<ushort> allIndices)
    {
        // Extract the vertices and indices for this specific blend mesh
        int vertexStart = header.MbnvStart;
        int vertexCount = header.MbnvCount;
        int indexStart = header.MbmiStart;
        int indexCount = header.MbmiCount;
        
        // Get the relevant vertices
        vertices = allVertices.Skip(vertexStart).Take(vertexCount).ToList();
        
        // Get the relevant indices
        indices = allIndices.Skip(indexStart).Take(indexCount).ToList();
        
        // Adjust indices to be relative to our local vertex array
        for (int i = 0; i < indices.Count; i++)
        {
            indices[i] = (ushort)(indices[i] - vertexStart);
        }
    }
    
    public Mesh BuildMesh()
    {
        var mesh = new Mesh();
        
        // Convert MBNV vertices to engine-specific vertex format
        mesh.SetVertices(vertices.Select(v => v.Position).ToArray());
        mesh.SetNormals(vertices.Select(v => v.Normal).ToArray());
        mesh.SetUVs(vertices.Select(v => v.TextureCoords).ToArray());
        mesh.SetColors(vertices.Select(v => v.Color1).ToArray());
        
        // Set triangles using indices
        // Assuming indices are stored as triplets representing triangles
        mesh.SetTriangles(indices.ToArray(), 0);
        
        return mesh;
    }
    
    public int GetTriangleCount()
    {
        return indices.Count / 3;
    }
}
```

## Usage Context

The MBMI chunk works in conjunction with the MBNV and MBMH chunks to define the geometry of blend meshes. These meshes are specifically designed to create smooth transitions between World of Warcraft's terrain and placed objects (WMOs).

Prior to the introduction of blend meshes in Mists of Pandaria, the game often had visible seams or harsh transitions where objects met the terrain. The blend mesh system resolved this issue by creating additional geometry that bridges these areas with proper visual blending.

When processing an ADT file, a client application would:
1. Load all MBNV vertex data
2. Load all MBMI index data
3. Use the MBMH entries to identify which ranges of vertices and indices belong to specific blend meshes
4. Construct the triangle meshes based on these indices
5. Apply the appropriate textures and rendering properties
6. Render the blend meshes in the correct world position to create seamless transitions

This indexing approach allows for efficient memory usage since the same vertex can be referenced by multiple triangles, reducing redundant data storage. It also facilitates more complex mesh topologies that would be difficult to represent with simple grid-based approaches. 