# MBMH (Map Blend Mesh Header)

## Type
ADT Chunk

## Source
ADT_v18.md

## Description
The MBMH chunk was introduced in Mists of Pandaria (MoP) and contains header information for blend meshes, which are used to create smooth terrain transitions between ADT terrain and WMO objects. It defines the properties of mesh blending for specific map objects, including references to vertex and index data in other chunks.

## Structure

```csharp
public struct MBMH
{
    public uint MapObjectID;      // Unique ID for the object
    public uint TextureId;        // Texture ID of the linked WMO
    public uint Unknown;          // Always zero?
    public uint MbmiCount;        // Record count in MBMI for this mesh
    public uint MbnvCount;        // Record count in MBNV for this mesh
    public uint MbmiStart;        // Start record index into MBMI for this mesh
    public uint MbnvStart;        // Start record index into MBNV for this mesh
}
```

## Properties

| Name | Type | Description |
|------|------|-------------|
| MapObjectID | uint32 | Unique identifier for the map object that this blend mesh applies to |
| TextureId | uint32 | Texture identifier of the linked WMO object |
| Unknown | uint32 | Reserved field, typically set to zero |
| MbmiCount | uint32 | Number of index records in the MBMI chunk for this mesh |
| MbnvCount | uint32 | Number of vertex records in the MBNV chunk for this mesh |
| MbmiStart | uint32 | Starting index into the MBMI array for this mesh's indices |
| MbnvStart | uint32 | Starting index into the MBNV array for this mesh's vertices |

## Dependencies

- **MBBB (C020)** - Contains bounding box information for each MBMH entry
- **MBNV (C021)** - Contains vertex data (positions, normals, texture coordinates) referenced by MBMH
- **MBMI (C022)** - Contains index data referenced by MBMH
- **MCBB (S010)** - Contains blend batches that reference MBMH data

## Implementation Notes

- Multiple MBMH entries can exist per map object when different textures are used
- The blend mesh system allows for smoother transitions between WMO objects and the surrounding terrain
- MBMH entries act as headers that organize vertex and index data stored in the MBNV and MBMI chunks
- These blend meshes are primarily used in higher-detail areas where terrain and structures need to blend seamlessly
- Found in both the root ADT file and LOD files

## Implementation Example

```csharp
public class BlendMeshManager
{
    private List<MBMH> _blendMeshHeaders = new List<MBMH>();
    private List<BoundingBox> _blendMeshBoundingBoxes = new List<BoundingBox>();
    private List<BlendMeshVertex> _vertices = new List<BlendMeshVertex>();
    private List<ushort> _indices = new List<ushort>();

    public void LoadBlendMeshData(string adtFilePath)
    {
        // Parse ADT file and extract MBMH, MBBB, MBNV, and MBMI chunks
        using (var reader = new BinaryReader(File.OpenRead(adtFilePath)))
        {
            // Read chunks...
            
            // Process MBMH entries
            foreach (var header in _blendMeshHeaders)
            {
                // Create mesh from vertices and indices
                var meshVertices = _vertices.Skip((int)header.MbnvStart)
                                           .Take((int)header.MbnvCount)
                                           .ToArray();
                
                var meshIndices = _indices.Skip((int)header.MbmiStart)
                                         .Take((int)header.MbmiCount)
                                         .ToArray();
                
                // Create and register the blend mesh
                CreateBlendMesh(header.MapObjectID, meshVertices, meshIndices);
            }
        }
    }
    
    private void CreateBlendMesh(uint mapObjectId, BlendMeshVertex[] vertices, ushort[] indices)
    {
        // Implementation to create and render the blend mesh
        // This would typically involve creating a mesh object in your rendering engine
    }
}

public struct BlendMeshVertex
{
    public Vector3 Position;
    public Vector3 Normal;
    public Vector2 TextureCoords;
    public Color[] Colors;  // Up to 3 colors depending on vertex format
}
```

## Usage Context

The MBMH chunk and related blend mesh chunks serve to enhance the visual quality of World of Warcraft terrain by creating smooth transitions between terrain and structures. Without blend meshes, there would be stark, unrealistic edges where WMO objects meet the terrain.

The blend mesh system was introduced in Mists of Pandaria to address issues with terrain-object integration, particularly in more detailed zones where the connection between structures and the surrounding landscape needed to appear more natural.

These meshes are typically small patches of triangles that connect a WMO's base to the surrounding terrain with appropriate texture blending. They're crucial for maintaining immersion, as they prevent players from seeing harsh transitions that would otherwise break the visual consistency of the game world. 