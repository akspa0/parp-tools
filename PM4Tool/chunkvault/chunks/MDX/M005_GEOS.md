# GEOS - MDX Geoset Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The GEOS (Geoset) chunk defines the geometry for the MDX model. It contains vertex positions, normals, texture coordinates, face definitions, vertex groups, and material references. Each geoset represents a discrete mesh component that can be animated independently.

## Structure

```csharp
public struct GEOS
{
    /// <summary>
    /// Array of geoset data
    /// </summary>
    // Array of MDLGEOSET structures
}

public struct MDLGEOSET
{
    /// <summary>
    /// Flags for the geoset
    /// </summary>
    public uint flags;
    
    /// <summary>
    /// Number of vertices in the geoset
    /// </summary>
    public uint numVertices;
    
    /// <summary>
    /// Vertex positions (XYZ)
    /// </summary>
    // Vector3 vertices[numVertices] follows
    
    /// <summary>
    /// Normal vectors for each vertex
    /// </summary>
    // Vector3 normals[numVertices] follows
    
    /// <summary>
    /// Texture coordinate type (typically 0, 1, or 2)
    /// </summary>
    public uint textureCoordType;
    
    /// <summary>
    /// Number of texture coordinate sets
    /// </summary>
    public uint numTextureCoordSets;
    
    /// <summary>
    /// Texture coordinates for each set
    /// </summary>
    // Vector2 textureCoords[numTextureCoordSets][numVertices] follows
    
    /// <summary>
    /// Material ID for this geoset
    /// </summary>
    public uint materialId;
    
    /// <summary>
    /// Selection group for the geoset
    /// </summary>
    public uint selectionGroup;
    
    /// <summary>
    /// Selection flags for the geoset
    /// </summary>
    public uint selectionFlags;
    
    /// <summary>
    /// Number of position indices (equal to number of vertices)
    /// </summary>
    public uint numPositionIndices;
    
    /// <summary>
    /// Indices of vertices used by faces
    /// </summary>
    // uint positionIndices[numPositionIndices] follows
    
    /// <summary>
    /// Number of faces
    /// </summary>
    public uint numFaces;
    
    /// <summary>
    /// Face indices (triangles, CCW winding)
    /// </summary>
    // ushort faceIndices[numFaces*3] follows
    
    /// <summary>
    /// Vertex group sizes (how many bones influence each vertex)
    /// </summary>
    // byte groupSizes[numVertices] follows
    
    /// <summary>
    /// Number of bone groups
    /// </summary>
    public uint numBoneGroups;
    
    /// <summary>
    /// Bone group matrices
    /// </summary>
    // uint boneGroups[numBoneGroups][?] follows
    
    /// <summary>
    /// Maximum number of bone influences per vertex
    /// </summary>
    public uint maxBonesPerVertex;
    
    /// <summary>
    /// Bone indices for vertex influences
    /// </summary>
    // uint boneIndices[?] follows
    
    /// <summary>
    /// Bone weights for vertex influences
    /// </summary>
    // float boneWeights[?] follows
    
    /// <summary>
    /// Minimum corner of geoset bounding box
    /// </summary>
    public Vector3 boundingBoxMin;
    
    /// <summary>
    /// Maximum corner of geoset bounding box
    /// </summary>
    public Vector3 boundingBoxMax;
    
    /// <summary>
    /// Radius of bounding sphere
    /// </summary>
    public float boundingRadius;
}
```

## Properties

### GEOS Chunk
The GEOS chunk itself has no properties beyond the geoset array. Its size is determined by the number and size of MDLGEOSET structures it contains.

### MDLGEOSET Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | flags | uint | Geoset flags |
| 0x04 | numVertices | uint | Number of vertices in the geoset |
| 0x08 | ... | Vector3[] | Vertex positions array (numVertices entries) |
| varies | ... | Vector3[] | Normal vectors array (numVertices entries) |
| varies | textureCoordType | uint | Type of texture coordinates |
| varies+4 | numTextureCoordSets | uint | Number of texture coordinate sets |
| varies+8 | ... | Vector2[][] | Texture coordinate sets array (numTextureCoordSets x numVertices) |
| varies | materialId | uint | Material ID referenced in MTLS chunk |
| varies+4 | selectionGroup | uint | Selection group for editor use |
| varies+8 | selectionFlags | uint | Selection type flags |
| varies+12 | numPositionIndices | uint | Number of vertex indices (should equal numVertices) |
| varies+16 | ... | uint[] | Position indices array (numPositionIndices entries) |
| varies | numFaces | uint | Number of triangular faces |
| varies+4 | ... | ushort[] | Face indices array (numFaces * 3 entries) |
| varies | ... | byte[] | Group sizes array (numVertices entries) |
| varies | numBoneGroups | uint | Number of bone groups |
| varies+4 | ... | uint[][] | Bone group matrices array |
| varies | maxBonesPerVertex | uint | Maximum bones influencing any vertex |
| varies | ... | uint[] | Bone indices for vertex influences |
| varies | ... | float[] | Bone weights for vertex influences |
| varies | boundingBoxMin | Vector3 | Minimum corner of bounding box |
| varies+12 | boundingBoxMax | Vector3 | Maximum corner of bounding box |
| varies+24 | boundingRadius | float | Radius of bounding sphere |

## Geoset Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | Unselectable | Geoset cannot be selected in the editor |
| 1-31 | Reserved | Reserved for future use, typically set to 0 |

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1400 (WoW Alpha) | Same structure as WC3 versions |
| 1500 (WoW Alpha) | Redesigned structure with additional fields for vertex color and optimizations |

## Dependencies
- MTLS - Referenced by materialId
- BONE - Referenced by bone indices for skinning

## Implementation Notes
- Vertices are organized as arrays of positions, normals, and texture coordinates
- Face indices are stored as unsigned shorts (16-bit values)
- Faces are triangles with counter-clockwise winding for front face determination
- The texture coordinate type is typically:
  - 0: Regular UV mapping
  - 1: Environment mapped (spherical)
  - 2: Environment mapped (cubic)
- Vertex skinning data connects vertices to bones:
  - Group sizes specify how many bones influence each vertex
  - Bone indices reference bones in the BONE chunk
  - Bone weights determine how much influence each bone has on a vertex
  - The sum of weights for a vertex should equal 1.0
- The material ID references a material in the MTLS chunk
- The selection group and flags are primarily for editor use
- Bounding box and radius are used for culling and collision detection
- Version 1500 made significant changes to the GEOS chunk structure

## Usage Context
The GEOS chunk provides:
- The 3D mesh data for rendering the model
- Vertex skinning information for skeletal animations
- Material assignments for texturing
- Bounding volumes for culling and collision
- Organizational grouping of model components

## Rendering System Integration
- Vertices, normals, and UVs provide the basic rendering data
- Material IDs connect geosets to textures and shaders
- Face indices define the triangles to render
- Bone indices and weights allow skeletal deformation
- Geoset flags control visibility and selection

## Implementation Example

```csharp
public class GEOSChunk : IMdxChunk
{
    public string ChunkId => "GEOS";
    
    public List<MdxGeoset> Geosets { get; private set; } = new List<MdxGeoset>();
    
    public void Parse(BinaryReader reader, long size, uint version)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + size;
        
        // Clear any existing geosets
        Geosets.Clear();
        
        // Read geosets until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var geoset = new MdxGeoset();
            
            // Handle different formats based on version
            if (version == 1500) // WoW Alpha v1500
            {
                ParseGeosetV1500(reader, geoset);
            }
            else // WC3 and WoW Alpha v1300-1400
            {
                ParseGeosetStandard(reader, geoset);
            }
            
            Geosets.Add(geoset);
        }
    }
    
    private void ParseGeosetStandard(BinaryReader reader, MdxGeoset geoset)
    {
        // Read flags and vertex count
        geoset.Flags = reader.ReadUInt32();
        geoset.NumVertices = reader.ReadUInt32();
        
        // Read vertex positions
        geoset.Vertices = new Vector3[geoset.NumVertices];
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            geoset.Vertices[i] = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
        }
        
        // Read normal vectors
        geoset.Normals = new Vector3[geoset.NumVertices];
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            geoset.Normals[i] = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle()
            );
        }
        
        // Read texture coordinate info
        geoset.TextureCoordType = reader.ReadUInt32();
        geoset.NumTextureCoordSets = reader.ReadUInt32();
        
        // Read texture coordinates
        geoset.TextureCoords = new Vector2[geoset.NumTextureCoordSets][];
        for (int set = 0; set < geoset.NumTextureCoordSets; set++)
        {
            geoset.TextureCoords[set] = new Vector2[geoset.NumVertices];
            for (int i = 0; i < geoset.NumVertices; i++)
            {
                geoset.TextureCoords[set][i] = new Vector2(
                    reader.ReadSingle(),
                    reader.ReadSingle()
                );
            }
        }
        
        // Read material, selection info
        geoset.MaterialId = reader.ReadUInt32();
        geoset.SelectionGroup = reader.ReadUInt32();
        geoset.SelectionFlags = reader.ReadUInt32();
        
        // Read position indices
        geoset.NumPositionIndices = reader.ReadUInt32();
        geoset.PositionIndices = new uint[geoset.NumPositionIndices];
        for (int i = 0; i < geoset.NumPositionIndices; i++)
        {
            geoset.PositionIndices[i] = reader.ReadUInt32();
        }
        
        // Read face indices
        geoset.NumFaces = reader.ReadUInt32();
        geoset.FaceIndices = new ushort[geoset.NumFaces * 3];
        for (int i = 0; i < geoset.NumFaces * 3; i++)
        {
            geoset.FaceIndices[i] = reader.ReadUInt16();
        }
        
        // Read vertex group info
        geoset.GroupSizes = new byte[geoset.NumVertices];
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            geoset.GroupSizes[i] = reader.ReadByte();
        }
        
        // Read bone groups
        geoset.NumBoneGroups = reader.ReadUInt32();
        geoset.BoneGroups = new uint[geoset.NumBoneGroups][];
        
        for (int i = 0; i < geoset.NumBoneGroups; i++)
        {
            // Read number of indices in this group
            uint numIndices = reader.ReadUInt32();
            geoset.BoneGroups[i] = new uint[numIndices];
            
            // Read indices
            for (int j = 0; j < numIndices; j++)
            {
                geoset.BoneGroups[i][j] = reader.ReadUInt32();
            }
        }
        
        // Read bone influence data
        geoset.MaxBonesPerVertex = reader.ReadUInt32();
        
        // Calculate total number of bone indices and weights
        int totalBoneInfluences = 0;
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            totalBoneInfluences += geoset.GroupSizes[i];
        }
        
        // Read bone indices
        geoset.BoneIndices = new uint[totalBoneInfluences];
        for (int i = 0; i < totalBoneInfluences; i++)
        {
            geoset.BoneIndices[i] = reader.ReadUInt32();
        }
        
        // Read bone weights
        geoset.BoneWeights = new float[totalBoneInfluences];
        for (int i = 0; i < totalBoneInfluences; i++)
        {
            geoset.BoneWeights[i] = reader.ReadSingle();
        }
        
        // Read bounding geometry
        geoset.BoundingBoxMin = new Vector3(
            reader.ReadSingle(),
            reader.ReadSingle(),
            reader.ReadSingle()
        );
        
        geoset.BoundingBoxMax = new Vector3(
            reader.ReadSingle(),
            reader.ReadSingle(),
            reader.ReadSingle()
        );
        
        geoset.BoundingRadius = reader.ReadSingle();
    }
    
    private void ParseGeosetV1500(BinaryReader reader, MdxGeoset geoset)
    {
        // Version 1500 has a different structure
        // This is a placeholder for the v1500 geoset parsing
        // The actual implementation would need to handle the differences in the v1500 format
        
        // Note: This would include parsing of vertex colors and other v1500-specific features
    }
    
    public void Write(BinaryWriter writer, uint version)
    {
        foreach (var geoset in Geosets)
        {
            if (version == 1500)
            {
                WriteGeosetV1500(writer, geoset);
            }
            else
            {
                WriteGeosetStandard(writer, geoset);
            }
        }
    }
    
    private void WriteGeosetStandard(BinaryWriter writer, MdxGeoset geoset)
    {
        // Write flags and vertex count
        writer.Write(geoset.Flags);
        writer.Write(geoset.NumVertices);
        
        // Write vertex positions
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            writer.Write(geoset.Vertices[i].X);
            writer.Write(geoset.Vertices[i].Y);
            writer.Write(geoset.Vertices[i].Z);
        }
        
        // Write normal vectors
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            writer.Write(geoset.Normals[i].X);
            writer.Write(geoset.Normals[i].Y);
            writer.Write(geoset.Normals[i].Z);
        }
        
        // Write texture coordinate info
        writer.Write(geoset.TextureCoordType);
        writer.Write(geoset.NumTextureCoordSets);
        
        // Write texture coordinates
        for (int set = 0; set < geoset.NumTextureCoordSets; set++)
        {
            for (int i = 0; i < geoset.NumVertices; i++)
            {
                writer.Write(geoset.TextureCoords[set][i].X);
                writer.Write(geoset.TextureCoords[set][i].Y);
            }
        }
        
        // Write material, selection info
        writer.Write(geoset.MaterialId);
        writer.Write(geoset.SelectionGroup);
        writer.Write(geoset.SelectionFlags);
        
        // Write position indices
        writer.Write(geoset.NumPositionIndices);
        for (int i = 0; i < geoset.NumPositionIndices; i++)
        {
            writer.Write(geoset.PositionIndices[i]);
        }
        
        // Write face indices
        writer.Write(geoset.NumFaces);
        for (int i = 0; i < geoset.NumFaces * 3; i++)
        {
            writer.Write(geoset.FaceIndices[i]);
        }
        
        // Write vertex group info
        for (int i = 0; i < geoset.NumVertices; i++)
        {
            writer.Write(geoset.GroupSizes[i]);
        }
        
        // Write bone groups
        writer.Write(geoset.NumBoneGroups);
        for (int i = 0; i < geoset.NumBoneGroups; i++)
        {
            writer.Write((uint)geoset.BoneGroups[i].Length);
            for (int j = 0; j < geoset.BoneGroups[i].Length; j++)
            {
                writer.Write(geoset.BoneGroups[i][j]);
            }
        }
        
        // Write bone influence data
        writer.Write(geoset.MaxBonesPerVertex);
        
        // Write bone indices
        for (int i = 0; i < geoset.BoneIndices.Length; i++)
        {
            writer.Write(geoset.BoneIndices[i]);
        }
        
        // Write bone weights
        for (int i = 0; i < geoset.BoneWeights.Length; i++)
        {
            writer.Write(geoset.BoneWeights[i]);
        }
        
        // Write bounding geometry
        writer.Write(geoset.BoundingBoxMin.X);
        writer.Write(geoset.BoundingBoxMin.Y);
        writer.Write(geoset.BoundingBoxMin.Z);
        
        writer.Write(geoset.BoundingBoxMax.X);
        writer.Write(geoset.BoundingBoxMax.Y);
        writer.Write(geoset.BoundingBoxMax.Z);
        
        writer.Write(geoset.BoundingRadius);
    }
    
    private void WriteGeosetV1500(BinaryWriter writer, MdxGeoset geoset)
    {
        // Version 1500 has a different structure
        // This is a placeholder for the v1500 geoset writing
        // The actual implementation would need to handle the differences in the v1500 format
    }
}

public class MdxGeoset
{
    // Basic properties
    public uint Flags { get; set; }
    public uint NumVertices { get; set; }
    
    // Geometry data
    public Vector3[] Vertices { get; set; }
    public Vector3[] Normals { get; set; }
    
    // Texture coordinate data
    public uint TextureCoordType { get; set; }
    public uint NumTextureCoordSets { get; set; }
    public Vector2[][] TextureCoords { get; set; }
    
    // Material and selection data
    public uint MaterialId { get; set; }
    public uint SelectionGroup { get; set; }
    public uint SelectionFlags { get; set; }
    
    // Face data
    public uint NumPositionIndices { get; set; }
    public uint[] PositionIndices { get; set; }
    public uint NumFaces { get; set; }
    public ushort[] FaceIndices { get; set; }
    
    // Bone influence data
    public byte[] GroupSizes { get; set; }
    public uint NumBoneGroups { get; set; }
    public uint[][] BoneGroups { get; set; }
    public uint MaxBonesPerVertex { get; set; }
    public uint[] BoneIndices { get; set; }
    public float[] BoneWeights { get; set; }
    
    // Bounding data
    public Vector3 BoundingBoxMin { get; set; }
    public Vector3 BoundingBoxMax { get; set; }
    public float BoundingRadius { get; set; }
    
    // Helper properties
    public bool Unselectable => (Flags & 0x1) != 0;
    
    // Version 1500 specific properties would be added here
}
``` 