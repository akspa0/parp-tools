# SKIN (Vertex Weights)

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The SKIN chunk defines how vertices are influenced by bones in the skeletal hierarchy, enabling smooth mesh deformation during animation. It contains vertex-to-bone binding information, including which bones affect each vertex and the weight (influence amount) of each bone. This data is essential for skeletal animation, allowing models to bend naturally at joints and creating realistic movement.

## Structure
The SKIN chunk consists of a vertex-to-bone mapping table. For each vertex in all geosets, it specifies which bones affect that vertex and with what weights.

```csharp
public class SKIN
{
    public string Magic { get; set; } // "SKIN"
    public int Size { get; set; }     // Size of the chunk data in bytes
    public List<MDLVERTEXWEIGHT> VertexWeights { get; set; }
}

// Vertex weight structure (variable size)
public struct MDLVERTEXWEIGHT
{
    public List<int> BoneIndices { get; set; }       // Indices of bones that influence this vertex
    public List<float> BoneWeights { get; set; }     // Corresponding weights for each bone
}
```

## Properties

### SKIN Chunk
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Magic | char[4] | "SKIN" |
| 0x04 | Size | uint32 | Size of the chunk data in bytes |
| 0x08 | VertexWeights | MDLVERTEXWEIGHT[] | Array of vertex weight data |

### MDLVERTEXWEIGHT Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | BoneCount | uint8 | Number of bones that influence this vertex (typically 1-4) |
| 0x01 | BoneIndices | uint8[] | Array of bone indices (length = BoneCount) |
| var | BoneWeights | float[] | Array of corresponding bone weights (length = BoneCount) |

## Weight Properties

1. **Normalization**: Weights for each vertex should sum to 1.0 (100% influence)
2. **Maximum Bones**: Most implementations limit vertices to be influenced by at most 4 bones
3. **Bone Indices**: Reference bones defined in the BONE chunk by their index
4. **Precision**: Weights are stored as 32-bit floating point values for precision

## Version Differences

| Version | Differences |
|---------|-------------|
| All | The SKIN chunk has maintained a consistent format across MDX versions |
| Early Models | May have fewer bones per vertex (1-2) for performance reasons |
| Advanced Models | May utilize the full 4 bones per vertex for smoother deformation |

## Dependencies
- **BONE**: Defines the bones referenced by indices in the SKIN chunk
- **GEOS**: Contains the vertices that these weights apply to
- **BPOS**: Contains bind pose matrices used together with skin weights for vertex transformation

## Implementation Notes

1. **Vertex Order**: Vertex weights are stored in the same order as vertices appear in the GEOS chunk
2. **Weight Distribution**: For realistic deformation, weights are typically higher for bones closest to the vertex
3. **Optimization**: Game engines may optimize by:
   - Discarding very small weights (< 0.01)
   - Renormalizing remaining weights
   - Limiting maximum bones per vertex based on hardware capabilities
4. **Memory Layout**: The data is packed to minimize storage:
   - First a count of influencing bones
   - Then all bone indices as consecutive bytes
   - Finally all corresponding weights as consecutive floats
5. **Pivot Points**: Weight calculation often considers the distance from vertex to bone pivot point

## Usage Context

The SKIN chunk is essential for:

1. **Character Animation**: Enables realistic movement of characters and creatures
2. **Cloth Simulation**: Creates basic cloth-like movement for clothing, capes, etc.
3. **Facial Animation**: Combined with specialized facial bones for expressions
4. **Vehicle Deformation**: Used for articulated vehicles and mechanical models
5. **Level of Detail**: Can be simplified for distant models to improve performance

## Implementation Example

```csharp
public class SkinningSystem
{
    private List<VertexWeight> vertexWeights = new List<VertexWeight>();
    private int totalVertexCount = 0;
    
    public class VertexWeight
    {
        public byte[] BoneIndices { get; set; }
        public float[] Weights { get; set; }
    }
    
    public void ParseSKIN(BinaryReader reader, int expectedVertexCount)
    {
        // Read chunk header
        string magic = new string(reader.ReadChars(4));
        if (magic != "SKIN")
            throw new Exception("Invalid SKIN chunk header");
            
        int chunkSize = reader.ReadInt32();
        long startPosition = reader.BaseStream.Position;
        
        totalVertexCount = expectedVertexCount;
        vertexWeights.Clear();
        
        // Read each vertex's weight information
        for (int vertexIndex = 0; vertexIndex < expectedVertexCount; vertexIndex++)
        {
            // Read number of bones influencing this vertex
            byte boneCount = reader.ReadByte();
            
            // Read bone indices
            byte[] boneIndices = new byte[boneCount];
            for (int i = 0; i < boneCount; i++)
            {
                boneIndices[i] = reader.ReadByte();
            }
            
            // Read bone weights
            float[] weights = new float[boneCount];
            for (int i = 0; i < boneCount; i++)
            {
                weights[i] = reader.ReadSingle();
            }
            
            // Add to our collection
            vertexWeights.Add(new VertexWeight
            {
                BoneIndices = boneIndices,
                Weights = weights
            });
        }
        
        // Verify we read the expected amount of data
        long endPosition = reader.BaseStream.Position;
        if (endPosition - startPosition != chunkSize)
            throw new Exception($"SKIN chunk size mismatch. Expected {chunkSize}, read {endPosition - startPosition}");
    }
    
    public void WriteSKIN(BinaryWriter writer)
    {
        if (vertexWeights.Count == 0)
            return; // Skip if no vertex weights
        
        // Calculate chunk size first
        int chunkSize = 0;
        foreach (var weight in vertexWeights)
        {
            // 1 byte for count + 1 byte per bone index + 4 bytes per weight
            chunkSize += 1 + weight.BoneIndices.Length + (weight.Weights.Length * 4);
        }
        
        // Write chunk header
        writer.Write("SKIN".ToCharArray());
        writer.Write(chunkSize);
        
        // Write each vertex's weight information
        foreach (var weight in vertexWeights)
        {
            // Write number of bones
            writer.Write((byte)weight.BoneIndices.Length);
            
            // Write bone indices
            for (int i = 0; i < weight.BoneIndices.Length; i++)
            {
                writer.Write(weight.BoneIndices[i]);
            }
            
            // Write bone weights
            for (int i = 0; i < weight.Weights.Length; i++)
            {
                writer.Write(weight.Weights[i]);
            }
        }
    }
    
    // Get weight information for a specific vertex
    public VertexWeight GetVertexWeight(int vertexIndex)
    {
        if (vertexIndex < 0 || vertexIndex >= vertexWeights.Count)
            throw new ArgumentOutOfRangeException(nameof(vertexIndex));
            
        return vertexWeights[vertexIndex];
    }
    
    // Normalize weights to ensure they sum to 1.0
    public void NormalizeWeights()
    {
        foreach (var weight in vertexWeights)
        {
            float sum = 0.0f;
            
            // Calculate sum of weights
            for (int i = 0; i < weight.Weights.Length; i++)
            {
                sum += weight.Weights[i];
            }
            
            // Skip if sum is zero or very close to it
            if (sum < 0.0001f)
                continue;
                
            // Normalize weights
            for (int i = 0; i < weight.Weights.Length; i++)
            {
                weight.Weights[i] /= sum;
            }
        }
    }
    
    // Apply vertex skinning using bone matrices
    public Vector3[] TransformVertices(Vector3[] baseVertices, Matrix4x4[] boneMatrices)
    {
        if (baseVertices.Length != vertexWeights.Count)
            throw new ArgumentException("Vertex count mismatch");
            
        Vector3[] transformedVertices = new Vector3[baseVertices.Length];
        
        for (int i = 0; i < baseVertices.Length; i++)
        {
            Vector3 position = baseVertices[i];
            VertexWeight weight = vertexWeights[i];
            
            // Initialize with zero vector
            Vector3 skinned = Vector3.Zero;
            
            // Apply each bone's influence
            for (int j = 0; j < weight.BoneIndices.Length; j++)
            {
                int boneIndex = weight.BoneIndices[j];
                float boneWeight = weight.Weights[j];
                
                if (boneIndex >= 0 && boneIndex < boneMatrices.Length && boneWeight > 0)
                {
                    // Transform vertex by bone matrix
                    Vector3 transformed = Vector3.Transform(position, boneMatrices[boneIndex]);
                    
                    // Add weighted contribution
                    skinned += transformed * boneWeight;
                }
            }
            
            transformedVertices[i] = skinned;
        }
        
        return transformedVertices;
    }
}
``` 