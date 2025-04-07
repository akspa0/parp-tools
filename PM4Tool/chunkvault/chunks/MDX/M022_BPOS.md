# BPOS (Bind Poses)

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The BPOS chunk defines bind poses for model bones, providing the base position and orientation for skinning vertices to bones. A bind pose represents the "rest position" of a model's skeleton - the reference pose used for calculating skinned vertex positions during animation. This data is essential for proper vertex skinning and ensuring animations deform the model correctly.

## Structure
The BPOS chunk contains a sequence of 4x4 transformation matrices, one for each bone in the model. These matrices define the rest pose for each bone in the skeletal hierarchy.

```csharp
public class BPOS
{
    public string Magic { get; set; } // "BPOS"
    public int Size { get; set; }     // Size of the chunk data in bytes
    public List<Matrix4x4> BindPoseMatrices { get; set; }
}

// 4x4 Transformation Matrix (64 bytes)
public struct Matrix4x4
{
    public float M11, M12, M13, M14;
    public float M21, M22, M23, M24;
    public float M31, M32, M33, M34;
    public float M41, M42, M43, M44;
}
```

## Properties

### BPOS Chunk
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | Magic | char[4] | "BPOS" |
| 0x04 | Size | uint32 | Size of the chunk data in bytes |
| 0x08 | BindPoseMatrices | Matrix4x4[] | Array of bind pose matrices (one per bone) |

### Matrix4x4 Structure
| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | M11 | float | First row, first column value |
| 0x04 | M12 | float | First row, second column value |
| 0x08 | M13 | float | First row, third column value |
| 0x0C | M14 | float | First row, fourth column value |
| 0x10 | M21 | float | Second row, first column value |
| 0x14 | M22 | float | Second row, second column value |
| 0x18 | M23 | float | Second row, third column value |
| 0x1C | M24 | float | Second row, fourth column value |
| 0x20 | M31 | float | Third row, first column value |
| 0x24 | M32 | float | Third row, second column value |
| 0x28 | M33 | float | Third row, third column value |
| 0x2C | M34 | float | Third row, fourth column value |
| 0x30 | M41 | float | Fourth row, first column value |
| 0x34 | M42 | float | Fourth row, second column value |
| 0x38 | M43 | float | Fourth row, third column value |
| 0x3C | M44 | float | Fourth row, fourth column value |

## Matrix Usage

The bind pose matrices serve several important purposes:

1. **Inverse Bind Poses**: The matrices are typically stored as inverse bind poses, which transform vertices from bone space to model space.
2. **Vertex Skinning**: During animation, these matrices are used together with the SKIN chunk data to position vertices relative to their influencing bones.
3. **Animation Calculations**: The inverse bind pose matrices are combined with bone animation matrices to calculate the final position of vertices.

## Version Differences

| Version | Differences |
|---------|-------------|
| All | The BPOS chunk has maintained a consistent format across MDX versions |
| Some Models | Not all models contain a BPOS chunk; those without it may use an identity matrix as the default bind pose |

## Dependencies
- **BONE**: Defines the bone hierarchy that corresponds to the bind pose matrices
- **SKIN**: Contains vertex weight information that uses the bind pose matrices
- **GEOS**: Contains the vertex data that will be transformed using these matrices

## Implementation Notes

1. **Matrix Order**: The bind pose matrices are stored in the same order as the bones defined in the BONE chunk.
2. **Matrix Format**: The matrices use a row-major format.
3. **Coordinate System**: Consistent with MDX format, uses a right-handed coordinate system with Y-up.
4. **Skinning Process**: To correctly skin a vertex:
   - Transform the vertex by each influencing bone's animation matrix
   - Transform by the bone's inverse bind pose matrix
   - Blend the results according to vertex weights from the SKIN chunk
5. **Storage Efficiency**: Although these are 4x4 matrices, the last row is typically [0,0,0,1] and could be omitted for storage efficiency (but is included in the file format).

## Usage Context

The BPOS chunk is essential for:

1. **Skinned Mesh Animation**: Provides the reference pose for calculating vertex skinning
2. **Export/Import Operations**: Required for transferring models between editing tools
3. **Animation Retargeting**: Used when adapting animations from one skeleton to another
4. **Physics Simulations**: Provides rest state for physics calculations
5. **Model Space Normalization**: Helps establish consistent coordinate systems between models

## Implementation Example

```csharp
public class BindPoseSystem
{
    private List<Matrix4x4> bindPoseMatrices = new List<Matrix4x4>();
    
    public void ParseBPOS(BinaryReader reader)
    {
        // Read chunk header
        string magic = new string(reader.ReadChars(4));
        if (magic != "BPOS")
            throw new Exception("Invalid BPOS chunk header");
            
        int chunkSize = reader.ReadInt32();
        int matrixCount = chunkSize / 64; // Each matrix is 64 bytes (16 floats * 4 bytes)
        
        bindPoseMatrices.Clear();
        
        // Read each bind pose matrix
        for (int i = 0; i < matrixCount; i++)
        {
            Matrix4x4 matrix = new Matrix4x4();
            
            // Read 16 floats for the 4x4 matrix
            matrix.M11 = reader.ReadSingle();
            matrix.M12 = reader.ReadSingle();
            matrix.M13 = reader.ReadSingle();
            matrix.M14 = reader.ReadSingle();
            
            matrix.M21 = reader.ReadSingle();
            matrix.M22 = reader.ReadSingle();
            matrix.M23 = reader.ReadSingle();
            matrix.M24 = reader.ReadSingle();
            
            matrix.M31 = reader.ReadSingle();
            matrix.M32 = reader.ReadSingle();
            matrix.M33 = reader.ReadSingle();
            matrix.M34 = reader.ReadSingle();
            
            matrix.M41 = reader.ReadSingle();
            matrix.M42 = reader.ReadSingle();
            matrix.M43 = reader.ReadSingle();
            matrix.M44 = reader.ReadSingle();
            
            bindPoseMatrices.Add(matrix);
        }
    }
    
    public void WriteBPOS(BinaryWriter writer)
    {
        if (bindPoseMatrices.Count == 0)
            return; // Skip if no bind pose matrices
            
        // Write chunk header
        writer.Write("BPOS".ToCharArray());
        writer.Write(bindPoseMatrices.Count * 64); // Size: matrix count * 64 bytes per matrix
        
        // Write each bind pose matrix
        foreach (Matrix4x4 matrix in bindPoseMatrices)
        {
            writer.Write(matrix.M11);
            writer.Write(matrix.M12);
            writer.Write(matrix.M13);
            writer.Write(matrix.M14);
            
            writer.Write(matrix.M21);
            writer.Write(matrix.M22);
            writer.Write(matrix.M23);
            writer.Write(matrix.M24);
            
            writer.Write(matrix.M31);
            writer.Write(matrix.M32);
            writer.Write(matrix.M33);
            writer.Write(matrix.M34);
            
            writer.Write(matrix.M41);
            writer.Write(matrix.M42);
            writer.Write(matrix.M43);
            writer.Write(matrix.M44);
        }
    }
    
    // Calculate the inverse of a bind pose matrix
    public Matrix4x4 GetInverseBindPose(int boneIndex)
    {
        if (boneIndex < 0 || boneIndex >= bindPoseMatrices.Count)
            return Matrix4x4.Identity; // Return identity matrix for invalid indices
            
        return Matrix4x4.Invert(bindPoseMatrices[boneIndex]);
    }
    
    // Apply skinning transform using bind pose matrices
    public Vector3 SkinVertex(Vector3 position, int[] boneIndices, float[] weights, Matrix4x4[] boneMatrices)
    {
        Vector3 skinnedPosition = Vector3.Zero;
        
        for (int i = 0; i < boneIndices.Length; i++)
        {
            int boneIndex = boneIndices[i];
            float weight = weights[i];
            
            if (weight > 0 && boneIndex >= 0 && boneIndex < bindPoseMatrices.Count)
            {
                // Calculate the skinning transform: BoneMatrix * InverseBindPose
                Matrix4x4 skinningTransform = Matrix4x4.Multiply(
                    boneMatrices[boneIndex],
                    GetInverseBindPose(boneIndex)
                );
                
                // Transform the vertex and add weighted contribution
                Vector3 transformedPos = Vector3.Transform(position, skinningTransform);
                skinnedPosition += transformedPos * weight;
            }
        }
        
        return skinnedPosition;
    }
}
``` 