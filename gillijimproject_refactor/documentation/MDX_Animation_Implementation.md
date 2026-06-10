# MDX Skeletal Animation Implementation Guide

## Current Status

### ✓ Working
- **MdxAnimator** - Bone hierarchy and matrix calculation
- **Keyframe interpolation** - Linear, Hermite, Bezier
- **Animation sequences** - Looping, global sequences
- **Bone transforms** - Translation, rotation, scaling

### ✗ Not Working
- **Vertex skinning** - Bone matrices not applied to vertices
- **Shader support** - No bone weight attributes in vertex shader
- **Bone weight upload** - VertexGroups/MatrixGroups not sent to GPU

## MDX Bone Weight System

### Data Structure
```
Geoset {
    Vertices: Vector3[]           // 3D positions
    Normals: Vector3[]            // Normals
    TexCoords: Vector2[]          // UVs
    Indices: ushort[]             // Triangle indices
    
    VertexGroups: byte[]          // Per-vertex: which matrix group to use
    MatrixGroups: uint[]          // Per-group: how many bone matrices
    MatrixIndices: uint[]         // Bone indices for each group
}
```

### How It Works

1. **VertexGroups** - One byte per vertex, indexes into MatrixGroups
   - `VertexGroups[vertexIndex]` → group index
   
2. **MatrixGroups** - Defines how many bones per group
   - `MatrixGroups[groupIndex]` → count of bones (usually 1-4)
   
3. **MatrixIndices** - Actual bone indices
   - Flattened array of all bone indices for all groups
   - Example: `[0, 1, 2, 5, 6, 7, 8]` for 2 groups (3 bones, 4 bones)

### Example
```
Vertex 0: VertexGroups[0] = 0 → MatrixGroups[0] = 1 → MatrixIndices[0] = 5
  → Vertex 0 is controlled by bone 5 only

Vertex 1: VertexGroups[1] = 1 → MatrixGroups[1] = 2 → MatrixIndices[1..2] = [3, 7]
  → Vertex 1 is controlled by bones 3 and 7 (50/50 weight)

Vertex 2: VertexGroups[2] = 2 → MatrixGroups[2] = 4 → MatrixIndices[3..6] = [1, 2, 3, 4]
  → Vertex 2 is controlled by bones 1, 2, 3, 4 (25% each)
```

## Implementation Plan

### Step 1: Preprocess Bone Weights (CPU)

Convert MDX bone weight structure to standard 4-bone skinning:

```csharp
struct VertexBoneData
{
    public Vector4 BoneIndices;  // 4 bone indices (as floats for shader)
    public Vector4 BoneWeights;  // 4 weights (sum = 1.0)
}

private VertexBoneData[] BuildBoneWeights(MdlGeoset geoset)
{
    var result = new VertexBoneData[geoset.Vertices.Count];
    int matrixOffset = 0;
    
    for (int v = 0; v < geoset.Vertices.Count; v++)
    {
        byte groupIdx = geoset.VertexGroups[v];
        uint boneCount = geoset.MatrixGroups[groupIdx];
        
        // Get bone indices for this group
        var indices = new float[4];
        var weights = new float[4];
        
        for (int b = 0; b < Math.Min(boneCount, 4); b++)
        {
            indices[b] = geoset.MatrixIndices[matrixOffset + b];
            weights[b] = 1.0f / boneCount; // Equal weights
        }
        
        result[v] = new VertexBoneData
        {
            BoneIndices = new Vector4(indices[0], indices[1], indices[2], indices[3]),
            BoneWeights = new Vector4(weights[0], weights[1], weights[2], weights[3])
        };
        
        matrixOffset += (int)boneCount;
    }
    
    return result;
}
```

### Step 2: Update Vertex Buffer Layout

Add bone data to vertex attributes:

```csharp
struct MdxVertex
{
    public Vector3 Position;     // location 0
    public Vector3 Normal;       // location 1
    public Vector2 TexCoord;     // location 2
    public Vector4 BoneIndices;  // location 3 (NEW)
    public Vector4 BoneWeights;  // location 4 (NEW)
}
```

### Step 3: Update Vertex Shader

```glsl
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;
layout(location = 3) in vec4 aBoneIndices;  // NEW
layout(location = 4) in vec4 aBoneWeights;  // NEW

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uBones[256];  // NEW - bone matrices
uniform int uHasBones;     // NEW - enable skinning

out vec3 vNormal;
out vec2 vTexCoord;
out vec3 vFragPos;
out vec3 vViewNormal;

void main()
{
    vec4 position = vec4(aPos, 1.0);
    vec3 normal = aNormal;
    
    // Apply bone skinning if enabled
    if (uHasBones > 0)
    {
        mat4 boneTransform = mat4(0.0);
        boneTransform += uBones[int(aBoneIndices.x)] * aBoneWeights.x;
        boneTransform += uBones[int(aBoneIndices.y)] * aBoneWeights.y;
        boneTransform += uBones[int(aBoneIndices.z)] * aBoneWeights.z;
        boneTransform += uBones[int(aBoneIndices.w)] * aBoneWeights.w;
        
        position = boneTransform * position;
        normal = mat3(boneTransform) * normal;
    }
    
    vec4 worldPos = uModel * position;
    gl_Position = uProj * uView * worldPos;
    
    vFragPos = worldPos.xyz;
    vNormal = normalize(mat3(uModel) * normal);
    vViewNormal = normalize(mat3(uView) * mat3(uModel) * normal);
    vTexCoord = aTexCoord;
}
```

### Step 4: Upload Bone Matrices Each Frame

```csharp
// In ModelRenderer.Render()
if (_animator != null && _animator.HasAnimation)
{
    _gl.Uniform1(_uHasBones, 1);
    
    // Upload all bone matrices
    var matrices = _animator.BoneMatrices;
    for (int i = 0; i < matrices.Length; i++)
    {
        var m = matrices[i];
        _gl.UniformMatrix4(_uBones + i, 1, false, (float*)&m);
    }
}
else
{
    _gl.Uniform1(_uHasBones, 0);
}
```

### Step 5: Update InitBuffers() in ModelRenderer

```csharp
private unsafe void InitBuffers()
{
    foreach (var geoset in _mdx.Geosets)
    {
        // Build bone weight data
        var boneData = BuildBoneWeights(geoset);
        
        // Interleave vertex data
        var vertexData = new List<float>();
        for (int i = 0; i < geoset.Vertices.Count; i++)
        {
            var v = geoset.Vertices[i];
            var n = geoset.Normals[i];
            var t = geoset.TexCoords[i];
            var b = boneData[i];
            
            // Position
            vertexData.Add(v.X);
            vertexData.Add(v.Y);
            vertexData.Add(v.Z);
            
            // Normal
            vertexData.Add(n.X);
            vertexData.Add(n.Y);
            vertexData.Add(n.Z);
            
            // TexCoord
            vertexData.Add(t.X);
            vertexData.Add(t.Y);
            
            // Bone indices
            vertexData.Add(b.BoneIndices.X);
            vertexData.Add(b.BoneIndices.Y);
            vertexData.Add(b.BoneIndices.Z);
            vertexData.Add(b.BoneIndices.W);
            
            // Bone weights
            vertexData.Add(b.BoneWeights.X);
            vertexData.Add(b.BoneWeights.Y);
            vertexData.Add(b.BoneWeights.Z);
            vertexData.Add(b.BoneWeights.W);
        }
        
        // Upload to GPU
        uint vao = _gl.GenVertexArray();
        uint vbo = _gl.GenBuffer();
        uint ebo = _gl.GenBuffer();
        
        _gl.BindVertexArray(vao);
        
        _gl.BindBuffer(BufferTargetARB.ArrayBuffer, vbo);
        fixed (float* ptr = vertexData.ToArray())
        {
            _gl.BufferData(BufferTargetARB.ArrayBuffer,
                (nuint)(vertexData.Count * sizeof(float)),
                ptr, BufferUsageARB.StaticDraw);
        }
        
        int stride = 16 * sizeof(float); // 3+3+2+4+4
        
        // Position (location 0)
        _gl.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, stride, (void*)0);
        _gl.EnableVertexAttribArray(0);
        
        // Normal (location 1)
        _gl.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, stride, (void*)(3 * sizeof(float)));
        _gl.EnableVertexAttribArray(1);
        
        // TexCoord (location 2)
        _gl.VertexAttribPointer(2, 2, VertexAttribPointerType.Float, false, stride, (void*)(6 * sizeof(float)));
        _gl.EnableVertexAttribArray(2);
        
        // Bone indices (location 3)
        _gl.VertexAttribPointer(3, 4, VertexAttribPointerType.Float, false, stride, (void*)(8 * sizeof(float)));
        _gl.EnableVertexAttribArray(3);
        
        // Bone weights (location 4)
        _gl.VertexAttribPointer(4, 4, VertexAttribPointerType.Float, false, stride, (void*)(12 * sizeof(float)));
        _gl.EnableVertexAttribArray(4);
        
        // Index buffer
        _gl.BindBuffer(BufferTargetARB.ElementArrayBuffer, ebo);
        fixed (ushort* ptr = geoset.Indices.ToArray())
        {
            _gl.BufferData(BufferTargetARB.ElementArrayBuffer,
                (nuint)(geoset.Indices.Count * sizeof(ushort)),
                ptr, BufferUsageARB.StaticDraw);
        }
        
        _geosets.Add(new GeosetBuffers
        {
            Vao = vao,
            Vbo = vbo,
            Ebo = ebo,
            IndexCount = geoset.Indices.Count,
            GeosetIndex = _geosets.Count,
            Visible = true
        });
    }
}
```

## Testing Strategy

1. **Static models** - Models with bones but no animation (should render unchanged)
2. **Simple animation** - Single bone rotation (torch flame)
3. **Complex animation** - Multi-bone character animation
4. **Verify transforms** - Check bone matrices are correct
5. **Performance** - Ensure 60 FPS with animated models

## Known Issues & Solutions

### Issue: MatrixGroups offset calculation
**Solution**: Track cumulative offset when iterating groups

### Issue: Bone index out of range
**Solution**: Clamp bone indices to valid range, log warnings

### Issue: Shader uniform limit (256 bones)
**Solution**: Most MDX models have <100 bones, should be fine

### Issue: Weight normalization
**Solution**: Ensure weights sum to 1.0 for each vertex

## File Changes Required

1. `ModelRenderer.cs`
   - Add `BuildBoneWeights()` method
   - Update `InitBuffers()` to include bone data
   - Update `InitShaders()` with new vertex shader
   - Add `_uBones` and `_uHasBones` uniforms
   - Upload bone matrices in `Render()`

2. `MdxAnimator.cs`
   - Add `GetBoneMatrix(int boneIndex)` public method
   - Ensure bone matrices are in correct space

## Timeline

- **Step 1-2**: Bone weight preprocessing - 1 hour
- **Step 3**: Shader updates - 30 minutes
- **Step 4-5**: Vertex buffer and rendering - 1 hour
- **Testing**: 1 hour
- **Total**: ~3.5 hours for complete skeletal animation

## References

- WoW M2 format: https://wowdev.wiki/M2
- MDX text format: Warcraft III model format
- Vertex skinning: Standard GPU skinning technique
- Matrix palette skinning: 4-bone weighted blending
