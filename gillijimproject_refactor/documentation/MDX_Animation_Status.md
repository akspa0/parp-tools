# MDX Animation - Current Status & Implementation Plan

## Problem Summary

**MDX animations are not working** because:
1. Bone matrices are calculated but **not applied to vertices**
2. Shader has no bone skinning support
3. Vertex buffers don't include bone weight data
4. Particles are not integrated into the rendering pipeline

## What's Already Working ✓

- `MdxAnimator.cs` - Complete bone hierarchy system
  - Keyframe interpolation (Linear, Hermite, Bezier)
  - Animation sequences with looping
  - Global sequences
  - Bone matrix calculation
  
- Animation update loop in `ModelRenderer.Render()`
  - Calls `_animator.Update(deltaMs)` each frame
  - Bone matrices are being calculated correctly

## What's Missing ✗

### 1. Bone Weight Data Processing
MDX stores bone weights in a complex structure:
- `VertexGroups[vertexIndex]` → group index (1 byte per vertex)
- `MatrixGroups[groupIndex]` → bone count for that group
- `MatrixIndices[]` → flattened array of bone indices

**Need to convert this to standard 4-bone skinning:**
```csharp
struct VertexBoneData {
    Vector4 BoneIndices;  // 4 bone indices
    Vector4 BoneWeights;  // 4 weights (sum = 1.0)
}
```

### 2. Vertex Buffer Update
Current vertex layout: `pos(3) + normal(3) + uv(2) = 8 floats`

**Need to expand to:** `pos(3) + normal(3) + uv(2) + boneIdx(4) + boneWt(4) = 16 floats`

### 3. Shader Bone Skinning
Shader already updated with:
- Bone matrix array uniform `uBones[256]`
- Bone attribute inputs `aBoneIndices`, `aBoneWeights`
- Skinning logic in vertex shader

**Still need to:**
- Upload bone matrices to shader each frame
- Set `uHasBones` flag
- Add bone vertex attributes to VAO

### 4. Particle System Integration
Particles are implemented but not connected:
- `ParticleSystem.cs` exists
- `ParticleRenderer.cs` exists
- Not instantiated in ModelRenderer
- Not rendered in scene

## Implementation Steps

### Step 1: Build Bone Weight Converter (30 min)
Add method to `ModelRenderer.cs`:

```csharp
private (Vector4[] indices, Vector4[] weights) BuildBoneWeights(MdlGeoset geoset)
{
    int vertCount = geoset.Vertices.Count;
    var indices = new Vector4[vertCount];
    var weights = new Vector4[vertCount];
    
    // Build group offset lookup
    var groupOffsets = new int[geoset.MatrixGroups.Count];
    int offset = 0;
    for (int g = 0; g < geoset.MatrixGroups.Count; g++)
    {
        groupOffsets[g] = offset;
        offset += (int)geoset.MatrixGroups[g];
    }
    
    // Process each vertex
    for (int v = 0; v < vertCount; v++)
    {
        byte groupIdx = geoset.VertexGroups[v];
        uint boneCount = geoset.MatrixGroups[groupIdx];
        int matrixOffset = groupOffsets[groupIdx];
        
        var idx = new float[4];
        var wt = new float[4];
        
        for (int b = 0; b < Math.Min(boneCount, 4); b++)
        {
            idx[b] = geoset.MatrixIndices[matrixOffset + b];
            wt[b] = 1.0f / boneCount; // Equal weights
        }
        
        indices[v] = new Vector4(idx[0], idx[1], idx[2], idx[3]);
        weights[v] = new Vector4(wt[0], wt[1], wt[2], wt[3]);
    }
    
    return (indices, weights);
}
```

### Step 2: Update InitBuffers() (30 min)
Change vertex data layout from 8 floats to 16 floats:

```csharp
// Build bone weights
var (boneIndices, boneWeights) = BuildBoneWeights(geoset);

// Interleave: pos(3) + normal(3) + uv(2) + boneIdx(4) + boneWt(4) = 16 floats
float[] vertexData = new float[vertCount * 16];
for (int v = 0; v < vertCount; v++)
{
    int offset = v * 16;
    
    // Position (0-2)
    vertexData[offset + 0] = geoset.Vertices[v].X;
    vertexData[offset + 1] = geoset.Vertices[v].Y;
    vertexData[offset + 2] = geoset.Vertices[v].Z;
    
    // Normal (3-5)
    if (hasNormals) {
        vertexData[offset + 3] = geoset.Normals[v].X;
        vertexData[offset + 4] = geoset.Normals[v].Y;
        vertexData[offset + 5] = geoset.Normals[v].Z;
    }
    
    // TexCoord (6-7)
    if (hasUVs) {
        vertexData[offset + 6] = geoset.TexCoords[v].U;
        vertexData[offset + 7] = geoset.TexCoords[v].V;
    }
    
    // Bone indices (8-11)
    vertexData[offset + 8] = boneIndices[v].X;
    vertexData[offset + 9] = boneIndices[v].Y;
    vertexData[offset + 10] = boneIndices[v].Z;
    vertexData[offset + 11] = boneIndices[v].W;
    
    // Bone weights (12-15)
    vertexData[offset + 12] = boneWeights[v].X;
    vertexData[offset + 13] = boneWeights[v].Y;
    vertexData[offset + 14] = boneWeights[v].Z;
    vertexData[offset + 15] = boneWeights[v].W;
}

// Update stride
uint stride = 16 * sizeof(float);

// Add bone attribute pointers
_gl.EnableVertexAttribArray(3);
_gl.VertexAttribPointer(3, 4, VertexAttribPointerType.Float, false, stride, (void*)(8 * sizeof(float)));
_gl.EnableVertexAttribArray(4);
_gl.VertexAttribPointer(4, 4, VertexAttribPointerType.Float, false, stride, (void*)(12 * sizeof(float)));
```

### Step 3: Upload Bone Matrices in Render() (15 min)
Add before rendering geosets:

```csharp
// Upload bone matrices if animated
if (_animator != null && _animator.HasAnimation)
{
    _gl.Uniform1(_uHasBones, 1);
    
    var matrices = _animator.BoneMatrices;
    for (int i = 0; i < Math.Min(matrices.Length, 256); i++)
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

### Step 4: Test & Debug (30 min)
1. Load animated MDX model
2. Verify bone matrices are uploaded
3. Check vertex attributes are correct
4. Verify animation plays

## Timeline

- **Bone weight conversion**: 30 minutes
- **Vertex buffer update**: 30 minutes  
- **Bone matrix upload**: 15 minutes
- **Testing & debugging**: 30 minutes
- **Total**: ~2 hours for working skeletal animation

## Test Models

Good test candidates:
- Torch models (simple fire animation)
- Character models (complex multi-bone)
- Creature models (skeletal animation)

## Next: Particles

After skeletal animation works:
1. Instantiate ParticleRenderer in ModelRenderer
2. Create ParticleEmitter instances from `_mdx.ParticleEmitters2`
3. Update particles in render loop
4. Render particles after transparent geometry
5. Load particle textures (fire, embers, smoke)

Estimated time: 2-3 hours
