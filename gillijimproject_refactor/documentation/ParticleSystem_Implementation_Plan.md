# MDX Particle System Implementation Plan

## Overview
Implementation of GPU-accelerated particle systems for MDX models to render fire, embers, smoke, and other visual effects.

## Completed Components

### 1. ParticleSystem.cs ✓
**Location**: `src/MdxViewer/Rendering/ParticleSystem.cs`

**Features**:
- `ParticleEmitter` class - manages particle spawning, physics, and lifecycle
- `Particle` struct - individual particle instance data
- Physics simulation with gravity and velocity
- Lifecycle interpolation (birth → mid → death)
- Color and size interpolation based on segment data
- Emission rate control with proper timing

**Key Methods**:
- `Update(deltaTime)` - updates physics and spawns particles
- `GetParticleColor(particle)` - interpolates RGB+Alpha across 3 segments
- `GetParticleSize(particle)` - interpolates size across 3 segments

### 2. ParticleRenderer.cs ✓
**Location**: `src/MdxViewer/Rendering/ParticleRenderer.cs`

**Features**:
- GPU-based rendering with billboarding
- Additive blending for fire/glow effects
- Camera-facing quads (billboard sprites)
- Batch rendering support
- Proper depth buffer handling (no depth write for particles)

**Shader Features**:
- Vertex shader: Billboard transformation using camera right/up vectors
- Fragment shader: Texture sampling with alpha blending
- Uniform support for view/projection matrices

## Next Steps

### Phase 1: Integration with MdxRenderer
**File**: `src/MdxViewer/Rendering/ModelRenderer.cs`

1. Add particle emitter list to MdxRenderer
2. Create ParticleRenderer instance
3. Initialize emitters from `_mdx.ParticleEmitters2`
4. Update emitter transforms based on bone hierarchy
5. Call particle update in render loop
6. Render particles after transparent geometry

**Code Changes Needed**:
```csharp
private ParticleRenderer? _particleRenderer;
private readonly List<ParticleEmitter> _particleEmitters = new();

// In constructor after InitBuffers():
InitParticleSystem();

private void InitParticleSystem()
{
    if (_mdx.ParticleEmitters2.Count == 0) return;
    
    _particleRenderer = new ParticleRenderer(_gl);
    
    foreach (var emitterDef in _mdx.ParticleEmitters2)
    {
        var emitter = new ParticleEmitter(emitterDef);
        _particleEmitters.Add(emitter);
    }
    
    ViewerLog.Info(ViewerLog.Category.Mdx, 
        $"Initialized {_particleEmitters.Count} particle emitters");
}

// In Render() method:
private void UpdateParticles(float deltaTime)
{
    foreach (var emitter in _particleEmitters)
    {
        // Update emitter transform from bone hierarchy
        if (emitter.Definition.ParentId >= 0 && _animator != null)
        {
            var boneMatrix = _animator.GetBoneMatrix(emitter.Definition.ParentId);
            var localPos = new Vector3(
                emitter.Definition.Position.X,
                emitter.Definition.Position.Y,
                emitter.Definition.Position.Z
            );
            emitter.Transform = Matrix4x4.CreateTranslation(localPos) * boneMatrix;
        }
        
        emitter.Update(deltaTime);
    }
}
```

### Phase 2: Particle Texture Loading
**File**: `src/MdxViewer/Rendering/ModelRenderer.cs` (LoadTextures method)

1. Load particle textures from MDX texture references
2. Handle replaceable texture IDs for particles
3. Support texture animation (rows/columns)
4. Default fallback texture for missing particle textures

**Particle Texture Paths** (common examples):
- `Textures\Particle\Fire.blp`
- `Textures\Particle\Ember.blp`
- `Textures\Particle\Smoke.blp`
- `Textures\Particle\Glow.blp`

### Phase 3: Animation Integration
**File**: `src/MdxViewer/Rendering/MdxAnimator.cs`

1. Extend animator to track particle emitter states
2. Support emission rate animation
3. Support color/alpha animation keyframes
4. Trigger particle bursts at specific animation frames

### Phase 4: Performance Optimization

1. **Instanced Rendering**: Replace per-particle draw calls with instanced rendering
2. **Particle Pooling**: Reuse particle objects instead of allocating/deallocating
3. **Spatial Culling**: Don't update/render particles outside view frustum
4. **LOD System**: Reduce particle count at distance

**Instanced Rendering Approach**:
```csharp
// Create instance buffer with particle data
struct ParticleInstance {
    Vector3 position;
    Vector4 color;
    float size;
}

// Upload to GPU as instance buffer
glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 4, particleCount);
```

### Phase 5: Advanced Features

1. **Particle Sorting**: Sort transparent particles back-to-front
2. **Soft Particles**: Depth-based fade near geometry
3. **Texture Animation**: Support for sprite sheet animation
4. **Ribbon Emitters**: Trail effects (already parsed in MDX)
5. **Collision**: Particle-terrain collision for ground effects

## Testing Plan

### Test Cases
1. **Torch Model**: Simple fire emitter test
2. **Campfire**: Multiple emitters (fire + smoke)
3. **Spell Effects**: Burst emitters with short lifespan
4. **Ambient Effects**: Continuous low-rate emitters

### Performance Targets
- 60 FPS with 1000 active particles
- 30 FPS with 5000 active particles
- Minimal impact on non-particle models

## Known Limitations

1. **Current Implementation**: One draw call per particle (inefficient)
2. **No Sorting**: Particles may have incorrect depth order
3. **No Collision**: Particles pass through geometry
4. **Fixed Texture**: All particles use same texture per emitter

## References

- MDX Format: `MdlParticleEmitter2` in `MdxModels.cs`
- WoW Particle Docs: https://wowdev.wiki/M2/.skin#Particle_emitters
- OpenGL Billboarding: Camera-facing quad technique
- Additive Blending: `glBlendFunc(GL_SRC_ALPHA, GL_ONE)`

## File Structure
```
src/MdxViewer/Rendering/
├── ParticleSystem.cs       ✓ Created
├── ParticleRenderer.cs     ✓ Created
├── ModelRenderer.cs        ⏳ Needs integration
└── MdxAnimator.cs          ⏳ Needs particle support
```

## Timeline Estimate
- Phase 1 (Integration): 2-3 hours
- Phase 2 (Textures): 1-2 hours
- Phase 3 (Animation): 2-3 hours
- Phase 4 (Optimization): 3-4 hours
- Phase 5 (Advanced): 4-6 hours

**Total**: 12-18 hours for complete implementation
