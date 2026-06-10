# Particle System

This document provides detailed analysis of the particle system in WoW Alpha 0.5.3, based on reverse engineering the WoWClient.exe binary.

## Overview

The particle system in WoW Alpha uses a sophisticated particle emitter architecture that supports multiple emitter types, particle physics, texture animation, and dynamic effects. The system is built around the `CParticleEmitter2` class and handles particles for spells, effects, environmental phenomena, and more.

## Key Classes and Structures

### CParticleEmitter2 (Particle Emitter)
```
Address: 0x0047c430+
Purpose: Main particle emitter class
Created by: CreateParticleEmitter2() @ 0x0073b440
Key Fields:
  - TSGrowableArray<CParticle2> m_particles: Active particles
  - TSGrowableArray<CParticle2_Model> m_modelParticles: Model-space particles
  - TSFixedArray<unsigned_int> m_alive: Stack of alive particle indices
  - TSFixedArray<unsigned_int> m_dead: Stack of dead particle indices
  - CParticleKey m_particleKeys[2]: Particle keyframe data
  - ParticleMaterial m_particleMaterial: Rendering properties
  - uint m_replaceableId: Replaceable texture ID
  - float m_particleEmissionRate: Particles per second
  - float m_particleLifeSpan: Particle lifetime in seconds
  - float m_particleVelocity: Initial velocity
  - float m_particleVelocityVariation: Random velocity variation
  - float m_particleAcceleration: Acceleration over time
  - float m_particleAngularVelocity: Rotation speed
  - float m_particleZsource: Z-axis source
  - uint m_textureRows: Texture sheet rows
  - uint m_textureColumns: Texture sheet columns
  - C3Vector m_windVector: Wind direction and strength
  - float m_windTime: Wind time offset
  - C3Vector m_followVector: Follow model direction
  - float m_followB, m_followM: Follow parameters
  - C3Vector m_xyAxis: XY plane orientation
  - uint m_particleType: PT_QUAD, PT_VERTEX, etc.
  - uint m_emitterType: PET_BASE_EMITTER, PET_PLANE, PET_SPHERE, PET_SPLINE
  - HTEXTURE__* m_hTex: Texture handle
```

### CParticle2 (Particle Instance)
```
Purpose: Individual particle instance
Fields:
  - C3Vector position: World position
  - C3Vector velocity: Current velocity
  - C3Vector acceleration: Current acceleration
  - float life: Remaining lifetime (0-1)
  - float maxLife: Maximum lifetime
  - float size: Particle size
  - float rotation: Current rotation angle
  - uint flags: Particle state flags
  - float twinklePhase: Twinkle effect phase
  - C3Vector modelSpacePos: Position in model space
```

### CParticleKey (Particle Keyframe)
```
Purpose: Stores particle animation keyframes
Fields:
  - uint time: Keyframe time
  - float value: Keyframe value (type depends on animation type)
```

### ParticleMaterial (Rendering Properties)
```
Purpose: Controls how particles are rendered
Fields:
  - EGxBlend alpha: Blend mode (GxBlend_Opaque, GxBlend_Blend, GxBlend_Add, GxBlend_AlphaKey)
  - uint flags: Rendering flags (bitfield with alpha test, etc.)
```

## Emitter Types (MDLEmitterType)

| Value | Name | Description |
|-------|------|-------------|
| 0 | PET_BASE_EMITTER | Standard point emitter |
| 1 | PET_PLANE_EMITTER | Planar emitter (rectangle) |
| 2 | PET_SPHERE_EMITTER | Spherical emitter |
| 3 | PET_SPLINE_EMITTER | Spline path emitter |

## Particle Types (MDLParticleType)

| Value | Name | Geometry |
|-------|------|----------|
| 0 | PT_QUAD | Billboarded quad (always faces camera) |
| 1 | PT_VERTEX | Vertex-based particle |
| 2 | PT_UPVERTEX | Up-facing vertex particle |

## Particle Keyframe Animations

The particle system supports animating particle properties over time using keyframes:

### Particle Keyframe Chunks

#### Emission Rate (KPEM / 0x4d45504b = "KPEM")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float emissionRate (particles/second)
```

#### Lifetime (KLIF / 0x46494c4b = "KLIF")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float lifetime (seconds)
```

#### Velocity (KVEL / 0x4c45564b = "KVEL")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float velocity
```

#### Gravity (KGRA / 0x4152474b = "KGRA")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float gravity
```

#### Scale (KSCALE / 0x454c4153 = "KSCA")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float scale
```

#### Color (KCOL / 0x4c4f434b = "KCOL")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - SplineKeyData[numKeys]:
    - uint32 time
    - C3Color color (RGB float 0-1)
    - C3Color inTangent
    - C3Color outTangent
```

#### Alpha (KALP / 0x504c414b = "KALP")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float alpha (0-1)
```

#### Rotation (KROT / 0x544f524b = "KROT")
```
Structure:
  - uint32 numKeys
  - uint32 globalSeqId
  - KeyframeData[numKeys]:
    - uint32 time
    - float rotation (radians)
```

## Texture Sheet Animation

The particle system supports texture sheet animation (sprite sheets):

```
Fields:
  - uint m_textureRows: Number of rows in texture sheet
  - uint m_textureColumns: Number of columns in texture sheet
  - float m_twinkleFPS: Twinkle animation FPS
  - float m_twinkleOnOff: Twinkle on/off ratio
  - float m_twinkleScaleMin: Minimum twinkle scale
  - float m_twinkleScaleMax: Maximum twinkle scale
```

## Wind Effects

Particles can be affected by wind:

```
Fields:
  - C3Vector m_windVector: Wind direction and strength
  - float m_windTime: Time offset for wind variation
```

## Follow Model Effects

Particles can follow the model:

```
Fields:
  - C3Vector m_followVector: Direction to follow
  - float m_followB: Follow parameter B
  - float m_followM: Follow parameter M
```

## Particle System Functions

### CParticleEmitter2 Constructor @ 0x0047c430
```c
CParticleEmitter2* CParticleEmitter2::CParticleEmitter2() {
  // Initialize reference count
  m_refCount = 1;
  
  // Initialize random seed
  NTempest::CRndSeed::SetSeed(&m_randSeed, rand() << 16 | rand() & 0xFFFF);
  
  // Initialize particle arrays
  m_particles.m_count = 0;
  m_particles.m_data = nullptr;
  m_modelParticles.m_count = 0;
  m_modelParticles.m_data = nullptr;
  
  // Initialize alive/dead stacks
  m_alive.m_stackPointer = 0;
  m_dead.m_stackPointer = 0;
  
  // Initialize particle keys (2 keyframes)
  for (int i = 0; i < 2; i++) {
    CParticleKey::CParticleKey(&m_particleKeys[i]);
  }
  
  // Default material (opaque)
  m_particleMaterial.alpha = GxBlend_Opaque;
  m_particleMaterial.flags = 7;
  
  // Initialize tumbling
  m_tumblex = {0, 0};
  m_tumbley = {0, 0};
  m_tumblez = {0, 0};
  
  // Initialize wind
  m_windVector = {0, 0, 0};
  m_windTime = 0;
  
  // Initialize follow
  m_followVector = {0, 0, 0};
  m_followB = 0;
  m_followM = 0;
  
  // Initialize XY axis
  m_xyAxis = {0, 0, 0};
  
  // Initialize transform
  m_modelToWorld = Matrix4::Identity();
  m_prevModelToWorldTrans = {0, 0, 0};
  
  // Default particle properties
  m_particleEmissionRate = 0;
  m_particleLifeSpan = 0;
  m_particleVelocity = 0;
  m_particleAcceleration = 0;
  m_particleVelocityVariation = 0.1f;
  m_particleAngularVelocity = 0;
  m_particleZsource = 0;
  m_particleType = PT_QUAD;
  m_emitterType = PET_BASE_EMITTER;
  
  // Texture info
  m_textureRows = 1;
  m_textureColumns = 1;
  
  // Initialize vtable
  *(void**)this = &_vftable_;
  
  return this;
}
```

### CreateParticle @ 0x0047d9d0
```c
CParticle2* CreateParticle(CParticleEmitter2* emitter) {
  // Check dead pool first
  if (emitter->m_dead.m_stackPointer > 0) {
    uint index = emitter->m_dead.m_stack[--emitter->m_dead.m_stackPointer];
    emitter->m_alive.m_stack[emitter->m_alive.m_stackPointer++] = index;
    return &emitter->m_particles.m_data[index];
  }
  
  // Allocate new particle if space available
  if (emitter->m_particles.m_count < MAX_PARTICLES) {
    uint index = emitter->m_particles.m_count++;
    emitter->m_alive.m_stack[emitter->m_alive.m_stackPointer++] = index;
    return &emitter->m_particles.m_data[index];
  }
  
  // No particle available
  return nullptr;
}
```

### EmitNewParticles @ 0x00480880
```c
void EmitNewParticles(CParticleEmitter2* emitter, float deltaTime) {
  // Calculate number of particles to emit
  float numToEmit = emitter->m_particleEmissionRate * deltaTime;
  
  // Get emission rate from keyframe if animated
  if (emitter->m_particleKeys[0].time > 0) {
    // Sample emission rate from keyframe track
    float currentEmission = SampleKeyframeTrack(
      emitter->m_particleKeys[0], 
      emitter->m_currentTime
    );
    numToEmit = currentEmission * deltaTime;
  }
  
  // Integer part - emit guaranteed particles
  int guaranteed = (int)numToEmit;
  for (int i = 0; i < guaranteed; i++) {
    EmitSingleParticle(emitter);
  }
  
  // Fractional part - probabilistic emission
  if (frac(numToEmit) > rand() / RAND_MAX) {
    EmitSingleParticle(emitter);
  }
}
```

### UpdateParticles @ 0x00480880
```c
void UpdateParticles(CParticleEmitter2* emitter, float deltaTime) {
  for (int i = 0; i < emitter->m_alive.m_stackPointer; i++) {
    uint index = emitter->m_alive.m_stack[i];
    CParticle2* particle = &emitter->m_particles.m_data[index];
    
    // Apply velocity
    particle->position += particle->velocity * deltaTime;
    
    // Apply acceleration
    particle->velocity += particle->acceleration * deltaTime;
    
    // Apply wind
    particle->velocity += emitter->m_windVector * deltaTime;
    
    // Apply gravity
    particle->velocity.z -= 9.81f * deltaTime;
    
    // Update lifetime
    particle->life -= deltaTime / particle->maxLife;
    
    // Check if dead
    if (particle->life <= 0) {
      // Move to dead pool
      emitter->m_dead.m_stack[emitter->m_dead.m_stackPointer++] = index;
      
      // Remove from alive list (swap with last)
      emitter->m_alive.m_stack[i--] = 
        emitter->m_alive.m_stack[--emitter->m_alive.m_stackPointer];
    }
  }
}
```

## Implementation Recommendations for MdxViewer

### 1. Particle System Class
```csharp
public class ParticleEmitter
{
    public EmitterType Type { get; set; }
    public ParticleType ParticleType { get; set; }
    public float EmissionRate { get; set; }
    public float LifeSpan { get; set; }
    public float Velocity { get; set; }
    public float VelocityVariation { get; set; }
    public float Acceleration { get; set; }
    public float AngularVelocity { get; set; }
    public Vector3 WindVector { get; set; }
    public float WindTime { get; set; }
    public Vector3 FollowVector { get; set; }
    public float FollowB { get; set; }
    public float FollowM { get; set; }
    public int TextureRows { get; set; } = 1;
    public int TextureColumns { get; set; } = 1;
    public BlendMode BlendMode { get; set; }
    public ReplaceableTexture Texture { get; set; }
    
    // Particle keyframe tracks
    public KeyframeTrack<float>? EmissionRateTrack { get; set; }
    public KeyframeTrack<float>? LifeSpanTrack { get; set; }
    public KeyframeTrack<float>? VelocityTrack { get; set; }
    public KeyframeTrack<float>? GravityTrack { get; set; }
    public KeyframeTrack<float>? ScaleTrack { get; set; }
    public KeyframeTrack<Color>? ColorTrack { get; set; }
    public KeyframeTrack<float>? AlphaTrack { get; set; }
    public KeyframeTrack<float>? RotationTrack { get; set; }
    
    // Active particles
    private List<Particle> _particles = new();
    private Stack<int> _deadPool = new();
    
    public void Emit(float deltaTime, Matrix4x4 transform)
    {
        float emissionRate = EmissionRate;
        if (EmissionRateTrack != null) {
            emissionRate = EmissionRateTrack.Evaluate(CurrentTime);
        }
        
        int count = (int)(emissionRate * deltaTime);
        for (int i = 0; i < count; i++) {
            CreateParticle(transform);
        }
    }
    
    private void CreateParticle(Matrix4x4 transform)
    {
        Particle particle = new();
        
        // Get properties from tracks
        float life = LifeSpan;
        if (LifeSpanTrack != null) {
            life = LifeSpanTrack.Evaluate(CurrentTime);
        }
        
        particle.MaxLife = life;
        particle.Life = 1.0f;
        
        // Initialize position based on emitter type
        particle.Position = GetEmitterPosition(transform);
        
        // Initialize velocity
        float vel = Velocity;
        if (VelocityTrack != null) {
            vel = VelocityTrack.Evaluate(CurrentTime);
        }
        particle.Velocity = GetRandomVelocity(vel) * transform.Forward;
        
        // Initialize color
        if (ColorTrack != null) {
            particle.Color = ColorTrack.Evaluate(CurrentTime);
        }
        
        _particles.Add(particle);
    }
    
    public void Update(float deltaTime)
    {
        foreach (var particle in _particles.ToList()) {
            particle.Life -= deltaTime / particle.MaxLife;
            
            if (particle.Life <= 0) {
                _particles.Remove(particle);
                _deadPool.Push(_particles.IndexOf(particle));
                continue;
            }
            
            // Update physics
            particle.Velocity += new Vector3(0, 0, -9.81f * deltaTime); // Gravity
            particle.Velocity += WindVector * deltaTime;
            particle.Position += particle.Velocity * deltaTime;
            
            // Update color/alpha
            if (ColorTrack != null) {
                particle.Color = ColorTrack.Evaluate(CurrentTime);
            }
            if (AlphaTrack != null) {
                particle.Alpha = AlphaTrack.Evaluate(CurrentTime);
            }
        }
    }
}
```

### 2. Particle Rendering
```csharp
public class ParticleRenderer
{
    public void Render(ParticleEmitter emitter, Camera camera)
    {
        foreach (var particle in emitter._particles) {
            // Calculate billboard quad
            Vector3 right, up;
            camera.GetBillboardVectors(out right, out up);
            
            float size = 1.0f;
            if (emitter.ScaleTrack != null) {
                size = emitter.ScaleTrack.Evaluate(emitter.CurrentTime);
            }
            
            // Draw billboard quad
            DrawQuad(
                particle.Position,
                right * size,
                up * size,
                particle.Color,
                emitter.BlendMode,
                emitter.Texture
            );
        }
    }
}
```

### 3. Blend Modes for Particles
```csharp
public enum ParticleBlendMode
{
    Opaque = 0,      // No blending
    Blend = 1,       // Standard alpha blending (SrcAlpha, InvSrcAlpha)
    Additive = 2,   // Additive blending (SrcAlpha, One)
    AlphaKey = 3    // Alpha testing (discard if alpha < threshold)
}

public void SetParticleBlendMode(ParticleBlendMode mode)
{
    switch (mode) {
        case ParticleBlendMode.Opaque:
            GL.Disable(Enable.Blend);
            break;
        case ParticleBlendMode.Blend:
            GL.Enable(Enable.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            break;
        case ParticleBlendMode.Additive:
            GL.Enable(Enable.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
            break;
        case ParticleBlendMode.AlphaKey:
            GL.Enable(Enable.Blend);
            GL.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.Zero);
            // Also enable alpha test
            GL.Enable(Enable.AlphaTest);
            GL.AlphaFunc(AlphaFunction.Greater, 0.5f);
            break;
    }
}
```

### 4. Texture Sheet Animation
```csharp
public class TextureSheetAnimator
{
    public int Rows { get; set; }
    public int Columns { get; set; }
    public float FrameTime { get; set; }
    public bool Loop { get; set; } = true;
    
    public void GetFrameUVs(float time, out Vector2 uvMin, out Vector2 uvMax)
    {
        float frameIndex = time / FrameTime;
        
        if (Loop) {
            frameIndex = frameIndex % (Rows * Columns);
        } else if (frameIndex >= Rows * Columns) {
            frameIndex = Rows * Columns - 1;
        }
        
        int frame = (int)frameIndex;
        int row = frame / Columns;
        int col = frame % Columns;
        
        uvMin = new Vector2(
            (float)col / Columns,
            (float)row / Rows
        );
        uvMax = new Vector2(
            (float)(col + 1) / Columns,
            (float)(row + 1) / Rows
        );
    }
}
```

## Key Functions Reference

| Function | Address | Purpose |
|----------|---------|---------|
| CParticleEmitter2::CParticleEmitter2 | 0x0047c430 | Constructor |
| CreateParticleEmitter2 | 0x0073b440 | Create emitter instance |
| CreateParticle | 0x0047d9d0 | Create single particle |
| DestroyParticle | 0x0047fdd0 | Destroy particle |
| EmitNewParticles | 0x00480880 | Emit new particles |
| UpdateParticles | 0x00480880 | Update particle physics |
| BufRenderParticles | 0x0047ee30 | Render particles |
| AnimObjectSetParticleEmissionRate2 | 0x00750c60 | Set emission rate |
| AnimObjectSetParticleGravity2 | 0x00751030 | Set gravity |
| AnimObjectSetParticleLifeSpan2 | 0x00752eb0 | Set lifetime |
| AnimObjectSetParticleSpeed2 | 0x00751f70 | Set speed |
| AnimObjectSetParticleVariation2 | 0x00751400 | Set variation |

## Debugging Tips

1. **No particles showing?** Check:
   - Emission rate > 0
   - Particle lifetime > 0
   - Blend mode is correct
   - Texture is loaded
   - Particles aren't immediately dying

2. **Particles all at origin?** Check:
   - Emitter transform is correct
   - Initial velocity is set
   - Wind isn't pushing them away

3. **Particles look wrong color?** Check:
   - Color values are 0-1 range
   - Alpha is applied correctly for blend mode
   - Texture has correct colors

4. **Texture not animating?** Check:
   - Texture sheet rows/columns are set
   - Frame time is correct
   - UV coordinates are calculated properly
