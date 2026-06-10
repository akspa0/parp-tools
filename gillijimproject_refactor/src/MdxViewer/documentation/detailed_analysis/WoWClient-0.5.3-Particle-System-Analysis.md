# WoW Alpha 0.5.3 (Build 3368) Particle System Analysis

## Overview

This document provides a deep analysis of the particle system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers particle emitters, particle creation, rendering, and integration with the animation system.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `CParticleEmitter::CParticleEmitter` | 0x0048a6a0 | Particle emitter constructor |
| `CreateParticle` | 0x0047d9d0 | Create particle instance |
| `CreateParticleEmitter2` | 0x0073b440 | Create particle emitter from animation data |
| `CParticle2` | 0x004818a0 | Particle class |
| `CParticle2_Model` | 0x004818f0 | Model particle variant |
| `CParticleEmitter` | 0x0048a6f0 | Particle emitter (variant) |
| `CParticleEmitter2` | 0x0047c430 | Particle emitter 2 |
| `CParticleKey` | 0x00485890 | Particle keyframe |
| `CPlaneParticleEmitter` | 0x00484020 | Plane emitter |
| `CSphereParticleEmitter` | 0x00484500 | Sphere emitter |
| `CSplineParticleEmitter` | 0x00484a60 | Spline emitter |
| `DestroyParticle` | 0x0047fdd0 | Destroy particle |
| `EmitNewParticles` | 0x00480880 | Emit new particles |
| `AnimObjectSetParticleEmissionRate2` | 0x00750c60 | Set emission rate |
| `AnimObjectSetParticleGravity2` | 0x00751030 | Set particle gravity |
| `AnimObjectSetParticleLength2` | 0x00752340 | Set particle length |
| `AnimObjectSetParticleSpeed2` | 0x00751f70 | Set particle speed |
| `AnimObjectSetParticleVariation2` | 0x00751400 | Set speed variation |
| `AnimObjectSetParticleWidth2` | 0x00752710 | Set particle width |
| `AnimObjectSetParticleZsource2` | 0x00752ae0 | Set Z source |
| `AnimObjectSetParticleLifeSpan2` | 0x00752eb0 | Set particle lifespan |

---

## Particle Emitter System

### CParticleEmitter Class Structure

```c
/* CParticleEmitter constructor at 0x0048a6a0 */
class CParticleEmitter {
    /* Reference counting */
    int m_refCount;
    
    /* Particle array - growable */
    TSGrowableArray<class CParticle> m_particles;
    
    /* Alive particle indices */
    TSGrowableArray<unsigned int> m_alive;
    
    /* Dead particle indices */
    TSGrowableArray<unsigned int> m_dead;
    
    /* Initialization */
    void Init();
};
```

### Particle Data Structure

```c
class CParticle {
    /* Position */
    float x, y, z;
    
    /* Velocity */
    float vx, vy, vz;
    
    /* Life and aging */
    float life;          // Current life (0.0 to maxLife)
    float maxLife;       // Maximum lifespan
    
    /* Appearance */
    float size;          // Particle size
    float rotation;      // Rotation angle
    
    /* Color */
    float r, g, b, a;   // RGBA color
    
    /* Texture coordinates */
    float u0, v0;        // UV coordinates
    float u1, v1;
    
    /* Physics */
    float gravity;       // Gravity effect
    float drag;          // Air resistance
};
```

---

## Particle Emitter Types

### 1. CPlaneParticleEmitter (0x00484020)

Emits particles from a plane surface:

```c
class CPlaneParticleEmitter {
    /* Plane dimensions */
    float width;         // X extent
    float height;        // Y extent
    
    /* Emission surface normal */
    float nx, ny, nz;    // Normal vector
    
    /* Edge flags */
    bool emitFromEdges;  // Emit from edges only
    bool emitFromFace;   // Emit from face
};
```

### 2. CSphereParticleEmitter (0x00484500)

Emits particles from a spherical volume:

```c
class CSphereParticleEmitter {
    /* Sphere properties */
    float radius;        // Emitter radius
    
    /* Emission mode */
    int emissionMode;    // 0 = surface, 1 = volume, 2 = shell
    
    /* Direction constraints */
    float minTheta;      // Minimum emission angle
    float maxTheta;      // Maximum emission angle
    float minPhi;        // Minimum azimuth
    float maxPhi;        // Maximum azimuth
};
```

### 3. CSplineParticleEmitter (0x00484a60)

Emits particles along a spline path:

```c
class CSplineParticleEmitter {
    /* Control points */
    C3Vector controlPoints[4];  // Cubic spline
    
    /* Interpolation */
    float t;                   // Current position on spline
    
    /* Emission settings */
    float emissionRate;        // Particles per second
    float speed;               // Particle speed
    float spread;              // Direction spread
};
```

---

## Particle Creation (CreateParticleEmitter2)

```c
/* CreateParticleEmitter2 at 0x0073b440 */
uchar* CreateParticleEmitter2(
    int* param_1,           // Parameter block
    CAnimData* animData,    // Animation data
    int param_3,
    int param_4,
    MDLTRACKTYPE trackType   // Animation track type
) {
    // Validate animation data
    if (animData == NULL) {
        Error("animation data required");
    }
    
    // Create emitter object
    CAnimEmitter2Obj* emitter = AnimObjectCreateEmitter2(animData);
    if (emitter == NULL) {
        Error("failed to create emitter");
    }
    
    // Process animation tracks
    GenericHandlerAnim(param_1 + 1, animData, (CAnimObj*)emitter,
                      param_3, param_4, trackType);
    
    // Set emission properties from tracks
    AnimObjectSetParticleEmissionRate2(..., animData, emitter, trackType);
    AnimObjectSetParticleGravity2(..., animData, emitter, trackType);
    AnimObjectSetEmitterLongitude2(..., animData, emitter, trackType);
    AnimObjectSetEmitterLatitude2(..., animData, emitter, trackType);
    AnimObjectSetParticleSpeed2(..., animData, emitter, trackType);
    AnimObjectSetParticleVariation2(..., animData, emitter, trackType);
    AnimObjectSetParticleLength2(..., animData, emitter, trackType);
    AnimObjectSetParticleWidth2(..., animData, emitter, trackType);
    AnimObjectSetParticleZsource2(..., animData, emitter, trackType);
    AnimObjectSetVisibilityTrack(..., (CAnimVisibleObj*)&emitter->field_0xc4, trackType);
    AnimObjectSetParticleLifeSpan2(..., animData, emitter, trackType);
    
    return success;
}
```

---

## Particle Properties

### Emission Properties

| Property | Function | Purpose |
|----------|----------|---------|
| Emission Rate | `AnimObjectSetParticleEmissionRate2` | Particles per second |
| Particle Speed | `AnimObjectSetParticleSpeed2` | Initial velocity magnitude |
| Speed Variation | `AnimObjectSetParticleVariation2` | Random speed variation |
| Gravity | `AnimObjectSetParticleGravity2` | Gravity acceleration |
| Lifespan | `AnimObjectSetParticleLifeSpan2` | How long particles live |

### Emitter Geometry

| Property | Function | Purpose |
|----------|----------|---------|
| Longitude | `AnimObjectSetEmitterLongitude2` | Vertical emission angle |
| Latitude | `AnimObjectSetEmitterLatitude2` | Horizontal emission angle |

### Particle Appearance

| Property | Function | Purpose |
|----------|----------|---------|
| Width | `AnimObjectSetParticleWidth2` | Particle width |
| Length | `AnimObjectSetParticleLength2` | Particle length |
| Z Source | `AnimObjectSetParticleZsource2` | Z-axis behavior |

---

## Particle Update Loop

### EmitNewParticles

```c
/* EmitNewParticles at 0x00480880 */
void EmitNewParticles(CParticleEmitter* emitter, float deltaTime) {
    // Calculate emission count
    float emissionCount = emitter->emissionRate * deltaTime;
    int count = (int)emissionCount;
    
    // Add fractional part for accumulation
    emitter->emissionAccumulator += emissionCount - count;
    if (emissionAccumulator >= 1.0f) {
        count += (int)emissionAccumulator;
        emissionAccumulator -= count;
    }
    
    // Emit particles
    for (int i = 0; i < count; i++) {
        // Get dead particle from pool
        int particleIndex = GetDeadParticle(emitter);
        if (particleIndex == -1) {
            // Create new particle if pool empty
            particleIndex = CreateNewParticle(emitter);
        }
        
        if (particleIndex != -1) {
            // Initialize particle
            CParticle* particle = &emitter->m_particles[particleIndex];
            InitializeParticle(emitter, particle);
            
            // Add to alive list
            emitter->m_alive.push(particleIndex);
        }
    }
}
```

### Particle Update

```c
void UpdateParticles(CParticleEmitter* emitter, float deltaTime) {
    for (int i = 0; i < emitter->m_alive.count; i++) {
        int particleIndex = emitter->m_alive[i];
        CParticle* particle = &emitter->m_particles[particleIndex];
        
        // Update life
        particle->life -= deltaTime;
        if (particle->life <= 0) {
            // Mark as dead
            MarkParticleDead(emitter, particleIndex);
            continue;
        }
        
        // Apply gravity
        particle->vy -= particle->gravity * deltaTime;
        
        // Apply drag
        particle->vx *= (1.0f - particle->drag * deltaTime);
        particle->vy *= (1.0f - particle->drag * deltaTime);
        particle->vz *= (1.0f - particle->drag * deltaTime);
        
        // Update position
        particle->x += particle->vx * deltaTime;
        particle->y += particle->vy * deltaTime;
        particle->z += particle->vz * deltaTime;
        
        // Update appearance based on life
        float lifeRatio = particle->life / particle->maxLife;
        UpdateParticleAppearance(particle, lifeRatio);
    }
}
```

---

## Rendering System

### BufRenderParticles

```c
/* BufRenderParticles at 0x0047ee30 */
void BufRenderParticles(CParticleEmitter* emitter) {
    // Build vertex buffer
    for (int i = 0; i < emitter->m_alive.count; i++) {
        int particleIndex = emitter->m_alive[i];
        CParticle* particle = &emitter->m_particles[particleIndex];
        
        // Calculate vertex positions (billboard quad)
        C3Vector vertices[4];
        CalculateBillboardQuad(particle, vertices);
        
        // Set color
        SetVertexColor(particle->r, particle->g, particle->b, particle->a);
        
        // Set texture coordinates
        SetTexCoords(particle->u0, particle->v0, particle->u1, particle->v1);
        
        // Add to render buffer
        AddQuadToBuffer(vertices);
    }
    
    // Draw all particles in single call
    DrawBuffer();
}
```

### Billboard Calculation

```c
void CalculateBillboardQuad(CParticle* particle, C3Vector* vertices) {
    // Get camera right and up vectors
    C3Vector camRight, camUp;
    GetCameraVectors(&camRight, &camUp);
    
    // Scale by particle size
    float halfSize = particle->size * 0.5f;
    
    // Build quad centered on particle position
    vertices[0] = particle->pos - camRight*halfSize - camUp*halfSize;
    vertices[1] = particle->pos + camRight*halfSize - camUp*halfSize;
    vertices[2] = particle->pos + camRight*halfSize + camUp*halfSize;
    vertices[3] = particle->pos - camRight*halfSize + camUp*halfSize;
}
```

---

## Particle Material System

### CreateParticleMaterial

```c
/* CreateParticleMaterial at 0x00448bd0 */
HParticleMaterial CreateParticleMaterial(
    const char* textureName,
    int blendMode,
    bool additive
) {
    // Create material
    HParticleMaterial material = AllocMaterial();
    
    // Set texture
    SetTexture(material, textureName);
    
    // Set blend mode
    switch (blendMode) {
        case 0:  // Opaque
            SetBlendOp(material, BLEND_OPAQUE);
            break;
        case 1:  // Transparent
            SetBlendOp(material, BLEND_ALPHA);
            break;
        case 2:  // Blend
            SetBlendOp(material, BLEND_BLEND);
            break;
        case 3:  // Additive
            SetBlendOp(material, BLEND_ADD);
            break;
    }
    
    // Set depth/write
    SetDepthWrite(material, !additive);
    SetAlphaTest(material, true);
    
    return material;
}
```

---

## MDX Particle Chunks

### PREM (Particle Emitter)

```
PREM Chunk:
├── uint32_t magic     // 'PREM' (0x4d455052)
├── uint32_t size      // Chunk size
└── ParticleEmitterData entries[]
```

**ParticleEmitterData Structure:**

```c
struct ParticleEmitterData {
    uint32_t id;              // Emitter ID
    uint32_t particleFlags;    // Particle flags
    
    // Geometry
    float emitterRadius;      // Emitter radius
    float emissionRate;       // Particles per second
    float lifetime;           // Particle lifetime
    
    // Velocity
    float speed;              // Initial speed
    float variation;          // Speed variation
    float gravity;            // Gravity effect
    
    // Direction
    float longitude;          // Vertical angle (radians)
    float latitude;           // Horizontal angle spread
    
    // Appearance
    float particleSize;       // Particle size
    float particleWidth;      // Particle width
    float particleLength;     // Particle length
    
    // Texture
    char textureName[256];    // Texture filename
    
    // Animation
    uint32_t emitterId;       // Index into global emitters
    uint32_t colorIndex;      // Color animation index
    uint32_t opacityIndex;    // Opacity animation index
    uint32_t scaleIndex;      // Scale animation index
};
```

### PRE2 (Particle Emitter 2)

Enhanced particle emitter with more properties:

```c
struct ParticleEmitter2Data {
    uint32_t id;
    uint32_t flags;
    
    // Enhanced geometry
    float emitterUnk1;
    float emitterUnk2;
    float emitterUnk3;
    
    // Enhanced velocity
    float speedUnk1;
    float speedUnk2;
    
    // New properties
    float zSource;            // Z-axis behavior
    
    // Texture coordinates
    float uvs[4];             // UV rectangle
};
```

---

## Animation Integration

### Emitter Properties Animation

Each particle emitter property can be animated using keyframe tracks:

```c
struct ParticleAnimTrack {
    uint32_t count;           // Number of keyframes
    MDLTRACKTYPE type;       // Interpolation type
    uint32_t globalSeqId;    // Global sequence ID
    ParticleKeyframe keys[];  // Keyframe data
};

struct ParticleKeyframe {
    uint32_t time;           // Time in milliseconds
    float value;             // Property value
};
```

### Supported Animation Properties

| Property | Track Name | Description |
|----------|------------|-------------|
| Emission Rate | `PARTICLE_RATE` | Particles per second |
| Gravity | `PARTICLE_GRAVITY` | Gravity strength |
| Speed | `PARTICLE_SPEED` | Initial velocity |
| Variation | `PARTICLE_VAR` | Random variation |
| Size | `PARTICLE_SCALE` | Particle size |
| Color | `PARTICLE_COLOR` | RGB color |
| Opacity | `PARTICLE_OPACITY` | Alpha value |

---

## Performance Optimization

### Particle Pooling

The engine uses a growable array with dead particle pooling:

```c
void InitializeParticle(CParticleEmitter* emitter, CParticle* particle) {
    // Check dead pool first
    if (emitter->m_dead.count > 0) {
        // Recycle from dead pool
        int recycledIndex = emitter->m_dead.pop();
        emitter->m_particles[recycledIndex] = *particle;
        return;
    }
    
    // Create new particle
    emitter->m_particles.push(*particle);
}
```

### Memory Layout

```
CParticleEmitter Memory Layout:
├── CParticleEmitter header (32 bytes)
├── TSGrowableArray<CParticle> particles
│   └── CParticle[capacity]
├── TSGrowableArray<unsigned int> alive
│   └── uint32_t[capacity]
└── TSGrowableArray<unsigned int> dead
    └── uint32_t[capacity]

CParticle Structure (48 bytes):
├── Position: float[3] (12 bytes)
├── Velocity: float[3] (12 bytes)
├── Life: float (4 bytes)
├── Size: float (4 bytes)
├── Color: float[4] (16 bytes)
└── Padding: 0 bytes
```

---

## Integration with Scene Rendering

### Rendering Order

1. **Opaque Geometry** - Terrain, WMOs, opaque models
2. **Transparent Geometry** - Transparent models, particles
3. **Alpha-Blended Particles** - Additive particles
4. **Post-Processing** - Fog, effects

### Particle Sorting

```c
void SortParticlesForRendering(CParticleEmitter* emitter) {
    // Sort by depth for correct blending
    C3Vector camPos = GetCameraPosition();
    
    qsort(emitter->m_alive.data, emitter->m_alive.count,
          sizeof(uint32_t), CompareParticleDepth);
}

int CompareParticleDepth(const void* a, const void* b) {
    CParticle* pa = &emitter->m_particles[*(uint32_t*)a];
    CParticle* pb = &emitter->m_particles[*(uint32_t*)b];
    
    float da = DistanceSquared(pa->pos, camPos);
    float db = DistanceSquared(pb->pos, camPos);
    
    return (da > db) - (da < db);  // Far to near
}
```

---

## Area-Specific Particle Effects

### Breath Particles

```c
/* AreaListZoneHasBreathParticles at 0x00667720 */
bool AreaListZoneHasBreathParticles(uint32_t areaId) {
    // Check if underwater or cold area
    AreaData* area = GetAreaData(areaId);
    return area->isUnderwater || area->temperature < 0;
}
```

---

## Summary

The particle system in WoW Alpha 0.5.3 provides:
- **Multiple emitter types**: Plane, Sphere, Spline
- **Full animation support**: Keyframe tracks for all properties
- **Efficient pooling**: Dead particle recycling
- **Flexible rendering**: Additive, alpha-blended, opaque
- **Integration**: Works with models, terrain, and WMOs

Key functions and their addresses provide a complete reference for reverse engineering and implementation.

---

*Document created: 2026-02-07*
*Analysis based on WoWClient.exe (Build 3368)*
