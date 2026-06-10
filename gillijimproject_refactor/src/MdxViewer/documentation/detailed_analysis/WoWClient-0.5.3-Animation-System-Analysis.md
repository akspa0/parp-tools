# WoW Alpha 0.5.3 (Build 3368) Animation System Analysis

## Overview

This document provides a deep analysis of the animation system in WoW Alpha 0.5.3 (Build 3368, Dec 11 2003), based on Ghidra reverse engineering of WoWClient.exe. It covers animation data structures, object creation, sequence handling, and interpolation methods.

## Related Functions

| Function | Address | Purpose |
|----------|---------|---------|
| `AnimObjectCreateAttachment` | 0x0074d870 | Create attachment point |
| `AnimObjectCreateBone` | 0x0074d8e0 | Create bone object |
| `AnimObjectCreateEmitter2` | 0x0074d970 | Create particle emitter |
| `AnimObjectCreateEvent` | 0x0074da50 | Create event object |
| `AnimObjectCreateHelper` | 0x0074d7a0 | Create helper object |
| `AnimObjectCreateLight` | 0x0074d800 | Create light object |
| `AnimObjectCreateRibbon` | 0x0074d9e0 | Create ribbon object |
| `AnimObjectSetAmbColor` | 0x00750010 | Set ambient color |
| `AnimObjectSetAmbIntensity` | 0x007504c0 | Set ambient intensity |
| `AnimObjectSetAttenuation` | 0x0074f1a0 | Set light attenuation |
| `AnimObjectSetColor` | 0x0074f790 | Set color |
| `AnimObjectSetEmitterLatitude2` | 0x00751ba0 | Set emitter latitude |
| `AnimObjectSetEmitterLongitude2` | 0x007517d0 | Set emitter longitude |
| `AnimObjectSetEventTrack` | 0x0074ddd0 | Set event track |
| `AnimObjectSetIndex` | 0x0074dac0 | Set index |
| `AnimObjectSetIntensity` | 0x0074fc40 | Set intensity |
| `AnimObjectSetParticleEmissionRate2` | 0x00750c60 | Set emission rate |
| `AnimObjectSetParticleGravity2` | 0x00751030 | Set particle gravity |
| `AnimObjectSetParticleLifeSpan2` | 0x00752eb0 | Set particle lifespan |
| `AnimObjectSetParticleSpeed2` | 0x00751f70 | Set particle speed |
| `AnimObjectSetParticleVariation2` | 0x00751400 | Set particle variation |
| `AnimObjectSetParticleWidth2` | 0x00752710 | Set particle width |
| `AnimObjectSetParticleLength2` | 0x00752340 | Set particle length |
| `AnimObjectSetParticleZsource2` | 0x00752ae0 | Set Z source |
| `AnimObjectSetVisibilityTrack` | 0x0074ddf0 | Set visibility track |
| `AnimApplyObjectFaceDir` | 0x007414a0 | Apply face direction |
| `AnimApplyObjectLookAt` | 0x00740e80 | Apply look-at |
| `AnimBuildObjectIdTranslation` | 0x0073a4d0 | Build ID translation |
| `AnimEnumObjects` | 0x0073e610 | Enumerate objects |
| `AnimGetEventObjectPosition` | 0x00741d00 | Get event position |
| `AnimGetObjectPosition` | 0x00741c00 | Get object position |
| `AnimGetObjectTimeScale` | 0x0074b680 | Get time scale |
| `AnimHasObjectId` | 0x00741960 | Check object exists |
| `AnimLockObjectSequence` | 0x00741840 | Lock sequence |
| `AnimAddCamera` | 0x007563e0 | Add camera |
| `AnimAddCameras` | 0x00755ff0 | Add multiple cameras |
| `AnimAnimateCameras` | 0x00743ec0 | Animate cameras |
| `AnimIsCameraEnabled` | 0x007407a0 | Check camera enabled |
| `AnimResetCameraOrdering` | 0x00740280 | Reset camera order |
| `AnimSetCameraOrdering` | 0x0073ff00 | Set camera order |

---

## Animation Data Structures

### CAnimData

Main animation data container:

```c
class CAnimData {
    /* Animation header */
    uint32_t animId;            // Animation ID
    uint32_t flags;             // Animation flags
    
    /* Object arrays */
    TSGrowableArray<CAnimBoneObj> boneObjs;        // Bone objects
    TSGrowableArray<CAnimLightObj> lightObjs;     // Light objects
    TSGrowableArray<CAnimEmitterObj> emitterObjs;  // Emitter objects
    TSGrowableArray<CAnimRibbonObj> ribbonObjs;    // Ribbon objects
    TSGrowableArray<CAnimCameraObj> cameraObjs;    // Camera objects
    TSGrowableArray<CAnimEventObj> eventObjs;      // Event objects
    
    /* Global sequences */
    TSGrowableArray<uint32_t> globalSeqIds;        // Global sequence IDs
    TSGrowableArray<uint32_t> globalSeqTimes;      // Global sequence durations
    
    /* Timestamp */
    uint32_t timestamp;          // Animation timestamp
};
```

### CAnimBoneObj

Bone animation object:

```c
class CAnimBoneObj {
    /* Bone identification */
    uint32_t boneId;            // Bone index
    uint32_t type;              // Object type
    
    /* Translation animation */
    CAnimTrack translation;      // Position keys
    
    /* Rotation animation */
    CAnimTrack rotation;        // Rotation keys
    
    /* Scaling animation */
    CAnimTrack scaling;         // Scale keys
    
    /* Parent bone */
    uint32_t parentBone;        // Parent bone index
    
    /* Flags */
    uint32_t flags;             // Bone flags
    
    /* Pivot point */
    C3Vector pivot;            // Local pivot point
    
    /* Index for lookup */
    uint32_t field_0x58;       // Index value
};
```

### CAnimLightObj

Light animation object:

```c
class CAnimLightObj {
    /* Light identification */
    uint32_t lightId;          // Light index
    uint32_t type;             // Object type
    
    /* Color animation */
    CAnimTrack color;          // RGB color keys
    
    /* Intensity animation */
    CAnimTrack intensity;       // Intensity keys
    
    /* Ambient color */
    CAnimTrack ambColor;       // Ambient color keys
    
    /* Ambient intensity */
    CAnimTrack ambIntensity;   // Ambient intensity keys
    
    /* Attenuation */
    CAnimTrack attenuation;    // Distance attenuation
    
    /* Visibility */
    CAnimTrack visibility;     // On/off keys
    
    /* Flags */
    uint32_t flags;             // Light flags
    
    /* Type */
    uint32_t lightType;        // 0=omni, 1=spot, 2=directional
    
    /* Index */
    uint32_t field_0x58;       // Index value
};
```

### CAnimEmitterObj

Particle emitter animation object:

```c
class CAnimEmitterObj {
    /* Emitter identification */
    uint32_t emitterId;        // Emitter index
    uint32_t type;             // Object type
    
    /* Emission rate */
    CAnimTrack emissionRate;   // Particles per second
    
    /* Gravity */
    CAnimTrack gravity;        // Gravity effect
    
    /* Lifetime */
    CAnimTrack lifetime;       // Particle lifespan
    
    /* Speed */
    CAnimTrack speed;          // Initial speed
    
    /* Variation */
    CAnimTrack variation;       // Speed variation
    
    /* Size */
    CAnimTrack scale;          // Particle size
    
    /* Position offset */
    C3Vector offset;           // Emitter offset
    
    /* Direction */
    float longitude;           // Vertical angle
    float latitude;           // Horizontal spread
    
    /* Flags */
    uint32_t flags;             // Emitter flags
    
    /* Texture */
    uint32_t textureId;       // Texture index
};
```

### CAnimRibbonObj

Ribbon emitter animation object:

```c
class CAnimRibbonObj {
    /* Ribbon identification */
    uint32_t ribbonId;         // Ribbon index
    uint32_t type;             // Object type
    
    /* Color animation */
    CAnimTrack color;          // RGBA color keys
    
    /* Visibility */
    CAnimTrack visibility;     // Visibility keys
    
    /* Height above */
    CAnimTrack heightAbove;     // Height above geometry
    
    /* Height below */
    CAnimTrack heightBelow;     // Height below geometry
    
    /* Texture slot */
    uint32_t textureSlot;      // Texture unit
    
    /* Flags */
    uint32_t flags;             // Ribbon flags
    
    /* Index */
    uint32_t field_0x58;       // Index value
};
```

---

## Animation Track Structure

### CAnimTrack

```c
struct CAnimTrack {
    uint32_t count;           // Number of keyframes
    MDLTRACKTYPE type;       // Interpolation type
    uint32_t globalSeqId;     // Global sequence ID (-1 if none)
    void* keys;               // Keyframe data
};
```

### MDLTRACKTYPE Enumeration

| Value | Name | Description |
|-------|------|-------------|
| 0x0 | TRACK_NO_INTERP | No interpolation |
| 0x1 | TRACK_LINEAR | Linear interpolation |
| 0x2 | TRACK_HERMITE | Hermite spline interpolation |
| 0x3 | TRACK_BEZIER | Bezier curve interpolation |
| 0x4 | TRACK_NUM_TYPES | Number of types |

### Keyframe Structures

```c
struct VectorKeyframe {
    uint32_t time;            // Time in milliseconds
    C3Vector value;           // Position/scale value
};

struct QuaternionKeyframe {
    uint32_t time;            // Time in milliseconds
    C4Quaternion value;       // Rotation (x, y, z, w)
};

struct FloatKeyframe {
    uint32_t time;            // Time in milliseconds
    float value;              // Scalar value
};

struct ColorKeyframe {
    uint32_t time;            // Time in milliseconds
    uint32_t color;           // ABGR color (0xAABBGGRR)
};
```

---

## Object Creation

### AnimObjectCreateBone

```c
/* AnimObjectCreateBone at 0x0074d8e0 */
CAnimBoneObj* AnimObjectCreateBone(CAnimData* animData) {
    // Validate animation data
    if (animData == NULL) {
        Error("animation data required");
        return NULL;
    }
    
    // Get current count
    uint32_t index = animData->boneObjs.m_count;
    
    // Allocate new bone
    CAnimBoneObj* bone = &animData->boneObjs.m_data[index];
    animData->boneObjs.m_count = index + 1;
    
    if (bone == NULL) {
        Error("failed to allocate bone");
        return NULL;
    }
    
    // Set index
    bone->field_0x58 = index;
    
    return bone;
}
```

### AnimObjectCreateLight

```c
/* AnimObjectCreateLight at 0x0074d800 */
CAnimLightObj* AnimObjectCreateLight(CAnimData* animData) {
    if (animData == NULL) {
        Error("animation data required");
        return NULL;
    }
    
    uint32_t index = animData->lightObjs.m_count;
    CAnimLightObj* light = &animData->lightObjs.m_data[index];
    animData->lightObjs.m_count = index + 1;
    
    if (light == NULL) {
        Error("failed to allocate light");
        return NULL;
    }
    
    light->field_0x58 = index;
    return light;
}
```

### AnimObjectCreateRibbon

```c
/* AnimObjectCreateRibbon at 0x0074d9e0 */
CAnimRibbonObj* AnimObjectCreateRibbon(CAnimData* animData) {
    if (animData == NULL) {
        Error("animation data required");
        return NULL;
    }
    
    uint32_t index = animData->ribbonObjs.m_count;
    CAnimRibbonObj* ribbon = &animData->ribbonObjs.m_data[index];
    animData->ribbonObjs.m_count = index + 1;
    
    if (ribbon == NULL) {
        Error("failed to allocate ribbon");
        return NULL;
    }
    
    ribbon->field_0x58 = index;
    return ribbon;
}
```

---

## Property Setting

### AnimObjectSetColor

```c
/* AnimObjectSetColor at 0x0074f790 */
void AnimObjectSetColor(CAnimLightObj* light, uint32_t trackType) {
    // Set up color track based on track type
    switch (trackType) {
        case TRACK_NO_INTERP:
            light->color.type = TRACK_NO_INTERP;
            break;
        case TRACK_LINEAR:
            light->color.type = TRACK_LINEAR;
            break;
        case TRACK_HERMITE:
            light->color.type = TRACK_HERMITE;
            break;
    }
}
```

### AnimObjectSetParticleSpeed2

```c
/* AnimObjectSetParticleSpeed2 at 0x00751f70 */
void AnimObjectSetParticleSpeed2(
    void* param,               // Parameter block
    int size,                  // Parameter size
    CAnimData* animData,
    CAnimEmitterObj* emitter,
    MDLTRACKTYPE trackType
) {
    // Read speed values from parameter block
    float* values = (float*)param;
    uint32_t count = size / sizeof(float);
    
    // Initialize track
    emitter->speed.type = trackType;
    emitter->speed.count = count;
    emitter->speed.keys = AllocateKeys(count);
    
    // Copy values with time
    for (uint32_t i = 0; i < count; i++) {
        FloatKeyframe* key = &emitter->speed.keys[i];
        key->time = values[i * 2];      // Time at even indices
        key->value = values[i * 2 + 1]; // Value at odd indices
    }
}
```

---

## Sequence Management

### AnimLockObjectSequence

```c
/* AnimLockObjectSequence at 0x00741840 */
void AnimLockObjectSequence(CAnimData* animData, uint32_t objectId) {
    // Find object by ID
    CAnimObj* obj = FindObjectById(animData, objectId);
    if (obj == NULL) {
        Error("object not found");
        return;
    }
    
    // Lock sequence
    obj->sequenceLocked = true;
    obj->lockedSequence = animData->currentSequence;
}
```

### AnimGetObjectTimeScale

```c
/* AnimGetObjectTimeScale at 0x0074b680 */
float AnimGetObjectTimeScale(CAnimData* animData, uint32_t objectId) {
    CAnimObj* obj = FindObjectById(animData, objectId);
    if (obj == NULL) {
        return 1.0f;  // Default time scale
    }
    
    return obj->timeScale;
}
```

---

## Position Queries

### AnimGetObjectPosition

```c
/* AnimGetObjectPosition at 0x00741c00 */
void AnimGetObjectPosition(
    CAnimData* animData,
    uint32_t objectId,
    uint32_t time,
    C3Vector* outPosition
) {
    CAnimBoneObj* bone = FindBoneById(animData, objectId);
    if (bone == NULL) {
        outPosition->x = 0;
        outPosition->y = 0;
        outPosition->z = 0;
        return;
    }
    
    // Evaluate translation track
    EvaluateVectorTrack(&bone->translation, time, outPosition);
}
```

---

## Interpolation Methods

### Linear Interpolation

```c
void InterpolateLinear(
    float t,                   // Interpolation factor [0, 1]
    float* start,              // Start value
    float* end,               // End value
    float* result             // Output
) {
    result[0] = start[0] + (end[0] - start[0]) * t;
    result[1] = start[1] + (end[1] - start[1]) * t;
    result[2] = start[2] + (end[2] - start[2]) * t;
}
```

### Hermite Interpolation

```c
void InterpolateHermite(
    float t,
    C3Vector* p0, C3Vector* p1,     // Points
    C3Vector* t0, C3Vector* t1,       // Tangents
    C3Vector* result
) {
    float t2 = t * t;
    float t3 = t2 * t;
    
    // Hermite basis functions
    float h00 = 2*t3 - 3*t2 + 1;
    float h10 = t3 - 2*t2 + t;
    float h01 = -2*t3 + 3*t2;
    float h11 = t3 - t2;
    
    result->x = h00*p0->x + h10*t0->x + h01*p1->x + h11*t1->x;
    result->y = h00*p0->y + h10*t0->y + h01*p1->y + h11*t1->y;
    result->z = h00*p0->z + h10*t0->z + h01*p1->z + h11*t1->z;
}
```

---

## Global Sequences

### Global Sequence Structure

```c
struct GlobalSequence {
    uint32_t seqId;           // Sequence ID
    uint32_t duration;        // Duration in milliseconds
    
    // Tracks shared across all animations
    TSGrowableArray<CAnimTrack> sharedTracks;
};
```

### Global Sequence Usage

When an animation uses a global sequence:
1. The `globalSeqId` field is set to the global sequence ID
2. The animation references times within the global sequence
3. All animations using the same global sequence stay synchronized

---

## MDX Animation Chunks

### SEQS (Sequences)

```
SEQS Chunk:
├── uint32_t magic     // 'SEQS' (0x53514553)
├── uint32_t size      // Chunk size
└── SequenceData entries[]
```

**SequenceData Structure:**

```c
struct SequenceData {
    char name[32];            // Sequence name
    uint32_t startTime;       // Start time (ms)
    uint32_t endTime;         // End time (ms)
    float moveSpeed;         // Movement speed
    uint32_t flags;           // Sequence flags
    C3Vector boundsMin;      // Minimum bounds
    C3Vector boundsMax;      // Maximum bounds
    float radius;            // Bounding radius
};
```

### GLBS (Global Sequences)

```
GLBS Chunk:
├── uint32_t magic     // 'GLBS' (0x53424c47)
├── uint32_t size      // Chunk size
└── GlobalSeqData entries[]
```

---

## Animation Evaluation Pipeline

```
Animation Evaluation:
1. Get current animation time
2. For each animated object:
   a. Find surrounding keyframes
   b. Calculate interpolation factor
   c. Interpolate value based on track type
   d. Apply to object
3. Update dependent objects (bones, attachments)
4. Recalculate transforms
5. Update GPU buffers
```

---

## Summary

The animation system in WoW Alpha 0.5.3 provides:
- **Multiple object types**: Bones, lights, emitters, ribbons, cameras, events
- **Flexible tracks**: Each property has its own animation track
- **Multiple interpolation types**: Linear, Hermite, Bezier
- **Global sequences**: Synchronized animations across multiple models
- **Sequence locking**: Prevent animation updates during specific states
- **Efficient evaluation**: Keyframe lookup with interpolation

Key functions and addresses provide a complete reference for reverse engineering and implementation.

---

*Document created: 2026-02-07*
*Analysis based on WoWClient.exe (Build 3368)*
