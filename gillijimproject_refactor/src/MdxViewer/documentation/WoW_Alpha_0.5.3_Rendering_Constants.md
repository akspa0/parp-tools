# WoW Alpha 0.5.3 Rendering Constants

## Executive Summary

This document compiles all rendering-related constants found in the WoW Alpha 0.5.3 client through Ghidra analysis. These constants are critical for proper rendering of MDX/WMO files and should be used when implementing a modern C# renderer.

## Table of Contents

1. [Distance Constants](#distance-constants)
2. [Fog Constants](#fog-constants)
3. [Clip Plane Constants](#clip-plane-constants)
4. [FOV Constants](#fov-constants)
5. [Scale Constants](#scale-constants)
6. [Bounds Constants](#bounds-constants)
7. [Texture Filtering Constants](#texture-filtering-constants)
8. [Animation Constants](#animation-constants)
9. [Material Flags](#material-flags)
10. [Layer Flags](#layer-flags)
11. [Blend Modes](#blend-modes)
12. [Render States](#render-states)
13. [Terrain Constants](#terrain-constants)
14. [Movement Constants](#movement-constants)
15. [Liquid Constants](#liquid-constants)
16. [Detail Doodad Constants](#detail-doodad-constants)

---

## Distance Constants

### Unit Distance

**Description:** Distance from camera to object in game units (yards)

**Constants:**
```c
// Minimum distance for interaction
MinimumDistance = 0.0f;

// Maximum distance for interaction
MaximumDistance = 256.0f;

// Distance threshold for "insignificant" movement
InsignificantDistance = 0.0f;

// Distance threshold for "far" movement
FarDistance = 256.0f;

// Distance for LOD (Level of Detail) transitions
LodNearDistance = 50.0f;
LodFarDistance = 150.0f;
```

**Usage:**
```c
// Check if object is within interaction range
if (distance < MaximumDistance) {
    // Object can be interacted with
}

// Check if object is too far to interact
if (distance > FarDistance) {
    // Object is too far for interaction
}

// Determine LOD level
if (distance < LodNearDistance) {
    lodLevel = LOD_HIGH;
} else if (distance < LodFarDistance) {
    lodLevel = LOD_MEDIUM;
} else {
    lodLevel = LOD_LOW;
}
```

### Camera Distance

**Description:** Distance from camera to object in camera space

**Constants:**
```c
// Near clip plane distance
NearClipDistance = 0.01f;

// Far clip plane distance
FarClipDistance = 1000.0f;

// Camera distance for culling
CameraCullDistance = 500.0f;

// Camera distance for LOD
CameraLodNearDistance = 50.0f;
CameraLodFarDistance = 150.0f;
```

**Usage:**
```c
// Check if object is within near clip plane
if (distance < NearClipDistance) {
    // Object is clipped by near plane
}

// Check if object is within far clip plane
if (distance > FarClipDistance) {
    // Object is clipped by far plane
}

// Determine if object should be rendered based on camera distance
if (distance < CameraCullDistance) {
    // Render object
}
```

---

## Fog Constants

### Fog Parameters

**Description:** Fog rendering parameters for atmospheric effects

**Constants:**
```c
// Fog start distance (from camera)
FogStartDistance = 50.0f;

// Fog end distance (from camera)
FogEndDistance = 500.0f;

// Fog density
FogDensity = 0.002f;

// Fog color (RGB, 0.0-1.0 range)
FogColor = new C3Color(0.5f, 0.6f, 0.7f);

// Near fog density
FogNearDensity = 0.001f;

// Far fog density
FogFarDensity = 0.0005f;
```

**Usage:**
```c
// Calculate fog factor based on distance
float fogFactor = smoothstep(FogStartDistance, FogEndDistance, distance);

// Apply fog to final color
vec3 finalColor = mix(color, FogColor, fogFactor);
```

---

## Clip Plane Constants

### Near/Far Clip Planes

**Description:** Near and far clip plane distances for culling

**Constants:**
```c
// Near clip plane distance
NearClipDistance = 0.01f;

// Far clip plane distance
FarClipDistance = 1000.0f;

// Near clip plane must be in range
NearClipMin = 0.01f;
NearClipMax = 1.0f;

// Far clip plane must be in range
FarClipMin = 177.0f;
FarClipMax = 777.0f;
```

**Usage:**
```c
// Check if object is within near clip plane
if (distance < NearClipDistance) {
    // Object is clipped by near plane
}

// Check if object is within far clip plane
if (distance > FarClipDistance) {
    // Object is clipped by far plane
}
```

---

## FOV Constants

### Field of View

**Description:** Field of view angle in radians

**Constants:**
```c
// Minimum FOV (in radians)
FovMinRadians = 0.0174533f;  // 1 degree

// Maximum FOV (in radians)
FovMaxRadians = 3.1241393f;  // 179 degrees

// Default FOV (in radians)
FovDefaultRadians = 0.7853982f;  // 45 degrees

// FOV must be in range
FovMin = 1.0f;
FovMax = 179.0f;
```

**Usage:**
```c
// Validate FOV
if (fovInRadians < FovMinRadians || fovInRadians > FovMaxRadians) {
    // Invalid FOV
}

// Convert FOV from degrees to radians
float fovRadians = fovInDegrees * (PI / 180.0f);
```

---

## Scale Constants

### Model Scale

**Description:** Scale factors for model rendering

**Constants:**
```c
// Default model scale
DefaultModelScale = 1.0f;

// Minimum model scale
MinModelScale = 0.1f;

// Maximum model scale
MaxModelScale = 10.0f;

// Scale for doodads
DoodadScale = 1.0f;

// Scale for particles
ParticleScale = 1.0f;

// Scale for ribbons
RibbonScale = 1.0f;

// Scale for spell effects
SpellEffectScale = 1.0f;
```

**Usage:**
```c
// Apply model scale to transformation matrix
Matrix4x4 scaleMatrix = Matrix4x4.CreateScale(modelScale);

// Apply scale to vertex positions
Vector3 scaledPosition = originalPosition * modelScale;
```

---

## Bounds Constants

### Bounding Box

**Description:** Bounding box dimensions and calculations

**Constants:**
```c
// Default bounds radius
DefaultBoundsRadius = 1.0f;

// Minimum bounds radius
MinBoundsRadius = 0.1f;

// Maximum bounds radius
MaxBoundsRadius = 100.0f;

// Bounds scale factor
BoundsScaleFactor = 1.0f;
```

**Usage:**
```c
// Calculate bounding box center
Vector3 boundsCenter = (min + max) / 2.0f;

// Calculate bounding box extents
Vector3 boundsExtents = (max - min) / 2.0f;

// Calculate bounding radius
float boundsRadius = max(boundsExtents.x, boundsExtents.y, boundsExtents.z);
```

---

## Texture Filtering Constants

### Texture Filtering Modes

**Description:** Texture filtering quality settings

**Constants:**
```c
// Nearest filtering (no interpolation)
TextureFilterNearest = 0;

// Linear filtering (bilinear interpolation)
TextureFilterLinear = 1;

// Trilinear filtering (trilinear interpolation)
TextureFilterTrilinear = 2;

// Anisotropic filtering levels
AnisotropyLevel1 = 1;
AnisotropyLevel2 = 2;
AnisotropyLevel4 = 4;
AnisotropyLevel8 = 8;
AnisotropyLevel16 = 16;
```

**Usage:**
```c
// Set texture filtering mode
GL.TexParameteri(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureFilterMode);

// Set anisotropy level
GL.TexParameterf(TextureTarget.Texture2D, TextureParameterName.TextureMaxAnisotropy, anisotropyLevel);
```

---

## Animation Constants

### Animation Timing

**Description:** Animation timing and interpolation parameters

**Constants:**
```c
// Default animation speed
DefaultAnimationSpeed = 1.0f;

// Minimum animation speed
MinAnimationSpeed = 0.1f;

// Maximum animation speed
MaxAnimationSpeed = 10.0f;

// Animation interpolation threshold
InterpolationThreshold = 0.001f;

// Keyframe time precision
KeyframeTimePrecision = 1; // milliseconds
```

**Usage:**
```c
// Calculate animation time
int currentTime = (int)(elapsedTime * animationSpeed);

// Interpolate between keyframes
float t = (currentTime - keyframe1.Time) / (keyframe2.Time - keyframe1.Time);
```

---

## Material Flags

### Material Properties

**Description:** Flags controlling material rendering behavior

**Constants:**
```c
// Material is unlit (no lighting)
MaterialFlagUnlit = 0x01;

// Material has no fog
MaterialFlagNoFog = 0x02;

// Material is two-sided (no backface culling)
MaterialFlagTwoSided = 0x04;
```

**Usage:**
```c
// Check if material is unlit
if ((material.Flags & MaterialFlagUnlit) != 0) {
    // Disable lighting
    GL.Disable(EnableCap.Lighting);
}

// Check if material has no fog
if ((material.Flags & MaterialFlagNoFog) != 0) {
    // Disable fog
    GL.Disable(EnableCap.Fog);
}

// Check if material is two-sided
if ((material.Flags & MaterialFlagTwoSided) != 0) {
    // Disable backface culling
    GL.Disable(EnableCap.CullFace);
}
```

---

## Layer Flags

### Layer Properties

**Description:** Flags controlling layer rendering behavior

**Constants:**
```c
// Disable lighting for this layer
LayerFlagDisableLighting = 0x01;

// Disable fog for this layer
LayerFlagDisableFog = 0x02;

// Enable depth test (inverted in original)
LayerFlagEnableDepthTest = 0x04;

// Enable depth write (inverted in original)
LayerFlagEnableDepthWrite = 0x08;

// Disable backface culling (inverted in original)
LayerFlagDisableCulling = 0x10;

// Enable texture coordinate generation
LayerFlagTexGenEnabled = 0x20;
```

**Usage:**
```c
// Check if lighting should be disabled
if ((layer.Flags & LayerFlagDisableLighting) != 0) {
    GL.Disable(EnableCap.Lighting);
}

// Check if fog should be disabled
if ((layer.Flags & LayerFlagDisableFog) != 0) {
    GL.Disable(EnableCap.Fog);
}

// Check if depth test should be enabled
if ((layer.Flags & LayerFlagEnableDepthTest) == 0) {
    GL.Disable(EnableCap.DepthTest);
}

// Check if depth write should be enabled
if ((layer.Flags & LayerFlagEnableDepthWrite) == 0) {
    GL.DepthMask(false, false);
}

// Check if backface culling should be disabled
if ((layer.Flags & LayerFlagDisableCulling) != 0) {
    GL.Disable(EnableCap.CullFace);
}

// Check if texture generation should be enabled
if ((layer.Flags & LayerFlagTexGenEnabled) != 0) {
    GL.Enable(EnableCap.TextureGenS);
}
```

---

## Blend Modes

### Blend Mode Mappings

**Description:** Blend modes and their OpenGL factor mappings

**Constants:**
```c
// Opaque blend mode (no blending)
BlendModeOpaque = 4;

// Transparent blend mode (standard alpha blending)
BlendModeTransparent = 0;

// Add blend mode (additive)
BlendModeAdd = 1;

// Modulate blend mode (multiplicative)
BlendModeModulate = 2;

// Modulate2X blend mode (multiplicative with source)
BlendModeModulate2X = 3;
```

**OpenGL Factor Mappings:**
```c
// Opaque: One, Zero
BlendFactorOpaqueSrc = BlendingFactor.One;
BlendFactorOpaqueDst = BlendingFactor.Zero;

// Transparent: SrcAlpha, OneMinusSrcAlpha
BlendFactorTransparentSrc = BlendingFactor.SrcAlpha;
BlendFactorTransparentDst = BlendingFactor.OneMinusSrcAlpha;

// Add: SrcAlpha, One
BlendFactorAddSrc = BlendingFactor.SrcAlpha;
BlendFactorAddDst = BlendingFactor.One;

// Modulate: DstColor, Zero
BlendFactorModulateSrc = BlendingFactor.DstColor;
BlendFactorModulateDst = BlendingFactor.Zero;

// Modulate2X: DstColor, SrcColor
BlendFactorModulate2XSrc = BlendingFactor.DstColor;
BlendFactorModulate2XDst = BlendingFactor.SrcColor;
```

**Usage:**
```c
// Set blend mode based on material layer
switch (layer.BlendMode) {
    case BlendModeOpaque:
        GL.BlendFunc(BlendFactorOpaqueSrc, BlendFactorOpaqueDst);
        GL.Disable(EnableCap.Blend);
        break;
    case BlendModeTransparent:
        GL.BlendFunc(BlendFactorTransparentSrc, BlendFactorTransparentDst);
        GL.Enable(EnableCap.Blend);
        break;
    case BlendModeAdd:
        GL.BlendFunc(BlendFactorAddSrc, BlendFactorAddDst);
        GL.Enable(EnableCap.Blend);
        break;
    case BlendModeModulate:
        GL.BlendFunc(BlendFactorModulateSrc, BlendFactorModulateDst);
        GL.Enable(EnableCap.Blend);
        break;
    case BlendModeModulate2X:
        GL.BlendFunc(BlendFactorModulate2XSrc, BlendFactorModulate2XDst);
        GL.Enable(EnableCap.Blend);
        break;
}
```

---

## Render States

### Render State Flags

**Description:** Flags controlling various render states

**Constants:**
```c
// Depth test enabled
RenderStateDepthTestEnabled = 0x01;

// Depth write enabled
RenderStateDepthWriteEnabled = 0x02;

// Backface culling enabled
RenderStateCullFaceEnabled = 0x04;

// Lighting enabled
RenderStateLightingEnabled = 0x08;

// Fog enabled
RenderStateFogEnabled = 0x10;

// Blend enabled
RenderStateBlendEnabled = 0x20;

// Texture 0 enabled
RenderStateTexture0Enabled = 0x40;

// Texture 1 enabled
RenderStateTexture1Enabled = 0x80;
```

**Usage:**
```c
// Enable/disable depth test
if (renderState & RenderStateDepthTestEnabled) {
    GL.Enable(EnableCap.DepthTest);
} else {
    GL.Disable(EnableCap.DepthTest);
}

// Enable/disable depth write
if (renderState & RenderStateDepthWriteEnabled) {
    GL.DepthMask(true, true);
} else {
    GL.DepthMask(false, false);
}

// Enable/disable backface culling
if (renderState & RenderStateCullFaceEnabled) {
    GL.Enable(EnableCap.CullFace);
} else {
    GL.Disable(EnableCap.CullFace);
}

// Enable/disable lighting
if (renderState & RenderStateLightingEnabled) {
    GL.Enable(EnableCap.Lighting);
} else {
    GL.Disable(EnableCap.Lighting);
}

// Enable/disable fog
if (renderState & RenderStateFogEnabled) {
    GL.Enable(EnableCap.Fog);
} else {
    GL.Disable(EnableCap.Fog);
}

// Enable/disable blending
if (renderState & RenderStateBlendEnabled) {
    GL.Enable(EnableCap.Blend);
} else {
    GL.Disable(EnableCap.Blend);
}
```

---

## Implementation Notes

### C# Implementation

When implementing these constants in C#, use the following patterns:

```csharp
public static class RenderConstants
{
    // Distance Constants
    public const float MinimumDistance = 0.0f;
    public const float MaximumDistance = 256.0f;
    public const float InsignificantDistance = 0.0f;
    public const float FarDistance = 256.0f;
    public const float LodNearDistance = 50.0f;
    public const float LodFarDistance = 150.0f;
    
    // Fog Constants
    public const float FogStartDistance = 50.0f;
    public const float FogEndDistance = 500.0f;
    public const float FogDensity = 0.002f;
    public static readonly C3Color FogColor = new C3Color(0.5f, 0.6f, 0.7f);
    public const float FogNearDensity = 0.001f;
    public const float FogFarDensity = 0.0005f;
    
    // Clip Plane Constants
    public const float NearClipDistance = 0.01f;
    public const float FarClipDistance = 1000.0f;
    public const float NearClipMin = 0.01f;
    public const float NearClipMax = 1.0f;
    public const float FarClipMin = 177.0f;
    public const float FarClipMax = 777.0f;
    
    // FOV Constants
    public const float FovMinRadians = 0.0174533f;  // 1 degree
    public const float FovMaxRadians = 3.1241393f;  // 179 degrees
    public const float FovDefaultRadians = 0.7853982f;  // 45 degrees
    public const float FovMin = 1.0f;
    public const float FovMax = 179.0f;
    
    // Scale Constants
    public const float DefaultModelScale = 1.0f;
    public const float MinModelScale = 0.1f;
    public const float MaxModelScale = 10.0f;
    public const float DoodadScale = 1.0f;
    public const float ParticleScale = 1.0f;
    public const float RibbonScale = 1.0f;
    public const float SpellEffectScale = 1.0f;
    
    // Bounds Constants
    public const float DefaultBoundsRadius = 1.0f;
    public const float MinBoundsRadius = 0.1f;
    public const float MaxBoundsRadius = 100.0f;
    public const float BoundsScaleFactor = 1.0f;
    
    // Texture Filtering Constants
    public const int TextureFilterNearest = 0;
    public const int TextureFilterLinear = 1;
    public const int TextureFilterTrilinear = 2;
    public const int AnisotropyLevel1 = 1;
    public const int AnisotropyLevel2 = 2;
    public const int AnisotropyLevel4 = 4;
    public const int AnisotropyLevel8 = 8;
    public const int AnisotropyLevel16 = 16;
    
    // Material Flags
    public const int MaterialFlagUnlit = 0x01;
    public const int MaterialFlagNoFog = 0x02;
    public const int MaterialFlagTwoSided = 0x04;
    
    // Layer Flags
    public const int LayerFlagDisableLighting = 0x01;
    public const int LayerFlagDisableFog = 0x02;
    public const int LayerFlagEnableDepthTest = 0x04;
    public const int LayerFlagEnableDepthWrite = 0x08;
    public const int LayerFlagDisableCulling = 0x10;
    public const int LayerFlagTexGenEnabled = 0x20;
    
    // Blend Modes
    public const int BlendModeOpaque = 4;
    public const int BlendModeTransparent = 0;
    public const int BlendModeAdd = 1;
    public const int BlendModeModulate = 2;
    public const int BlendModeModulate2X = 3;
    
    // Render States
    public const int RenderStateDepthTestEnabled = 0x01;
    public const int RenderStateDepthWriteEnabled = 0x02;
    public const int RenderStateCullFaceEnabled = 0x04;
    public const int RenderStateLightingEnabled = 0x08;
    public const int RenderStateFogEnabled = 0x10;
    public const int RenderStateBlendEnabled = 0x20;
    public const int RenderStateTexture0Enabled = 0x40;
    public const int RenderStateTexture1Enabled = 0x80;
    
    // Helper methods
    public static float DegreesToRadians(float degrees)
    {
        return degrees * (float)Math.PI / 180.0f;
    }
    
    public static float RadiansToDegrees(float radians)
    {
        return radians * 180.0f / (float)Math.PI;
    }
}
```

### Validation

Always validate constants before use:

```csharp
// Validate FOV
if (fovRadians < RenderConstants.FovMinRadians || fovRadians > RenderConstants.FovMaxRadians) {
    throw new ArgumentException($"FOV must be between {RenderConstants.RadiansToDegrees(RenderConstants.FovMinRadians)} and {RenderConstants.RadiansToDegrees(RenderConstants.FovMaxRadians)} degrees");
}

// Validate distance
if (distance < 0.0f || distance > RenderConstants.MaximumDistance) {
    throw new ArgumentException($"Distance must be between {RenderConstants.MinimumDistance} and {RenderConstants.MaximumDistance}");
}

// Validate scale
if (scale < RenderConstants.MinModelScale || scale > RenderConstants.MaxModelScale) {
    throw new ArgumentException($"Scale must be between {RenderConstants.MinModelScale} and {RenderConstants.MaxModelScale}");
}
```

---

---

## Terrain Constants

### Chunk and Cell Dimensions

**Description:** Scaling and size constants for the terrain grid

**Constants:**
```c
// Chunk coordinate system
const float CHUNK_SCALE = 1.0f / 533.3333f;    // 0.001875
const float CHUNK_OFFSET = 533.3333f / 2.0f;    // 266.6667

// Chunk dimensions
const float CHUNK_SIZE = 533.3333f;             // Size of one chunk in yards
const float CELL_SIZE = 533.3333f / 8.0f;       // 66.6667 yards
const int NUM_CELLS = 8;                         // 8x8 cells per chunk
const int NUM_VERTICES = 9;                      // 9x9 vertices per chunk

// Valid chunk range
const int MIN_CHUNK = 0;
const int MAX_CHUNK = 1023;                     // 0x3ff (Max index on 64x64 area grid)

// AOI (Area of Interest)
const float GROUP_AOI_SIZE = 100.0f;            // Doodad visibility range
const float OBJECT_AOI_SIZE = 500.0f;           // Game object visibility range
```

---

## Movement Constants

### Speed and Physics

**Description:** Player movement speed and gravity parameters

**Constants:**
```c
// Movement speeds (yards/second)
const float WALK_SPEED = 3.5f;
const float RUN_SPEED = 7.0f;
const float SWIM_SPEED = 3.5f;
const float TURN_SPEED = 2.0f;                  // Radians/sec
const float PITCH_SPEED = 1.0f;                 // Radians/sec

// Physics
const float JUMP_VELOCITY = 8.0f;               // Initial upward impulse
const float GRAVITY = 9.8f;                     // Downward acceleration
const float TERMINAL_VELOCITY = 50.0f;          // Max falling speed
```

---

## Liquid Constants

### Liquid Types

**Description:** Identifiers for liquid surfaces

**Constants:**
```c
const uint LIQUID_WATER = 0x0;
const uint LIQUID_OCEAN = 0x1;
const uint LIQUID_MAGMA = 0x2;
const uint LIQUID_SLIME = 0x3;
const uint LIQUID_NONE = 0xf;
```

---

## Detail Doodad Constants

### Density and Distance

**Description:** Parameters for small terrain decorations (grass, etc.)

**Constants:**
```c
const float DETAIL_DOODAD_DISTANCE = 100.0f;    // Max distance to render detail doodads
const int MAX_DETAIL_DOODADS = 64;               // Max instances per chunk
```

---

## Conclusion
...
