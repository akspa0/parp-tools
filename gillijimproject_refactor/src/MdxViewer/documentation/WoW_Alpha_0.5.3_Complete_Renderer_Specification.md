# WoW Alpha 0.5.3 Complete Renderer Specification

## Executive Summary

This document provides a comprehensive specification of the WoW Alpha 0.5.3 rendering pipeline based on Ghidra decompilation. It covers the complete rendering system including world rendering, WMO rendering, MDX model rendering, doodads, shaders, and texture layering. This specification is designed to enable implementation of a modern, portable C# renderer that matches the behavior of the original WoW Alpha 0.5.3 client.

## Table of Contents

1. [Complete Rendering Pipeline](#complete-rendering-pipeline)
2. [World Rendering](#world-rendering)
3. [WMO Rendering](#wmo-rendering)
4. [MDX Model Rendering](#mdx-model-rendering)
5. [Doodad Rendering](#doodad-rendering)
6. [Shader System](#shader-system)
7. [Texture Layering and Combining](#texture-layering-and-combining)
8. [Modern C# Renderer Specification](#modern-c-renderer-specification)

---

## Complete Rendering Pipeline

### Main Rendering Function

**Function:** `OnWorldRender` @ 0x004f3440

**Rendering Order:**

```
1. Push render state
2. Update player name plates (early)
3. Set up world environment (lighting, fog, etc.)
4. Render world (WMOs, terrain, etc.)
5. Update/resort unit nameplates
6. Render unit footprints
7. Draw cursor shadow
8. Update unit effects
9. Render particle emitters
10. Render ribbon emitters
11. Render fade-out models
12. Render bow strings
13. Set texture LOD bias
14. Render opaque models (MDX models)
15. Render alpha world (transparent WMOs, etc.)
16. Render spell visuals
17. Render transparent models
18. Render collision info
19. Render day/night glares
20. Update player name plates (late)
21. Pop render state
```

**Code:**
```c
void __thiscall CGWorldFrame::OnWorldRender(CGWorldFrame *this)
{
    uint uVar2 = GxRsStackOffset();
    GxRsPush();
    
    // Early updates
    PlayerNameUpdateEarly();
    CWorld::SetEnvironment();
    
    // World rendering
    CWorld::Render();
    
    // Unit nameplates
    if ((this->m_flags & 3) != 0) {
        CGUnit_C::ResortAllUnitNameplates(this);
    }
    if ((this->m_flags & 1) != 0) {
        CGUnit_C::UpdateUnitNameplates(this);
    }
    
    // Footprints
    UnitFootprintRenderSplats(&cameraPos);
    
    // Cursor shadow
    if ((DAT_008467b8 < 2) && ((&DAT_00dddcfc)[DAT_008467b8] != 0)) {
        DrawCursorShadow();
    }
    
    // Effects
    UnitEffectUpdate(this->m_camera);
    ParticleSystemManager::RenderEmitters(this_00);
    RibbonManager::RenderEmitters(this_01);
    
    // Fade-out models
    RenderFadeOutModels(cameraPos.x, cameraPos.y, cameraPos.z, target.x, target.y, target.z);
    
    // Bow strings
    CGUnit_C_RenderBowStrings(&cameraPos);
    
    // Texture LOD bias
    GxRsSet(GxRs_TexLodBias0, extraout_EDX);
    
    // Opaque models
    ModelRenderSceneOpaque((CStatus *)0x0);
    
    // Alpha world
    CWorld::RenderAlpha();
    
    // Spell visuals
    SpellVisualsRender();
    
    // Transparent models
    ModelRenderSceneTransparent((CStatus *)0x0);
    
    // Collision info
    RenderCollisionInfo();
    
    // Day/night glares
    DayNightRenderGlares();
    
    // Late updates
    PlayerNameUpdateLate();
    
    // Restore state
    GxRsPop();
    
    // Cleanup
    Player_C_ClearGuildIDs();
    SmartScreenRectClearAllGrids();
}
```

### Render State Management

**Functions:**
- `GxRsPush()` - Push current render state to stack
- `GxRsPop()` - Pop render state from stack
- `GxRsStackOffset()` - Get current stack depth

**Usage Pattern:**
```c
// Save state before rendering
GxRsPush();

// ... render operations ...

// Restore state after rendering
GxRsPop();

// Validate stack balance
uint before = GxRsStackOffset();
// ... operations ...
uint after = GxRsStackOffset();
if (before != after) {
    // Error: stack imbalance
}
```

---

## World Rendering

### World Environment Setup

**Function:** `CWorld::SetEnvironment()`

**Sets up:**
- Global lighting parameters
- Fog parameters (color, density, start/end distances)
- Ambient color
- Sky parameters

### Frustum Culling

**Main Function:** `CWorldScene::UpdateFrustum` @ 0x0066a460

**Description:**
Updates the 6 frustum planes (Left, Right, Top, Bottom, Near, Far) from the current view-projection matrix for visibility testing.

**Algorithm:**
```c
void UpdateFrustum(C44Matrix* viewProjMatrix) {
    // Left plane: row4 + row1
    frustumPlanes[0].a = viewProjMatrix->m41 + viewProjMatrix->m11;
    // ... repeat for all 6 planes ...
    
    // Normalize planes for accurate distance testing
    for (int i = 0; i < 6; i++) {
        float length = sqrt(frustumPlanes[i].a * frustumPlanes[i].a + ...);
        frustumPlanes[i].a /= length;
        // ...
    }
}
```

**Visibility Tests:**
- `IsInFrustum` @ 0x0066a4a0 (Point)
- `IsChunkInFrustum` @ 0x0066a4e0
- `IsWMOInFrustum` @ 0x0066a520
- `IsDoodadInFrustum` @ 0x0066a560

### Map Chunk Rendering

**Main Entry Point:** `CMapChunk::Render` @ 0x006a6d80

**Pipeline Phase Logic:**
The chunk rendering is split into multiple passes to handle different transparency and layering requirements.

1. **Opaque Pass (`RenderOpaque` @ 0x006a6e00):** Renders the terrain base layers and opaque doodads.
2. **Transparent Pass (`RenderTransparent` @ 0x006a6e50):** Handles alpha-blended terrain layers and transparent doodads.
3. **Liquid Pass (`RenderLiquid` @ 0x006a6ea0):** Renders water, lava, or slime surfaces.
4. **Detail Doodad Pass (`RenderDetailDoodads` @ 0x006a6ef0):** Renders grass, flowers, and other small decorative meshes.

### Liquid Rendering

**Preparation Function:** `CWorldScene::PrepareRenderLiquid` @ 0x0066a590
**Status Query:** `CMap::QueryLiquidStatus` @ 0x00664e70

**Key Characteristics:**
- **Liquid Types:** Water (0x0), Ocean (0x1), Magma (0x2), Slime (0x3). No liquid is 0xF.
- **Particle Effects:**
  - Water/Ocean: Scale 0.027777778 (1/36).
  - Magma: Scale 0.11111111 (1/9).
- **Environment Updates:** Forces a full Day/Night cycle update if the liquid type changes (e.g., player submerging).

### World Rendering (Opaque)

**Function:** `CWorld::Render()`

**Renders:**
- Terrain chunks (Base layers)
- WMO (World Model Object) opaque parts
- Water (opaque pass)
- Static geometry

### World Rendering (Transparent)

**Function:** `CWorld::RenderAlpha()`

**Renders:**
- WMO transparent parts
- Water (alpha pass)
- Alpha-blended geometry

**Rendering Order:**
- Transparent objects are sorted by distance (back-to-front)
- Uses priority queue for sorting

---

## WMO Rendering

### WMO Structure

WMO (World Model Object) files contain:
- Group definitions
- Material definitions
- Portal definitions
- Light definitions
- Doodad definitions
- Vertex/index data

### WMO Rendering Pipeline

**Key Functions:**
- `CWorld::Render()` - Renders opaque WMO parts
- `CWorld::RenderAlpha()` - Renders transparent WMO parts

**Rendering Characteristics:**
1. **Batched Rendering:** Multiple WMO groups are batched together
2. **Material Sorting:** Groups are sorted by material for efficiency
3. **Portal Culling:** Portals are used for visibility culling
4. **Light Baking:** WMOs use pre-baked lightmaps

---

## MDX Model Rendering

### MDX File Structure

```
MDLX (magic)
├── VERS (version)
├── MODL (model info)
├── SEQS (sequences/animations)
├── GLBS (global sequences)
├── MTLS (materials)
├── TEXS (textures)
├── GEOS (geosets)
│   ├── VRTX (vertices)
│   ├── NRMS (normals)
│   ├── PTYP (primitive types)
│   ├── PCNT (primitive counts)
│   ├── PVTX (primitive vertices/indices)
│   ├── GNDX (group indices)
│   ├── MTGC (matrix group counts)
│   ├── MATS (matrix indices)
│   ├── UVAS (UV set count)
│   ├── UVBS (UV coordinates)
│   ├── BIDX (bone indices)
│   ├── BWGT (bone weights)
│   └── ATSQ (geoset animation tracks - Alpha 0.5.3 specific)
├── BONE (bones)
├── HELP (helpers)
├── PIVT (pivot points)
├── ATCH (attachments)
├── LITE (lights)
├── PREM/PRE2 (particle emitters)
├── RIBB (ribbon emitters)
├── EVTS (events)
├── CAMS (cameras)
├── CLID (collision)
├── HTST (hit test shapes)
├── TXAN (texture animations)
└── CORN (PopcornFX emitters)
```

### Opaque Model Rendering

**Function:** `IModelRenderSceneOpaque` @ 0x0042eea0

**Rendering Process:**
1. Sort opaque layers by priority
2. For each opaque layer:
   - Fill in render data
   - Set render states (lighting, fog, depth, culling, blend)
   - Bind textures (up to 2 textures)
   - Draw primitives
3. Restore render state

**Render States for Opaque:**
- Depth test: Enabled (usually)
- Depth write: Enabled (usually)
- Blend mode: Opaque (One, Zero)
- Backface culling: Enabled (usually)

### Transparent Model Rendering

**Function:** `IModelRenderSceneTransparent` @ 0x004316d0

**Rendering Process:**
1. Sort transparent objects by distance (back-to-front)
2. For each transparent object:
   - Push render state
   - Fill in render data
   - Set render states
   - Bind textures
   - Draw primitives
   - Pop render state

**Render States for Transparent:**
- Depth test: Enabled (usually)
- Depth write: Disabled (usually)
- Blend mode: Transparent/Add/Modulate
- Backface culling: Enabled (usually)

**Supported Object Types:**
- `SORTOBJ_GEOSET` - Geosets
- `SORTOBJ_EMITTER2` - Particle emitters
- `SORTOBJ_RIBBON` - Ribbon emitters
- `SORTOBJ_CUSTOM_GEO` - Custom geosets
- `SORTOBJ_CUSTOM_MODEL` - Custom models

### Geoset Rendering

**Function:** `RenderGeoset` @ 0x00431ad0

**Rendering Process:**
1. Prepare geoset data
2. Render geoset layers
3. Restore fog if needed

**Visibility Check:**
```c
if (((param_2[1] & 1) == 0) &&
   (*(char *)(*(int *)(param_3 + 0xbc) * 0x10 + 3 + *(int *)(param_1 + 4)) != '\0')) {
    RenderGeoset(param_1_00, param_2_00, param_3, param_4);
}
```

**Key Findings:**
- Geosets with zero alpha are completely skipped
- Geosets with flag bit 0 set are skipped

### Geoset Layer Rendering

**Function:** `RenderUniformUVMapLayers` @ 0x00430510

**Rendering Process:**
1. For each material layer:
   - Set material diffuse color (geoset color)
   - Set material emissive color
   - Set texture generation modes
   - Set lighting/fog/depth/culling/blend states
   - Bind textures (up to 2)
   - Draw primitives
   - Push/pop render state for multi-layer rendering

**Geoset Color Application:**
```c
// Calculate geoset color (multiplied by layer alpha)
color._0_3_ = (uint3)CONCAT11(
    (char)((uVar3 >> 0x10 & 0xff) * (uVar2 >> 0x10 & 0xff) + 0xff >> 8),
    (char)((uVar3 >> 8 & 0xff) * (uVar2 >> 8 & 0xff) + 0xff >> 8)
) << 8 | (uint3)((uVar3 & 0xff) * (uVar2 & 0xff) + 0xff >> 8) & 0xff;

color.a = (uchar)(((uVar2 >> 0x18) *
              (uint)*(byte *)(*(int *)(param_3 + 0x14) + 0x1c + param_4 * 0x20)) / 0xff);
```

**Key Findings:**
- Geoset color is multiplied by layer color: `geosetColor * layerColor / 255`
- Geoset alpha is multiplied by layer alpha: `geosetAlpha * layerAlpha / 255`
- Each RGB component is clamped to 0-255 range
- Alpha is clamped to 0-255 range

---

## Doodad Rendering

### Doodad Types

1. **Simple Doodads:** Low-detail models with simple rendering
2. **Detail Doodads:** Higher-detail models with full rendering

### Simple Doodad Rendering

**Function:** `CSimpleDoodad::RenderScene` @ 0x006a8020

**Rendering Process:**
1. Push render state
2. Set depth write to enabled
3. Set diffuse color
4. Select vertex shader (PassThru)
5. Push world transform
6. For each simple doodad:
   - For each geoset:
     - Get texture
     - Set texture blend mode
     - Bind texture
     - Set culling and blend modes
     - Lock buffer and render
     - Apply matrix transformations
7. Pop world transform
8. Pop render state

**Code:**
```c
GxRsPush();
GxRsSet(GxRs_DepthWrite, 1);
GxRsSet(GxRs_MatDiffuse, diffuseColor);
GxVertexShaderSelect(GxVS_PassThru);
GxXformPush(GxXform_World);

// For each doodad and geoset:
GxRsSet(GxRs_TexBlend0, 1);
GxRsSet(GxRs_Texture0, texture);
GxRsSet(GxRs_Culling, (props & 1) == 0);
GxRsSet(GxRs_Blend, (props & 2) != 0);
GxBufLock(gxBufDyn);
GxBufRender(&gxBatch);
GxBufUnlock();

GxXformPop(GxXform_World);
GxRsPop();
```

### Detail Doodad Rendering

**Main Entry Point:** `CWorldScene::RenderDoodads` @ 0x0066d8a0
**Creation Function:** `CMapChunk::CreateDetailDoodads` @ 0x006a6cf0

Detail doodads are small decorative meshes like grass or flowers that are automatically populated on the terrain surface.

#### Data Structures

```c
struct CDetailDoodadInst {
    TSLink<CDetailDoodadGeom> lameAssLink;  // Link to geometry
    CDetailDoodadGeom* geom[64];            // Up to 64 geometry instances
    CGxBuf* gxBuf[64];                      // Associated graphics buffers
};

struct CDetailDoodadGeom {
    C3Vector position;      // Position in world space
    C3Vector rotation;      // Rotation (Euler angles)
    C3Vector scale;         // Scale
    uint modelId;           // Model ID
    uint textureId;         // Texture ID
};
```

#### Creation Logic
1. **Distance Check:** Chunk must be within `detailDoodadDist` (default 100.0f).
2. **Density:** Controlled by `detailDoodadDensity`.
3. **Placement:** random X/Y within chunk, Z queried from terrain height.
4. **Randomization:** Rotation (0-360) and Scale (0.5-1.5) are randomized for variety.

#### Rendering Process
1. **Transform:** A unique world matrix is constructed for each doodad instance using its position, rotation, and scale.
2. **Batched Rendering:** Doodads are rendered using their pre-created graphics buffers.
3. **State:** Standard model rendering states apply, often with alpha testing for foliage.

#### Console Commands
- `detailDoodadAlpha` @ 0x00665ff0
- `detailDoodadTest` @ 0x00665fb0
- `showDetailDoodads` @ 0x00665770

---

## Shader System

### Shader Types

**Vertex Shaders:**
- `GxVS_PassThru` - Pass-through vertex shader (default)
- Other vertex shaders for special effects

**Pixel Shaders:**
- Default pixel shader for standard rendering
- Custom pixel shaders for special effects

### Shader Selection

**Function:** `GxVertexShaderSelect` @ 0x0058e490

**Code:**
```c
void GxVertexShaderSelect(GxVertexShader shader)
{
    // Select vertex shader for rendering
    // This is a wrapper around the Gx API
}
```

### Texture Shader Application

**Function:** `GetTextureShader` @ 0x0044c8b0

**Code:**
```c
undefined4 GetTextureShader(int textureId, uint flags)
{
    if ((textureId != -1) && ((flags & 0x100) == 0)) {
        return 1;  // Use shader
    }
    return 0;  // Don't use shader
}
```

**Key Findings:**
- Shader is used if texture ID is valid and flag 0x100 is not set
- Flag 0x100 disables shader usage

### Shader Parameters

**Functions:**
- `ISetShaderParamList` @ 0x0059ff60
- `ISetShaderParameters` @ 0x00595130

**Sets up:**
- Shader parameter lists
- Uniform values
- Texture bindings

---

## Texture Layering and Combining

### Multi-Texture Support

**Maximum Textures per Layer:** 2 (Texture0 and Texture1)
**Texture Coordinate Sets:** UV0 (Base/Alpha), UV1 (Detail)

**Texture Binding Logic:**
```c
GxRsSet(GxRs_Texture0, layer->textureId);     // Base texture
GxRsSet(GxRs_Texture1, layer->alphaMapId);   // Alpha blend mask (if applicable)
```

### Texture Blend Modes

Standard blend mode mapping used by the Alpha 0.5.3 engine:

| Enum | Name | Mapping | Formula |
|---|---|---|---|
| 0 | `BLEND_ALPHA` | `SrcAlpha`, `InvSrcAlpha` | `src*alpha + dst*(1-alpha)` |
| 1 | `BLEND_ADD` | `SrcAlpha`, `One` | `src*alpha + dst` |
| 2 | `BLEND_MODULATE` | `DstColor`, `Zero` | `src * dst` |
| 3 | `BLEND_MODULATE2X` | `DstColor`, `SrcColor` | `src * dst * 2` |
| 4 | `BLEND_OPAQUE` | `One`, `Zero` | `src` |

### Alpha Maps

Alpha maps are typically 64x64 grayscale textures used to blend terrain layers.

**Sampling Logic:**
```c
float SampleAlphaMap(AlphaMap* alphaMap, float u, float v) {
    int x = Clamp((int)(u * 64), 0, 63);
    int y = Clamp((int)(v * 64), 0, 63);
    return alphaMap->data[y * 64 + x] / 255.0f;
}
```

### Multi-Layer Rendering Pipeline

**Main Entry Point:** `CMapChunk::Render` @ 0x006a6d80

For terrain layers 1-3 (multi-layering), the engine uses a specific push/pop state management strategy to blend on top of the base layer.

**Logic:**
```c
for (int i = 0; i < numLayers; i++) {
    CMapChunkLayer* layer = &chunk->layers[i];
    
    // Set up textures and shaders
    GxTextureSet(0, layer->textureId);
    if (i > 0) GxTextureSet(1, layer->alphaMapId);
    GxBlendSet(layer->blendMode);
    
    RenderLayer(chunk, layer);
    
    // Push state for subsequent blending passes
    if (i > 0) GxRenderStatePush();
}

// Balance the stack
for (int i = 1; i < numLayers; i++) {
    GxRenderStatePop();
}
```

**Key Findings:**
- The base layer (index 0) is typically opaque.
- Layers 1+ use alpha maps to modulate their visibility over previous layers.
- Multi-layering requires strict stack discipline to avoid state leaks between chunks.

---

## Modern C# Renderer Specification

### Architecture Overview

```
Renderer (Main)
├── WorldRenderer
│   ├── TerrainRenderer
│   ├── WmoRenderer
│   └── WaterRenderer
├── ModelRenderer
│   ├── MdxRenderer
│   └── AnimationSystem
├── DoodadRenderer
│   ├── SimpleDoodadRenderer
│   └── DetailDoodadRenderer
├── ParticleRenderer
├── RibbonRenderer
├── SpellRenderer
└── PostProcessRenderer
```

### Core Components

#### 1. Render State Manager

**Purpose:** Manages OpenGL render state with push/pop functionality

**Interface:**
```csharp
public interface IRenderStateManager
{
    void Push();
    void Pop();
    int GetStackDepth();
    void ValidateStackBalance();
}

public class OpenGLRenderStateManager : IRenderStateManager
{
    private Stack<RenderState> _stateStack = new();
    
    public void Push()
    {
        _stateStack.Push(CaptureCurrentState());
    }
    
    public void Pop()
    {
        if (_stateStack.Count == 0)
            throw new InvalidOperationException("Cannot pop empty state stack");
        
        var state = _stateStack.Pop();
        RestoreState(state);
    }
    
    public int GetStackDepth() => _stateStack.Count;
    
    public void ValidateStackBalance()
    {
        if (_stateStack.Count != 0)
            throw new InvalidOperationException($"Render state stack imbalance: {_stateStack.Count}");
    }
    
    private RenderState CaptureCurrentState()
    {
        return new RenderState
        {
            BlendEnabled = IsBlendEnabled(),
            BlendFuncSrc = GetBlendFuncSrc(),
            BlendFuncDst = GetBlendFuncDst(),
            DepthTestEnabled = IsDepthTestEnabled(),
            DepthWriteEnabled = IsDepthWriteEnabled(),
            CullFaceEnabled = IsCullFaceEnabled(),
            FogEnabled = IsFogEnabled(),
            // ... other states
        };
    }
    
    private void RestoreState(RenderState state)
    {
        if (state.BlendEnabled)
            GL.Enable(EnableCap.Blend);
        else
            GL.Disable(EnableCap.Blend);
        
        GL.BlendFunc(state.BlendFuncSrc, state.BlendFuncDst);
        // ... restore other states
    }
}
```

#### 2. Material System

**Purpose:** Manages materials, textures, and blend modes

**Interface:**
```csharp
public class Material
{
    public List<MaterialLayer> Layers { get; set; } = new();
    public MaterialFlags Flags { get; set; }
}

public class MaterialLayer
{
    public int TextureId { get; set; }
    public int TextureId2 { get; set; } = -1;  // Second texture
    public BlendMode BlendMode { get; set; }
    public TextureCombiner Combiner { get; set; }
    public bool TexGenEnabled { get; set; }
    public float StaticAlpha { get; set; } = 1.0f;
    public C3Color Color { get; set; } = new C3Color(1.0f, 1.0f, 1.0f);
    public LayerFlags Flags { get; set; }
}

public enum BlendMode
{
    Opaque = 4,      // One, Zero
    Transparent = 0,  // SrcAlpha, OneMinusSrcAlpha
    Add = 1,         // SrcAlpha, One
    Modulate = 2,     // DstColor, Zero
    Modulate2X = 3    // DstColor, SrcColor
}

public enum TextureCombiner
{
    Modulate = 0,
    Add = 1,
    Replace = 2
}

[Flags]
public enum LayerFlags
{
    None = 0,
    DisableLighting = 0x01,
    DisableFog = 0x02,
    EnableDepthTest = 0x04,  // Inverted in original
    EnableDepthWrite = 0x08,  // Inverted in original
    EnableCulling = 0x10,  // Inverted in original
    TexGenEnabled = 0x20
}

[Flags]
public enum MaterialFlags
{
    None = 0,
    Unlit = 0x01,
    NoFog = 0x02,
    TwoSided = 0x04
}
```

**Blend Mode Mapping to OpenGL:**
```csharp
public static class BlendModeMapper
{
    public static (BlendingFactor Src, BlendingFactor Dst) ToOpenGL(BlendMode mode)
    {
        switch (mode)
        {
            case BlendMode.Opaque:
                return (BlendingFactor.One, BlendingFactor.Zero);
            case BlendMode.Transparent:
                return (BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
            case BlendMode.Add:
                return (BlendingFactor.SrcAlpha, BlendingFactor.One);
            case BlendMode.Modulate:
                return (BlendingFactor.DstColor, BlendingFactor.Zero);
            case BlendMode.Modulate2X:
                return (BlendingFactor.DstColor, BlendingFactor.SrcColor);
            default:
                throw new ArgumentException($"Unknown blend mode: {mode}");
        }
    }
}
```

#### 3. Animation System

**Purpose:** Manages MDX animations including ATSQ geoset animations

**Interface:**
```csharp
public class AnimationSystem
{
    private Dictionary<int, GeosetAnimation> _geosetAnimations = new();
    private int _animationTime = 0;
    private int _currentSequence = 0;
    
    public void LoadGeosetAnimations(List<GeosetAnimation> animations)
    {
        foreach (var anim in animations)
        {
            _geosetAnimations[anim.GeosetId] = anim;
        }
    }
    
    public void Update(int deltaTimeMs)
    {
        _animationTime += deltaTimeMs;
    }
    
    public void SetSequence(int sequenceId)
    {
        _currentSequence = sequenceId;
    }
    
    public (C3Color color, float alpha) EvaluateGeosetAnimation(int geosetId)
    {
        if (!_geosetAnimations.TryGetValue(geosetId, out var anim))
        {
            // Return default values if no animation
            return (new C3Color(1.0f, 1.0f, 1.0f), 1.0f);
        }
        
        float alpha = anim.EvaluateAlpha(_animationTime);
        C3Color color = anim.EvaluateColor(_animationTime);
        
        return (color, alpha);
    }
}
```

#### 4. MDX Renderer

**Purpose:** Renders MDX models with proper material and animation support

**Interface:**
```csharp
public class MdxRenderer : ISceneRenderer
{
    private readonly GL _gl;
    private readonly MdxFile _mdx;
    private readonly AnimationSystem _animationSystem;
    private readonly IRenderStateManager _stateManager;
    private readonly Dictionary<int, uint> _textures = new();
    private readonly List<GeosetBuffers> _geosets = new();
    
    public MdxRenderer(GL gl, MdxFile mdx, AnimationSystem animationSystem, 
                      IRenderStateManager stateManager)
    {
        _gl = gl;
        _mdx = mdx;
        _animationSystem = animationSystem;
        _stateManager = stateManager;
        
        InitShaders();
        InitBuffers();
        LoadTextures();
        _animationSystem.LoadGeosetAnimations(mdx.GeosetAnimations);
    }
    
    public unsafe void Render(Matrix4x4 modelMatrix, Matrix4x4 view, Matrix4x4 proj)
    {
        _stateManager.Push();
        
        // Set matrices
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&modelMatrix);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        
        // Render opaque geosets
        RenderOpaqueGeosets();
        
        // Render transparent geosets
        RenderTransparentGeosets();
        
        _stateManager.Pop();
        _stateManager.ValidateStackBalance();
    }
    
    private void RenderOpaqueGeosets()
    {
        // Sort opaque geosets by material priority
        var sortedGeosets = _geosets
            .Where(g => g.IsOpaque)
            .OrderBy(g => g.MaterialPriority)
            .ToList();
        
        foreach (var gb in sortedGeosets)
        {
            RenderGeoset(gb, isOpaque: true);
        }
    }
    
    private void RenderTransparentGeosets()
    {
        // Sort transparent geosets by distance (back-to-front)
        var sortedGeosets = _geosets
            .Where(g => !g.IsOpaque)
            .OrderByDescending(g => g.DistanceToCamera)
            .ToList();
        
        foreach (var gb in sortedGeosets)
        {
            RenderGeoset(gb, isOpaque: false);
        }
    }
    
    private void RenderGeoset(GeosetBuffers gb, bool isOpaque)
    {
        var geoset = _mdx.Geosets[gb.GeosetIndex];
        
        // Get animated geoset color and alpha
        var (geoColor, geoAlpha) = _animationSystem.EvaluateGeosetAnimation(gb.GeosetIndex);
        
        if (geoset.MaterialId >= 0 && geoset.MaterialId < _mdx.Materials.Count)
        {
            var material = _mdx.Materials[geoset.MaterialId];
            
            for (int l = 0; l < material.Layers.Count; l++)
            {
                var layer = material.Layers[l];
                
                // Push state for multi-layer rendering
                if (l > 0)
                    _stateManager.Push();
                
                // Apply geoset color and alpha
                float r = geoColor.R * layer.Color.R / 255.0f;
                float g = geoColor.G * layer.Color.G / 255.0f;
                float b = geoColor.B * layer.Color.B / 255.0f;
                float alpha = geoAlpha * layer.StaticAlpha;
                
                // Set render states
                ApplyRenderStates(layer, isOpaque);
                
                // Bind textures
                BindTextures(layer);
                
                // Set uniforms
                _gl.Uniform4(_uColor, r, g, b, alpha);
                
                // Draw
                _gl.BindVertexArray(gb.Vao);
                _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, 
                               DrawElementsType.UnsignedShort, null);
                
                // Pop state for multi-layer rendering
                if (l > 0)
                    _stateManager.Pop();
            }
        }
    }
    
    private void ApplyRenderStates(MaterialLayer layer, bool isOpaque)
    {
        // Depth test
        if ((layer.Flags & LayerFlags.EnableDepthTest) != 0)
            _gl.Enable(EnableCap.DepthTest);
        else
            _gl.Disable(EnableCap.DepthTest);
        
        // Depth write
        if (isOpaque && ((layer.Flags & LayerFlags.EnableDepthWrite) != 0))
            _gl.DepthMask(true, true);
        else
            _gl.DepthMask(false, false);
        
        // Backface culling
        if ((layer.Flags & LayerFlags.EnableCulling) != 0)
            _gl.Enable(EnableCap.CullFace);
        else
            _gl.Disable(EnableCap.CullFace);
        
        // Blend mode
        if (isOpaque)
        {
            _gl.Disable(EnableCap.Blend);
        }
        else
        {
            _gl.Enable(EnableCap.Blend);
            var (src, dst) = BlendModeMapper.ToOpenGL(layer.BlendMode);
            _gl.BlendFunc(src, dst);
        }
        
        // Lighting
        if ((layer.Flags & LayerFlags.DisableLighting) == 0)
            _gl.Enable(EnableCap.Lighting);
        else
            _gl.Disable(EnableCap.Lighting);
        
        // Fog
        if ((layer.Flags & LayerFlags.DisableFog) == 0)
            _gl.Enable(EnableCap.Fog);
        else
            _gl.Disable(EnableCap.Fog);
    }
    
    private void BindTextures(MaterialLayer layer)
    {
        // Bind first texture
        if (layer.TextureId >= 0 && _textures.TryGetValue(layer.TextureId, out uint tex0))
        {
            _gl.ActiveTexture(TextureUnit.Texture0);
            _gl.BindTexture(TextureTarget.Texture2D, tex0);
        }
        
        // Bind second texture
        if (layer.TextureId2 >= 0 && _textures.TryGetValue(layer.TextureId2, out uint tex1))
        {
            _gl.ActiveTexture(TextureUnit.Texture1);
            _gl.BindTexture(TextureTarget.Texture2D, tex1);
        }
    }
}
```

#### 5. Doodad Renderer

**Purpose:** Renders simple and detail doodads

**Interface:**
```csharp
public class DoodadRenderer
{
    private readonly GL _gl;
    private readonly IRenderStateManager _stateManager;
    private readonly List<SimpleDoodad> _simpleDoodads = new();
    private readonly List<DetailDoodad> _detailDoodads = new();
    
    public void RenderSimpleDoodads(Matrix4x4 view, Matrix4x4 proj)
    {
        _stateManager.Push();
        
        // Set up simple doodad rendering
        _gl.Enable(EnableCap.DepthTest);
        _gl.DepthMask(true, true);
        _gl.Disable(EnableCap.Blend);
        
        foreach (var doodad in _simpleDoodads)
        {
            RenderSimpleDoodad(doodad, view, proj);
        }
        
        _stateManager.Pop();
    }
    
    private void RenderSimpleDoodad(SimpleDoodad doodad, Matrix4x4 view, Matrix4x4 proj)
    {
        // Apply doodad transform
        var modelMatrix = CalculateDoodadTransform(doodad);
        
        _gl.UniformMatrix4(_uModel, 1, false, (float*)&modelMatrix);
        _gl.UniformMatrix4(_uView, 1, false, (float*)&view);
        _gl.UniformMatrix4(_uProj, 1, false, (float*)&proj);
        
        // Render each geoset
        foreach (var geoset in doodad.Geosets)
        {
            // Bind texture
            if (_textures.TryGetValue(geoset.TextureId, out uint tex))
            {
                _gl.ActiveTexture(TextureUnit.Texture0);
                _gl.BindTexture(TextureTarget.Texture2D, tex);
            }
            
            // Set blend mode
            if ((geoset.Flags & 0x02) != 0)
                _gl.Enable(EnableCap.Blend);
            else
                _gl.Disable(EnableCap.Blend);
            
            // Set culling
            if ((geoset.Flags & 0x01) == 0)
                _gl.Enable(EnableCap.CullFace);
            else
                _gl.Disable(EnableCap.CullFace);
            
            // Draw
            _gl.BindVertexArray(geoset.Vao);
            _gl.DrawElements(PrimitiveType.Triangles, geoset.IndexCount, 
                           DrawElementsType.UnsignedShort, null);
        }
    }
    
    public void RenderDetailDoodads(Matrix4x4 view, Matrix4x4 proj)
    {
        // Similar to simple doodads but with full MDX rendering
        // Uses MdxRenderer for each detail doodad
    }
}
```

#### 6. World Renderer

**Purpose:** Renders terrain, WMOs, and water

**Interface:**
```csharp
public class WorldRenderer
{
    private readonly GL _gl;
    private readonly IRenderStateManager _stateManager;
    private readonly WmoRenderer _wmoRenderer;
    private readonly TerrainRenderer _terrainRenderer;
    private readonly WaterRenderer _waterRenderer;
    
    public void RenderOpaque(Matrix4x4 view, Matrix4x4 proj)
    {
        _stateManager.Push();
        
        // Set up environment
        SetupEnvironment();
        
        // Render terrain
        _terrainRenderer.RenderOpaque(view, proj);
        
        // Render WMOs (opaque)
        _wmoRenderer.RenderOpaque(view, proj);
        
        // Render water (opaque)
        _waterRenderer.RenderOpaque(view, proj);
        
        _stateManager.Pop();
    }
    
    public void RenderTransparent(Matrix4x4 view, Matrix4x4 proj)
    {
        _stateManager.Push();
        
        // Render WMOs (transparent)
        _wmoRenderer.RenderTransparent(view, proj);
        
        // Render water (transparent)
        _waterRenderer.RenderTransparent(view, proj);
        
        _stateManager.Pop();
    }
    
    private void SetupEnvironment()
    {
        // Set global lighting
        _gl.Uniform3(_uAmbientColor, _ambientColor.R, _ambientColor.G, _ambientColor.B);
        
        // Set fog
        _gl.Enable(EnableCap.Fog);
        _gl.Fog(_fogMode, _fogDensity, _fogStart, _fogEnd);
        _gl.FogColor(_fogColor.R, _fogColor.G, _fogColor.B, _fogColor.A);
    }
}
```

### Shader Specification

#### Vertex Shader

```glsl
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;

uniform int uLightingEnabled;
uniform int uFogEnabled;
uniform float uFogDensity;
uniform float uFogStart;
uniform float uFogEnd;
uniform vec4 uFogColor;

out vec3 vFragPos;
out vec3 vNormal;
out vec2 vTexCoord;
out float vFogFactor;

void main()
{
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(uModel))) * aNormal;
    vTexCoord = aTexCoord;
    
    // Calculate fog factor
    if (uFogEnabled != 0)
    {
        float dist = length((uView * worldPos).xyz);
        vFogFactor = smoothstep(uFogStart, uFogEnd, dist);
    }
    else
    {
        vFogFactor = 0.0;
    }
    
    gl_Position = uProj * uView * worldPos;
}
```

#### Fragment Shader

```glsl
#version 330 core

in vec3 vFragPos;
in vec3 vNormal;
in vec2 vTexCoord;
in float vFogFactor;

uniform sampler2D uSampler0;
uniform sampler2D uSampler1;

uniform vec4 uColor;
uniform int uHasTexture0;
uniform int uHasTexture1;
uniform int uLightingEnabled;
uniform vec3 uAmbientColor;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform int uFogEnabled;
uniform vec4 uFogColor;

out vec4 FragColor;

void main()
{
    // Sample textures
    vec4 texColor;
    if (uHasTexture0 != 0 && uHasTexture1 != 0)
    {
        vec4 tex0 = texture(uSampler0, vTexCoord);
        vec4 tex1 = texture(uSampler1, vTexCoord);
        texColor = tex0 * tex1;  // Modulate combiner
    }
    else if (uHasTexture0 != 0)
    {
        texColor = texture(uSampler0, vTexCoord);
    }
    else if (uHasTexture1 != 0)
    {
        texColor = texture(uSampler1, vTexCoord);
    }
    else
    {
        texColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
    
    // Apply material color
    vec4 finalColor = texColor * uColor;
    
    // Apply lighting
    if (uLightingEnabled != 0)
    {
        vec3 normal = normalize(vNormal);
        float diff = max(dot(normal, uLightDir), 0.0);
        vec3 lighting = uAmbientColor + uLightColor * diff;
        finalColor.rgb *= lighting;
    }
    
    // Apply fog
    if (uFogEnabled != 0)
    {
        finalColor = mix(finalColor, uFogColor, vFogFactor);
    }
    
    FragColor = finalColor;
}
```

### Implementation Recommendations

1. **Use Modern Graphics API:**
   - OpenGL 3.3+ Core Profile
   - Vulkan (optional, for future)
   - DirectX 12 (optional, for Windows)

2. **Portable Libraries:**
   - Silk.NET for OpenGL/Vulkan
   - OpenTK for cross-platform OpenGL
   - Veldrid for Vulkan

3. **Architecture:**
   - Component-based design
   - Interface-driven for testability
   - Dependency injection for flexibility

4. **Performance Optimizations:**
   - Batch rendering where possible
   - Instancing for repeated objects
   - Frustum culling
   - LOD (Level of Detail) system
   - Async texture loading

5. **Features to Implement:**
   - ATSQ geoset animation support
   - Proper depth buffer management
   - Layer sorting (priority for opaque, distance for transparent)
   - Multi-texture support
   - Render state push/pop
   - Lighting and fog control
   - WMO rendering
   - Doodad rendering
   - Particle and ribbon systems

---

## Conclusion

This specification provides a complete blueprint for implementing a modern, portable C# renderer that matches the behavior of the WoW Alpha 0.5.3 client. The key insights from the Ghidra analysis are:

1. **Complete Rendering Pipeline:** Clear separation of opaque and transparent passes
2. **Geoset Animations:** ATSQ chunk provides alpha/color animations per geoset
3. **Material System:** Multi-layer rendering with proper state management
4. **Blend Modes:** Five distinct blend modes with specific OpenGL mappings
5. **Render State Management:** Push/pop system for multi-layer rendering
6. **World Rendering:** Separate terrain, WMO, and water rendering
7. **Doodad System:** Simple and detail doodad types with different rendering paths
8. **Shader System:** Vertex and pixel shaders with proper parameter binding

Implementing this specification will enable a modern, cross-platform MDX/WMO viewer that no longer relies on the ancient WoW Alpha 0.5.3 renderer.
