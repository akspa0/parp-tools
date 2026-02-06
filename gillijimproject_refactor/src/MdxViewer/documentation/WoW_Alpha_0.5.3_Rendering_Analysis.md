# WoW Alpha 0.5.3 Rendering Analysis

## Executive Summary

This document provides a comprehensive analysis of the rendering pipeline in the WoW Alpha 0.5.3 client based on Ghidra decompilation. It covers geoset rendering, material/texture application, animation effects, blending modes, and compares findings with the current C# MDX viewer implementation.

## Table of Contents

1. [Rendering Pipeline Overview](#rendering-pipeline-overview)
2. [Geoset Rendering Process](#geoset-rendering-process)
3. [Material and Texture Application](#material-and-texture-application)
4. [Animation Rendering Effects](#animation-rendering-effects)
5. [Blending Modes and Alpha Handling](#blending-modes-and-alpha-handling)
6. [Opaque vs Transparent Rendering](#opaque-vs-transparent-rendering)
7. [Comparison with C# Implementation](#comparison-with-c-implementation)
8. [Recommendations](#recommendations)
9. [World Rendering Analysis](#world-rendering-analysis)
10. [References to Extended Analysis](#references-to-extended-analysis)

---

## Rendering Pipeline Overview

### Main Rendering Functions

| Function | Address | Purpose |
|-----------|----------|---------|
| `IModelRenderSceneOpaque` | 0x0042eea0 | Renders all opaque geosets and layers |
| `IModelRenderSceneTransparent` | 0x004316d0 | Renders all transparent objects (geosets, emitters, ribbons) |
| `GeosetComplexRender` | 0x004325d0 | Entry point for complex geoset rendering |
| `RenderGeoset` | 0x00431ad0 | Main geoset rendering function |
| `RenderGeosetLayers` | 0x00431b90 | Renders all material layers for a geoset |
| `RenderUniformUVMapLayers` | 0x00430510 | Renders layers with uniform UV mapping |

### Rendering Flow

```
IModelRenderSceneOpaque (Opaque Pass)
├── Sort opaque layers by priority
├── For each opaque layer:
│   ├── FillInRenderData (prepare render data)
│   ├── Set render states (lighting, fog, depth, culling, blend)
│   ├── Bind textures (up to 2 textures)
│   └── Draw primitives
└── Restore render state

IModelRenderSceneTransparent (Transparent Pass)
├── Sort transparent objects by distance
├── For each transparent object:
│   ├── Push render state
│   ├── FillInRenderData
│   ├── Set render states
│   ├── Bind textures
│   └── Draw primitives
│   └── Pop render state
└── Restore render state
```

---

## Geoset Rendering Process

### 1. Geoset Visibility Check

**Function:** `RenderGeosetCheckVis` @ 0x00432600

```c
if (((param_2[1] & 1) == 0) &&
   (*(char *)(*(int *)(param_3 + 0xbc) * 0x10 + 3 + *(int *)(param_1 + 4)) != '\0')) {
    RenderGeoset(param_1_00,param_2_00,param_3,param_4);
}
```

**Key Findings:**
- Geosets are only rendered if:
  1. Flag bit 0 is NOT set (param_2[1] & 1 == 0)
  2. Geoset color alpha is NOT zero (color.a != '\0')

**Implication:** Geosets with zero alpha are completely skipped, not rendered with alpha=0.

### 2. Geoset Preparation

**Function:** `RenderGeosetPrep` @ 0x00430030

Prepares vertex data for rendering, including:
- Vertex buffer setup
- Index buffer setup
- UV coordinate setup
- Normal setup

### 3. Geoset Layer Rendering

**Function:** `RenderGeosetLayers` @ 0x00431b90

```c
iVar2 = SingleUvMapping((int)CVar1);
if ((iVar2 == 0) && ((*(byte *)(param_1_00 + 0x1c) & 2) == 0)) {
    RenderGeosetMultiUvMapping(param_1_00,(int)param_2_00,CVar1,param_3);
    return;
}
RenderGeosetOneUvMapping(param_1_00,param_2_00,(int)CVar1,param_3);
```

**Key Findings:**
- Supports both single UV mapping and multi UV mapping
- Multi UV mapping is used when:
  1. SingleUvMapping returns 0 (multiple UV sets)
  2. Flag bit 1 is NOT set (param_1_00 + 0x1c & 2 == 0)

### 4. Single UV Mapping Rendering

**Function:** `RenderGeosetOneUvMapping` @ 0x00430340

**Critical Code:**
```c
// Calculate geoset color (multiplied by layer alpha)
color._0_3_ = (uint3)CONCAT11(
    (char)((uVar3 >> 0x10 & 0xff) * (uVar2 >> 0x10 & 0xff) + 0xff >> 8),
    (char)((uVar3 >> 8 & 0xff) * (uVar2 >> 8 & 0xff) + 0xff >> 8)
) << 8 | (uint3)((uVar3 & 0xff) * (uVar2 & 0xff) + 0xff >> 8) & 0xff;

color.a = (uchar)(((uVar2 >> 0x18) *
              (uint)*(byte *)(*(int *)(param_3 + 0x14) + 0x1c + param_4 * 0x20)) / 0xff);

RenderSingleUVMapPrep(param_1_00,(int)param_2_00,param_3,iVar1,param_4,param_5);
RenderUniformUVMapLayers(param_1_00,param_2_00,param_3,iVar1,&color,param_6);
```

**Key Findings:**
1. **Geoset Color is Multiplied by Layer Alpha:**
   - RGB: `geosetColor * layerColor / 255`
   - Alpha: `geosetAlpha * layerAlpha / 255`

2. **Color Calculation:**
   - Each RGB component is multiplied and divided by 255
   - Result is clamped to 0-255 range

3. **Alpha Calculation:**
   - Geoset alpha is multiplied by layer alpha
   - Result is stored as a byte (0-255)

---

## Material and Texture Application

### 1. Render State Setup

**Function:** `RenderUniformUVMapLayers` @ 0x00430510

**Render States Set:**

```c
// Material properties
GxRsSet(GxRs_MatDiffuse, *param_5);           // Diffuse color (geoset color)
GxRsSet(GxRs_MatEmissive, *(CImVector *)(param_3 + 0x20)); // Emissive color

// Texture generation modes
GxRsSet(GxRs_TexGen0, (*pbVar9 & 1) ? 6 : 0);  // 0=disabled, 6=enabled
GxRsSet(GxRs_TexGen1, (*pbVar9 & 1) ? 6 : 0);

// Lighting and fog
GxRsSet(GxRs_Lighting, ((*(byte *)(param_1_00 + 0x1c) & 4) == 0) && 
                         ((*(byte *)(iVar4 + 4) & 1) == 0) ? 1 : 0);
GxRsSet(GxRs_Fog, ((*(byte *)(param_1_00 + 0x1c) & 8) == 0) && 
                    ((*(byte *)(iVar4 + 4) & 2) == 0) ? 1 : 0);

// Depth testing and writing
GxRsSet(GxRs_DepthTest, ~*(uint *)(iVar4 + 4) >> 2 & 1);
GxRsSet(GxRs_DepthWrite, ~*(uint *)(iVar4 + 4) >> 3 & 1);

// Backface culling
GxRsSet(GxRs_Culling, ~*(uint *)(iVar4 + 4) >> 4 & 1);

// Blend mode
GxRsSet(GxRs_Blend, *(int *)(iVar4 + 8));
```

**Key Findings:**

1. **Lighting Control:**
   - Enabled by default
   - Can be disabled by:
     - Global flag bit 2 (param_1_00 + 0x1c & 4)
     - Layer flag bit 0 (iVar4 + 4 & 1)

2. **Fog Control:**
   - Enabled by default
   - Can be disabled by:
     - Global flag bit 3 (param_1_00 + 0x1c & 8)
     - Layer flag bit 1 (iVar4 + 4 & 2)

3. **Depth Testing:**
   - Controlled by layer flag bit 2 (inverted)
   - `~flags >> 2 & 1` means bit 2 = 0 enables depth test

4. **Depth Writing:**
   - Controlled by layer flag bit 3 (inverted)
   - `~flags >> 3 & 1` means bit 3 = 0 enables depth write

5. **Backface Culling:**
   - Controlled by layer flag bit 4 (inverted)
   - `~flags >> 4 & 1` means bit 4 = 0 enables culling

### 2. Texture Binding

**Code:**
```c
EVar10 = GxRs_Texture0;
do {
    if (piVar8[1] == -1) {
        GxRsSet(EVar10, 0);  // No texture
    }
    else if ((*(int *)(iVar4 + 8) == 0) && ((*(byte *)(param_1_00 + 0x1c) & 2) != 0)) {
        GxRsSet(EVar10, 0);  // Opaque mode with flag bit 1 set
    }
    else {
        pCVar6 = TextureGetGxTex(*(HTEXTURE__ **)(*(int *)(param_1_00 + 0x10) + piVar8[1] * 8), 1, param_6);
        GxRsSet(EVar10 + GxRs_MatSpecularExp, *piVar8);  // Texture combiner
        GxRsSet(EVar10, pCVar6);  // Bind texture
    }
    piVar8 = piVar8 + 2;
    EVar10 = EVar10 + GxRs_MatDiffuse;
} while (uVar1 < 2);
```

**Key Findings:**
1. **Supports up to 2 textures per layer** (Texture0 and Texture1)
2. **Texture ID -1 means no texture** (untextured rendering)
3. **Opaque mode with flag bit 1 set disables textures**
4. **Texture combiner is set** for each texture unit

---

## Animation Rendering Effects

### 1. Geoset Color Calculation

**Function:** `CalcGeosetColor` @ 0x0074adc0

```c
SetGeosetAlpha(param_1, (CKeyFrameTrackBase *)param_2, (int)param_3, (int)param_4);
SetGeosetColor(param_1, (int)param_2, &param_3->color, (undefined1 *)param_4);

bVar1 = (param_3->base).flags;
if ((param_4->animatedColor).a != '\0') {
    (param_3->base).flags = bVar1 | 1;  // Set animated flag
    return;
}
(param_3->base).flags = bVar1 & 0xfe;  // Clear animated flag
```

**Key Findings:**
1. **Both alpha and color are calculated from animation tracks**
2. **Animated flag is set if color alpha is non-zero**
3. **Flag bit 0 indicates the geoset has animated color**

### 2. Geoset Alpha Calculation

**Function:** `SetGeosetAlpha` @ 0x0074b1f0

**Critical Code:**
```c
if (param_2_00->m_numKeyFrames == 0) {
    return;  // No keyframes, use default
}

uVar5 = CKeyFrameTrackBase::SetAnimTime(param_2_00, (CBaseStatus *)(param_3 + 0x18), pCVar1, param_1_00);

if (uVar5 < 2) {
    // Static or single keyframe
    if ((*(byte *)(param_3 + 0x19) & 0x10) == 0) {
        return;  // Not animated
    }
    if (uVar5 != 0) {
        pCVar6 = CKeyFrameTrackBase::GetKeyFrame(param_2_00, pCVar1->currKey);
        visibility = (float)pCVar6[1].time;  // Get alpha value
    }
    else {
        visibility = 1.0f;  // Default alpha
    }
}
else {
    // Interpolated animation
    CKeyFrameTrack<float,float>::Interpolate((CKeyFrameTrack<float,float> *)param_2_00, pCVar1,
                                         pCVar3[uVar5].time.h - pCVar3[uVar5].time.l, &visibility);
}

// Clamp and convert to byte
fVar4 = visibility * *(float *)(param_4 + 0xc) * 255.0f;
*(char *)(param_4 + 3) = (char)((uint)(fVar4 + 128.0f) >> 7);
```

**Key Findings:**
1. **Alpha is multiplied by 255.0f** to convert from 0.0-1.0 to 0-255
2. **Alpha is clamped to 0-255 range**
3. **Animation flag bit 4 (0x10) controls whether alpha is animated**
4. **Default alpha is 1.0f (fully opaque)**

### 3. Geoset Color Calculation

**Function:** `SetGeosetColor` @ 0x0074ae50

**Critical Code:**
```c
if (*(int *)(param_2_00 + 0x20) != 0) {
    uVar4 = CKeyFrameTrackBase::SetAnimTime(this, (CBaseStatus *)(param_3 + 2), param_3, param_1_00);
    
    if (uVar4 < 2) {
        // Static or single keyframe
        if ((param_3[2].currKey & 0x1000) == 0) {
            return;  // Not animated
        }
        if (uVar4 == 0) {
            color.b = 0.0f;
            color.g = 0.0f;
            color.r = 0.0f;
        }
        else {
            pCVar5 = CKeyFrameTrackBase::GetKeyFrame(this, param_3->currKey);
            color.b = (float)pCVar5[1].time;
            color.g = (float)pCVar5[2].time;
            color.r = (float)pCVar5[3].time;
        }
    }
    else {
        // Interpolated animation
        CKeyFrameTrack<class_C3Color,class_C3Color>::Interpolate(
            (CKeyFrameTrack<class_C3Color,class_C3Color> *)this, param_3,
            (pCVar6->time).h - (pCVar6->time).l, &color);
    }
    
    // Clamp and convert each component to byte
    fVar3 = color.r * 255.0f;
    *param_4 = (char)((uint)(fVar3 + 128.0f) >> 7);
    
    fVar3 = color.g * 255.0f;
    param_4[1] = (char)((uint)(fVar3 + 128.0f) >> 7);
    
    fVar3 = color.b * 255.0f;
    param_4[2] = (char)((uint)(fVar3 + 128.0f) >> 7);
}
```

**Key Findings:**
1. **Each RGB component is multiplied by 255.0f** to convert from 0.0-1.0 to 0-255
2. **Each component is clamped to 0-255 range**
3. **Animation flag bit 12 (0x1000) controls whether color is animated**
4. **Default color is black (0, 0, 0)**

---

## Blending Modes and Alpha Handling

### 1. Blend Mode Mapping

**Function:** `SetMaterialBlendMode` @ 0x00448cb0

```c
switch(param_1) {
    case 0:  // Transparent
        *param_2 = 2;  // SrcAlpha, OneMinusSrcAlpha
        param_2[1] = param_2[1] & 0xfffffffb;
        return;
    case 1:  // Add
        *param_2 = 3;  // SrcAlpha, One
        param_2[1] = param_2[1] & 0xfffffffb;
        return;
    case 2:  // Modulate
        *param_2 = 4;  // DstColor, Zero
        param_2[1] = param_2[1] & 0xfffffffb;
        return;
    case 3:  // Modulate2x
        *param_2 = 5;  // DstColor, SrcColor
        param_2[1] = param_2[1] & 0xfffffffb;
        return;
    case 4:  // Opaque
        *param_2 = 1;  // One, Zero
}
```

**Blend Mode Table:**

| MDX Blend Mode | GxBlend Mode | Src Factor | Dst Factor | Description |
|----------------|---------------|-------------|-------------|-------------|
| 0 (Transparent) | 2 | SrcAlpha | OneMinusSrcAlpha | Standard alpha blending |
| 1 (Add) | 3 | SrcAlpha | One | Additive blending |
| 2 (Modulate) | 4 | DstColor | Zero | Multiplicative blending |
| 3 (Modulate2x) | 5 | DstColor | SrcColor | Multiplicative with source |
| 4 (Opaque) | 1 | One | Zero | No blending (opaque) |

### 2. Multi-Layer Blending

**Code from RenderUniformUVMapLayers:**
```c
if ((layersDrawn == 0) || (!bVar3)) {
    bVar3 = false;
}
else {
    bVar3 = true;
    GxRsPush();  // Save render state
    GxRsSet(GxRs_FogColor, extraout_EDX_00);  // Save fog color
}

// Draw the layer
if ((((byte)param_2_00[0xc0] & 0x10) == 0) || (DAT_00cc4348 == 0)) {
    GxPrimDrawElements();  // Normal draw
}
else {
    Project2d(param_2_00, param_5);  // 2D projection
}

if ((*(byte *)(param_1_00 + 0x1c) & 2) != 0) break;  // Single layer mode

if (bVar3) {
    GxRsPop();  // Restore render state
}

layersDrawn = layersDrawn + 1;
```

**Key Findings:**
1. **First layer is drawn without state push/pop**
2. **Subsequent layers push render state before drawing**
3. **Render state is restored after each subsequent layer**
4. **Fog color is saved/restored for multi-layer rendering**
5. **Flag bit 1 (param_1_00 + 0x1c & 2) enables single-layer mode**

---

## Opaque vs Transparent Rendering

### 1. Opaque Rendering

**Function:** `IModelRenderSceneOpaque` @ 0x0042eea0

**Characteristics:**
- Layers are sorted by priority (not by distance)
- No depth sorting needed (opaque objects can be drawn in any order)
- Optimized for batch rendering
- Uses `RenderGeosetSingleLayer` for single-layer geosets
- Uses `RenderSortedGeoset` for multi-layer geosets

**Render States for Opaque:**
- Depth test: Enabled (usually)
- Depth write: Enabled (usually)
- Blend mode: Opaque (One, Zero)
- Backface culling: Enabled (usually)

### 2. Transparent Rendering

**Function:** `IModelRenderSceneTransparent` @ 0x004316d0

**Characteristics:**
- Objects are sorted by distance (back-to-front)
- Uses priority queue for sorting
- Supports multiple object types:
  - SORTOBJ_GEOSET: Geosets
  - SORTOBJ_EMITTER2: Particle emitters
  - SORTOBJ_RIBBON: Ribbon emitters
  - SORTOBJ_CUSTOM_GEO: Custom geosets
  - SORTOBJ_CUSTOM_MODEL: Custom models

**Render States for Transparent:**
- Depth test: Enabled (usually)
- Depth write: Disabled (usually)
- Blend mode: Transparent/Add/Modulate
- Backface culling: Enabled (usually)

**Sorting Algorithm:**
```c
// Priority-based sorting
while (1 < uVar9) {
    uVar10 = uVar9 >> 1;
    bVar3 = CTransparentObject::HasHigherPriority(obj1, obj2);
    if (!bVar3) break;
    // Swap objects
    uVar9 = uVar10;
}
```

---

## Comparison with C# Implementation

### Current C# Implementation

**File:** [`Rendering/ModelRenderer.cs`](Rendering/ModelRenderer.cs)

**Current Approach:**
```csharp
for (int i = 0; i < _geosets.Count; i++)
{
    var gb = _geosets[i];
    if (!gb.Visible) continue;

    var geoset = _mdx.Geosets[gb.GeosetIndex];
    if (geoset.MaterialId >= 0 && geoset.MaterialId < _mdx.Materials.Count)
    {
        var material = _mdx.Materials[geoset.MaterialId];
        for (int l = 0; l < material.Layers.Count; l++)
        {
            var layer = material.Layers[l];
            
            // Set blend mode for layers after the first
            if (l > 0)
            {
                _gl.Enable(EnableCap.Blend);
                switch (layer.BlendMode)
                {
                    case MdlTexOp.Transparent:
                        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                        break;
                    case MdlTexOp.Add:
                    case MdlTexOp.AddAlpha:
                        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                        break;
                    case MdlTexOp.Modulate:
                    case MdlTexOp.Modulate2X:
                        _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
                        break;
                    default:
                        _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                        break;
                }
            }
            
            // Bind texture
            if (texId >= 0 && _textures.TryGetValue(texId, out uint glTex))
            {
                _gl.ActiveTexture(TextureUnit.Texture0);
                _gl.BindTexture(TextureTarget.Texture2D, glTex);
                _gl.Uniform1(_uHasTexture, 1);
            }
            
            // Set alpha
            float alpha = layer.StaticAlpha;
            _gl.Uniform4(_uColor, 1.0f, 1.0f, 1.0f, alpha);
            
            // Draw
            _gl.DrawElements(PrimitiveType.Triangles, gb.IndexCount, DrawElementsType.UnsignedShort, null);
            
            if (l > 0)
                _gl.Disable(EnableCap.Blend);
        }
    }
}
```

### Differences and Issues

| Aspect | WoW Alpha 0.5.3 | Current C# Implementation | Issue |
|---------|-------------------|------------------------|--------|
| **Geoset Color** | Multiplied with layer color | Not applied | Missing geoset color tinting |
| **Geoset Alpha** | Multiplied with layer alpha | Not applied | Missing geoset alpha modulation |
| **Animation Effects** | Applied via CalcGeosetColor | Not implemented | ATSQ animations not rendered |
| **Layer Sorting** | Opaque: by priority, Transparent: by distance | No sorting | Incorrect rendering order |
| **Depth Write** | Disabled for transparent layers | Always enabled | Transparent objects write to depth buffer |
| **Fog** | Controlled by flags | Not implemented | Missing fog effects |
| **Lighting** | Controlled by flags | Simple directional light | Missing proper lighting control |
| **Backface Culling** | Controlled by flags | Always enabled | Missing culling control |
| **Multi-Texture** | Up to 2 textures per layer | Only 1 texture | Missing multi-texture support |
| **Texture Combiner** | Set per texture unit | Not implemented | Missing texture combining |
| **Render State Push/Pop** | Used for multi-layer rendering | Not used | Incorrect state management |

### Critical Missing Features

1. **ATSQ Animation Support:**
   - Geoset alpha animations not evaluated
   - Geoset color animations not evaluated
   - Animation time not tracked

2. **Geoset Color/Alpha Application:**
   - Geoset color from ATSQ not multiplied with layer color
   - Geoset alpha from ATSQ not multiplied with layer alpha

3. **Proper Layer Rendering:**
   - No render state push/pop for multi-layer rendering
   - No fog color save/restore

4. **Depth Buffer Management:**
   - Transparent layers should disable depth write
   - Opaque layers should enable depth write

5. **Sorting:**
   - Opaque layers should be sorted by priority
   - Transparent objects should be sorted by distance

---

## Recommendations

### 1. Implement ATSQ Animation Support

**Add to MdxRenderer:**
```csharp
private Dictionary<int, GeosetAnimation> _geosetAnimations = new();
private int _animationTime = 0;

public void UpdateAnimation(int deltaTimeMs)
{
    _animationTime += deltaTimeMs;
    
    // Update geoset colors and alphas from animations
    foreach (var kvp in _geosetAnimations)
    {
        int geosetId = kvp.Key;
        var anim = kvp.Value;
        
        float alpha = anim.EvaluateAlpha(_animationTime);
        C3Color color = anim.EvaluateColor(_animationTime);
        
        // Store for rendering
        _geosetAnimatedColors[geosetId] = color;
        _geosetAnimatedAlphas[geosetId] = alpha;
    }
}
```

### 2. Apply Geoset Color/Alpha During Rendering

**Modify RenderWithTransform:**
```csharp
for (int i = 0; i < _geosets.Count; i++)
{
    var gb = _geosets[i];
    var geoset = _mdx.Geosets[gb.GeosetIndex];
    
    // Get animated geoset color and alpha
    C3Color geoColor = _geosetAnimatedColors.GetValueOrDefault(gb.GeosetIndex, new C3Color(1.0f, 1.0f, 1.0f));
    float geoAlpha = _geosetAnimatedAlphas.GetValueOrDefault(gb.GeosetIndex, 1.0f);
    
    // Multiply with layer color and alpha
    for (int l = 0; l < material.Layers.Count; l++)
    {
        var layer = material.Layers[l];
        
        // Calculate final color: geosetColor * layerColor / 255
        float r = geoColor.R * layer.Color.R / 255.0f;
        float g = geoColor.G * layer.Color.G / 255.0f;
        float b = geoColor.B * layer.Color.B / 255.0f;
        
        // Calculate final alpha: geosetAlpha * layerAlpha
        float alpha = geoAlpha * layer.StaticAlpha;
        
        _gl.Uniform4(_uColor, r, g, b, alpha);
        
        // Draw...
    }
}
```

### 3. Implement Proper Depth Buffer Management

**Modify layer rendering:**
```csharp
for (int l = 0; l < material.Layers.Count; l++)
{
    var layer = material.Layers[l];
    
    // First layer (opaque)
    if (l == 0)
    {
        _gl.DepthMask(true, true);  // Enable depth write
        _gl.Disable(EnableCap.Blend);
    }
    // Subsequent layers (transparent)
    else
    {
        _gl.DepthMask(false, false);  // Disable depth write
        _gl.Enable(EnableCap.Blend);
        
        // Set blend mode
        switch (layer.BlendMode)
        {
            case MdlTexOp.Transparent:
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.OneMinusSrcAlpha);
                break;
            case MdlTexOp.Add:
            case MdlTexOp.AddAlpha:
                _gl.BlendFunc(BlendingFactor.SrcAlpha, BlendingFactor.One);
                break;
            case MdlTexOp.Modulate:
                _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.Zero);
                break;
            case MdlTexOp.Modulate2X:
                _gl.BlendFunc(BlendingFactor.DstColor, BlendingFactor.SrcColor);
                break;
        }
    }
    
    // Draw...
}
```

### 4. Implement Layer Sorting

**Add sorting for opaque layers:**
```csharp
// Sort opaque layers by priority
var sortedLayers = material.Layers
    .Select((layer, index) => new { Layer = layer, Index = index })
    .OrderBy(x => x.Layer.Priority)
    .ToList();

foreach (var item in sortedLayers)
{
    // Render layer...
}
```

**Add sorting for transparent objects:**
```csharp
// Sort transparent objects by distance (back-to-front)
var transparentObjects = _geosets
    .Where(g => IsTransparent(g))
    .OrderByDescending(g => CalculateDistance(g))
    .ToList();

foreach (var obj in transparentObjects)
{
    // Render object...
}
```

### 5. Implement Render State Push/Pop

**For multi-layer rendering:**
```csharp
for (int l = 0; l < material.Layers.Count; l++)
{
    if (l > 0)
    {
        // Push render state before drawing subsequent layers
        _gl.PushAttrib(AttribMask.ColorBufferBit | AttribMask.DepthBufferBit | AttribMask.EnableBit);
    }
    
    // Draw layer...
    
    if (l > 0)
    {
        // Pop render state after drawing
        _gl.PopAttrib();
    }
}
```

### 6. Add Lighting and Fog Control

**Add render state flags:**
```csharp
public class MaterialFlags
{
    public bool DisableLighting { get; set; }
    public bool DisableFog { get; set; }
    public bool DisableDepthTest { get; set; }
    public bool DisableDepthWrite { get; set; }
    public bool DisableCulling { get; set; }
}

// Apply during rendering
if (!material.Flags.DisableLighting)
    _gl.Enable(EnableCap.Lighting);
else
    _gl.Disable(EnableCap.Lighting);

if (!material.Flags.DisableFog)
    _gl.Enable(EnableCap.Fog);
else
    _gl.Disable(EnableCap.Fog);
```

---

---

## World Rendering Analysis

### 1. Rendering Command List

The WoW Alpha 0.5.3 engine exposed several console commands for real-time renderer control and debugging:

| Command | Purpose |
|---------|---------|
| `gxRestart` | Restarts the graphics engine |
| `gxWindow` | Toggles windowed/fullscreen mode |
| `renderWorld` | Toggles rendering of the world scene |
| `renderTerrain` | Toggles terrain chunk rendering |
| `renderModels` | Toggles MDX model rendering |
| `renderWMO` | Toggles WMO building rendering |
| `renderDoodads` | Toggles doodad rendering |
| `detailDoodads` | Toggles detail doodad (grass, etc.) |
| `farclip` | Sets the far clipping plane distance |
| `horizonalAOI` | Sets the horizontal area of interest |

### 2. Terrain Lighting System

**Analysis:**
The terrain lighting system in 0.5.3 uses a hybrid approach:
- **Vertex Lighting:** Each of the 145 vertices in a chunk has a pre-calculated RGB color.
- **Shadow Mapping:** A dedicated shadow map texture (baked) is applied over the terrain.
- **Blending Color:** The alpha channel of the vertex color is used for smoothing transitions between layers or for baked AO.

### 3. Chunk Pipeline Entry Points

Additional key entry points for the world rendering sub-systems:

- `CMapChunk::Render` @ 0x006a6d80
- `CWorldScene::PrepareRenderLiquid` @ 0x0066a590
- `CMapChunk::CreateDetailDoodads` @ 0x006a6cf0
- `CWorldScene::UpdateFrustum` @ 0x0066a460

---

## References to Extended Analysis

The following documents contain deep-dive technical findings for specific world rendering sub-systems:

- **[01_Terrain_System.md](World_Rendering_Analysis/01_Terrain_System.md)**: Hierarchical data structures and chunk loading.
- **[02_Collision_Detection.md](World_Rendering_Analysis/02_Collision_Detection.md)**: Ray-casting and subchunk traversal.
- **[03_Player_Movement.md](World_Rendering_Analysis/03_Player_Movement.md)**: Input processing and physics constants.
- **[05_Liquid_Rendering.md](World_Rendering_Analysis/05_Liquid_Rendering.md)**: Water, magma, and slime pass details.
- **[06_Detail_Doodads.md](World_Rendering_Analysis/06_Detail_Doodads.md)**: Decorative mesh placement and rendering.
- **[07_Frustum_Culling.md](World_Rendering_Analysis/07_Frustum_Culling.md)**: Plane-based visibility tests.
- **[09_Texture_Layering.md](World_Rendering_Analysis/09_Texture_Layering.md)**: Multi-layer state management logic.
- **[10_Chunk_Rendering.md](World_Rendering_Analysis/10_Chunk_Rendering.md)**: Sequential pass breakdown.

---

## Conclusion

The WoW Alpha 0.5.3 client uses a sophisticated rendering pipeline...
