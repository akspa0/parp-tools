# MDX Rendering Quality Fix Plan

## Goal
Fix MDX doodad rendering issues visible in Stormwind/Ironforge WMO scenes. Trees, fountains, lanterns, and other doodads have incorrect transparency, missing alpha cutouts, and visual artifacts.

## Reference Implementations Studied
- **Noggit-Red**: `lib/noggit-red/src/noggit/rendering/ModelRender.cpp` — `prepareDraw()` method
- **Barncastle mdx-viewer**: `lib/wow-mdx-viewer/src/renderer/model/modelRenderer.ts` — `setLayerProps()` method

Both implementations agree on the same 7 blend modes with specific GL states.

## File to Edit
`gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs`

---

## Phase 1: Fix Alpha Discard Thresholds (HIGH IMPACT, SMALL CHANGE)

### Problem
Current code uses `uAlphaTest` as a boolean (0 or 1). The fragment shader does:
```glsl
float outAlpha = (uAlphaTest == 0) ? 1.0 : texColor.a;
```
This is wrong. Different blend modes need different alpha discard thresholds.

### Fix — Fragment Shader
Replace the current alpha logic with a proper discard threshold:
```glsl
uniform float uDiscardAlpha;  // rename from uAlphaTest

void main() {
    // ... existing lighting calc ...
    vec4 texColor = ...;
    
    // Always use texture alpha (don't conditionally force 1.0)
    vec4 finalColor = vec4(texColor.rgb * lighting, texColor.a) * uColor;
    
    // Discard fragments below threshold
    if (finalColor.a < uDiscardAlpha)
        discard;
    
    FragColor = finalColor;
}
```

### Fix — Per-BlendMode GL State (in RenderWithTransform)
Update the blend mode switch to set proper thresholds:

| BlendMode | Name | GL Blend | DepthWrite | uDiscardAlpha |
|-----------|------|----------|------------|---------------|
| 0 (Load/Opaque) | Opaque | Disabled | ON | 0.0 (no discard) |
| 1 (Transparent) | Alpha_Key | SRC_ALPHA, 1-SRC_ALPHA | **ON** | **0.75** |
| 2 (Blend) | Alpha | SRC_ALPHA, 1-SRC_ALPHA | OFF | 1/255 ≈ 0.004 |
| 3 (Add) | Additive | ONE, ONE | OFF | 1/255 |
| 4 (AddAlpha) | AddAlpha | SRC_ALPHA, ONE | OFF | 1/255 |
| 5 (Modulate) | Modulate | DST_COLOR, ZERO | OFF | 1/255 |
| 6 (Modulate2X) | Modulate2x | DST_COLOR, SRC_COLOR | OFF | 1/255 |

**Key insight**: BlendMode 1 (Alpha_Key/Transparent) is NOT true transparency — it's a hard cutout. It keeps depth writes ON and uses a high discard threshold (0.75). This is critical for things like tree leaves and fences.

### Specific Code Changes

1. **Rename uniform**: `_uAlphaTest` → `_uDiscardAlpha` (or keep name, change semantics to float threshold)
2. **Fragment shader**: Remove `(uAlphaTest == 0) ? 1.0 : texColor.a` conditional — always use `texColor.a`
3. **Opaque pass**: Set `uDiscardAlpha = 0.0` (no discard, but always pass through texture alpha)
4. **Transparent pass**: Set per-mode thresholds:
   - BlendMode 1: `uDiscardAlpha = 0.75f`, `DepthMask(true)`, blend ON
   - BlendMode 2-6: `uDiscardAlpha = 0.004f`, `DepthMask(false)`, blend ON with mode-specific func

---

## Phase 2: Per-Geoset Color and Alpha (MEDIUM IMPACT)

### Problem
Current code only uses `layer.StaticAlpha` for the color uniform. Animated geoset alpha and color from `GeosetAnims` are not applied.

### What Barncastle Does
```typescript
// Per-geoset: find animated alpha and color
this.rendererData.geosetAlpha[i] = this.findAlpha(i);
this.rendererData.geosetColor[i] = this.findColor(i);

// Skip invisible geosets
if (this.rendererData.geosetAlpha[i] < 1e-6) continue;

// Apply to shader
this.gl.uniform4f(colorUniform, color[0], color[1], color[2], alpha);
```

### What Noggit Does
```cpp
// mesh_color starts as (1,1,1, model.trans)
// If color_index exists: mesh_color = animated color + opacity
// If transparency_combo_index exists: mesh_color.w *= animated transparency
// If mesh_color.w <= 0: skip pass entirely
```

### Fix
- In `RenderWithTransform`, before rendering each geoset:
  1. Look up `GeosetAnims` for this geoset index
  2. Use `DefaultAlpha` (already parsed in MdxFile) as base alpha
  3. Multiply with `layer.StaticAlpha`
  4. Set `uColor = (1, 1, 1, finalAlpha)` — or use color keys if available
  5. Skip geoset entirely if alpha ≈ 0

### Data Already Available
`MdxFile.cs` already parses `GeosetAnimations` with `DefaultAlpha`, `AlphaKeys`, `ColorKeys`, and `GeosetId`. The `ModelRenderer.cs` already has `_mdx.GeosetAnimations` accessible.

---

## Phase 3: WMO Lighting (SEPARATE TASK)

Not part of this plan. See TODO item 15b:
- v14-16: grayscale lightmap from MOLM/MOLD chunks
- v17: vertex colors from MOCV chunk
- Requires changes to `WmoRenderer.cs` vertex shader (add vertex color attribute)

---

## Phase 4: MDX Particles (LARGER FEATURE, LATER)

Not part of this immediate plan. Requires:
- Separate particle shader program
- Billboard quad generation per particle
- Particle lifecycle management (emit → update → die)
- Per-particle blend modes (Blend, Additive, Modulate, AlphaKey)
- Rendered after all model geometry

Reference: `lib/wow-mdx-viewer/src/renderer/model/particles.ts` (820 lines)

---

## Testing

1. Load **Stormwind.wmo** — check trees, lanterns, fences (AlphaKey cutouts)
2. Load **Ironforge.wmo** — check grating transparency, lava visibility
3. Load standalone MDX models with known transparency (particle effects, etc.)
4. Verify no regression in opaque geometry rendering

## Build & Run
```
cd gillijimproject_refactor/src/MdxViewer
dotnet build --no-restore
dotnet run --no-restore
```
