# Plan: WMO Object Overlay for Synthesized Minimaps

## Goal

Add `--include-wmos` to `WowViewer.Tool.Harvest synthetic-minimap` so placed WMO geometry is rendered with the same solar lighting and composited onto the terrain-only minimap output. This produces minimaps that include building/structure silhouettes and shadows, matching real authored minimaps more closely.

## Scope

- WMO only (no MDX/M2 per user direction)
- Above-ground WMOs only (compare placement Z against terrain height)
- Same orthographic top-down camera as the terrain minimap
- Same light direction (`--time-hours` solar direction)
- Composite: terrain image * (1 - wmo_mask) + wmo image * wmo_mask

## Existing Infrastructure

| Component | Purpose | Location |
|-----------|---------|----------|
| `ObjectCaptureRenderer` | Headless top-down single-object capture | `Core.Renderer/ObjectCapture/` |
| `WmoObjectRenderer` | Renders one WMO model with `ObjectCaptureShader` | `Core.Renderer/ObjectCapture/WmoObjectRenderer.cs` |
| `ObjectCaptureShader` | GLSL shader for textured + masked rendering | `Core.Renderer/ObjectCapture/ObjectCaptureShader.cs` |
| `HeadlessContext` | Offscreen OpenGL context | `Core.Renderer/Headless/` |
| `WmoFullLoader` | Loads WMO from catalog into `WmoV14Data` | `Core.IO/Converters/WmoFullLoader.cs` |
| `TerrainTileTensorPack` | Holds `PlacementModfData` (MODF placements), `PlacementModfNames` | `Core/Maps/TerrainTileTensorPack.cs` |
| `TerrainMinimapCompositor` | CPU terrain-only minimap compositor | `Core.IO/Maps/TerrainMinimapCompositor.cs` |

## Changes

### 1. `ObjectCaptureShader` — make light direction a uniform

**Current**: hardcoded `vec3 lightDir = normalize(vec3(0.4, -0.5, -0.85));`

**Change**: add `uniform vec3 uLightDir;` and set it from the terrain's `TerrainSolarDirection.Evaluate(gameTime)`. This ensures WMO lighting matches the terrain lighting exactly.

### 2. `WmoObjectRenderer` — add per-instance transform support

**Current**: `Render(Matrix4x4 viewProj, bool maskMode)` sets `_shader.SetModel(Matrix4x4.Identity)`.

**Change**: add a `RenderWithTransform(Matrix4x4 viewProj, Matrix4x4 model, bool maskMode)` overload that sets the model matrix to the WMO's world-space placement transform. The placement transform is:
```
Matrix4x4.CreateTranslation(posX, posY, posZ) *
Matrix4x4.CreateRotationZ(rotZ) *
Matrix4x4.CreateRotationY(rotY) *
Matrix4x4.CreateRotationX(rotX)
```

### 3. `ObjectCaptureRenderer` — add `CaptureTileWmos()` method

New method that:
1. Sets up the orthographic camera to match the tile's world bounds:
   ```
   tileWorldSize = 533.33333f
   mapOrigin = 32f * tileWorldSize
   tileCenter.X = mapOrigin - (tileY + 0.5f) * tileWorldSize
   tileCenter.Y = mapOrigin - (tileX + 0.5f) * tileWorldSize
   orthoHalf = tileWorldSize * 0.5f
   cameraHeight = 500f  // well above any WMO
   ```
2. Iterates MODF placements, loads each WMO via `WmoFullLoader`, renders at its world transform
3. Returns `(byte[] imageRgba, byte[] maskRgba)` at the minimap resolution
4. Builds `WmoObjectRenderer` per unique WMO model (not per placement — cache by model path)

### 4. Harvest tool `Program.cs` — wire up the flag

**New flag**: `--include-wmos` (no argument)

**In `RunSyntheticMinimap()`**:
- Parse `--include-wmos`
- If set, create a `HeadlessContext` + `ObjectCaptureRenderer` once
- Set the shader's light direction from the terrain's `TerrainSolarDirection.Evaluate(gameTime)`
- After terrain compositing, call `capture.CaptureTileWmos(...)` 
- Composite: for each pixel, if WMO mask alpha > 0, blend WMO image over terrain image

### 5. Compositing logic

Simple alpha blending in the terrain compositor's output loop:
```
for each pixel (x, y):
    if wmo_mask[y, x] > 0:
        result[y, x] = wmo_image[y, x]
    else:
        result[y, x] = terrain_image[y, x]
```

This is a simple overlay — WMO fragments occlude terrain behind them. Since the renderer uses depth testing, WMOs correctly occlude each other and the terrain (which is rendered first as the clear color).

### 6. Above-ground filter

Skip WMOs whose bounding box is entirely below the tile's terrain surface. Compare `bbMaxZ` (top of the WMO's bounding box) against the tile's maximum terrain height from `Height257`. If `bbMaxZ < terrain_max_height`, the WMO is underground and skipped.

## Files Modified

| File | Change |
|------|--------|
| `src/core/WowViewer.Core.Renderer/ObjectCapture/ObjectCaptureShader.cs` | Add `uLightDir` uniform, remove hardcoded lightDir |
| `src/core/WowViewer.Core.Renderer/ObjectCapture/WmoObjectRenderer.cs` | Add `RenderWithTransform()` overload |
| `src/core/WowViewer.Core.Renderer/ObjectCapture/ObjectCaptureRenderer.cs` | Add `CaptureTileWmos()` method, `SetLightDirection()` |
| `tools/harvest/WowViewer.Tool.Harvest/Program.cs` | Add `--include-wmos` flag, headless GL context, compositing |

## Output

The `--include-wmos` flag produces the same tile PNGs as before, but with WMO buildings/structures rendered on top with matching lighting. The manifest records:
```
"wmo_overlay": true,
"wmo_lighting_source": "solar_direction_at_time_hours"
```

## Not in Scope

- MDX/M2 doodad rendering (user excluded)
- WMO interior rendering (only exterior shells)
- Per-object shadow maps (MCSH is terrain-only; WMO self-shadowing is a future concern)
- GPU texture atlasing (each WMO loads its textures individually)