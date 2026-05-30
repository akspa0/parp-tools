# Roof Capture Overhaul Plan

**Branch**: `v0.5.0-clean` (hard-reverted to 630dee49)
**Target builds**: `0_5_3_3368`, `3_0_1_8303`, `4_0_0_11927`
**Output**: per-asset Zarr with `roof_rgb`, `roof_mask`, `roof_confidence`, `build_code`, `metadata`

---

## Problem Summary

The GPU roof capture pipeline (`MdxViewer --capture-roof`) produces empty/solid black PNGs across all builds. The rendering works nominally (no crashes, files created) but geometry doesn't render. Root cause chain:

1. M2/MDX files in 3.x+ MPQs have `.mdx` extension in the listfile but `.m2` extension on disk → `IDataSource.ReadFile` returns null → MdxRenderer never created → transparent output
2. Animation crash loop: corrupted M2 animations throw on frame update → `inst.Renderer = null` kills all future doodad renders (the 03ec7598 regression)
3. Camera distance uses bounding box diagonal but doesn't account for model aspect ratio → geometry clipped on tall/skinny objects
4. FBO framebuffer state corruption between frames → OpenGL state leaks

---

## Data Flow

```
AssetList JSON (.wmo/.mdx paths)
        │
        v
  [Fix] PathResolver: .mdx → .m2 probe for 3.x+ clients
        │
        v
  [Fix] MdxRenderer: disable animation update (static pose only)
        │
        v
  [Fix] CameraSolver: adaptive distance from geometry raycast
        │
        v
  GPU Render → offscreen FBO → RGBA pixels
        │
        v
  [Fix] FBO restore between frames (clear color+depth each render)
        │
        v
  Encode → NPZ/Zarr chunk (roof_rgb, roof_mask)
        │
        v
  Pack into per-build Zarr store with resume support
```

---

## Fix 1: MDX/M2 Path Resolution

**File**: `ViewerApp_CaptureAutomation.cs` / `ScreenshotRenderer.cs`

**Problem**: 3.x+ clients store M2 files in MPQ as `file.m2` but the listfile reports them as `file.mdx`. The current pipeline only tries `.mdx` → null → no render.

**Fix**: Before reading an asset file, probe for alternate extensions:
- If asset ends in `.mdx` and `ReadFile(mdxPath)` returns null, try `ReadFile(m2Path)` (swap extension to `.m2`)
- If asset ends in `.m2` and `ReadFile(m2Path)` returns null, try `ReadFile(mdxPath)` (swap extension to `.mdx`)
- Alpha builds (0.5.x) only use `.mdx` → no swap needed

```csharp
byte[]? ReadModelBytes(IDataSource ds, string path) {
    byte[]? data = ds.ReadFile(path);
    if (data != null) return data;
    string swapped = Path.ChangeExtension(path, 
        path.EndsWith(".mdx") ? ".m2" : ".mdx");
    return ds.ReadFile(swapped);
}
```

**Affected call sites** (ScreenshotRenderer.cs lines ~151-162, ~181-193, CaptureMdxMultiAngle, CaptureMdxRoofTopDown, CaptureMdxRoofTopDownByPath):
- `CaptureMdxAllAngles` line 153: `ds.ReadFile(modelPath.Replace('/', '\\'))`
- `CaptureMdxRoofTopDownByPath` line 183: same
- `CaptureMdxMultiAngle`: same pattern

---

## Fix 2: Disable Animation in Roof Capture

**File**: `ScreenshotRenderer.cs` — `RenderMdxRoofTopDown` and `RenderMdxMultiAngle`

**Problem**: M2 animation update can crash on corrupted bones/tracks. The roof capture only needs a static t-pose or bind-pose render.

**Fix**: Pass a flag to skip `renderer.UpdateAnimation()` during roof capture. In `MdxRenderer`, add a property or parameter to skip animation evaluation and use bind-pose bone matrices.

**Alternative**: simpler — wrap the UpdateAnimation call in a try/catch in the capture path only (not the main viewer loop):

```csharp
try { renderer.UpdateAnimation(); } 
catch { /* static pose only */ }
```

This preserves the main viewer's animation system while preventing capture crashes.

---

## Fix 3: Adaptive Camera Distance

**File**: `ScreenshotRenderer.cs` — `RenderWmoRoofTopDown`, `RenderMdxRoofTopDown`

**Problem**: Current code computes camera distance from `boundsDiagonal * 1.15` but doesn't account for the orthographic or perspective projection's field-of-view. Tall objects get clipped at the top/bottom.

**Fix**: For **orthographic top-down roof capture**, compute the bounding box spans in XY (top-down) and set the orthographic projection to exactly cover `max(spanX, spanY) * 1.15`. Camera position = `center + (0,0, spanZ * 3)`.

```csharp
// Orthographic roof camera
float spanX = boundsMax.X - boundsMin.X;
float spanY = boundsMax.Y - boundsMin.Y;
float spanZ = boundsMax.Z - boundsMin.Z;
float maxSpan = Math.Max(spanX, spanY) * 1.15f;
Vector3 center = (boundsMin + boundsMax) * 0.5f;
Vector3 eyePos = center + new Vector3(0, 0, spanZ * 3);
var view = Matrix4x4.CreateLookAt(eyePos, center, Vector3.UnitY);
var proj = Matrix4x4.CreateOrthographic(maxSpan, maxSpan, 0.01f, spanZ * 10f);
```

For **perspective multi-angle**, use the existing bounds-corner projection but add a `Math.Max(aspectRatio, 1/aspectRatio)` multiplier to the max distance.

---

## Fix 4: FBO State Restoration

**File**: `ScreenshotRenderer.cs` — `RenderWmoRoofTopDown`, the `EnsureFbo` method

**Problem**: After rendering one model, the FBO binding, viewport, and clear color may be stale for the next model. If a model fails mid-render (e.g. missing texture), the FBO state leaks.

**Fix**: At the start of each render call:
```csharp
_gl.BindFramebuffer(FramebufferTarget.Framebuffer, _fbo);
_gl.Viewport(0, 0, (uint)width, (uint)height);
_gl.ClearColor(0, 0, 0, 0);  // black transparent background
_gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);
_gl.Enable(EnableCap.DepthTest);
_gl.DepthFunc(DepthFunction.Less);
```

Then in the `finally` block after render+readback:
```csharp
_gl.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
```

---

## Fix 5: Zarr Output (no intermediate PNGs)

**New file**: `ViewerApp_ZarrOutput.cs` or integrate into existing pack step.

Instead of writing per-asset `roof_topdown.png` to disk → MdxViewer packs rendered RGBA pixels into a memory buffer, then writes to a Zarr store via a controlled append operation.

**Design**:
1. MdxViewer opens a Zarr store file (or stdout pipe) at startup
2. For each captured asset, encodes `roof_rgb` (128×128×3 uint8) + `roof_mask` (128×128 float32) + metadata
3. Appends to the Zarr store in fixed-size chunks
4. On crash/resume, scans the Zarr for completed entries and skips those assets

**Resume mechanism**: A `_resume_state.json` inside the Zarr directory tracks which asset paths have been captured. On startup, load the completed set and skip.

**Metadata parquet**: Write `roof_exemplars.parquet` alongside the Zarr with schema:
- `asset_path` (string)
- `build` (string) 
- `success` (bool)
- `failure_reason` (string, nullable)
- `render_time_ms` (uint32)

---

## Fix 6: Underground Group Culling

**File**: `ScreenshotRenderer.cs` — WMO capture path

**Problem**: WMO groups below the terrain surface (basements, underground rooms) render as invisible black pixels wasted on disk.

**Fix**: After loading the WMO, iterate groups and skip rendering any group whose bounding box `max.Z < wmoBounds.min.Z + epsilon` (i.e. entirely below the WMO's base). For Alpha WMOs (v14), the Z-up convention puts ground at `boundsMin.Z`.

```csharp
float groundZ = wmo.BoundsMin.Z;
for each WMO group:
    if group.BoundsMax.Z <= groundZ + 0.5f:
        continue; // skip underground group
    render group
```

This also reduces render time per WMO.

---

## Implementation Order

| Step | File(s) | Description | Risk |
|------|---------|-------------|------|
| 1 | `ScreenshotRenderer.cs` | Fix 4: FBO state restoration between renders | Low |
| 2 | `ScreenshotRenderer.cs` | Fix 1: MDX/M2 extension probing | Low |
| 3 | `ScreenshotRenderer.cs` | Fix 2: Wrap animation update in try/catch | Low |
| 4 | `ScreenshotRenderer.cs` | Fix 3: Adaptive orthographic camera for roof | Medium |
| 5 | `ScreenshotRenderer.cs` | Fix 6: Underground group culling | Low |
| 6 | New: `ViewerApp_ZarrOutput.cs` | Fix 5: Direct-to-Zarr output with resume | Medium |

Steps 1-6 each add ~20-40 lines. Total ~200 lines changed/new.

---

## Validation

After each fix, test by capturing a small asset list (e.g. 10 WMOs from `duskwood`) on build `3_0_1_8303` and check:
- Non-zero pixel output (no all-black renders)
- No OpenGL errors in console output
- Camera distance covers full model
- M2 models render (check a known M2 like `world/azeroth/passivedoodads/...`)

Final validation: full capture of `0_5_3_3368`, `3_0_1_8303`, `4_0_0_11927` at 512×512, all assets, direct to Zarr.
