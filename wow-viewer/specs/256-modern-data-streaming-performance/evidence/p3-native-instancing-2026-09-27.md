# Receipt — T041 P3 native M2 GPU instancing (2026-09-27)

Operator: "Start P3 instancing" (T040: native instanced path).

## Files changed
- `src/viewer/WoWViewer/Rendering/M2Renderer.Instancing.cs` (new, 193 lines): per-renderer instance buffer
  (mat4 + fade, attributes 4–8, divisor 1, disabled outside instanced draws); `Begin/Queue/EndNativeGpuInstanceBatch`;
  one `DrawElementsInstanced` per visible, non-pending, opaque section.
- `src/viewer/WoWViewer/Rendering/M2Renderer.cs`:
  - native `RequiresUnbatchedWorldRender` = wireframe (was always true) and `SupportsGpuInstancedOpaque` = native gate
    (GL alive, instance buffer, not wireframe, `MdxRenderer.GpuInstancingEnabled`);
  - `IGpuInstancedModelRenderer` methods route to the native batch when there is no legacy renderer;
  - shader: `uInstanced` selects the instance matrix and `vInstanceFade`, which is 1.0 when not instanced;
  - per-section uniforms moved into `ApplySectionUniforms`, shared by `RenderCore` and the instanced draw.

## Parity argument (per instance, opaque pass)
| Quantity | `RenderInstance(M, Opaque, f)` | Instanced |
|---|---|---|
| Model matrix | `uModel = M` (row-major bytes, transpose false) | same 16 floats as instance attributes 4–7 → identical `mat4` |
| Section color | `clamp(animColor) · clamp(f, 0.1, 1)` | `uBaseColor = clamp(animColor)`, × `clamp(vInstanceFade, 0.1, 1)` in shader |
| Alpha | `clamp(f · animAlpha, 0, 1)` | `clamp(animAlpha · vInstanceFade, 0, 1)` in shader (blend is off in this pass either way) |
| Geometry / animation | the model's VBO (CPU-skinned once per frame per model) | same VBO |
| Blend, depth, cull | blend off, depth write on, cull off | same |
Non-instanced draws set `uInstanced = 0` → `vInstanceFade = 1.0`; both new shader factors are then exact identities.

## Verification
| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 0 | 0 errors; compiler warnings identical to base (NU1903 excluded) |
| `dotnet test WowViewer.slnx -c Debug` | 1 | 26 failures (Core.Tests 18, PM4 8), identical to the pre-existing E4 environmental set |
| `glslangValidator m2.vert m2.frag -l` (shader sources extracted from `M2Renderer.cs`; glslang 15.1.0) | 0 | both stages compile and link as `#version 330 core` |

## Criterion → evidence
| Criterion | Evidence |
|---|---|
| Native M2s take the instanced path | Code only; Runtime Stats "Models opaque inst/hoisted/unbatched" must show a non-zero instanced count (T042) |
| Frame time drops, visuals unchanged | **Not proven.** Operator capture + visual A/B owed (T042). A/B switch: PM4 workbench sidebar → "GPU instancing for opaque models (Spec 202)" |
