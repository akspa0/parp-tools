# Spec 259 Operator Amendment Receipt — Ground-Effect Lighting Propagation

Date: 2026-10-14 · Scope: carry terrain MCCV and MCSH-derived per-instance tint through the existing MDX material path.

## Files changed

- `src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementModels.cs`
- `src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs`
- `src/viewer/WoWViewer/Rendering/IModelRenderer.cs`
- `src/viewer/WoWViewer/Rendering/ModelRenderer.cs`
- `src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs`
- `tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectPlacementTests.cs`
- Spec 259 `spec.md`, `plan.md`, and `tasks.md`

## Verification commands and results

- `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:q` — exit 0; 0 errors.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~DetailDoodads" -v:q` — exit 0; 8 passed, 0 failed.
- Tint assertions: neutral missing-MCCV yields `(1,1,1)`; green MCCV with shadow yields an RGB multiplier near `(0,1.098,0)`; `IgnoreMCCV` with shadow yields approximately `(0.7,0.7,0.7)`.

## Criterion evidence

| Criterion | Result | Evidence |
|---|---|---|
| AC-009 code path: MCCV and MCSH tint reaches the model material while directional/ambient lighting remains active | **PASS (code/unit scope)** | `DetailDoodadInstance.RenderTint` carries the computed multiplier; `GroundEffectSceneService` passes it via `IModelRenderer.RenderInstance`; `MdxRenderer.RenderGeosets` multiplies material RGB. The focused tests above cover neutral, tinted, and shadowed values; viewer build validates integration. |
| AC-009 runtime appearance on authentic terrain with MCCV and MCSH | **OPEN — operator-owned** | No real-client visual capture was performed. Tracked in T015 and must be observed before claiming visual proof. |
