# Spec 259 Operator Amendment Receipt — Ground-Effect Lighting Propagation

Date: 2026-10-14 · Scope: carry terrain MCCV and MCSH-derived per-instance tint through the existing MDX material path.

> **Historical / reverted:** this receipt describes the initial implementation from commit `048b098e`.
> After release, the operator reported that the viewer could no longer load content. The code and tint
> tests were reverted on 2026-10-14. Do not use this receipt as proof that the current branch implements
> AC-009 or that the attempted change's root cause has been identified.

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
| AC-009 code path: MCCV and MCSH tint reaches the model material while directional/ambient lighting remains active | **REVERTED / OPEN** | The symbols described above were removed after the operator's load failure report; prior unit/build output does not identify the regression cause. A safe reimplementation is not present in current source. |
| AC-009 runtime appearance on authentic terrain with MCCV and MCSH | **OPEN — operator-owned** | No real-client visual capture was performed. T015 remains open for a future safe fix. |
