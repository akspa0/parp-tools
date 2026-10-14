# Ground-Effect Load Regression Revert — Evidence

Date: 2026-10-14 · Operator report: “the ground effect doodad changes make the viewer unable to load anything.”

## Action and scope

Reverted the new per-instance ground-effect tint path only: `TerrainLightingTint`/`RenderTint`, the
`IModelRenderer` tint overload, MDX material tint multiplication, the tint tuple in
`GroundEffectSceneService`, and the assertions added for those values. Existing ground-effect placement,
MDDL decoding, normal lighting, shared CLI packaging, and tool builds remain unchanged.

No crash log or stack trace was available in the workspace, and no client-load reproduction was run by
the agent. The triggering code delta is removed; the precise runtime failure mechanism remains unknown.

## Verification

- `git diff --exit-code b6f54a0b -- wow-viewer/src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs wow-viewer/src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementModels.cs wow-viewer/src/viewer/WoWViewer/Rendering/IModelRenderer.cs wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs wow-viewer/src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs wow-viewer/tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectPlacementTests.cs` — exit 0; these source/test files match the pre-tint baseline.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~DetailDoodads" -v:q` — exit 0; 8 passed, 0 failed.
- `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:q` — exit 0; 0 errors.

## Criterion evidence

| Criterion | Result | Evidence |
|---|---|---|
| Restore baseline ground-effect renderer behavior | **PASS (source-level)** | Exact `git diff --exit-code` baseline comparison above. |
| Viewer can load a real client/map after rollback | **NOT VERIFIED** | Operator report is recorded; agent did not perform real-client visual/runtime proof. Replacement build is pending; operator must verify a map loads. |
| New MCCV/MCSH material tint | **REVERTED / OPEN** | No tint API or shader multiplication remains in current source. Track a new isolated design/fix separately. |
