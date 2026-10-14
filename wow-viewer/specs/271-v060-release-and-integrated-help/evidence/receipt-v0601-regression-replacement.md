# v0.6.0.1 Replacement Receipt — Ground-Effect Tint Reverted

Date: 2026-10-14 · Replacement tag commit: `5ba5eb2a934b78dc46c31e7267cc67069c03a2f9` ·
Release: https://github.com/akspa0/parp-tools/releases/tag/v0.6.0.1

## Files changed

- Ground-effect tint/API changes reverted in `GroundEffectPlacementGenerator.cs`,
  `GroundEffectPlacementModels.cs`, `IModelRenderer.cs`, `ModelRenderer.cs`,
  `GroundEffectSceneService.cs`, and `GroundEffectPlacementTests.cs`.
- `eng/Version.props`, `CHANGELOG.md`, and `docs/releases/v0.6.0.1.md` identify the replacement build
  and defer tint propagation.
- Spec 259/271 specifications, plans, tasks, receipts, `specs/STATUS.md`, `memory-bank/activeContext.md`,
  and `memory-bank/progress.md` record the regression, rollback, and remaining operator proof.
- CLI packaging/workflow changes and the separate, dependency-deduplicated tool archive are retained.
- Pre-existing dirty `imgui.ini` files were not staged or committed.

## Verification commands and results

- `git diff --exit-code b6f54a0b -- wow-viewer/src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementGenerator.cs wow-viewer/src/core/WowViewer.Core.Runtime/DetailDoodads/GroundEffectPlacementModels.cs wow-viewer/src/viewer/WoWViewer/Rendering/IModelRenderer.cs wow-viewer/src/viewer/WoWViewer/Rendering/ModelRenderer.cs wow-viewer/src/viewer/WoWViewer/Workbench/Services/GroundEffects/GroundEffectSceneService.cs wow-viewer/tests/WowViewer.Core.Tests/DetailDoodads/GroundEffectPlacementTests.cs` — exit 0; ground-effect source/test files match the pre-tint baseline.
- `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~DetailDoodads" -v:q` — exit 0; 8 passed, 0 failed.
- `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore -v:q` — exit 0; 0 errors (warnings remain).
- Branch matrix run [38512662293](https://github.com/akspa0/parp-tools/actions/runs/38512662293) — conclusion `success`; all four RIDs rebuilt viewer + shared-output tools and uploaded separate artifacts.
- `gh run watch 38513490254 --exit-status` — exit 0; replacement release workflow succeeded.
- `gh release view v0.6.0.1 --json tagName,isDraft,isPrerelease,assets,url` — exit 0; release contains four viewer ZIPs and four CLI-tools ZIPs.

## Acceptance evidence

| Criterion | Result | Evidence |
|---|---|---|
| AC-C1 — separate viewer/tool archives, nine tools, shared dependencies once | **PASS** | Four-RID branch matrix passed; public v0.6.0.1 release lists eight assets. The Windows ZIP inspection in `receipt-phase-c-v0601.md` verified 9 apphosts, 1,342 DBD files, and one common Core DLL. |
| AC-C2 — successful tagged release | **PASS** | Replacement release workflow `38513490254` succeeded; tag resolves to `5ba5eb2a`; eight ZIP assets are attached. |
| AC-C3 — ground-effect lighting applied | **DEFERRED / OPEN** | Tint code was reverted after the operator reported the viewer could not load content. No tint claim is made for this replacement. Root cause is not independently reproduced; real-client map-load verification remains operator-owned (Spec 271 C4). |
