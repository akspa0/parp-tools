# Spec 294 Phases 3–4 — Manual Curation and Persistence Receipt

**Date:** 2026-11-05  
**Scope:** T013–T022. Manual placement/association UI and curation-only persistence are implemented. Real-client visual and saved-project runtime proof remains operator-owned.

## Files changed

| File | Change |
|---|---|
| [`ManualCurationProject.cs`](../../src/core/WowViewer.Core/Maps/ManualCurationProject.cs) | Added manual placement/hull identity models and explicit add/update/remove/replace policies, including exact-duplicate rejection. |
| [`ManualCurationProjectFile.cs`](../../src/core/WowViewer.Core/Maps/ManualCurationProjectFile.cs) | Added versioned JSON load/save with same-directory temporary write and overwrite, separate from source map files. |
| [`PhaseLayerProjectFile.cs`](../../src/core/WowViewer.Core/Maps/PhaseLayerProjectFile.cs) | Persists/restores exact donor-group transform metadata. Missing metadata remains backward-compatible (`null`). |
| [`ManualCurationWorkbenchService.cs`](../../src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/ManualCurationWorkbenchService.cs) | Adds in-scene position/yaw edits, selected client-catalog placement, explicit object↔hull pairing, duplicate reporting, remove/replace, separate project save/reload, and project-to-scene restoration. |
| [`AssetCatalogView.cs`](../../src/viewer/WoWViewer/Catalog/AssetCatalogView.cs) | Exposes the selected catalog entry to the curation workbench without creating a second catalog. |
| [`Pm4WorkbenchSidebarService.cs`](../../src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs) | Connects selected scene objects and PM4 hulls to the curation workbench. |
| [`ManualCurationProjectPolicyTests.cs`](../../tests/WowViewer.Core.Tests/Maps/ManualCurationProjectPolicyTests.cs) | Tests duplicate rejection, shared-hull distinct objects, explicit remove/replace, invalid input, and relation preservation when moving a placement. |
| [`ManualCurationProjectFileTests.cs`](../../tests/WowViewer.Core.Tests/Maps/ManualCurationProjectFileTests.cs) | Tests placement/association round-trip, unsupported versions, and source-map sentinel bytes unchanged by project save. |
| [`PhaseLayerProjectFileTests.cs`](../../tests/WowViewer.Core.Tests/Maps/PhaseLayerProjectFileTests.cs) | Extends layer-project round-trip coverage to the authored group transform. |
| [`ui-controls.json`](../../eng/ui/ui-controls.json), [`UI-REFERENCE.md`](../../docs/WoWViewer/UI-REFERENCE.md) | Registers manual curation controls and regenerates the UI reference. |
| [`plan.md`](../plan.md), [`tasks.md`](../tasks.md), [`STATUS.md`](../../STATUS.md), [`activeContext.md`](../../../memory-bank/activeContext.md), [`progress.md`](../../../memory-bank/progress.md) | Updates implementation state, receipts, and handoff. |

## Verification

| Command | Exit | Result |
|---|---:|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DonorGroupTransformServiceTests|FullyQualifiedName~ManualCurationProjectPolicyTests|FullyQualifiedName~ManualCurationProjectFileTests|FullyQualifiedName~PhaseLayerProjectFileTests" --nologo` | 0 | 22 passed, 0 failed, 0 skipped. |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | Inventory 152 rows / 140 distinct labels; 140 registry entries; 0 errors, 0 warnings. |
| `dotnet build wow-viewer/WowViewer.slnx -c Debug --no-restore --nologo -v minimal` | 0 | Full solution build succeeded, 0 errors; 266 warnings from package advisories and existing project diagnostics. |
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --no-restore --nologo -v minimal` | 0 | Cross-platform viewer build succeeded, 0 errors; 134 warnings, including pre-existing diagnostics and two NU1902 package warnings. |

## Requirement → evidence

| Requirement | Evidence |
|---|---|
| FR-008 — hand move/rotate and explicit pairing | PM4 workbench exposes finite X/Y/Z and arbitrary-yaw edits for the selected scene placement. The curation section stores the placement snapshot and requires a separate user action to pair it with a selected hull. Move/yaw-to-scene behavior still needs operator visual proof. |
| FR-009 — existing and catalog placements | Existing selected tile placements can be recorded/moved; the existing client Asset Catalog selection is exposed, and its selected model can create a curation-only placement on the selected hull's tile. Catalog placement is not automatically paired: the user explicitly adds the association. Runtime load/appearance remains an operator gate. |
| FR-010 — visible members and duplicate prevention | The workbench lists object model/path and hull tile/CK24/part/group. `ManualCurationProjectPolicyTests` proves exact duplicate rejection and permits distinct object placements to share a hull. |
| FR-011 — curation-only persistence | `ManualCurationProjectFile` saves placements and associations to `output/projects/manual_curation/<map>.manual-curation.json`; `PhaseLayerProjectFile` saves the group transform. The round-trip test verifies the map-source sentinel is unchanged; real-client source hashes remain operator-owned. |
| FR-012 — explicit remove/replace | UI exposes separate Remove and Replace actions; policy tests prove each only changes the requested relation and invalid/duplicate replacements preserve the original association. |
| SC-002 / SC-004 — save/reopen state | File-level round-trip tests preserve exact group transform, manual placement transforms, and hull associations. In-scene reapplication is implemented for resident and catalog-created placements; the actual viewer save/reopen pass remains owed. |
| AC-015 — shared UI registry | `Verify-UiReference.ps1` passed with zero errors and warnings. |

## Operator gate owed

Use a real client/map and a selected PM4 hull. Place an asset selected from the client Asset Catalog, move/rotate an existing object, pair and inspect both members, exercise duplicate/remove/replace, save/reload the curation project and layer project, and compare source-map hashes before/after. Record client build/root fingerprint and artifacts. This receipt does not claim visual, runtime, or real-client proof.
