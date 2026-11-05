# Spec 294 Phase 1 — Donor Group Transform Core Receipt

**Date:** 2026-11-05  
**Scope:** T001–T006 only. UI, persistence, and real-client proof are not included in this receipt.

## Files changed

| File | Change |
|---|---|
| [`DonorGroupTransformService.cs`](../../src/core/WowViewer.Core/Maps/DonorGroupTransformService.cs) | Added UI-free angle normalization, default/custom pivot handling, group translation and rotation, grid snapping, clone-based previews, structured bounds/collision/gap issues. |
| [`DonorGroupTransformServiceTests.cs`](../../tests/WowViewer.Core.Tests/Maps/DonorGroupTransformServiceTests.cs) | Added tests for translation, 17°/fractional angles, angle normalization, single-tile pivot, pivot fallback, bounds, collision, gaps, clone isolation, invalid input, and empty groups. |
| [`plan.md`](../plan.md) | Recorded operator approval and tile-grid snap rule for the core transform. |
| [`tasks.md`](../tasks.md) | Marked Phase 1 implementation and test tasks complete with this receipt. |

## Verification

| Command | Exit | Result |
|---|---:|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DonorGroupTransformServiceTests" --nologo` | 0 | 13 passed, 0 failed, 0 skipped. |

## Requirement → evidence

| Requirement | Evidence |
|---|---|
| FR-001 — transform multiple donor tiles as one group | Translation and rotation tests transform all placements around the resolved group pivot; per-placement donor identity and metadata remain intact. Target tile indices snap after the continuous rigid transform, as recorded in [`plan.md`](../plan.md). Viewer-level selection and visual proof remain for Phase 2/operator runtime validation. |
| FR-002 — translate while retaining group arrangement | `CreatePreview_TranslatesAllPlacementsTogether_AndPreservesPlacementMetadata` checks the same translation on each target and unchanged donor identity/lock/include/Z metadata. |
| FR-003 — arbitrary rotation | Theory cases cover 17°, 17.5° (via 377.5° normalization), -17°, and 720°; the 17° case checks transformed target coordinates and the preview retains the exact normalized angle. |
| FR-007 — non-destructive project state; no terrain resampling | Preview uses `PhaseLayerSettings.Clone`; serialized source-layer state is identical before and after preview/cancel simulation. The service only changes candidate target mappings and does not touch donor data or terrain channels. Persistence and runtime proof remain in later phases. |

This receipt proves the Phase 1 core behavior and unit-test output only. It does not claim the user-facing workflow, real-map rendering, save/reopen, or unchanged real source files; those remain explicit later-phase/operator gates.
