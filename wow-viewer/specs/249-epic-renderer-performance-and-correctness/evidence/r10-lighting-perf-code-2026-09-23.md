# Receipt — R-10 modern-data lighting performance, code slice (2026-09-23)

Epic 249 item R-10 (operator P1, scope approved 2026-09-23 "Full bundle"). This receipt covers the
**code** tasks R10-T002, T004, T005, T006. It does **not** claim any FPS, draw-call or visual result:
those are R10-T003 (baseline) and R10-T007 (after-capture), operator-owned.

## Files changed

| File | Change |
|---|---|
| `src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs` | Candidate gains `ReachedBySceneLight` / `EmitsSceneLights` (optional, default false); lit placements go to the per-placement path; plan reports `BatchedPlacementCount`, `LitFallbackCount`, `SelfLitFallbackCount` |
| `src/core/WowViewer.Core.Runtime/World/SceneLightingFrameStats.cs` (new) | Per-frame light workload record |
| `src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs` | `SceneLighting` init property |
| `src/viewer/WoWViewer/Rendering/SceneLightManager.cs` | Uniform XY grid (64-unit cells, lazy rebuild, large-radius list, full-scan fallback for huge queries); `AnyAffecting`; filtered `AddRange`; `TakeFrameCounters`; `RecordWmoPartition`; nearest-first with insertion-order tie break; no per-query allocations |
| `src/viewer/WoWViewer/Rendering/FrustumCuller.cs` | `TestSphereIgnoringFarPlane` (side + near planes) |
| `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` | `EmitsSceneLights`, `GetWorldBounds` (wrap existing `TransformAabb`) |
| `src/viewer/WoWViewer/Terrain/WorldScene.cs` | Edits inside existing members only (no new members, AGENTS.md §10): per-placement gate at the WMO opaque candidate loop; partition recorded; batch path uploads `sceneLights: null`; `RebuildSceneLights` keeps only lights whose sphere (+256 margin) touches the view; frame stats carry `SceneLighting` |
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | Two lines in *Submission efficiency (this frame)*: WMO placements batched / lit per-placement (self-lit), scene lights kept/collected, queries, candidates tested |
| `tests/WowViewer.Core.Tests/WorldObjectPassCoordinatorTests.cs` | +2 tests (lit partition, determinism) |
| `tests/WowViewer.Core.Tests/SceneLightManagerTests.cs` (new) | 5 tests incl. 40 random scenes × 200 queries vs a reference linear scan |

## Commands and exit status

The operator's viewer (PID 4248) held `src/viewer/WoWViewer/bin/Debug`, so builds used an isolated
artifacts path inside the repo.

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug `
  --artifacts-path I:/parp/parp-tools/wow-viewer/output/tmp/r10-artifacts `
  --filter "FullyQualifiedName~SceneLightManagerTests|FullyQualifiedName~WorldObjectPassCoordinatorTests"
# Passed! Failed: 0, Passed: 18, Total: 18 — exit 0

dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug `
  --artifacts-path I:/parp/parp-tools/wow-viewer/output/tmp/r10-artifacts
# Build succeeded (all projects). Core.Editor.Tests 91/91, Core.Curation.Tests 39/39,
# Core.Tests 1586 passed / 10 failed / 1 skipped, Core.PM4.Tests 102 passed / 1 failed — exit 1
```

The 11 failures are outside the changed files (none of the failing code paths is in the diff):
`WorldFramePassCoordinatorTests` ×3 (expect a `wdl` stage the coordinator no longer emits),
`WtfLineClassifierTests` ×2, `AdtV23SummaryReaderTests` (chunk overrun in `ChunkedFileReader`),
`ModelFootprintReaderTests`, `LkToAlphaRoundTripTests.AlphaToLk_FlagContract…` (archived Spec 221's
pinned-red open defect), `V18StorePlacementsReaderTests`, `EnrichmentStreamFormatTests`,
`Pm4RegionObjectGrouperTests` (development corpus). **Not re-run on a clean `HEAD`** to prove they
pre-date this change; that baseline is still owed.

## Criterion → evidence

| Criterion (Epic 249 R-10 / archived 242) | Evidence | Status |
|---|---|---|
| FR-001 batch decision per placement | `WorldScene` candidate loop uses `SceneLightManager.AnyAffecting(placement world AABB)`; planner test `LitPlacementsKeepThePerPlacementPath` | code + unit test |
| FR-002 lit placements keep their own light set | lit → fallback → existing `RenderWithTransform(…, _sceneLightManager)`; batched path uploads zero lights because no light reaches any batched placement | code + unit test |
| FR-003 unlit placements eligible for instancing | same test: placements 0 and 2 batch | unit test |
| FR-004 deterministic partition | `PartitionIsDeterministic` | unit test |
| R-10d query results identical to the linear scan | `QueryAffecting_MatchesLinearScan_ForRandomScenes` (8,000 queries incl. inverted bounds, huge radii, huge boxes, equal-distance ties) + Z/huge-light/rebuild tests | unit test |
| R-10c no drawn pixel loses a light | exact geometric argument: a light whose sphere lies outside every side/near plane contributes nothing inside the view; far plane not used. **Caveat**: terrain renders before the rebuild and uses the previous frame's set; the 256-unit margin covers ordinary motion, a very fast turn could delay a light entering at the screen edge by one frame on terrain only | reasoning; **operator visual check owed** |
| R-10a counters visible | panel lines in *Frame history → Submission efficiency* | **operator check owed** |
| SC-001/SC-002 draw calls + ≥2× FPS on `wow_classic_beta` 1.60.1 `Azeroth` | — | **not claimed** (R10-T003/T007) |
| SC-003 lit shading unchanged | — | **not claimed** (operator side-by-side) |
| FR-006 no new `WorldScene`/`ViewerApp` members | diff: edits inside existing methods only | code review |
