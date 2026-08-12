# Implementation Plan: Camera Capture Paths

**Branch**: `144-camera-capture-paths` | **Date**: 2026-08-11 | **Spec**: [spec.md](spec.md)

## Summary

Recover the existing capture automation route in the Tools > Utilities panel and add a reusable M2-style camera path contract. The core owns map-bound keyframes, spline evaluation, M2/MDX camera sampling, and round-trip tests. The viewer owns loaded-client asset selection, ImGui editing, timeline scrubbing, active-map validation, opt-in collision resolution, playback, bounded path preloading, and integration with the existing still/video capture queue.

## Technical Context

**Language/Version**: C# / .NET 10
**Primary Dependencies**: Existing WowViewer.Core.Runtime M2 track sampler, shared M2/MDX readers and MDX camera resolver, ImGui.NET, Silk.NET window/input
**Storage**: Project JSON path plus native camera-only `.m2` and `.m2.json` metadata sidecar
**Testing**: xUnit focused core tests and Debug project build
**Target Platform**: Windows desktop viewer, with non-Windows file-picker fallback unchanged
**Project Type**: Desktop viewer and reusable core libraries
**Performance Goals**: Path evaluation is constant-time over the key list; capture warmup retains only sampled path tiles and queues their objects before recording without enabling full-map residency
**Constraints**: Do not replace the existing capture queue, do not rewrite format readers, and do not claim interactive runtime proof from compilation alone
**Scale/Scope**: One active authored path at a time, with arbitrary ordered key points and imported M2 camera samples

## Constitution Check

- New code stays in `wow-viewer`.
- Existing M2 readers and capture backend are reused.
- Viewer runtime proof remains user-run; this slice provides focused tests/build only.
- The existing sidebar remains available; this is additive route recovery, not a shell rewrite.
- Capture warmup is a scoped lease over sampled available tiles. `TerrainManager` owns tile retention and `WorldScene`/`WorldAssetManager` own placement-asset queues; the lease is released on stop or completion.

## Source Structure

- `src/core/WowViewer.Core.Runtime/M2/M2CameraPath.cs`: path model, evaluator, M2 import sampling, persisted collision settings.
- `src/core/WowViewer.Core.Runtime/M2/MdxCameraPathImporter.cs`: shared MDX CAMS sampling into the path model.
- `src/core/WowViewer.Core.Runtime/M2/M2CameraPathWriter.cs`: camera-only native M2 writer.
- `src/viewer/WoWViewer/ViewerApp_CameraPaths.cs`: editor, playback, save/load, capture integration,
  and the Tools > Utilities > Capture panel route.
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: existing Utilities tab host and compatibility routing.
- `tests/WowViewer.Core.Tests/M2CameraPathTests.cs` and `MdxCameraPathImporterTests.cs`: interpolation, native round-trip, and MDX sampling proof.
- `src/viewer/WoWViewer/Terrain/WorldScene.cs`: opt-in heightfield and resident-WMO-bound camera-path collision sweep.

## Validation Gates

1. Core tests prove path evaluation and native M2 read-back.
2. Viewer Debug build passes with no errors.
3. User manually loads a map, selects a client M2/MDX camera, imports and scrubs it, verifies collision-constrained playback, and runs path video capture.
