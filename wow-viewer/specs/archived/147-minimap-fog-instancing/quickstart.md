# Quickstart: Spec 147 Validation

This document describes the bounded proof sequence. It does not claim real-client visual or FPS
proof from source tests/builds.

## Focused source checks

Run from PowerShell 7:

```powershell
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter FullyQualifiedName~MinimapInteraction
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter FullyQualifiedName~Tile
dotnet test I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --filter FullyQualifiedName~Batch
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
```

The current focused source proof is six tests: four MinimapInteractionTests and two
AlphaAreaAudioCatalogTests. The full solution build remains the release check; the cross-platform
viewer build is sufficient for the bounded viewer source slice when the Windows target is blocked
by environment-specific packaging.

If the implementation uses different test names, the task owner must update these filters rather
than silently treating a zero-test result as proof.

## User-run minimap gate

1. Use a configured client root and load a map with visible minimap tiles.
2. Open the full-screen minimap.
3. Drag the map and release; confirm the map pans and the camera position is unchanged.
4. Click one valid target three times without moving the pointer; confirm teleport occurs on click
   three and the status/active tile changes.
5. Drag between clicks or change target; confirm the sequence resets and no premature teleport
   occurs.

## User-run fog/residency gate

Record the exact client root/build, map, camera tile/path, resolution, warm-up policy, active fog
source/start/end, detail controls, and capture-preload state. During a short movement capture:

- change camera direction without moving far and confirm nearby side tiles do not disappear;
- cross a tile boundary and confirm near-field content remains stable;
- change the active fog profile and confirm the reported coverage window changes;
- verify tiles outside fog are excluded unless explicitly marked retained/preloaded/full-load;
- verify WMO containment remains stable.

## User-run doodad gate

Use a dense repeated-doodad location and capture the structured frame report. Compare unique assets,
compatible batches, instances, fallback instances, animation updates, draw submissions, and stage
times before/after. Inspect transparent, animated, particle/ribbon, WMO-internal, and static opaque
examples separately. Do not infer a visual/FPS win from counters alone.

## Required handoff fields

- client root and exact build/fingerprint;
- map and camera tile/path;
- screen resolution and v-sync state;
- warm-up duration/frame count;
- active fog source/start/end and user overrides;
- detailed/retained tile controls;
- capture preload/full-load state;
- tile, WMO, MDX/M2, batch, fallback, and frame-stage counters;
- observed visual result and whether the result is user runtime proof.
