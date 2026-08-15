# Phase 0 Research: Portal-Aware Rendering, Game Mode, and Simple Surface

## Native 0.5.3 Ghidra Evidence

The active Ghidra REST bridge was queried directly at `http://127.0.0.1:8089` for the open
`WoWClient.exe` in project `0.5.3`. The image is the x86 client at image base `0x00400000`.

Relevant native anchors:

- `CMapObj::RRenderThruPortals @ 0x0069bf60` takes a current group, previous group, current clip rectangle, and recursion level. It stops at a native maximum recursion level, skips invalid/same-group references, transforms each portal once per render count, rejects a portal on the camera-facing plane test, intersects its clip rectangle with the current rectangle, pushes a narrowed frustum, and recursively visits the destination group.
- `CMapObj::RTransformPortal @ 0x0069c3a0` transforms portal vertices, marks portals behind the near clip plane, calls `CWorldScene::ClipPortal`, and derives a normalized screen rectangle from the clipped vertices. Degenerate/near-clip cases are marked rather than trusted.
- `CWorldScene::ClipPortal @ 0x0066b520` clips a portal polygon against the current viewport planes using a bounded scratch buffer. It returns no polygon when all points are rejected.
- `CMapObj::StabPortals @ 0x0069b630` repeatedly resolves portal transitions until the group is stable, then treats a group flag bit as a special interior/exterior result. The exact flag semantic remains inferred and must not be hardcoded without fixture validation.
- `CMapObj::VectorIntersectPortals @ 0x00693af0` tests a segment against candidate group bounds and each group portal, then returns the crossing groups and distance. `CMapObj::VectorIntersectPortal @ 0x00693d90` performs a group-local ray/portal triangle test.
- `CMapObj::RenderPortals @ 0x0069dc90` and `@ 0x0069de30` triangulate portal polygons for the native portal visualization/debug pass; they are not evidence that the viewer should draw portal surfaces in the normal pass.

These findings are clean-room design evidence only. No native code is ported and no native runtime
dependency is introduced.

## Existing Renderer Findings

- `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` already owns decoded WMO groups, portal vertices,
  portal records, portal references, group bounds, group frustum culling, group visibility buffers,
  and doodad visibility collection.
- Its current `UpdateRuntimeVisibility` path builds undirected group adjacency, chooses an interior
  group from expanded bounds or exterior groups from the frustum, then traverses neighbors using
  portal-center distance and a depth limit. It does not perform native-style portal polygon clipping
  or camera-side tests.
- The current path returns all groups visible when the camera is inside the root or near the root.
  That is a safe rendering fallback but forfeits most interior WMO savings exactly where portal
  traversal is most valuable.
- `src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalVisibilityEvaluator.cs`
  already provides a graph-side portal-volume evaluator with bounded breadth-first traversal and
  fail-open diagnostics. It is currently a diagnostic/scene-graph contract; final WMO group
  submission remains in `WmoRenderer`.
- `src/core/WowViewer.Core.Runtime/World/SceneGraph/WorldScenePortalAdapter.cs` is the existing
  reader-owned adapter for decoded portal geometry and references. It correctly rejects malformed
  geometry without reading client files itself.
- `src/viewer/WoWViewer/Terrain/WorldScene.cs` already records separate WMO visibility and WMO
  submission timings and visible counts, which are sufficient for a first controlled comparison.

## Existing Camera, Collision, UI, and Logging Findings

- `src/viewer/WoWViewer/Rendering/Camera.cs` is a mutable free-fly camera with position, yaw, pitch,
  roll, and planar WASD movement. `ViewerApp.cs` owns input, mouse-look, frame update, and camera
  persistence.
- `WorldScene.TryResolveCameraPathCollision` already exposes terrain-height and conservative WMO
  placement-bound collision for camera paths. It intentionally permits free-fly inspection and is
  not yet a player-body physics contract.
- The viewer has a tabbed/dockable shell and many optional diagnostic windows. This supports a
  separate surface/profile without replacing the advanced data-explorer path.
- `src/viewer/WoWViewer/Logging/ViewerLog.cs` defaults console output to `Important` and keeps a
  bounded history, but every log call still enters the history lock before level filtering. The
  interactive profile should avoid formatting/emitting raw debug-route work at its callers and
  should make the retained diagnostic budget explicit rather than deleting forensic history.

## Decisions

1. Make native-style portal clipping the first implementation slice because it is independently
   testable, directly supported by existing decoded data, and addresses a concrete render-cost gap.
2. Keep the existing graph evaluator as a reusable diagnostic/validation seam, but do not make two
   unrelated portal authorities. The final plan must identify one shared visibility decision or a
   documented bridge between graph and renderer paths.
3. Add game-mode physics as a pure, testable runtime contract first; integrate input/camera wiring
   after the deterministic core is proven.
4. Implement the simple surface as an explicit UI profile that hides expensive/raw panels and
   selects the interactive diagnostic policy. Advanced inspection remains available by opt-in.
5. Treat runtime visual/FPS/audio proof against `H:\CLIENTS` as user-owned. Build and focused tests
   can establish structural correctness only.

## Open Research Boundaries

- The exact semantic of the native group flag tested by `StabPortals` remains inferred. The first
  implementation must use existing decoded flags only where current code already documents them,
  and retain a conservative fallback for ambiguity.
- The current WMO read model may not retain enough per-reference side information for exact native
  portal plane traversal in every client-era asset. If so, the implementation must prove a safe
  approximation and fall back to current behavior when side/geometry data is insufficient.
- Head attachment naming and collision shapes vary by M2/MDX era. The game-mode contract therefore
  needs a model-owned anchor provider with a finite height fallback, not a hardcoded attachment ID.
