# Spec 142 Graph and Opt-In Traversal Quickstart

This slice proves the scene-graph, synthetic-workload, object-adapter, and runtime traversal
contracts. The per-ADT graph traversal is default-off in the viewer, with `Use ADT Scene Graph`
available as an explicit runtime investigation toggle. The flat path now uses conservative
maintenance-time tile/chunk buckets, and WDL GPU meshes stream around the camera's fog window.
This quickstart does not launch the viewer, a GPU capture,
a real-client load, training, or a long benchmark.

## Focused proof

From `I:\parp\parp-tools\wow-viewer`:

```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneGraphTests|FullyQualifiedName~SyntheticWorldWorkloadTests|FullyQualifiedName~WorldSceneGraphObjectAdapterTests|FullyQualifiedName~WorldScenePortalGraphTests|FullyQualifiedName~WorldScenePortalAdapterTests|FullyQualifiedName~WorldScenePortalViewVolumeTests|FullyQualifiedName~WorldScenePortalVisibilityEvaluatorTests"
```

The graph/traversal foundation can be checked together with:

```powershell
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~WorldSceneGraphTests|FullyQualifiedName~SyntheticWorldWorkloadTests|FullyQualifiedName~WorldSceneTraversalTests|FullyQualifiedName~WorldSceneGraphObjectAdapterTests|FullyQualifiedName~WorldScenePortalGraphTests"
```

The viewer integration seam can be compile-checked without launching a client or capture:

```powershell
dotnet build src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

The normal runtime now has two deliberately separate tile sets:

- `LastSelectedTiles` is the directional active submission set. Its size follows the ADT Detail
  Tiles control from 1 through 25 and gates detailed terrain, liquids, scene-graph work, and
  WMO/MDX object admission. The active tile and immediate 3×3 neighborhood are admitted before
  remaining budget expands through bounded forward-cone rings. The selector and camera tile
  conversion use the established 533.333-yard ADT span represented by `WoWConstants.ChunkSize`.
- `LastRetainedTiles` is the bounded camera-centered residency window. Its default radius is two
  tiles and the runtime control permits radius three. It controls streaming and unload protection;
  retained tiles do not become visible objects merely because they are resident.

The viewer diagnostics report `Active Tiles`, `Retained Tiles`, `Retained Radius`, and detailed
draw-call counts together. Compare those values when evaluating performance; a larger retained
count is expected without a corresponding increase in active submission.

## User-run production render diagnostic

`profile-render` is the first headless path that invokes the current production `WorldScene.Render`
loop; it is not the terrain-only native adapter or a 2-D preview. It opens a hidden OpenGL context,
so run it yourself against a named client and local WDT after the build/test proof above.

```powershell
$ClientRoot = "H:\CLIENTS\World of Warcraft Cata beta 11927"
$Wdt = "World\Maps\Azeroth\Azeroth.wdt"
$BuildLabel = "4.0.0.11927"
dotnet run --project tools/validation-capture/WowViewer.Tool.ValidationCapture/WowViewer.Tool.ValidationCapture.csproj -- profile-render --client-root $ClientRoot --map-input $Wdt --output output\diagnostics\azeroth-32-32.json --build $BuildLabel --tile-x 32 --tile-y 32 --warmup-frames 30 --frames 120
```

For a standard-era client, `--map-input` should be the WDT virtual path inside that same client,
not an unrelated extracted or custom map file. Local WDT input remains supported for Alpha clients
whose terrain adapter requires a disk path. Use `--load-all-tiles` only when intentionally profiling full terrain residency. The JSON schema is
`world-render-diagnostic-v1`; inspect `findings`, `stages`, `workload`, and the raw per-frame stats.
The command currently attributes CPU stages and client I/O counters. It explicitly reports that
per-stage GPU/driver timing is not yet captured.

The current full-map evidence is a failure, not a benchmark win: `Azeroth 32_32` on
4.0.0.11927 took 66.4 seconds to synchronously materialize 839 ADTs, and its `overlay` stage
stalled for roughly 40-44 seconds on alternating frames. Do not use `--load-all-tiles` as a normal
viewer startup path. Phase 8J first attributes/fixes overlay admission; Phase 8K then makes normal
whole-map operation index-first and budgeted. Keep the command available as an explicit stress
comparison after each bounded phase.

For a fresh implementation session, start with
[Phase 8J overlay recovery](phase-8j-overlay-recovery.md), not generic renderer optimization.
Its first slice is attribution only; it supplies the exact focused build/test and user-run capture
commands and blocks queues, caching, residency, and modern GPU work until one overlay owner is
proven dominant.

`Azeroth` tile `32_32` is the standard cross-client anchor for this diagnostic. When a tile is
specified, `profile-render` verifies that exact ADT exists in the configured client and positions
its production camera at the tile center instead of using the map's inferred startup camera.
While the profile is running, its requested JSON path contains `world-render-diagnostic-progress-v1`
with the last entered phase and completed-frame counts. It is replaced by the final
`world-render-diagnostic-v1` report only after the measured frames finish, or by a terminal
`status: failed` progress document containing the exception and stack trace if managed execution
fails before the report is written. The capture tool also registers an unhandled-exception
checkpoint for native access violations surfaced through CoreCLR; a hard fail-fast or external
termination can still stop before any failure document is possible.

## Full runtime-library proof

```powershell
dotnet build src/core/WowViewer.Core.Runtime/WowViewer.Core.Runtime.csproj -c Debug
dotnet test tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug
```

## Expected evidence

- Two builds from the same `fixture_name` and `seed` produce the same graph snapshot and node IDs.
- A nested WMO/M2/PM4 fixture has one reachable root, explicit parent ownership, and conservative
  bounds.
- Detaching a tile removes its complete subtree from lookup and enumeration.
- Invalid duplicate IDs, cycles, second-parent attachment, and non-finite rejectable bounds fail
  closed.
- A synthetic minimap/image record is not accepted as a synthetic world workload.
- A rejected synthetic region skips all descendant visibility tests and reports the skipped count.
- A node with unknown bounds is included without being passed to the visibility predicate.
- Existing `WorldObjectInstance` placements adapt into stable tile/external graph buckets.
- Client-backed WMO group summaries mount beneath their placement node as nested `WmoGroup` nodes;
  missing or malformed group bounds remain fail-open.
- Graph-only portal adjacency rejects malformed links and reports cycle, missing-entry, absent-data,
  and depth-limit fallback diagnostics without claiming portal geometry or renderer parity.
- Existing `WmoRenderDocument` portal read models adapt to stable group IDs and preserve valid
  geometry; malformed geometry and unknown groups remain explicit fallback cases.
- A child portal volume preserves its parent planes and adds doorway-cone planes; depth limits,
  unknown sides, degenerate edges, invalid geometry, and camera-on-plane cases return fallback
  diagnostics instead of narrowing visibility.
- With the opt-in graph selector, already-loaded WMO renderers can populate placement-keyed portal
  adapter diagnostics using the same nested group IDs; this is a compile-checked bridge only and
  does not change WMO visibility.
- The opt-in graph traversal evaluates reachable WMO groups through nested portal volumes; uncertain
  camera/portal data returns all groups, and WMO placement collection/submission remains unchanged.
- Resident non-skybox ADT M2 placements receive deterministic `map -> tile -> chunk -> M2` graph
  ownership; rejecting a known chunk skips its ordinary doodad descendants, while unknown bounds fail
  open.
- Traversal diagnostics attribute individually tested nodes, rejected subtree roots, and skipped
  descendants by node kind; the focused chunk proof reports one rejected Chunk and two skipped M2
  descendants.
- The opt-in traversal defers graph-level leaf visibility only for ordinary ADT M2 placements under
  Chunk nodes; the existing M2 collector still performs exact leaf visibility and asset-readiness
  checks.
- The opt-in graph set gives every resident ADT tile its own `Tile`-rooted scene graph. WorldScene
  traverses those graphs independently; external M2/WMO content remains in a separate graph.
- An ADT root retains authoritative finite tile bounds even if streamed M2/WMO descendants have not
  resolved bounds. It expands around all resolved placements, so an off-camera tile rejects its
  unresolved subtree before object visibility runs; ordinary child buckets remain fail-open.
- External M2 spawns, skyboxes, WMO placements, and WMO-internal doodad-set submission remain
  outside this chunk-bucket slice.
- Unknown object bounds keep their bucket and map fail-open.
- The viewer project compiles with the reversible `WorldScene.UseHierarchicalSceneTraversal` seam
  and its legacy-path toggle. The selector is default-off because real Azeroth captures showed
  graph traversal adding roughly 150 ms per frame to WMO/MDX visibility.
- WDL loading indexes the available low-resolution height tiles but only builds GPU meshes inside
  a fog-centered residency window; detailed ADTs continue to use the existing AOI stream.
- Runtime stats expose graph active/inactive state, resident ADT graph roots, traversal
  visited/tested/rejected/skipped counts, AOI camera and retained counts, and the last unloaded ADT
  plus its WMO placement count.
- The bounded directional tile contract exposes `TerrainManager.LastSelectedTiles`,
  `LastFrameActiveTileCount`, `LastFrameDetailedTileDrawCalls`, and
  `LastDirectionalTileInvariantPassed`. With verbose logging enabled, the render boundary emits
  the paired `Active Tiles` and `Detailed Draw Calls` values. Normal camera admission follows the
  ADT Detail Tiles control from 1 through 25; capture preloads and `--full-load` remain explicit
  exceptions. In this renderer's established coordinate contract, `WoWConstants.ChunkSize` is
  the 533.333-yard ADT span; `WoWConstants.TileSize` is a legacy 16-span aggregate and must not
  be used for camera ADT selection.
- WorldScene object admission consumes the selected camera tiles instead of traversing every
  resident ADT graph. The flat WMO/MDX collectors and deferred bounds promotion use the same gate;
  capture-preload tiles remain an explicit render-path lease. `--full-load` retains residency for
  stress analysis but does not make every resident tile object-visible.
- Residency-triggered graph rebuilds use cached WMO summaries only; a rebuild must not synchronously
  read or parse resident WMO files merely to mount optional `WmoGroup` metadata.
- Deferred WMO doodad model loads are advanced once per scene frame through `WorldAssetManager`,
  independent of the number of visible WMO placements.
- Minimap archive/loose-file reads use one background reader against the shared client data source;
  completed BLP textures still upload through the existing bounded render-thread queue.
- `profile-render` executes the production `WorldScene.Render` path in a hidden OpenGL context and
  emits every existing CPU-stage timer plus visibility/submission, queue, initialization, and
  client-read evidence. A terrain-only adapter or 2-D preview does not satisfy this proof.

## Not yet run by this phase

The four-scale performance ladder, runtime WMO portal submission/doorway parity, current-vs-new
traversal comparison, pass/query parity, per-stage GPU timing, and whole-map residency work remain
later user-run gates. Do not interpret this test/build pass as an FPS result.
