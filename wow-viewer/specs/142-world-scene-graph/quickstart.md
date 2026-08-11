# Spec 142 Graph and Opt-In Traversal Quickstart

This slice proves the scene-graph, synthetic-workload, object-adapter, and runtime traversal
contracts. The per-ADT graph traversal is now default-on in the viewer, with `Use ADT Scene Graph`
available as a runtime fallback toggle. This quickstart does not launch the viewer, a GPU capture,
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

`Azeroth` tile `32_32` is the standard cross-client anchor for this diagnostic. When a tile is
specified, `profile-render` verifies that exact ADT exists in the configured client and positions
its production camera at the tile center instead of using the map's inferred startup camera.
While the profile is running, its requested JSON path contains `world-render-diagnostic-progress-v1`
with the last entered phase and completed-frame counts. It is replaced by the final
`world-render-diagnostic-v1` report only after the measured frames finish.

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
- External M2 spawns, skyboxes, WMO placements, and WMO-internal doodad-set submission remain
  outside this chunk-bucket slice.
- Unknown object bounds keep their bucket and map fail-open.
- The viewer project compiles with the default-on `WorldScene.UseHierarchicalSceneTraversal` seam
  and its reversible legacy-path toggle.
- Runtime stats expose graph active/inactive state, resident ADT graph roots, traversal
  visited/tested/rejected/skipped counts, AOI camera and retained counts, and the last unloaded ADT
  plus its WMO placement count.
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
