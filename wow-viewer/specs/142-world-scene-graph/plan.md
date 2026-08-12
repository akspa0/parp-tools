# Implementation Plan: World Scene Graph and Spatial Partitioning

**Branch**: `142-world-scene-graph` | **Date**: 2026-08-10 | **Spec**: [spec.md](spec.md)

## Summary

Phase 1 establishes the library-owned runtime contract that later renderer migration will consume:
a nestable scene graph with stable node identity, transforms, conservative bounds, attach/detach
ownership, and deterministic synthetic-world workload manifests. The first slice is deliberately
below the renderer: it proves graph structure, replayability, and invariant enforcement before
moving the existing `WorldScene` type collections behind the new traversal.

The synthetic fixture is a workload generator, not a second renderer. It produces the same node
and render-pass descriptors that a real client scene will use, while keeping payloads generated and
client-independent. Image-only synthetic minimaps remain outside this benchmark contract.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: `System.Numerics`, `System.Text.Json`, existing `WowViewer.Core.Runtime`
  contracts, xUnit test stack

**Storage**: In-memory graph and JSON-serializable workload manifest; no client data or generated
  scene assets are committed

**Testing**: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug`
  plus focused graph/manifest tests and `dotnet build` for the runtime project

**Target Platform**: Windows desktop viewer runtime; the core graph and fixture contracts remain
  platform-neutral

**Project Type**: Library-first runtime model with a later viewer integration

**Performance Goals**: Phase 1 must make resident/visible counts, graph depth, subtree rejection
  potential, and workload identity measurable without a benchmark-only renderer. It does not claim
  an FPS improvement until the graph is wired into real frame traversal.

**Constraints**: Do not rewrite existing format readers, do not launch GPU captures or long runs,
  do not hardcode client roots, do not make image-only minimap data satisfy renderer proof, and do
  not modify the legacy reference repository.

**Scale/Scope**: Support sparse maps, nested asset sub-graphs, four replayable synthetic scales,
  repeated asset references, portal descriptors, and explicit render-pass/material descriptors.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo independence**: PASS. New source and tests stay under `wow-viewer/`.
- **Library first**: PASS. The graph and workload model live in `WowViewer.Core.Runtime`; no
  viewer-only duplicate is introduced.
- **Real-data validation**: PASS for this phase. No real-data claim is made; real-client parity is
  a later gate and remains user-run.
- **Format ownership**: PASS. No parser or writer is changed.
- **Streaming/user-run heavy work**: PASS. The slice is in-memory and deterministic; it launches no
  training, harvest, GPU capture, or broad benchmark.
- **One phase at a time**: PASS. This plan stops after graph/manifest proof; renderer integration,
  spatial indexing, portal culling, and migration remain later phases.
- **Documentation/memory**: PASS. Spec 142, plan/tasks, quickstart, and memory-bank are updated
  with the slice.

## Project Structure

### Documentation

```text
wow-viewer/specs/142-world-scene-graph/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── synthetic-world-workload.schema.md
├── checklists/requirements.md
└── tasks.md
```

### Source and Tests

```text
wow-viewer/src/core/WowViewer.Core.Runtime/World/SceneGraph/
├── WorldSceneNode.cs
├── WorldSceneNodeKind.cs
├── WorldSceneGraph.cs
├── WorldSceneGraphSnapshot.cs
├── WorldSceneTraversal.cs
├── WorldSceneGraphObjectAdapter.cs
├── SyntheticWorldWorkload.cs
└── SyntheticWorldWorkloadBuilder.cs

wow-viewer/tests/WowViewer.Core.Tests/World/
├── WorldSceneGraphTests.cs
└── SyntheticWorldWorkloadTests.cs
```

**Structure Decision**: Keep graph ownership in the existing runtime library so the viewer,
validation capture, future spatial queries, and future renderer migration consume one contract.
The fixture builder stays beside the contract but has no OpenGL or client-file dependency. A later
CLI may serialize/replay manifests, but this phase does not create a parallel renderer tool.

## Phase 0 — Research Decisions

1. Preserve existing `System.Numerics.Matrix4x4` and min/max bounds conventions used by runtime
   world objects.
2. Use stable caller-provided node IDs rather than generated object references; this makes replay
   and current/new path comparison possible.
3. Treat a node's bounds as conservative world-space rejection bounds. A parent that cannot prove
   containment is marked non-rejectable instead of silently shrinking to its children.
4. Represent shared asset structure as a reusable fixture descriptor plus per-placement nodes; do
   not copy the entire asset tree for every placement in the first contract.
5. Keep portal links as graph metadata in Phase 1. Portal frustum construction and culling belong
   to the later traversal phase.

## Phase 1 — Graph and Workload Contract

**Status**: Complete for the bounded foundation slice on 2026-08-10. The runtime graph and
deterministic manifest build; focused proof is 8 passing tests. No renderer integration or GPU
performance claim is made.

1. Add immutable node identity/kind/render metadata and guarded parent/child attachment.
2. Add graph enumeration, lookup, subtree detach, world-transform propagation, and invariant
   validation.
3. Add the versioned synthetic-world workload manifest and deterministic builder for sparse tiles,
   nested WMO/M2/PM4 nodes, repeated assets, render-pass mix, and portal descriptors.
4. Add snapshot/count output sufficient to compare current and future traversal without running a
   GPU renderer.
5. Add focused tests for identity, bounds containment, cycle prevention, detach cleanup,
   deterministic replay, and explicit separation from image-only minimap workloads.

## Phase 2 — Conservative Shared Traversal

**Status**: Complete for the library slice on 2026-08-10. The traversal rejects a complete subtree
after one failed node test, preserves non-rejectable nodes, and reports visited/tested/skipped/
visible counts. Its graph-validation pass is optional so a graph proven at rebuild time does not
pay a full invariant walk on every frame.

1. Traverse one graph with an injected visibility predicate and a renderable-node selector.
2. Attribute the rejected region and count descendants skipped without visiting them.
3. Preserve unknown/incomplete bounds as visible-but-non-rejectable.
4. Prove the behavior with synthetic fixed-camera-style tests before viewer integration.

## Phase 3 — Runtime Object Adapter and WorldScene Traversal

**Status**: Complete for the bounded selector slice on 2026-08-10. Existing `WorldScene` object
lists can be adapted to `map -> tile/external bucket -> placement`, and client-backed WMO group
summaries mount as nested children when available. The selector uses one graph traversal before the
existing WMO/MDX visibility and asset-readiness checks, but real Azeroth captures measured roughly
68 ms of WMO visibility plus 82 ms of MDX visibility per frame on this path. The selector is now
default-off; the flat collectors remain the production runtime path until graph traversal has a
bounded measured cost. No FPS, GPU, portal-traversal, pass-order, or real-client parity claim is
made.

1. Adapt resolved `WorldObjectInstance` placements without reopening format readers; use existing
   client-backed WMO group summaries when available and fail open for malformed bounds.
2. Rebuild the graph only when object residency or resolved bounds change.
3. Use one conservative frustum traversal to feed the existing WMO and M2 collectors when
   `UseHierarchicalSceneTraversal` is explicitly enabled; keep the flat collectors as the default
   until real-scene cost is bounded.
4. Expose graph snapshot and traversal diagnostics for later validation-capture reporting.
5. Prove the adapter with stable IDs, tile grouping, unknown-bound fail-open, and replay tests;
   compile the viewer project to verify the integration seam.

## Phase 4 — Bounded Portal Adjacency Contract

**Status**: Complete for the graph-only adjacency slice on 2026-08-10. `WorldScenePortalGraph`
accepts stable group IDs and read-model links, rejects malformed edges without inventing
connectivity, and reports deterministic bounded traversal diagnostics for absent portal data,
cycles, missing entries, and maximum-depth fallback. Four new focused tests bring the Spec 142
graph proof to 19 passing tests.

This does not claim WMO portal geometry, doorway clipping, nested frustum construction, or
`WorldScene` integration. The existing `WmoRenderer` remains the runtime portal behavior owner
until an adapter can consume its existing read models without changing format readers.

## Phase 5 — WMO Portal Read-Model Adapter

**Status**: Complete for the read-model adaptation slice on 2026-08-10. `WorldScenePortalAdapter`
consumes existing `WmoRenderDocument` portal vertices, portal geometry, and group references,
maps them to stable graph IDs, preserves accepted geometry for later clipping, and converts
unknown groups or malformed geometry into explicit graph fallback diagnostics. Four adapter tests
bring the focused Spec 142 proof to 23 passing tests.

This still does not change a format reader, duplicate `WmoRenderer`'s current portal handling,
construct nested frusta, or enable graph portal traversal in `WorldScene`.

## Phase 6 — Bounded Portal View-Volume Contract

**Status**: Complete for the library-only nested-volume slice on 2026-08-10. The view-volume
builder preserves parent planes, adds a camera-to-doorway edge cone and destination-side portal
plane, and returns explicit fallback for depth limits, unknown sides, invalid geometry, degenerate
edges, and camera-on-plane cases. Three new focused tests bring the Spec 142 proof to 26 passing
tests.

This contract is not yet consumed by `WorldScene` or `WmoRenderer`, and it does not establish
doorway parity or a performance result.

## Phase 7 — Opt-In Runtime Portal Bridge

**Status**: Complete for the preparatory bridge slice on 2026-08-10. Loaded `WmoRenderer` portal
data is exposed through the existing graph read-model contract, and opt-in `WorldScene` graph
rebuilds cache placement-keyed portal adapters whose group IDs align with nested `WmoGroup`
children. The viewer build is the proof owner for this bridge; no visibility behavior changed.

Unloaded WMOs remain absent from the bridge and therefore fail open. Runtime nested-volume
traversal, renderer parity, and performance evidence are still not claimed.

## Phase 8 — Graph-Side Runtime Portal Traversal

**Status**: Complete for the opt-in diagnostic traversal slice on 2026-08-10. The portal
visibility evaluator finds the camera's containing group, walks graph adjacency through nested
portal volumes, and fail-opens to all graph groups when camera ownership, geometry, depth, or
portal data is uncertain. Opt-in `WorldScene` traversal applies this result to `WmoGroup` graph
nodes while preserving whole-WMO collection and legacy `WmoRenderer` submission.

This proves graph traversal mechanics only; it is not doorway parity or a performance result.

## Phase 8A — ADT M2 Doodad Chunk Partition

**Status**: Complete for the bounded opt-in object-population slice on 2026-08-10. Existing
resident, non-skybox ADT M2 placements now receive deterministic spatial-bucket metadata and
mount as `map -> tile -> chunk -> M2` nodes. Chunk bounds are unions of resolved placement bounds;
unknown members keep the chunk and its ancestors non-rejectable. The opt-in traversal can reject a
chunk before testing its ordinary doodad descendants, while external M2 spawns, skyboxes, WMOs, and
WMO-internal doodad-set submission remain unchanged.

The focused adapter proof is 7 passing tests and the runtime/viewer builds are clean apart from
existing repository warnings. This is still not a real-scene performance result.

## Phase 8B — Traversal Rejection Attribution

**Status**: Complete for the ADT M2 chunk evidence slice on 2026-08-10. Aggregate traversal counters do not
show whether the new terrain-chunk buckets actually saved individual M2 tests. This slice adds
per-node-kind counts for individually tested nodes, rejected subtree roots, and skipped descendants.
The proof owner remains the core traversal diagnostics; no renderer submission behavior changes.

## Phase 8C — Deferred ADT M2 Leaf Visibility

**Status**: Complete for the opt-in ADT M2 path on 2026-08-10. Graph traversal retains terrain-chunk subtree
rejection but defers exact visibility of M2 leaves under those chunks to the existing M2 collector,
preventing a duplicate graph-level leaf frustum test. Deferred leaf counts are attributed by kind.
No external spawn, skybox, WMO, WMO doodad-set, or renderer submission behavior changes.

## Phase 8D — Independent ADT Scene-Graph Roots

**Status**: Complete for the contract, viewer integration, and focused proof on 2026-08-10. The current opt-in
adapter no longer materializes one global map graph whose tile nodes own every ADT placement.
This slice changes the ownership boundary to one independently traversable `Tile`-rooted graph per
resident ADT, with external content kept separate; it does not rewrite terrain loading, WMO/M2
submission, or claim a measured performance win before capture.
The graph-set contract is the proof owner for partitioning; `WorldScene` is the integration proof owner.

## Phase 8E — Runtime Activation and Residency Diagnostics

**Status**: In progress on 2026-08-10. The per-ADT graph path is now default-on in `WorldScene`, with
a runtime toggle back to the legacy path for A/B diagnosis. Viewer runtime stats expose graph
activation and rejection counts, camera/retained ADT counts, and the last ADT unload with its WMO
placement count. This identifies whether a disappearing WMO is a residency event or an object-cull
event before changing retention policy. No runtime capture or measured performance claim is made.

1. Keep the selector transition invalidating graph state so toggling paths cannot reuse stale
   visibility or placement lists.
2. Expose graph roots, traversal counts, AOI camera/retention counts, and last WMO-bearing tile
   unload in existing investigation/runtime-stat surfaces.
3. Run the focused graph proof and viewer build; defer real-client A/B capture and any AOI/cull
   policy change until the user witnesses the diagnostics on the affected map.

## Phase 8F — Residency-Safe Graph Rebuild

**Status**: Complete on 2026-08-10. Graph rebuilds now use only cached WMO summaries when mounting
optional nested group nodes. A residency event no longer synchronously reads and parses every
resident WMO merely to rebuild graph metadata. Missing summaries remain fail-open and can be mounted
on a later graph rebuild after the asset is already loaded. The focused graph suite remains 34
passing tests; no runtime FPS claim is made by this phase.

## Phase 8G — Scene-Wide Deferred Asset I/O Bound

**Status**: Complete for the bounded runtime ownership slice on 2026-08-10. The graph rebuild path
remains metadata-only, and deferred WMO doodad model loading is now advanced once per `WorldScene`
frame through `WorldAssetManager`; it is no longer triggered by each visible WMO placement. The
minimap reader is also limited to one background client-data reader because it shares the active
`IDataSource` with terrain/object streaming.

This prevents placement count from multiplying synchronous client reads, but does not establish
that model parsing, terrain uploads, GPU submission, or driver wait are within budget. The next
proof owner is a user-run real-client capture using the existing stage and asset I/O diagnostics.

## Phase 8H — Production Headless Render Diagnostics

The diagnostic accepts either a local WDT or a standard-client virtual WDT path. Standard runs use
the latter so the WDT, ADTs, WMO/M2 assets, and minimaps are sourced from one client build; the
no-GPU `--dry-run` verifies that archive resolution before a renderer profile begins.
The canonical runtime anchor is `Azeroth` tile `32_32`; its tile coordinates are verified against
the same client catalog and directly determine the profiling camera position.
The requested output path is an in-progress `world-render-diagnostic-progress-v1` document until
the final report is complete, with stdout phase markers around data-source construction, GL setup,
scene construction, full-residency opt-in, warmup, and measurement. Progress writes are atomic;
managed failures replace the running document with a terminal `status: failed` record containing
the last completed frame and exception stack trace. A native/process-level crash may still leave
the last running checkpoint, which is evidence of the crash boundary rather than a completed run.
The `scene_maintenance` CPU stage owns the otherwise hidden pre-pass costs of PM4 completion and
instance/scene-graph rebuilding; it is the first follow-up attribution point after a 577.7 ms
tile-32_32 frame reported only ~12.8 ms across the prior named stages.

Before judging full-map frame cost, normal ADT success-path logging is verbose-only: console output
and its shared history lock must not become benchmark workload. Invalid or malformed ADT findings
remain visible by default.

**Status**: Complete for the CPU-stage diagnostic harness on 2026-08-10. The
`ValidationCapture` tool now has `profile-render`, which opens a hidden OpenGL surface and invokes
the actual `WorldScene.Render` loop after production scene construction. It records all existing
frame-stage timings, per-frame visibility/submission counts, streaming queues, initialization time,
and `MpqDataSource` cache statistics in `world-render-diagnostic-v1` JSON. The report also names
unsettled streaming, absent object-path coverage, CPU budget stalls, and the current missing
per-stage GPU timer-query attribution.

The focused runtime report tests and validation-capture build prove the contract and command
wiring. A named real-client profile remains user-run evidence; this phase neither launches that
profile nor claims GPU timing or an FPS improvement.

## Phase 8I — Authoritative ADT Root Rejection

**Status**: Complete for the graph contract and focused proof on 2026-08-10. A whole-Azeroth
diagnostic loaded 839 ADTs containing 243,585 M2 and 3,173 WMO placements, then spent 91.2 ms in
M2 visibility and 107.8 ms in WMO visibility despite admitting zero M2 instances. The cause was
contractual: any unresolved streamed child disabled `CanRejectSubtree` on its ADT root, forcing
all resident tiles into traversal.

Each per-ADT root now owns finite native tile bounds, expanded by resolved placement bounds, and is
the only node allowed to retain rejection authority when descendants are unresolved. This safely
rejects off-camera ADTs before their WMO/M2 buckets are visited while leaving ordinary buckets and
placements fail-open. The next proof owner is a user-run, post-fix full-map `profile-render` report;
no measured speedup is claimed until that report exists.

## Phase 8I.1 — Flat Collector Spatial Admission

**Status**: Implemented as a bounded runtime slice on 2026-08-11; real-client performance proof is
pending. The graph's `tile -> chunk -> placement` structure now guides maintenance-time flat
visibility buckets for resident WMO and M2 placements. The production render loop does not traverse
graph nodes: it rejects only conservative bucket AABBs, then delegates surviving instances to the
existing collectors. Unresolved bounds and cross-tile ownership remain fail-open. Viewer build
passes with 0 errors; no FPS improvement is claimed until a user-run capture measures the stages.

1. Build flat tile/chunk candidate buckets when residency or resolved bounds change.
2. Reject only whole buckets whose aggregate AABB is safely outside frustum/range; retain the
   existing per-instance visibility collector as the correctness authority.
3. Keep the graph selector default-off and compare flat-bucket and pre-bucket visibility counts on
   the canonical `Azeroth 32_32` client scene.

## Phase 8I.2 — Fog-Window WDL Residency

**Status**: Implemented as a bounded far-field residency slice on 2026-08-11; real-client frame
time and visual movement proof are pending. `WdlTerrainRenderer` now retains parsed WDL height data
as a compact CPU index and builds/evicts GPU tile meshes around the camera using the active fog range
with hysteresis. Draw-time frustum and fog-distance admission remain in place. Detailed ADT
streaming and object residency ownership are unchanged.

1. Keep WDL map inventory discoverable without constructing every tile's GPU mesh at world load.
2. Build a bounded number of nearest fog-window meshes per render frame.
3. Evict GPU meshes outside the retained fog window while preserving CPU WDL data for re-entry.
4. Preserve ADT hide/show transitions for WDL meshes that are built after detailed terrain arrives.

**Exit evidence**: the viewer build passes; a user-run `profile-render` or interactive movement
check must confirm that WDL residency stays near the fog window and that no visual seam/regression
appears while crossing tile boundaries.

## Phase 8I.3 — Tile-Windowed WL* Liquids and WDL Horizon

**Status**: Implemented as a bounded render-admission correction on 2026-08-11; real-client
frame-time and visual horizon proof remain user-owned.

1. Preserve map-wide WL* source discovery, but partition each logical liquid body's geometry into
   the existing 64x64 terrain tile buckets before GPU upload.
2. Enumerate only camera-window WL* buckets for the normal liquid pass; retain an explicit
   out-of-grid bucket for editor data that cannot be assigned safely.
3. Keep WDL GPU residency bounded by the existing fog/horizon window, but do not permanently
   remove a WDL tile merely because its detailed ADT is resident. WDL is an underlay and detailed
   terrain depth should win in the near field while the WDL mesh remains available at the horizon.
4. Keep synthesized minimaps terrain-only: their shared compositor already applies the selected
   solar direction, ambient, baked shadow, and optional cast-shadow signals. A skybox is a 3D
   world-composition concern and must not be painted into top-down terrain minimap pixels.

**Exit evidence**: viewer build passes; the user-run viewer confirms WL* visible draw counts track
the camera tile window, WDL survives as a distant horizon after ADT residency, and the sky dome
shows the active sun/moon composition without reintroducing the prior native draw crash.

## Phase 8J — Overlay Work Attribution and Admission

**Status**: `selection_bounds` was identified from user-run owner evidence on 2026-08-10/11 and is
now fixed, confirmed by a second user-run capture. Its
36.8-second sample prepared/submitted zero primitives, proving full-placement admission/filter work
as the immediate blocker. A follow-up report proved the visible-list loop was not sufficient: the
slow frames still had only 1,144-1,469 visible MDX and 3-11 WMO entries. The root cause was a
render-time `SelectedInstance` read synchronously rebuilding the full placement/scene-graph index
after deferred bounds promotion. That accessor now fails closed while dirty, and bounds promotion
updates existing graph nodes in place. The follow-up report shows 0.0021 ms P95 for
`selection_bounds` and 0.0068 ms P95 for coarse `overlay`; the remaining stress-path bottlenecks
are 66.2 s scene initialization, 93.7 ms P95 WMO visibility, and 85.7 ms P95 M2 visibility.
No-change/cache proof remains pending as a separate Phase 8J concern.

1. Split the `overlay` frame stage into named owner records with invalidation key, cache/rebuild
   result, input/output counts, and duration; add report-contract tests.
2. Identify the owner from a real `Azeroth 32_32` capture and add a focused no-change-frame test
   proving it does not rebuild without an invalidation event.
3. Introduce a bounded overlay work queue with a visible per-frame budget and deferred-work
   diagnostic; preserve the current output after convergence.
4. Add a cache-key contract that invalidates only on named map, camera, renderer-setting, or source
   content changes; fail visibly on ambiguous ownership.
5. Compile and run focused diagnostic/overlay tests; hand off one user-run real-client capture.

**Exit evidence**: no un-attributed overlay duration; unchanged frames do not run a full overlay
rebuild; the report proves whether overlay work is complete, cached, or deliberately deferred.

The fresh-session execution contract is [phase-8j-overlay-recovery.md](phase-8j-overlay-recovery.md).
It freezes the first slice to attribution only and lists the owner taxonomy, explicit non-goals,
cache/invalidation rules, later admission conditions, commands, and commit/documentation boundary.

## Phase 8K — Index-First, Budgeted Whole-Map Residency

**Status**: Planned; blocked by Phase 8J attribution. The 66.4-second `--load-all-tiles` workload
is useful stress evidence but not an acceptable normal startup model. This phase changes normal
map ownership from synchronous all-ADT materialization to lightweight full-map indexing plus
camera-prioritized, budgeted promotions.

1. Define `TileResidencyRecord` state transitions and invariant tests in the runtime library.
2. Build a full-map tile/bounds index without ADT decode, terrain mesh upload, or object-instance
   creation; prove inventory parity against the configured client map.
3. Route camera/AOI selection through an explicit priority queue and a per-frame CPU decode budget.
4. Separate CPU decode completion from bounded render-thread GPU upload and graph materialization.
5. Add retained-tile eviction that detaches graph, terrain, liquid, and object state together while
   preserving the index record.
6. Make `--load-all-tiles` an explicitly labeled stress admission mode with per-substage timing;
   it must not become viewer startup behavior.
7. Prove stream-in/out leak safety and target-tile visual parity; hand off a real-client movement
   capture after focused tests/build pass.

**Exit evidence**: map selection becomes interactive after index discovery; no normal frame drains
unbounded tile/object work; full-map stress reports decode, mesh, upload, object, and graph time
separately.

## Phase 8M — Strict Directional Tile Admission Baseline

**Status**: Source slice landed; real-client movement evidence is still required before widening
the visible tile cone. This is intentionally a smaller prerequisite than the later dense-submit
work and does not replace the existing WMO/M2 batching fallbacks.

1. Select the active tile and at most three immediately forward-facing neighbors with a pure,
   deterministic selector. The baseline uses a 45-degree cone half-angle and never searches a
   second ring or radial fog footprint.
2. Route normal ADT admission and retention through that selector. Fog distance and the manual
   detail control may reduce the set but cannot expand it beyond four normal tiles.
3. Keep explicit capture-path preload leases and `--full-load` stress mode separate and labeled;
   they are not normal camera-driven admission.
4. Expose paired active-tile and detailed-terrain-draw diagnostics at the render boundary and
   fail the baseline invariant if either normal count exceeds four.
5. Hand off a user-run real-client movement capture. Only after it passes may a later slice radiate
   admission outward inside the directional cone.

**Exit evidence**: focused selector tests pass; the viewer builds; a user-run normal movement
capture reports no more than four active detailed tiles and no more than four detailed terrain
draw calls per frame. FOV radiation, unique-model submission redesign, and FPS claims remain open.

## Phase 8L — Capability-Gated Modern Dense Submission

**Status**: Planned after Phase 8K and the Phase 8M baseline gate. This phase coordinates with Spec 138; it does not change 4.x
format/profile semantics. It consumes stabilized visible lists and residency states.

1. Add a renderer capability record and deterministic fallback selection tests.
2. Build shared immutable asset buffers plus a static-compatible instance-buffer path.
3. Add material/texture-array grouping only where the selected GL capability and asset contract
   permit it; retain current binding fallback.
4. Add multi-draw or indirect submission behind the capability record and count calls/state changes.
5. Keep animated, transparent, particle/ribbon, WMO-group, and unsupported-driver paths on named
   correctness fallbacks.
6. Compare dense 4.x and older-client scenes with exact visible identities before accepting a win.

**Exit evidence**: each modern path has a real-scene before/after report, GPU/driver attribution
where available, and a tested fallback with no unexplained visual difference.

## Later Phases (Not Started In This Slice)

- **Phase 9**: Integrate graph portal volumes into runtime WMO submission and prove doorway
  parity with the existing renderer.
- **Phase 10**: Per-pass visible/non-visible queues, shared animation update ownership, and query
  reuse.
- **Phase 11**: Terrain mesh/chunk graph migration beyond the ADT M2 object-population buckets,
  synthetic four-scale measurements, per-stage GPU timer-query attribution, and named real-client
  parity captures.

## Complexity Tracking

No constitution violations are required. The workload manifest is additional contract surface,
not a second runtime or renderer; it exists because synthetic image data and synthetic 3-D scene
data answer different performance questions.
