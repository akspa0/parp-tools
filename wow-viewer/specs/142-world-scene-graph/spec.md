# Feature Specification: World Scene Graph and Spatial Partitioning

**Feature Branch**: `142-world-scene-graph`

**Created**: 2026-08-10

**Status**: Draft

**Input**: User description: "Runtime scene graph and spatial partitioning architecture for the viewer, modeled on the client's CWorld scene graph (reference module 09). Replace the current type-partitioned flat per-instance culling with a unified hierarchical scene node model where the world is a scene graph of nested sub-scene-graphs: map -> tile -> chunk -> object node, and each asset (WMO with its groups, M2 with its attachments, PM4 structures) is itself a sub-graph mounted into a parent node. Provide a nestable view frustum stack with WMO portal-based occlusion culling, visible/non-visible node lists, state-sorted render pass buckets, and a single shared traversal used by both rendering and spatial queries (picking/raycast). Goal is to fix the god-awful CPU-bound performance of the current implementation and make the viewer able to hold whole maps."

## Why This Spec Exists

The viewer is CPU-bound on visibility, not on GPU work. The cause is structural, not a
collection of local inefficiencies: the world is partitioned **by object type** rather than
**by space**, so every frame the viewer walks type-specific flat collections and re-derives
visibility per item from scratch.

The observed baseline (measured 2026-08-10 against the current branch):

| Observation | Location |
|---|---|
| Terrain iterates a single flat chunk list spanning **all loaded tiles**, with no tile-level early-out; per-chunk distance and AABB tests run every frame regardless of tile visibility | `src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` chunk loop |
| Objects use a two-level scheme only: a per-tile bucket bounds test, then a **linear scan of every instance** in every visited tile | `src/viewer/WoWViewer/Terrain/WorldScene.cs` `CollectVisibleWmoInstances` / `CollectVisibleMdxBuckets` |
| Each surviving instance runs a cascade of seven-plus independent heuristics (bounds distance, vision cone, cone cull distance, no-cull radius, frustum AABB, max view distance, projected size, fade) | `src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs` |
| One global frustum with no nesting; no portal, occluder, or interior/exterior handling | `src/viewer/WoWViewer/Rendering/FrustumCuller.cs` |
| WMO portal data is fully parsed into read models but **never consulted for visibility** | `src/core/WowViewer.Core/Wmo/WmoPortal*.cs` |
| A WMO is culled as a single root volume; its groups are never culled individually | `src/viewer/WoWViewer/Rendering/WmoRenderer.cs` |
| Terrain, WMO, MDX, PM4, and taxi actors are parallel disjoint collections, each with its own duplicated visibility and submission logic | `WorldScene.cs` (~14,000 lines, single `Render` method body) |
| Picking walks its own separate traversal rather than reusing visibility results | `WorldScene.cs` `CollectSceneObjectPickHits` |
| No split between "visible" and "culled but still ticking" content; Spec 136 had to add renderer-level deduplication as a workaround for redundant per-instance animation updates | Spec 136 |

The reference client solves the same problem with a hybrid multi-layer spatial scene graph
(`.reference_data/4.0.0.11792/09_CWORLD_SCENE_GRAPH_DEEP_DIVE.md`): a global spatial grid whose
leaves are chunk nodes, one node base type (`CMapBaseObj`) that terrain chunks, WMO definitions,
WMO groups, and entities all derive from, a nestable frustum stack (`CWorldView`) that pushes a
clipped child frustum per visible portal, per-pass visibility queues sorted to minimize graphics
state changes, and a single spatial structure that also answers raycasts and sphere intersections.

No existing spec covers this. Spec 138 indexes six of the nineteen reference modules and module 09
is not among them; its performance requirement (FR-007) asks for batched or instanced **submission**
of already-visible instances, which is a different stage of the frame. Spec 136 is a narrow
two-line submission fix. Archived spec 045 built a scene-graph **tree view UI** over existing
collections, not a runtime graph. Archived spec 020 corrected cull **coordinates**, not structure.
Batching cannot fix a frame whose cost is paid before submission begins.

This spec owns the visibility and traversal architecture. Submission batching, shader work,
lighting, and format evolution remain owned by Specs 136 and 138.

## Performance Grounding and Workload Boundaries

The word "synthetic" currently describes more than one thing in this repository. Those
workloads are useful for different questions and MUST NOT be treated as interchangeable
performance evidence.

| Workload class | What it contains | What it can prove | What it cannot prove |
|---|---|---|---|
| `synthetic_world_scene` | Deterministic in-memory scene nodes, generated bounds, transforms, proxy geometry, materials, and optional portal topology | Spatial-index scaling, traversal cost, pass planning, query cost, and controlled CPU/GPU stress | Correct decoding, placement semantics, or parity with a real client build |
| `synthetic_minimap_asset` | Generated RGB, height, mask, or texture-preview data from the data/model pipeline | Image preparation, upload, texture display, and dataset-viewer cost | 3-D scene-graph, culling, WMO portal, or world-object performance unless the data is explicitly converted through the same world-runtime path |
| `real_client_scene` | A named map and client build loaded through the existing viewer runtime | Ground-truth visibility, format integration, coordinate correctness, and user-facing parity | Unbounded scaling beyond the available client content |
| `mixed_validation_scene` | Real client scene plus explicitly identified synthetic overlays or fixtures | Interaction between known real content and a controlled added load | A clean attribution unless real and synthetic nodes are separately counted |

The current viewer already exposes useful measurement anchors: `WorldScene` owns the frame
result and stage statistics, `TerrainRenderer` keeps separate tile/chunk residency and draw
counters, `WorldObjectVisibilityCollector` owns the object visibility cascade, and the
validation-capture path can render a runtime frame through the hidden-window GPU preview. These
are grounding points for instrumentation, not permission to assume that a statistic proves a
different stage. A synthetic minimap generated by the harvester is not a synthetic world scene
until it enters the same runtime scene and render path.

Every benchmark result MUST declare its workload class, scene source, and proof level. A result
from a synthetic world scene may establish a scaling trend; it may not close real-scene parity.
A result from an image-only synthetic minimap may establish a data or texture-preview result; it
MUST NOT be reported as evidence that the 3-D renderer is fast.

Before the new graph is allowed to claim an improvement, the existing path MUST have a recorded
baseline containing at least: repository commit, operating system, CPU/GPU identity, graphics
API and resolution, synchronization/v-sync state, client root and exact build for real scenes,
map and tile coordinates, resident tile/object counts, camera pose, enabled visibility settings,
warm-up policy, sample count, and separate CPU stage timings. When available, GPU time, driver
wait time, allocation volume, and upload time MUST be recorded separately rather than hidden in
one frame-time number.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One Node Model For Every World Element (Priority: P1)

As a viewer developer, I want every element of the loaded world — terrain chunks, placed WMOs,
WMO groups, M2 doodads, model attachments, PM4 structures, taxi actors, and overlays — to be the
same kind of scene node in one graph, so that visibility, transforms, bounds, lifetime, and
diagnostics are written once instead of re-implemented per content type.

**Why this priority**: Every other story in this spec depends on there being a single thing to
traverse. Today each content type carries its own bespoke visibility cascade, which is why a fix
applied to one type never benefits the others and why the render method has grown to fourteen
thousand lines.

**Independent Test**: Load a map with terrain, WMOs, doodads, and a PM4 overlay. Enumerate the
graph from its root and confirm every rendered element is reachable as a node with a resolved
world transform, world-space bounds, a node kind, and a parent, and that the enumerated node
count reconciles with the counts reported by the existing per-type diagnostics.

**Acceptance Scenarios**:

1. **Given** a loaded map, **When** the scene graph is enumerated from the root, **Then** each
   rendered element appears exactly once as a node carrying its kind, parent, world transform,
   and world-space bounds, and no rendered element exists outside the graph.
2. **Given** a placed WMO, **When** its node is inspected, **Then** its groups are child nodes
   with their own bounds, and its doodad set members are child nodes beneath the group that owns
   them, rather than being flattened into a global doodad list.
3. **Given** an asset that is instanced many times, **When** the graph is inspected, **Then** the
   asset's internal structure is shared across placements and only per-placement state is
   duplicated.
4. **Given** an unloaded or streamed-out tile, **When** its subtree is detached, **Then** every
   descendant node is removed from traversal in one operation, and no orphaned instance remains
   reachable from any visibility list.

---

### User Story 2 - Cull By Subtree, Not By Instance (Priority: P1)

As a viewer user, I want the cost of a frame to scale with what is actually near and visible
rather than with how much of the world is loaded, so that loading more of a map does not make the
viewer unusable.

**Why this priority**: This is the reported problem. The current traversal touches every loaded
terrain chunk and every instance in every visited tile every frame, so frame cost grows with
residency instead of with visibility.

**Independent Test**: Capture frame time and per-stage visibility timings on a fixed camera and
scene at increasing tile-residency counts. Confirm visibility cost tracks the visible node count
rather than the resident node count, and that a subtree rejected at a high level reports its
descendants as culled without those descendants being individually tested.

**Acceptance Scenarios**:

1. **Given** a node whose bounds fail the visibility test, **When** traversal reaches it, **Then**
   its entire subtree is rejected without evaluating any descendant, and the descendant count is
   attributed to the culled total.
2. **Given** a fixed camera and a growing number of resident tiles, **When** residency increases
   from one tile to a full map region, **Then** per-frame visibility time grows with the visible
   node count and does not grow proportionally with the resident node count.
3. **Given** a terrain tile fully outside the view volume, **When** the frame is drawn, **Then**
   none of its chunk nodes are individually distance- or bounds-tested.
4. **Given** a camera that does not move between two frames, **When** the second frame is drawn,
   **Then** visibility results are reused rather than fully recomputed, and any reuse is reported
   in diagnostics so it can be disabled for comparison.
5. **Given** the same scene rendered with the new traversal and with the current path, **When**
   the two outputs are compared, **Then** the visible content matches within a declared tolerance
   and any difference is attributable to a named, intentional behavior change.

---

### User Story 3 - Interiors Cull Through Portals (Priority: P2)

As a viewer user, I want to be able to stand inside a building and have only the rooms actually
visible through doorways be drawn, and to look out of a window without the entire exterior world
being submitted, so that dense interiors are as responsive as open terrain.

**Why this priority**: Portal data is already parsed and unused; interiors are where the current
whole-WMO cull is least selective. It is scoped after the traversal foundation because a portal
frustum is a nested traversal and has nothing to nest into until Story 2 lands.

**Independent Test**: Position the camera inside a multi-room WMO with known adjacency, and
verify from diagnostics that rooms not reachable through a visible portal chain are excluded, that
the frustum nesting depth is reported, and that walking through the building produces no popping
or missing geometry at portal boundaries.

**Acceptance Scenarios**:

1. **Given** a camera inside a WMO group, **When** the frame is culled, **Then** neighboring
   groups are visited only through visible portals, and each traversal step through a portal
   restricts the active view volume to that portal's opening.
2. **Given** a portal chain deeper than the supported nesting limit, **When** traversal reaches
   the limit, **Then** it terminates safely with a diagnostic record and does not omit geometry
   the user can see or recurse without bound.
3. **Given** a WMO with missing, malformed, or degenerate portal data, **When** it is rendered,
   **Then** the viewer falls back to whole-object visibility, records why, and never drops the
   building.
4. **Given** a camera looking from an interior through a window to the outside world, **When** the
   frame is culled, **Then** exterior content is restricted to the opening rather than being
   culled against the full screen frustum.
5. **Given** a camera transitioning through a doorway, **When** frames are captured across the
   transition, **Then** no frame shows a hole in the world or a room that should be occluded.

---

### User Story 4 - Visibility Results Are Ordered For Cheap Submission (Priority: P2)

As a viewer developer, I want traversal to produce pass-specific, state-ordered lists of visible
nodes, so that submission consumes a prepared plan instead of re-deriving grouping and ordering,
and so that batching work has a stable contract to build on.

**Why this priority**: It converts the traversal's output into the form the submission layer and
Spec 138's batching work need, and it removes the last reason for content-type-specific render
code paths. It is P2 because Stories 1 and 2 already deliver the primary frame-time win.

**Independent Test**: Render a scene containing opaque, alpha-tested, translucent, and liquid
content, then inspect the produced per-pass lists and confirm each visible node appears in exactly
the passes its materials require, ordered so that consecutive entries share render state wherever
possible; verify the resulting state-change count against the current path.

**Acceptance Scenarios**:

1. **Given** a culled frame, **When** the visibility result is inspected, **Then** it exposes
   separate ordered lists per render pass, and each visible node appears in exactly the passes its
   materials require.
2. **Given** a set of visible nodes sharing materials, **When** the pass list is produced, **Then**
   they are adjacent in submission order, and the frame reports its resulting state-change count.
3. **Given** content that is culled but still requires time-based updates such as animation or
   emitters, **When** the frame is processed, **Then** it appears on a separate non-visible list
   that is updated without being submitted for drawing, and each unique animated asset is advanced
   at most once per frame.
4. **Given** translucent content, **When** its pass list is produced, **Then** it is ordered for
   correct blending independently of the opaque ordering.

---

### User Story 5 - One Traversal Answers Picking And Spatial Queries (Priority: P3)

As a viewer user, I want clicking, hovering, teleporting, and measuring in the world to use the
same spatial structure the renderer uses, so that what I can select always matches what I can see
and query cost does not scale with world residency.

**Why this priority**: It removes a second, divergent traversal and a class of
"picked something that is not drawn" inconsistencies, but the viewer remains usable without it.

**Independent Test**: With a known scene, pick against terrain, a WMO group, and a doodad, and
verify each query returns the same node identity the renderer used, filtered by an explicit
content mask, with query cost independent of the number of resident tiles.

**Acceptance Scenarios**:

1. **Given** a ray from the camera, **When** a query runs, **Then** it walks the same spatial
   structure as rendering and returns node identity, hit position, and distance.
2. **Given** a query restricted to a content mask such as terrain only or objects only, **When**
   it runs, **Then** only nodes matching the mask are considered.
3. **Given** an object hidden by a user visibility toggle, **When** a query runs, **Then** the
   result honors the same visibility state the renderer honored.
4. **Given** a growing number of resident tiles, **When** the same query is repeated, **Then**
   query time does not grow proportionally with residency.

---

### User Story 6 - Synthetic Stress Is Grounded In The Real Renderer (Priority: P1)

As a viewer developer, I want deterministic synthetic world scenes that exercise the same
runtime traversal and submission contracts as real maps, so that I can expose scaling failures
without waiting for a particular client build while still knowing whether a result applies to
the renderer or only to the data-preview path.

**Why this priority**: The current renderer is already too slow on controlled synthetic content.
Without a repeatable workload ladder, a change can appear faster simply because it renders fewer
objects, bypasses the real traversal, or measures only image preparation. Synthetic stress must
make the cost visible and attributable before the graph is expanded to whole real maps.

**Independent Test**: Run the same deterministic workload manifest through the current traversal
and the new traversal at four resident-scene scales, with a fixed camera and fixed visible
region. Confirm that both paths report the same scene inventory and visible identities, and that
the report separates graph construction, residency/update work, visibility, pass ordering,
submission, GPU/driver wait, and spatial-query time.

**Acceptance Scenarios**:

1. **Given** a synthetic world-scene manifest and seed, **when** it is generated twice, **then**
   node counts, parent/child identities, bounds, transforms, material/pass descriptors, portal
   topology, and camera poses are identical.
2. **Given** a synthetic scene containing terrain chunks, WMO groups, M2 placements, repeated
   animated assets, PM4 overlays, and sparse tiles, **when** it is rendered, **then** those
   elements travel through the same graph, visibility, pass, and query contracts used by real
   content; a benchmark-only renderer is not allowed to stand in for the runtime path.
3. **Given** a fixed visible region and four increasing resident-scene scales, **when** the
   resident count grows, **then** the report shows which stage grows, how many subtrees were
   rejected, and whether cost tracks visible nodes or resident nodes.
4. **Given** a generated RGB or height minimap that is displayed without conversion into world
   nodes, **when** it is benchmarked, **then** the result is labeled `synthetic_minimap_asset`
   and cannot satisfy a scene-graph or renderer success criterion.
5. **Given** a synthetic result that improves scaling but a real client scene that loses visible
   content or exceeds its CPU budget, **when** the result is reviewed, **then** the real-scene
   failure blocks promotion and the synthetic result remains diagnostic only.

---

### Edge Cases

- A placement's bounds are unknown, degenerate, or zero-sized, so no meaningful subtree rejection
  can be made from them.
- A node's bounds are valid but its children extend beyond them, which would let a parent-level
  rejection wrongly discard visible children.
- An asset is still streaming when traversal reaches its node, so its real bounds and sub-graph do
  not yet exist.
- A tile is streamed out during traversal, or while its nodes sit in a visibility list.
- A camera sits exactly on a portal plane, or inside geometry, so interior and exterior
  classification is ambiguous.
- A WMO's portal graph is cyclic, disconnected, or references groups that failed to load.
- A single asset is placed so many times that per-placement node state itself becomes the
  dominant memory or traversal cost.
- Content is deliberately exempt from culling, such as the sky dome, debug overlays, or a
  currently selected object that must remain visible.
- PM4 overlay structures mount into a graph whose coordinate frame is established by a different
  file format.
- A map's tiles are non-contiguous or sparse, so the spatial structure must not assume a dense grid.
- A synthetic fixture contains uniform proxy geometry that is cheaper than a real asset, so its
  result could falsely imply that asset decoding or material submission was fixed.
- A synthetic minimap is shown in a 2-D preview surface and is incorrectly counted as a 3-D
  renderer benchmark.
- The CPU is idle while the GPU or driver is saturated, or the GPU is idle while CPU traversal
  dominates; one aggregate frame-time number obscures the limiting resource.
- A deterministic fixture is regenerated with a different seed, ordering, or camera and is
  incorrectly compared as though it were the same scene.
- Synthetic portal topology is valid but unlike any loaded WMO, so it proves stack mechanics but
  not portal correctness against client data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST represent the loaded world as a single hierarchical scene graph in
  which every renderable and queryable element is a node with a stable identity, a node kind, a
  parent, a world transform, and world-space bounds.
- **FR-002**: The graph MUST support mounting a sub-graph as a child of any node, so that an asset
  with internal structure — a WMO with groups and doodad sets, a model with attachments, a PM4
  structure — contributes its own graph rather than being flattened into a global list.
- **FR-003**: Sub-graph structure that is identical across placements MUST be shared, with only
  per-placement state duplicated; the memory cost of a placement MUST NOT scale with the asset's
  internal complexity.
- **FR-004**: The world MUST be spatially partitioned so that traversal can reject a region and all
  its contents in one test; the partition MUST NOT assume a dense or contiguous set of tiles.
- **FR-005**: Traversal MUST evaluate visibility hierarchically, and a rejected node MUST NOT cause
  any descendant to be individually evaluated.
- **FR-006**: Traversal MUST support a nestable view volume, so that a restricted volume derived
  from a portal or opening can be pushed for a sub-traversal and popped afterward, up to a declared
  maximum nesting depth that is enforced and reported.
- **FR-007**: Interior visibility MUST use WMO portal geometry to restrict traversal between
  groups, and MUST fall back to whole-object visibility with a recorded reason when portal data is
  absent, malformed, or exceeds the nesting limit.
- **FR-008**: Traversal MUST produce, per frame, ordered per-render-pass lists of visible nodes,
  and a separate list of culled nodes that still require time-based updates.
- **FR-009**: Per-pass lists MUST be ordered to group nodes that share render state, and the frame
  MUST report the resulting state-change and draw-submission counts.
- **FR-010**: Time-based updates for a shared asset MUST be performed at most once per frame
  regardless of how many placements reference it.
- **FR-011**: Spatial queries — including picking, hover, ray intersection, and volume
  intersection — MUST run against the same spatial structure and node identities used by rendering,
  and MUST accept an explicit content-kind mask.
- **FR-012**: Spatial queries MUST honor the same user visibility state the renderer honors, so a
  hidden object is not selectable and a visible object is always selectable.
- **FR-013**: Adding or removing map tiles, assets, or overlays MUST attach or detach a subtree as
  a single operation, leaving no reachable orphaned node and no stale entry in any visibility list.
- **FR-014**: A node whose bounds are missing, degenerate, or not yet known MUST be handled by an
  explicit declared policy rather than by silent inclusion or silent omission, and the policy's
  effect MUST be visible in diagnostics.
- **FR-015**: A parent's bounds MUST be guaranteed to contain its descendants' bounds, or the
  parent MUST be marked as non-rejectable; the viewer MUST NOT discard visible geometry through a
  parent-level rejection.
- **FR-016**: The viewer MUST expose per-frame traversal diagnostics including nodes visited,
  nodes rejected with the level at which rejection occurred, subtree rejections, visible counts per
  pass, nesting depth reached, and the time spent in each traversal stage.
- **FR-017**: Frame-to-frame reuse of visibility results MUST be correctness-preserving, MUST be
  invalidated by camera, transform, residency, or visibility-state changes, and MUST be
  independently disableable for comparison measurement.
- **FR-018**: The new traversal MUST be selectable against the existing path at runtime for the
  duration of the migration, and both paths MUST report comparable visibility and frame statistics
  so that parity and performance can be measured on the same scene.
- **FR-019**: Visible content under the new traversal MUST match the current path on a named set of
  comparison scenes within a declared tolerance; any intentional difference MUST be named and
  justified rather than absorbed as noise.
- **FR-020**: Every performance claim MUST be supported by before-and-after measurements on a named
  real scene with a recorded camera, client build, and residency count, and MUST identify the
  limiting stage.
- **FR-021**: The scene graph MUST be the authoritative runtime model of the loaded world, and
  inspection surfaces such as the scene tree, object lists, and selection panels MUST read from it
  rather than maintaining parallel collections.
- **FR-022**: Culling behavior MUST be attributable: for any element the user expects to see but
  does not, diagnostics MUST identify which node rejected it and by which test.
- **FR-023**: Every performance workload MUST declare one of the workload classes
  `synthetic_world_scene`, `synthetic_minimap_asset`, `real_client_scene`, or
  `mixed_validation_scene`; reports MUST NOT merge results from different classes without
  preserving per-class metrics.
- **FR-024**: A `synthetic_world_scene` benchmark MUST exercise the same scene-node, spatial
  traversal, visibility-result, render-pass, and spatial-query interfaces used by a real client
  scene. A separate benchmark-only renderer or theoretical node-count loop MUST NOT satisfy this
  requirement.
- **FR-025**: Synthetic world-scene fixtures MUST be parameterized by at least resident tile or
  region count, chunks per tile, WMO placements and groups, M2 placements, repeated asset count,
  PM4 overlay count, portal topology, render-pass/material mix, animation/update load, and camera
  pose. The selected values MUST be recorded in the workload manifest.
- **FR-026**: Synthetic world-scene generation MUST be deterministic from a versioned fixture
  schema and explicit seed. The generated inventory, transforms, bounds, portal links, camera
  path, and expected visible identities MUST be serializable for replay.
- **FR-027**: Benchmark capture MUST report separate warm-up and steady-state distributions for
  graph build, attach/detach or streaming, per-frame update, spatial visibility, portal
  traversal, pass ordering, opaque submission, transparent ordering/submission, GPU/driver wait,
  and spatial queries. At minimum, median and 95th-percentile values MUST be retained; one FPS
  or total-frame-time value is insufficient.
- **FR-028**: Current-path and new-path comparisons MUST consume the same workload manifest,
  camera sequence, feature toggles, residency state, and render resolution. A change in scene
  content or culling settings MUST invalidate the comparison rather than being silently accepted.
- **FR-029**: A `synthetic_minimap_asset` result MUST be labeled as data-preparation or
  texture-preview evidence and MUST NOT be used to close scene-graph, culling, portal, or
  world-object performance requirements unless it is explicitly promoted through the same
  world-runtime scene path.
- **FR-030**: Every benchmark report MUST identify the limiting stage and resource owner — CPU
  traversal, CPU submission preparation, GPU execution, driver synchronization, asset upload, or
  query — and MUST retain enough counters to reproduce that attribution.
- **FR-031**: Synthetic performance improvements MUST NOT override real-scene correctness or
  parity. Promotion requires both a passing synthetic scaling gate and passing named real-client
  comparison scenes; if only the synthetic gate passes, the result remains diagnostic.
- **FR-032**: Benchmark reports MUST record repository commit, workload schema/version, fixture
  seed, device/runtime settings, scene source or client provenance, camera, residency, node
  counts, visible counts, draw/state counts, and all enabled culling/portal/reuse policies.
- **FR-033**: The viewer MUST provide an explicit diagnostic state for "not a renderer
  benchmark" when a workload measures only image decoding, tensor preparation, texture upload,
  or a 2-D preview surface.

### Key Entities *(include if feature involves data)*

- **Scene node**: The single element type in the graph. Carries identity, kind, parent, children,
  local and world transform, world-space bounds, visibility state, and the payload needed to render
  or query it. Terrain chunks, WMO placements, WMO groups, model placements, attachments, PM4
  structures, and overlays are all node kinds.
- **Spatial region node**: An interior node whose purpose is rejection rather than rendering — the
  map root, a tile, and a chunk-level grouping. Owns bounds that contain all descendants.
- **Asset sub-graph**: The internal node structure implied by a file format, mounted under a
  placement node and shared across all placements of that asset.
- **View volume stack**: The nestable set of active view volumes for the frame, with the base
  camera volume at the bottom and portal-restricted volumes pushed above it, bounded by a maximum
  depth.
- **Visibility frame**: The per-frame result of traversal — ordered per-pass visible lists, the
  non-visible-but-updating list, and the traversal diagnostics.
- **Spatial query**: A ray or volume request with a content-kind mask, answered from the same
  structure, returning node identity and hit information.
- **Traversal diagnostic record**: Per-frame counters and timings, including per-level rejection
  attribution, sufficient to explain both frame cost and any missing content.
- **Performance workload manifest**: A versioned description of the workload class, scene source,
  fixture seed, resident content dimensions, camera sequence, render settings, culling policies,
  and expected inventory used for a replayable comparison.
- **Performance sample**: One warm-up or steady-state observation with CPU stage timings, GPU or
  driver timing when available, allocations/uploads, node and pass counts, and limiting-stage
  attribution.
- **Synthetic world-scene fixture**: A deterministic generated scene that uses runtime node and
  render contracts while making its content counts, bounds, transforms, portals, materials, and
  update load explicit.
- **Grounding record**: The provenance and proof-level declaration that says whether a result is
  synthetic scaling evidence, image/data-preview evidence, mixed evidence, or real-client parity
  evidence.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a named dense scene that currently runs below 10 frames per second with doodads
  enabled, the viewer sustains at least 60 frames per second at the same camera, residency, and
  visual settings.
- **SC-002**: With the camera fixed, increasing resident tiles by a factor of ten increases
  per-frame visibility time by no more than a factor of two.
- **SC-003**: The viewer holds a full real map region resident and remains interactive — at least
  30 frames per second and responsive camera control — at a residency count at least ten times the
  largest the current path sustains at the same frame rate.
- **SC-004**: On a named dense scene, the number of nodes individually visibility-tested per frame
  is reduced by at least 90% relative to the current path, with the reduction attributed to
  subtree-level rejections in diagnostics.
- **SC-005**: On a named interior scene, enabling portal traversal reduces submitted geometry by at
  least 50% relative to whole-object WMO visibility, with no frame in a scripted walkthrough
  showing missing or wrongly visible geometry.
- **SC-006**: Across a named set of at least five comparison scenes spanning early, 1.x/3.x, and
  4.x clients, the set of visible elements under the new traversal matches the current path within
  the declared tolerance, with zero unexplained differences.
- **SC-007**: Time-based updates for shared assets are performed exactly once per unique asset per
  frame, verified on a scene with at least 500 placements of a single animated model.
- **SC-008**: Render state changes per frame on a named dense scene are reduced by at least 50%
  relative to the current path at equal visible content.
- **SC-009**: Picking and hover queries return the same node identity the renderer used in 100% of
  a scripted set of at least 20 picks spanning terrain, WMO groups, doodads, and PM4 structures,
  and query time does not grow more than 20% when residency grows tenfold.
- **SC-010**: For every element reported missing during comparison testing, diagnostics identify
  the rejecting node and test in 100% of cases, with no unattributable disappearances.
- **SC-011**: Streaming a tile in and out 100 times leaves no reachable orphaned node, no stale
  visibility-list entry, and no net growth in resident node count.
- **SC-012**: A versioned synthetic world-scene workload can be replayed with the same seed and
  produces identical inventory, bounds, transforms, portal topology, camera poses, and expected
  visible identities in two independent runs.
- **SC-013**: The synthetic performance ladder contains at least four resident-scene scales and
  reports median and 95th-percentile timings for every declared stage, with CPU traversal,
  submission preparation, GPU/driver wait, and query cost separately attributable.
- **SC-014**: On each synthetic scale, the current and new paths consume the same manifest and
  produce identical visible node identities and pass membership before any intentional behavior
  change is accepted.
- **SC-015**: Any image-only synthetic minimap benchmark is reported as `not_renderer_benchmark`
  and cannot mark FR-004 through FR-012, FR-016, or SC-001 through SC-009 as passing.
- **SC-016**: A synthetic scaling improvement is promotable only when at least one named dense
  real-client scene and one named interior real-client scene pass visibility parity and their
  declared CPU stage budgets; synthetic-only wins remain diagnostic.
- **SC-017**: For every benchmark run, the report identifies one limiting stage and resource
  owner, and no performance claim is accepted when more than 10% of steady-state samples lack
  the timing needed to support that attribution.

## Validation Gates and Evidence Order

The work proceeds through evidence gates so that a fast synthetic fixture cannot conceal a
regression in the actual viewer.

1. **Baseline gate**: Record the current path on named real scenes and the existing frame-stage
   counters before changing traversal ownership. Include a dense exterior, an interior WMO, and a
   sparse or partial-tile scene.
2. **Synthetic identity gate**: Add the versioned fixture manifest and deterministic replay check.
   Prove that the fixture uses the same runtime scene and render interfaces; do not use an
   image-only minimap as a substitute.
3. **Synthetic scaling gate**: Run the fixed-camera resident-scale ladder and identify whether
   the limiting stage is traversal, submission preparation, GPU work, synchronization, or upload.
   This gate measures scaling, not client parity.
4. **Real parity gate**: Run the same camera/scripted comparison scenes through the current and
   new paths using configured client roots, exact build identities, and captured residency. No
   unexplained missing or extra content is acceptable.
5. **Promotion gate**: Promote the new path only when the synthetic scaling gate and real parity
   gate pass together. A passing unit test, a faster proxy fixture, or a good image-preview frame
   alone is not promotion evidence.

Long-running real-scene captures and GPU measurements are user-run operations. The implementation
must provide PowerShell-ready commands and machine-readable reports; it must not silently launch
training, harvesting, or broad capture work as part of ordinary tests.

## Assumptions

- The new traversal is developed behind a runtime selector alongside the existing path and
  migrates content type by content type, rather than replacing the current render path in one
  change. The current path remains the parity reference until every comparison scene passes, and
  is removed only after that.
- "Whole map" means every tile a real map actually defines, which is far fewer than the theoretical
  64-by-64 grid; sparse and non-contiguous tile sets are the normal case.
- Level-of-detail selection, streaming policy, and asset residency budgets are inputs to this
  architecture, not part of it. This spec decides *what is visible*; it does not decide what is
  loaded or at which detail level.
- Submission batching and instancing, shader work, lighting, fog, and format evolution remain owned
  by Specs 136 and 138. This spec delivers the ordered per-pass lists those efforts consume, and
  must not restate their requirements as its own.
- Ground clutter instancing in the manner of the client's detail-doodad batching is a consumer of
  this architecture and is deliberately out of scope here.
- The reference module is evidence about how the client structured the problem, not a contract to
  reproduce its class layout, its 32-deep frustum limit, or its exact queue set. Structural
  decisions must be justified by measurement on this viewer, not by resemblance to the decompiled
  original.
- Existing format readers are complete and are not reopened by this work; the graph is built from
  what they already produce.
- WMO portal read models already exist and are sufficient to drive portal traversal; if they are
  not, closing that gap is part of Story 3 rather than a new format effort.
- Performance measurement scenes, client builds, and camera positions are named and recorded per
  the project's real-data validation requirement; long captures and heavy scene runs are executed
  by the user with handed-off commands.
- Synthetic world fixtures are project-owned controls and may use generated proxy payloads, but
  they must pass through the same runtime graph and renderer contracts. They are not a replacement
  for client-backed parity scenes.
- Synthetic minimap image generation and 2-D dataset preview are separate data/texture workflows.
  They become renderer evidence only if an explicit adapter feeds them through the same 3-D world
  runtime, with that adapter and its limitations recorded.
- Existing validation-capture or hidden-window GPU paths are measurement surfaces only after the
  report proves which runtime stages they exercise; a capture wrapper is not assumed to represent
  the full interactive viewer.
- PM4 overlay structures mount into the graph using the coordinate frame already established for
  them; this spec does not revisit PM4 coordinate resolution.
