# MdxViewer Renderer Performance Plan

Date: 2026-03-31
Target: `gillijimproject_refactor/src/MdxViewer`
Priority: render-engine performance first, lighting completion second, graveyard overlay after renderer work

## Goal

Make camera movement on real world maps usable again by reducing per-frame draw-call and state-churn cost in the active `MdxViewer` world path.

This plan is intentionally about the active renderer in `WorldScene`, not a speculative future engine. The first job is to make the existing world path collect visibility once, own explicit render layers, and submit less redundant work every frame.

## Current State

The current hot path is still centered in `src/MdxViewer/Terrain/WorldScene.cs`:

- `Render(...)` resolves lighting, renders sky or WDL or terrain, collects visible WMOs, updates MDX animation, collects visible MDXs, runs opaque MDX, runs liquids, sorts transparent MDX, then continues into PM4/debug/overlay work.
- `RenderQueue.cs` and `FrustumCuller.cs` exist under `src/MdxViewer/Rendering`, but only `FrustumCuller` is active in the world path today.
- visibility collection already improved in narrow slices:
  - visible WMO scratch bucket exists
  - visible MDX scratch bucket exists
  - WMO doodad loading is already deferred incrementally
- `LightService` already feeds shared world lighting state through `TerrainLighting`, but that does not solve the bigger frame-cost problem by itself.
- area POI and taxi overlays already exist as lazy-loaded world/minimap overlays, so graveyards from `WorldSafeLocs.dbc` should be implemented as a sibling overlay only after the main render frame is stabilized.

## Main Problems To Solve

1. `WorldScene.Render(...)` still owns too many responsibilities and too much GL state choreography.
2. Submission is only partly normalized:
   - terrain has its own pass ownership
   - WMO visibility is collected once, but WMO renderer-local pass order still hides scene-wide submission control
   - MDX batching is still narrow and renderer-local
3. Draw-call and state churn are not measured explicitly enough to target the worst offenders with confidence.
4. Overlays and debug/editor layers still live too close to the main world submission path.

## Non-Goals For The First Slice

- not a shader-family rewrite
- not a full historical visual-parity rewrite
- not a `wow-viewer` migration task
- not a graveyard feature slice yet
- not another PM4-focused pass unless PM4 overlays are proven to be part of the active frame-cost bottleneck

## Phase Order

### Phase 1: Instrument The Active Frame

Purpose: stop guessing where frame time goes.

Work:

- add per-frame counters and timings for:
  - terrain render
  - WDL render
  - WMO visibility collection
  - WMO submission
  - MDX animation update
  - MDX visibility collection
  - MDX opaque submission
  - liquid render
  - MDX transparent submission
  - deferred asset-load drain
  - PM4/debug/editor overlay submission
- count visible instances and draw submissions per layer
- count renderer-local batching effectiveness for MDX
- expose a compact renderer stats block in the viewer so dense-map flythroughs can be compared before and after each slice

Primary files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/ViewerApp.cs`

Exit condition:

- on the development map, the viewer reports concrete per-layer timings and counts during live movement

### Phase 2: Build An Explicit World Render Frame

Purpose: pull visibility and submission ownership out of the current monolithic `Render(...)` flow.

Work:

- introduce a per-frame world render contract, for example `WorldRenderFrame` or equivalent, owned by `WorldScene`
- fill it once per frame with explicit buckets such as:
  - opaque terrain or WDL
  - opaque WMO shell
  - opaque MDX instances
  - liquids
  - transparent WMO shell
  - transparent MDX instances
  - late overlays and debug
- keep the existing visible-WMO and visible-MDX scratch lists as seed inputs instead of throwing them away
- move layer ordering into explicit scene-owned sequencing instead of leaving it implicit inside one long method

Primary files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/RenderQueue.cs` or a replacement contract in the same area

Exit condition:

- `WorldScene.Render(...)` becomes an orchestration method over a named frame object rather than the place where every visibility and submission decision lives inline

### Phase 3: Batch MDX Submission By Compatible Renderer Or State

Purpose: reduce redundant MDX world work during camera movement.

Work:

- keep one visibility pass, but stop treating the first visible renderer as the de facto batch owner
- batch compatible MDX instances by renderer/material/state requirements instead of one long visible-instance loop
- split explicitly between:
  - batched opaque MDX
  - unbatched opaque MDX
  - batched transparent MDX
  - unbatched transparent MDX
- only sort the transparent subset that actually needs sorting
- preserve current conservative M2 compatibility mode while restructuring submission; do not reopen richer M2 semantics in the same slice

Primary files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/MdxRenderer.cs`
- `src/MdxViewer/Rendering/ModelRenderer.cs`
- `src/MdxViewer/Rendering/WarcraftNetM2Adapter.cs` only if required to avoid regression, not as a design target

Exit condition:

- live counters show fewer MDX submissions or reduced opaque/transparent world-pass churn on the development map

### Phase 4: Split WMO Scene Submission From WMO Renderer-Local Pass Ownership

Purpose: stop WMO rendering from remaining a large black box inside the scene frame.

Work:

- preserve current WMO correctness first, then expose separate WMO scene submissions for shell or liquid or transparent work where practical
- keep deferred doodad loading and current culling wins intact
- move toward scene-level ownership of WMO opaque versus liquid versus transparent ordering so WMO no longer blocks a cleaner world render frame
- do not attempt cross-WMO material batching before the frame/layer split exists and is measured

Primary files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Rendering/WmoRenderer.cs`
- `src/MdxViewer/Terrain/WorldAssetManager.cs`

Exit condition:

- WMO shell/liquid/transparent cost is visible separately in frame stats, and at least one WMO-local pass boundary has moved under scene-level ownership

### Phase 5: Move PM4, Debug, And Editor Layers To Explicit Late Passes

Purpose: make exploration tooling stop polluting the main world frame.

Work:

- isolate PM4 overlays, debug bounds, archaeology overlays, POIs, taxi helpers, and similar tools into explicit late layers
- ensure these paths can be toggled without causing hidden work in the main opaque or transparent world passes
- verify that overlay visibility and hit-testing do not force unnecessary render-side scene walks

Primary files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/ViewerApp.cs`

Exit condition:

- turning off overlays measurably removes their work from frame counters rather than only hiding pixels

### Phase 6: Finish DBC Light Integration After The Render Frame Is Stable

Purpose: improve lighting correctness without hiding structural performance problems.

Work:

- keep `LightService` as the shared source of zone light state
- verify all relevant world passes consume the same frame light data instead of partially bypassing it
- expand this only after render-layer ownership is explicit, so lighting work is not mixed with the performance refactor
- treat fog distance, zone overrides, and time-of-day as shared frame inputs, not per-renderer side decisions

Primary files:

- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/Terrain/TerrainLighting.cs`
- `src/MdxViewer/Terrain/LightService.cs`
- relevant renderer files as needed

Exit condition:

- world passes read one shared lighting contract consistently, and lighting completion is no longer blocked by render-order ambiguity

### Phase 7: Add Graveyard POIs From WorldSafeLocs.dbc

Purpose: add the requested feature only after the frame can absorb more overlays cleanly.

Work:

- implement a `WorldSafeLocs`-backed loader parallel to `AreaPoiLoader` and `TaxiPathLoader`
- add graveyard markers as a sibling overlay in world view and minimap
- reuse the existing lazy-load and toggle model used by area POIs and taxi paths

Primary files:

- new loader under `src/MdxViewer/Terrain`
- `src/MdxViewer/Terrain/WorldScene.cs`
- `src/MdxViewer/ViewerApp.cs`

Exit condition:

- graveyards are their own optional overlay and do not add hidden per-frame cost when disabled

## Suggested First Implementation Slice

The first actual code slice should be `Phase 1` plus the smallest viable `Phase 2` setup:

- add frame counters and timers
- introduce an explicit frame data contract in `WorldScene`
- move existing visible WMO/MDX scratch ownership under that contract
- leave renderer internals mostly alone in that first landing

That is the lowest-risk path that gives immediate measurement value and sets up the batching/culling work the user actually asked for.

## Validation Rules

Builds are necessary but not sufficient.

Each phase should be validated on real viewer data, especially:

- `gillijimproject_refactor/test_data/development/World/Maps/development`
- the user’s normal camera-movement workflow on dense terrain/object areas

At minimum, capture before/after evidence for:

- frame time while moving the camera
- visible WMO/MDX counts
- draw submissions by layer
- deferred asset-load drain cost
- overlay-on versus overlay-off cost where relevant

Do not describe any phase as complete based only on compile success or synthetic tests.

## Recommended File Order

1. `src/MdxViewer/Terrain/WorldScene.cs`
2. `src/MdxViewer/Rendering/RenderQueue.cs` or a replacement frame contract
3. `src/MdxViewer/ViewerApp.cs` for stats visibility
4. `src/MdxViewer/Rendering/MdxRenderer.cs`
5. `src/MdxViewer/Rendering/WmoRenderer.cs`
6. overlay loaders and UI after the renderer frame is stable

## Decision Summary

- renderer performance comes first
- batching/culling work should be grounded in the active `WorldScene` path, not in dormant abstractions alone
- DBC lighting is already partly wired and should be finished after layer ownership is explicit
- graveyards belong after renderer cleanup as a sibling overlay to area POIs and taxi paths