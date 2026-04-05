# wow-viewer Reference Renderer Performance Plan

Date: 2026-04-04
Target: `wow-viewer`
Status: active
Priority: renderer architecture and real-data performance first, consumer cutover second

## Goal

Turn `wow-viewer` into the canonical C# reference renderer for old WoW data, not just a parser stack.

That means one version-aware engine that can:

- read Alpha-era and later 3.x-era assets through shared format ownership
- preserve genuine historical rendering artifacts instead of flattening them away
- render the same fixed real maps repeatedly with stable, measurable performance
- become the design owner for world rendering, M2 rendering, WMO rendering, terrain, liquids, sky, and lighting

The target is not another compatibility-only viewer. The target is a real runtime that can serve as the implementation reference for this data family in C#.

## Why This Plan Exists

- current `MdxViewer` work proved that surface-level fixes are not enough; the remaining bottlenecks are architectural
- load-time improvements alone still leave the active world path with very poor standstill frame rate because visibility admission, animation update, submission, and renderer-local pass ownership are still too expensive
- current `MDX batched` counters in the active viewer are not true GPU instancing; they mostly reflect shared shader setup while still issuing per-instance draw loops
- current `WMO` rendering is also still renderer-local and per-instance, which blocks scene-level pass control and broader batching
- `wow-viewer` already has the right repo shape and the first runtime seams; it now needs an integrated renderer program instead of more isolated parser slices

## Core Principles

### 1. One Engine, Profile-Driven

Do not build separate Alpha and Wrath renderers.

Build one runtime with explicit per-build or per-era profiles for:

- format interpretation
- material and effect routing
- skin-profile behavior
- lighting and fog rules
- runtime flags and submission behavior

### 2. Artifact Fidelity Is A Feature

The engine should preserve real historical behavior when it is known, even when that makes the runtime less generic.

Do not erase:

- unresolved flags
- build-specific skin behavior
- old effect-family differences
- early-format lighting and fog differences
- WMO or M2 material routing differences across eras

### 3. Performance Is Architecture, Not Cleanup

The goal is not to keep patching hotspots in a monolithic `WorldScene`-style host.

The goal is to own:

- explicit frame contracts
- explicit visibility services
- explicit render-entry families
- explicit scene submission and batching seams
- explicit residency and streaming policy

### 4. Real Data Only

No phase is complete because a build passed or a synthetic fixture rendered.

Every meaningful renderer phase needs fixed real-data proof from:

- Alpha-era maps and models
- 3.3.5-era maps and models
- at least one dense object scene where standstill and movement performance can be compared

### 5. wow-viewer Owns The Design

`MdxViewer` remains a compatibility harness or proof consumer when needed.

New renderer design ownership belongs in:

- `wow-viewer/src/core/WowViewer.Core`
- `wow-viewer/src/core/WowViewer.Core.IO`
- `wow-viewer/src/core/WowViewer.Core.Runtime`
- `wow-viewer/src/viewer/WowViewer.App`

## Success Criteria

This plan is successful when all of the following are true:

1. `wow-viewer` has an explicit runtime-owned world frame with named passes and measurable costs.
2. `wow-viewer` has typed render-entry families for terrain, WMO, M2, liquids, sky, overlays, and debug/editor work.
3. scene visibility uses explicit spatial ownership instead of full placement-set walks every frame.
4. M2 and WMO submission paths have real batching or amortized submission control, not only shared shader state.
5. Alpha and 3.x data both run through the same engine with build-aware profile differences rather than separate ad hoc codepaths.
6. fixed real-data captures show stable frame-time improvement at standstill and while moving.
7. `wow-viewer` can be described honestly as a renderer reference, not just a file reader.

## Explicit Non-Goals

- not a one-shot renderer rewrite
- not an immediate promise of full visual parity for every build family
- not a UI-polish plan
- not another parser-only milestone presented as runtime closure
- not a claim that the current `MdxViewer` compatibility regressions are already solved

## Workstreams

## Workstream A - Measurement, Golden Corpus, And Proof Harness

Purpose:
make every renderer slice measurable against fixed real scenes.

Deliverables:

- a fixed golden scene set for Alpha and 3.3.5
- saved camera shots for standstill and movement comparisons
- per-pass CPU timings and submission counts in the runtime contract
- persistent performance reports or capture metadata for before/after comparisons

Required proof:

- same scene, same camera, same settings, before/after metrics
- no claims of closure based only on code inspection or build success

## Workstream B - Version Profile System For Rendering Rules

Purpose:
ensure Alpha and 3.x remain one engine with explicit rule differences.

Deliverables:

- typed runtime profile layer for build family or era
- explicit hooks for M2 skin behavior, WMO material rules, lighting, fog, and effect routing
- profile-driven runtime switches instead of hardcoded compatibility branches spread across consumers

Required proof:

- one Alpha map and one 3.3.5 map both run through the same runtime service graph with different profiles

## Workstream C - World Frame And Visibility Ownership

Purpose:
move world frame collection and pass sequencing into runtime-owned services.

Deliverables:

- `WorldRenderFrame` or equivalent in `wow-viewer`
- scene-owned pass buckets for terrain, WMO, M2, liquids, sky, overlays
- visible-set extraction services moved out of monolithic viewer host code
- scene statistics that report both visible counts and submission counts honestly

Required proof:

- active consumer no longer owns visibility and pass sequencing inline

## Workstream D - Spatial Indexing, Residency, And Streaming

Purpose:
stop scanning or loading the world as if all placements were equally relevant every frame.

Deliverables:

- spatial indices for WMO and M2 placements
- tile or cell ownership separated from render admission
- explicit residency states: unloaded, queued, resident, known-missing, suppressed-failure
- predictable background I/O and load budgets

Required proof:

- visible-set collection and idle frame time no longer scale linearly with total placement count on dense maps

## Workstream E - WMO Runtime Decomposition

Purpose:
stop treating WMO rendering as a per-instance opaque black box.

Deliverables:

- explicit WMO runtime contracts for shell, liquids, doodads, transparent groups
- scene-level ownership of WMO pass sequencing where practical
- reduced per-instance repeated work for group selection, transparent ordering, and doodad handling

Required proof:

- WMO visibility and WMO submission are separately measurable in the new runtime
- at least one repeated per-instance WMO cost center is removed or amortized

## Workstream F - M2 Runtime, Submission, And Real Batching

Purpose:
own M2 as a first-class runtime instead of a generic model renderer path.

Deliverables:

- typed active-section and material/effect state from `wow-viewer` runtime
- animation, lighting, and effect state kept in the runtime contract
- render-entry families for opaque, alpha-cutout, transparent, additive, particles, ribbons, and special cases
- real batching or amortized submission by compatible renderer or state, not just shared shader setup
- explicit handling for cases that must remain unbatched

Required proof:

- `batched` counts correspond to actual submission reduction rather than only shared program state
- dense repeated-model scenes show measurable submission-count reduction

## Workstream G - Terrain, Liquids, Sky, And Shared Lighting

Purpose:
keep terrain-family ownership aligned with the world frame instead of ad hoc renderer-local decisions.

Deliverables:

- one shared frame lighting contract used by terrain, WMO, M2, liquids, sky, and fog
- explicit terrain vs WDL vs liquid pass ownership
- profile-driven era differences for Alpha and later terrain or fog behavior

Required proof:

- no world pass bypasses the shared frame lighting contract silently

## Workstream H - Consumer Cutover And Reference App

Purpose:
make `wow-viewer` itself the first-class renderer consumer.

Deliverables:

- `WowViewer.App` consuming runtime-owned world and M2 seams directly
- `MdxViewer` reduced to compatibility-only reuse where still needed
- fixed parity harness scenes routed through the `wow-viewer` app path

Required proof:

- one real world scene opens through the `wow-viewer` viewer app path using the extracted runtime seams

## Ordered Phases

### Phase 0 - Proof Harness And Performance Contract

First slice because everything else depends on honest measurement.

Scope:

- formalize fixed Alpha and 3.3.5 scene set
- extend runtime frame stats so they can be persisted or compared across runs
- define target metrics: standstill CPU ms, movement CPU ms, visible counts, submission counts, queue depth

Exit condition:

- future slices can be judged with concrete before/after evidence

### Phase 1 - World Runtime Extraction: Visible Sets And Pass Ownership

Scope:

- continue the `WorldScene` split into `wow-viewer` runtime services
- extract visible-set collection, pass sequencing, and frame scratch ownership
- keep active viewer compatibility only as a proof harness

Exit condition:

- `WorldScene` is no longer the design owner of world visibility and pass ordering

### Phase 2 - Spatial Index And Residency System

Scope:

- build placement indexing and runtime residency tracking
- remove full placement-set idle scans from the hot path
- keep known-missing and failure suppression explicit and reusable

Exit condition:

- idle frame cost scales with admitted scene cells or buckets, not full map placement volume

### Phase 3 - WMO Runtime Pass Decomposition

Scope:

- split WMO shell, liquids, doodads, and transparent work into explicit runtime-owned seams
- reduce per-instance repeated renderer-local pass work

Exit condition:

- WMO cost is measured and controlled at scene level instead of being trapped inside one renderer path

### Phase 4 - M2 Runtime Completion: Sections, Materials, Animation, Effects

Scope:

- finish the open M2 runtime slices already documented in `wow_viewer_m2_runtime_plan_2026-03-31.md`
- keep build-aware profile differences explicit
- do not flatten unresolved semantics away

Exit condition:

- M2 runtime state is owned in `wow-viewer`, not inferred ad hoc during draw

### Phase 5 - Real M2 Submission And Batching

Scope:

- replace the current shared-shader-only world path with actual compatible submission grouping
- separate true batching from necessary unbatched cases
- preserve correctness for alpha-cutout, transparent, particle, and ribbon families

Exit condition:

- repeated-model scenes show real submission-count reduction with fixed-shot evidence

### Phase 6 - Shared Lighting And Era Parity

Scope:

- unify frame lighting ownership across terrain, WMO, M2, liquids, sky, and fog
- use the version profile layer to keep Alpha and 3.x differences explicit

Exit condition:

- one engine, one frame contract, profile-driven behavior differences

### Phase 7 - wow-viewer App Consumer And Reference Positioning

Scope:

- make `WowViewer.App` the direct runtime consumer
- keep `MdxViewer` only for compatibility-only harness work still not cut over
- document honest proof level and current gaps

Exit condition:

- `wow-viewer` can be presented as the canonical C# renderer reference for these formats with explicit remaining boundaries

## Immediate Next Slices

The next implementation work should not be another isolated `MdxViewer` hotfix.

The best next slices are:

1. `wow-viewer` world-runtime slice: visible-set runtime extraction
   - move visibility contracts and frame scratch ownership out of `MdxViewer.WorldScene`
   - this is the prerequisite for honest scene-level batching and spatial indexing

2. `wow-viewer` M2 runtime slice: scene submission and batching design
   - convert the current `shared shader == batching` lie into an explicit runtime-owned submission model
   - define which families can batch and which must remain unbatched

3. performance proof harness slice
   - lock the fixed Alpha and 3.3.5 comparison scenes and capture format so later slices can prove real gains

## Validation Rules

- default implementation proof for `wow-viewer` stays:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- compatibility build only when needed:
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- runtime claims require fixed real-data evidence from both Alpha and 3.3.5 families when the slice affects cross-version renderer behavior
- do not claim performance closure based on parser ownership, synthetic scenes, or one build-only success

## Practical Deliverables To Produce Over Time

- runtime performance dashboards or structured reports
- golden-scene capture pack for Alpha and 3.3.5
- version-profile matrix for renderer-relevant behavior
- explicit WMO and M2 render-entry family contracts
- explicit residency and visibility service contracts
- one `wow-viewer` viewer app scene proving the engine path directly

## What This Plan Explicitly Does Not Claim Yet

- it does not claim that current `wow-viewer` runtime ownership is already deep enough
- it does not claim that Alpha and 3.3.5 parity is already solved
- it does not claim that batching alone fixes everything
- it does not claim that `MdxViewer` should disappear before `wow-viewer` has a real consumer path

## Routing Note

Use this plan as the high-level parent for:

- `wow_viewer_world_runtime_service_plan_2026-03-31.md`
- `wow_viewer_m2_runtime_plan_2026-03-31.md`
- future renderer-performance proof harness work

Those plans remain the narrower execution surfaces. This file is the integrated renderer program.