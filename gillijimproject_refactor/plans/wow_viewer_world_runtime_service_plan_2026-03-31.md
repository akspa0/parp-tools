# wow-viewer World Runtime Service Plan

## Status

- status: active
- intent: split `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs` into explicit runtime services owned by `wow-viewer`
- current proof floor:
  - renderer telemetry contracts and optimization-hint logic already live in `wow-viewer/src/core/WowViewer.Core.Runtime/World`
  - `MdxViewer` currently consumes that seam successfully
  - slice 01 (negative lookup suppression) has meaningful compatibility-path progress in `MdxViewer`, but the full runtime extraction sequence is still open

## Apr 03, 2026 Status Snapshot

- slice 01 (negative asset lookup suppression): partial
- slice 02 (visible set runtime extraction): partial
- slice 03 (world pass service extraction): partial
- slice 04 (WorldScene host thinning): open
- slice 05 (wow-viewer app runtime consumer): open

## Why This Plan Exists

- the user wants the long-term world renderer design to move into `wow-viewer`, not stay trapped in `MdxViewer`
- the remaining `WorldScene` file is still too large and mixes asset churn, visible-set collection, pass sequencing, and overlay/debug work
- repeated `.skin` miss churn and failed MDX retry spam are active performance noise and should be reduced before deeper renderer extraction so later measurements stay interpretable

## Immediate Live Blocker

- after the slice-01 miss-suppression work, the shared renderer fallback fix, and the stale-build correction, the live viewer still misses a large world-M2 set on the development map, especially giant root structures
- latest screenshots show those objects still resolve in tooltip selection and still draw world bounding boxes, which means placement and instance metadata are present
- that evidence does not prove shaded triangle submission succeeded; it only proves registration/bounds
- before treating slice 01 as runtime-closed or pushing harder on extraction slices, the next active-viewer diagnostic step should isolate whether adapted M2s are:
  - not reaching the opaque/transparent draw passes in `WorldScene`, or
  - reaching `ModelRenderer.RenderGeosets(...)` but becoming invisible because of material-state / blend / alpha routing from `WarcraftNetM2Adapter`
- this is still compatible with the longer-term `wow-viewer` extraction plan; it just means the active `MdxViewer` proof path still needs one more honest render-path investigation slice first

## Ordered Slices

### Slice 01 - Negative Asset Lookup Suppression

- status update:
  - compatibility-path work has already reduced repeated miss churn in the active viewer path
  - this slice is still only partial relative to the full plan because the seam is not yet completed as a reusable runtime-owned service extraction in `wow-viewer`

- target problem:
  - repeated `.skin` candidate searches
  - repeated failed MDX retry paths in `WorldAssetManager`
  - noisy asset-miss logging that hides other hot paths
- likely files:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
  - `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
  - optional narrow shared contract in `wow-viewer/src/core/WowViewer.Core.Runtime`
- proof goal:
  - known-missing companion `.skin` and known-failed MDX loads stop redoing the same work every frame or deferred load pass
  - logs report the miss once or in bounded form instead of flooding
  - build validation plus optional fixed-shot capture smoke

### Slice 02 - Visible Set Runtime Extraction

- target problem:
  - `WorldScene` still owns visible WMO/MDX/taxi collection and frame scratch lists
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/*`
- proof goal:
  - `WorldScene` no longer owns the visible-set contracts or scratch orchestration directly

#### Apr 04, 2026 implementation-ready build plan

Use this as the actual execution surface for slice 02. Do not treat the parent renderer plan as the build sheet.

Current assumed boundary in `WorldScene` today:

- visibility admission for WMO, MDX, and taxi actors is still owned inline by:
  - `CollectVisibleWmoInstances(...)`
  - `CollectVisibleMdxInstances(...)`
- the currently extracted `wow-viewer` runtime seam stops at frame telemetry and optimization hints:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderOptimizationAdvisor.cs`
- `WorldScene` still owns the concrete visibility bucket shapes and scratch state:
  - nested `VisibleWmoInstance`
  - nested `VisibleMdxInstance`
  - nested `WorldRenderFrame` visible lists and reset logic
- current collection code also mixes three responsibilities that should not stay fused:
  - pure visibility admission math
  - compatibility-host asset-ready lookups (`TryGetQueuedWmo` / `TryGetQueuedMdx`)
  - compatibility-host pending-load prioritization (`TrackPendingVisibleLoad(...)`, flush helpers)

Single next extraction seam:

- move only the pure visibility admission and reusable visible-bucket ownership into `wow-viewer`
- keep renderer resolution, pending-load queueing, animation advance, and actual draw submission in `WorldScene`
- this is the correct narrow seam because it makes `WorldScene` smaller for real without forcing a fake graphics abstraction or premature pass-service extraction

Exact runtime contracts to add in `wow-viewer`:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs`
  - shared replacement for the current `WorldScene.ObjectInstance` data contract
  - keep it data-only: model key/path/name, transform, placement position, bounds, unique id, placement index, tile coordinates, bounds-resolved flag
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityContext.cs`
  - camera position
  - camera forward
  - fog end
  - object streaming range multiplier
  - `CullSmallDoodadsOnly`
  - `CountAsTaxiActor`
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibleWmoEntry.cs`
  - `WorldObjectInstance Instance`
  - `float CenterDistanceSq`
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibleMdxEntry.cs`
  - `WorldObjectInstance Instance`
  - `float CenterDistanceSq`
  - `float OpaqueFade`
  - `float TransparentFade`
  - `bool IsTaxiActor`
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibilityFrame.cs`
  - runtime-owned visible-set scratch only
  - `List<WorldVisibleWmoEntry> VisibleWmos`
  - `List<WorldVisibleMdxEntry> VisibleMdx`
  - `int VisibleTaxiMdxCount`
  - `Reset()`
- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs`
  - pure static or sealed service that performs visibility admission only
  - no OpenGL
  - no renderer references
  - no `WorldAssetManager` dependency

Exact compatibility-host code that should remain in `WorldScene` after slice 02:

- asset-readiness lookup:
  - `TryGetQueuedWmo(...)`
  - `TryGetQueuedMdx(...)`
- pending-load prioritization and bounded queue flush:
  - `_pendingVisibleMdxLoadDistances`
  - `_pendingVisibleWmoLoadDistances`
  - `TrackPendingVisibleLoad(...)`
  - `FlushPendingVisibleMdxLoads()`
  - `FlushPendingVisibleWmoLoads()`
- animation update for admitted renderers
- opaque and transparent submission loops
- transparent sort scratch and pass-local batching counters
- GL state sequencing and all terrain/WDL/liquid/overlay work

Exact `WorldScene` edits for the first slice:

1. Replace the current `WorldScene.ObjectInstance` definition with the shared runtime contract from `WowViewer.Core.Runtime.World`.
2. Delete nested `VisibleWmoInstance` and `VisibleMdxInstance` from `WorldScene`.
3. Replace the nested frame-owned visible lists with a runtime-owned `WorldVisibilityFrame` field.
4. Replace `CollectVisibleWmoInstances(...)` with a thin adapter call into `WorldObjectVisibilityCollector.CollectVisibleWmos(...)`.
5. Replace `CollectVisibleMdxInstances(...)` with a thin adapter call into `WorldObjectVisibilityCollector.CollectVisibleMdx(...)`.
6. Keep missing-asset handling in the host by iterating the runtime-produced visible entries and resolving renderers there.
7. Keep transparent sorting in the host for now; this belongs to slice 03 pass extraction, not slice 02.

Compatibility bridge rule:

- do not let `WowViewer.Core.Runtime` reference `WmoRenderer`, `IModelRenderer`, `WorldAssetManager`, or `FrustumCuller`
- if the current frustum test must be reused immediately, pass a host-supplied predicate into the collector such as `Func<Vector3, Vector3, bool> isBoundsVisible`
- if that delegate makes the slice noisy, add a tiny runtime interface in `WowViewer.Core.Runtime` and keep the `FrustumCuller` adapter in `MdxViewer`
- do not move pending-load queue logic into the runtime in this slice; the point is visible-set ownership first, not asset policy migration

First PR-sized file set:

- add:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityContext.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibleWmoEntry.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibleMdxEntry.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibilityFrame.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs`
- change:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
  - optional small test file under `wow-viewer/tests/WowViewer.Core.Runtime.Tests` or the nearest existing runtime-capable test project if that test project does not exist yet

Required proof for this slice:

- build proof:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- test proof:
  - add focused collector tests over synthetic visibility inputs that prove:
    - off-frustum + rear-cone rejection
    - near-hold preservation
    - small-doodad distance cull behavior
    - MDX fade output stays deterministic for the same input context
    - taxi actors increment `VisibleTaxiMdxCount` through the runtime frame
- structural proof:
  - `WorldScene` no longer contains the concrete visible-entry structs or the inline admission loops
  - visible-set scratch ownership is now runtime-owned even though render submission is still host-owned

Out of scope for slice 02:

- no terrain/WDL/liquid extraction
- no WMO or MDX render-pass service extraction
- no real batching redesign
- no migration of `_pendingVisible*LoadDistances` into runtime policy
- no claim that `WowViewer.App` is a direct consumer yet

What slice 03 should inherit immediately afterward:

- `WorldVisibilityFrame` becomes the pass input to explicit runtime-owned WMO/MDX pass coordinators
- the host should stop owning transparent-sort scratch once pass services move
- the current host-local asset-ready bridge can then be revisited as either a runtime residency query seam or a pass-preparation seam

#### Apr 04, 2026 implementation status update

- first bridge slice is now landed:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldObjectInstance.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityContext.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibleWmoEntry.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibleMdxEntry.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldVisibilityFrame.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs`
- focused runtime proof also landed in:
  - `wow-viewer/tests/WowViewer.Core.Tests/WorldObjectVisibilityCollectorTests.cs`
- active compatibility bridge landed in:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs`
- current boundary after the landed slice:
  - visible-set bucket ownership and pure visibility admission now live in `wow-viewer`
  - `WorldScene` still owns asset-ready lookups, pending-load queueing, animation advance, transparent sort, and actual draw submission
- proof completed:
  - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldObjectVisibilityCollectorTests`
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
  - no real-data runtime capture or viewer-side performance signoff was done in this slice
  - render-pass ownership is still not extracted; that remains slice 03

### Slice 03 - World Pass Service Extraction

- target problem:
  - `WorldScene.Render(...)` still sequences terrain, WMO, MDX, liquids, and overlays inline
- likely destination:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/*`
  - coordinator in `wow-viewer/src/core/WowViewer.Core.Runtime/World/*`
- proof goal:
  - explicit runtime-owned pass services exist for terrain/WMO/MDX/overlay work

#### Apr 04, 2026 implementation status update

- first object-pass coordinator bridge is now landed:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassFrame.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs`
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs`
- focused proof also landed in:
  - `wow-viewer/tests/WowViewer.Core.Tests/WorldObjectPassCoordinatorTests.cs`
  - `wow-viewer/tests/WowViewer.Core.Tests/WorldFramePassCoordinatorTests.cs`
- active compatibility host now consumes that seam in:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs`
- current boundary after the landed sub-slice:
  - runtime now owns object-pass scratch for transparent MDX ordering and MDX animation dedup plus concrete pass-family iteration helpers for WMO opaque, MDX animation, MDX opaque, and MDX transparent
  - runtime now also owns the current top-level frame order from lighting through terrain and from object-phase preparation through liquid/overlay via `WorldFramePassCoordinator`, while `WorldScene` supplies the host callbacks
  - `WorldScene` still owns GL state sequencing inside those callbacks, renderer resolution, batch `BeginBatch(...)` timing, frustum and asset-ready host bridges, and the actual renderer method calls
- proof completed:
  - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "WorldFramePassCoordinatorTests|WorldObjectPassCoordinatorTests|WorldObjectVisibilityCollectorTests"`
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - `dotnet build i:/parp/parp-tools/gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug`
- proof boundary:
  - this is still a partial bridge, not full world-pass extraction into renderer-independent services
  - terrain, WDL, liquid, sky, and overlay sequencing is now runtime-owned, but the concrete pass work remains host-owned inside `WorldScene`
  - startup capture queueing is now available without UI for fixed runs, but no real-data capture or performance signoff was done in this slice

### Slice 04 - WorldScene Host Thinning

- target problem:
  - even after service extraction, `WorldScene` can still act like the design owner if delegation remains vague
- proof goal:
  - `WorldScene` becomes a thin host or adapter over runtime services instead of the orchestration owner

### Slice 05 - wow-viewer App Runtime Consumer

- target problem:
  - extracted runtime services still need a first direct consumer in `wow-viewer`
- proof goal:
  - `wow-viewer/src/viewer/WowViewer.App` exercises the extracted runtime seams directly without claiming full production parity

## Prompt Surface

- root router:
  - `.github/prompts/wow-viewer-world-runtime-plan-set.prompt.md`
- ordered prompts:
  - `.github/prompts/wow-viewer-world-runtime/01-negative-asset-lookup-suppression.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/02-visible-set-runtime-extraction.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/03-world-pass-service-extraction.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/04-world-scene-host-thinning.prompt.md`
  - `.github/prompts/wow-viewer-world-runtime/05-wow-viewer-app-runtime-consumer.prompt.md`

## Validation Rules

- default `wow-viewer` proof stays `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` and `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- build `gillijimproject_refactor/src/MdxViewer/MdxViewer.sln -c Debug` only when a slice changes compatibility or current-viewer behavior
- treat capture automation in `src/MdxViewer/ViewerApp_CaptureAutomation.cs` as a useful smoke aid for fixed-shot before/after checks, not as runtime signoff by itself

## Explicit Non-Claims

- this plan does not claim that the renderer can move in one jump
- this plan does not claim that every performance issue is `.skin` related
- this plan does not claim that a wow-viewer build or test pass proves viewer-runtime closure