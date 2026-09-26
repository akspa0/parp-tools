# Plan — Spec 255 WorldScene Decomposition

**Status**: proposed 2026-09-26, awaiting operator approval (tasks T001). Nothing below is authorised
until T001 is checked.

## Technique (proven by U-01 E1 and the ViewerApp campaign)

`WorldScene` is a single non-partial class whose private fields are shared by every feature, so each step:

1. Picks a cluster with the Roslyn member map + closure (members used only by the cluster move with it,
   including the fields that are its own state).
2. Moves members verbatim into an owned class under `Terrain/Scene/<Feature>/`, namespace
   `WoWViewer.Terrain` (same as `WorldScene`, so type names resolve identically). Members from one source
   keep that file's usings.
3. Reaches remaining scene state only through an `IWorldSceneHost` interface implemented explicitly by
   `WorldScene` (as `IPm4OverlayHost` does for E1); mutable fields are `ref`-returning members; the
   service re-declares what it uses as private same-named bridge members, so moved bodies do not change.
   E1's `IPm4OverlayHost` is folded into `IWorldSceneHost` only if that is a pure rename (optional W0 task).
4. `WorldScene` keeps one field per service, built in its constructors where `_pm4Overlay` is built today;
   remaining references are rewritten `X` → `_service.X` by the syntax-aware pass; external callers
   (`ViewerApp` services) change receiver only: `_worldScene.X` → `_worldScene.<Service>.X`.
5. Only `private` → `internal` visibility changes. Hard stops: bare `this`, unqualified
   `GetType()`/`ToString()`, a new type name that already exists.
6. Per step: extract → visibility fixes → audit → full-solution build → commit only on 0 errors.

The ViewerApp extraction scripts are rebuilt for `WorldScene` (class name and host interface are
parameters); they stay out of the repository unless the operator asks for them under `tools/`.

## Proposed steps

Ordered so independent clusters go first and render-path clusters wait for R-10.

| Step | Cluster(s) | ≈ Lines out | Target | Gate | Operator smoke |
|---|---|---|---|---|---|
| W0 | Frame record types: `WorldRenderFrame`, visibility buckets and other nested records → own files (type move only) | 351 | `Terrain/Scene/Frame/*.cs` | none | build-only step; covered by W8 smoke |
| W1 | Hover / pick / wireframe reveal (collection side of E4; `HoveredAssetInfo` formatting) | 1,099 | `SceneHoverPickController` | none | hover/click on WMO, doodad-in-WMO, MDX, liquid; wireframe reveal; click-disambiguation list |
| W2 | Taxi actors & routes | 404 | `TaxiActorScene` | none | taxi routes/nodes draw; actor models ride routes; actor model override; taxi-ride camera |
| W3 | UniqueId / object-path filters + archaeology layer queries | 352 | `SceneObjectFilters` | none | UniqueId range/layer filters, path filters hide/show; archaeology layers list |
| W4 | Selected-object resolution + selected placement edit/move | 578 | `SceneSelectionState` | after W1 | select WMO/MDX/doodad; selection survives tile rebuild; move a placement and save |
| W5 | Lighting / LIT / fog + skybox | 330 | `SceneAtmosphere` | none | LIT fallback, fog range restore, skybox on a map with a skybox, stars fallback |
| W6 | External (SQL) spawns; PM4 MPRL yaw helpers → PM4 overlay | 320 | `ExternalSpawnLayer`; `Terrain/Pm4` | none | SQL spawns load/stream; PM4 MPRL-based yaw readouts unchanged |
| W7 | Camera-path collision & terrain height sampling | 243 | `SceneTerrainQueries` | none | camera path with collision on; terrain height readouts |
| W8 | Tile streaming & instance build; bounds & visibility buckets | 870 | `SceneInstanceStore` | **after R-10 after-capture** | fly a legacy and a modern map: tiles stream in/out, objects appear, frame counters unchanged |
| W9 | Frame visibility collection; render-path planning & deferred asset loads; frame stats & time of day | 805 | `SceneFrameVisibility`, `SceneAssetStreaming` | **after R-10**, with/after W8 | same flights; draw and deferred-load counters unchanged |
| E2 | `Render()` → frame orchestrator + per-pass classes (existing U-01 E2 / U01-T004) | 1,844 | `Terrain/Passes/*` | **after R-10** | as U01-T005 |

Expected result: after W0–W7 `WorldScene.cs` ≈ 4,900 lines; after W8, W9 and E2 ≈ 1,300–1,700 lines
(state, constructors, composition, frame orchestration, host implementation).

## Risks

| Risk | Mitigation |
|---|---|
| `Render()` references most clusters, so every step rewrites some of its lines | receiver-only rewrites, audited; render-path clusters wait for R-10 so its measurement is not disturbed |
| Hot per-frame paths gain an interface hop per bridged field | bridges are non-virtual private properties over one interface call; W8/W9 get before/after frame-counter captures from the operator |
| Nested types referenced from `ViewerApp` services by `WorldScene.X` | qualification pass + compiler; W0 moves types first |
| Nested classes holding a `WorldScene` reference and calling moved members | compiler finds them; receiver-only fix, recorded in the receipt |
