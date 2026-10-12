# Receipt — Spec 255 W0–W7: WorldScene independent clusters extracted (2026-09-27)

Covers tasks **T010–T017** (plan approved as written 2026-09-27, T001). Every step is a behaviour-preserving
move; no runtime, visual, FPS or input behaviour is claimed. Operator smoke for everything moved here is
**T018** (checklist at the end). W8, W9 and E2 stay gated on the R-10 after-capture.

`WorldScene.cs`: **8,275 → 4,922 lines** (plan estimate ≈ 4,900). 253 members + 9 types moved into 8 classes.

## Steps

| Step | Commit | New owner (under `src/viewer/WoWViewer/`) | Members / lines moved | Receiver edits outside `WorldScene` | `private`→`internal` | Hand edits |
|---|---|---|---|---|---|---|
| W0 | `a5ca359` | `Terrain/Scene/Frame/{WorldRenderFrame, FlatVisibilityBucket, TerrainAssetLoadPolicy}.cs`, `Terrain/Scene/Selection/SelectedSceneObjectKey.cs` | 4 nested types / 296 | 0 | `AreFiniteOrderedBounds` | none |
| W1 | `215bcc1` | `Terrain/Scene/HoverPick/SceneHoverPickController.cs` (+ `ObjectType`, `HoveredAssetInfo`, `SceneObjectPickHit` files) | 47 / 843 | 42 (incl. 1 in E1's PM4 bridge) | 4 moved members + 4 `WorldScene` statics they bridge to | none |
| W2 | `b1341fb` | `Terrain/Scene/Taxi/TaxiActorScene.cs` (+ `TaxiActorPose.cs`) | 54 / 473 | 137 | 5 + 1 | 3 receivers + 1 flip for the mixed static/instance `TryGetTaxiRouteSelectionPoint` |
| W3 | `26e8507` | `Terrain/Scene/Filters/SceneObjectFilters.cs` (+ `UniqueIdVisibilityScope`, `UniqueIdArchaeologyLayer`) | 28 / 232 | 63 | 1 | none |
| W4 | `733f7d1` | `Terrain/Scene/Selection/SceneSelectionState.cs` | 32 / 564 | 76 | 1 | none |
| W5 | `f046fda` | `Terrain/Scene/Atmosphere/SceneAtmosphere.cs` | 68 / 398 | 82 | 12 | 2 qualified *type* uses of `LitLoader` reverted (type and property share the name); 1 property pattern `{ Atmosphere.HasUserFogRangeOverride: true }` |
| W6a | `0d8c2b2` | `Terrain/Pm4/Pm4OverlayMprlHelpers.cs` (internal static) | 9 / 139 | 1 (E1's PM4 bridge) | 0 | none |
| W6b | `6a8c6e4` | `Terrain/Scene/ExternalSpawns/ExternalSpawnLayer.cs` | 7 / 141 | 7 | 0 | none |
| W7 | `bced801` | `Terrain/Scene/TerrainQueries/SceneTerrainQueries.cs` | 8 / 211 | 1 | 1 | none |

Supporting files: `Terrain/Scene/IWorldSceneHost.cs` (new, 33 members), implemented explicitly in
`WorldScene.cs` (33 one-line impls, block marked `WORLD-SCENE-HOST-IMPL-END`); one field + one public
accessor per service (`HoverPick`, `TaxiActors`, `ObjectFilters`, `Selection`, `Atmosphere`,
`ExternalSpawns`, `TerrainQueries`), each built first in both constructors (beside `_pm4Overlay`).

Callers changed only receivers: `_worldScene.X` → `_worldScene.<Accessor>.X`, `WorldScene.C` →
`<Service>.C`, in `Workbench/Services/**` (ViewerApp services) and `Terrain/Pm4/Pm4OverlayScene.cs` (E1 bridge
lines). Files touched outside `Terrain/`: `ViewerChromeService.Sidebars.cs`, `InvestigationService.Investigation.cs`,
`NavigatorPanelService.cs`, `SceneHoverAndPickService{,.ClickSelection,.Investigation}.cs`,
`WorkbenchPanelsService.cs`, `WorldObjectsPanelService.cs`, `TaxiAndAreaPoiSelectionService.cs` and other
taxi/lighting/selection/settings/camera-path panels (full list: `git diff --stat 450c181..bced801`).

## Method (the ViewerApp technique, adapted)

1. Roslyn member map + closure over `WorldScene.cs` (members used only by the cluster move with it).
2. Verbatim move into a `public sealed` class (`internal` constructor, E1's pattern) in namespace
   `WoWViewer.Terrain`; the new file carries `WorldScene.cs`'s using block unchanged. (The SDK's IDE0005
   unused-using fixer was tried and made no change, so no directive was hand-pruned.)
3. Every `WorldScene` member the moved code still uses is re-declared in the class as a same-named private
   bridge: instance → `IWorldSceneHost` (`ref` properties for mutable fields), static → `WorldScene.X`.
   Statics are bridged explicitly rather than via `using static WorldScene`, so a class member still wins
   name lookup over the PM4 `using static` imports exactly as it did inside `WorldScene`.
4. References inside `WorldScene` rewritten `X` → `_field.X` / `Service.X` by the syntax-aware qualify pass.
5. Compiler-driven fixer: `CS1061/CS0117` on `WorldScene` → receiver inserted at the reported column;
   `CS0122` → `private` → `internal` on that declaration only. Everything else stops for review.
6. Line-multiset audit, full-solution build (0 errors), warning comparison, full test run; commit only then.

Hard stops (bare `this`, unqualified object-member calls, volatile bridges, type-name collisions) — none hit.
The tool warned on one mixed static/instance overload (W2, W7) and one member/type name collision (W5);
each was resolved by hand as listed above.

## Scope decisions within the approved steps (spec-sync)

- **W0** moved the four *nested* types only. The top-level public types in `WorldScene.cs` moved with the
  step that owns their feature (W1: `ObjectType`, `HoveredAssetInfo`, `SceneObjectPickHit`; W2:
  `TaxiActorPose`; W3: `UniqueIdVisibilityScope`, `UniqueIdArchaeologyLayer`). `SelectedSceneObjectKey`
  went to `Scene/Selection/`, not `Scene/Frame/`. The optional fold of `IPm4OverlayHost` into
  `IWorldSceneHost` was not done.
- **W1** left shared geometry statics (`RayAABBIntersect`, `ScreenToRay`, `TransformBounds`, chunk-key
  helpers) in `WorldScene`; click selection went to W4.
- **W4** left `ResolveMdxSelectionBounds` (instance build uses it) for W8.
- **W5** left `SetDbcCredentials` (taxi/audio use it), `_skyDome` (readonly, constructor-assigned), object
  fog, scene lights and time of day (render path / W9).
- **W6** put the PM4 helpers in a new `Pm4OverlayMprlHelpers` static class. Eight of the nine are
  **unreachable** — no callers, or called only by another unreachable helper (`TryComputeExpectedMprlYawRadians`,
  both yaw-delta helpers, `ComputeUndirectedAngleDelta`, `NormalizeSignedRadians`,
  `DecodeRawMprlPackedAngleRadians`, `ToCoreCoordinateMode`, `NearestPositionRefDistanceSquared`); only
  `ConvertMprlPositionToWorld` is live. They were moved, not deleted. The external instance lists stay in `WorldScene` (instance store, W8).
- **W7** also took `IsMdxFullyOccludedByTerrain` (built on the height sampler).

## Verification (run per step; the table shows the final state at `bced801`)

| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true --no-incremental` | 0 | 0 errors every step |
| warning comparison vs baseline at `450c181` (path/line-normalised) | — | compiler warnings identical every step; `NU1903` (NuGet audit of `Snappier` 1.0.0, restore-time) varies run to run on an unchanged tree (72/72/68/70 observed), so it is compared separately |
| `dotnet test WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 1 | identical 26 pre-existing environmental failures every step (Core.Tests 1,598 passed / 18 failed / 1 skipped; Core.PM4.Tests 95/8; Editor 91/0; Curation 39/0) |
| line-multiset audit of each step's diff | — | only headers, bridges, host impl/interface, service fields/constructors and the edits listed above |

Linux build note: `-p:EnableWindowsTargeting=true` is required on the agent host; the operator's
Windows command below needs no flag.

## Criterion → evidence

| Spec 255 acceptance criterion | Evidence | Status |
|---|---|---|
| 1. Each step moves one cluster; bodies verbatim | per-step line-multiset audits; hand edits enumerated above | met for W0–W7 |
| 2. No new `WorldScene` members beyond one field per service + host plumbing; no partial file; every file < 2,000 | 7 service fields + 7 accessors (E1's `Pm4Overlay` pattern) + 33 explicit `IWorldSceneHost` impls; no partial file; largest new file 973 lines (`SceneHoverPickController.cs`) | met; `WorldScene.cs` itself is still 4,922 (criterion 4) |
| 3. Build 0 errors, test failure set unchanged, warnings compared, receipt | table above; this file | met |
| 4. `WorldScene.cs` ≤ ~2,000 after all steps incl. E2 | 4,922 after W0–W7 | open — W8, W9, E2 (gated on R-10) |
| 5. Runtime behaviour only from operator smoke | not claimed | **T018 (operator)** |

## Operator smoke (T018, PowerShell 7)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

On a map with WMOs, doodads, taxi paths and WL liquids (the same session can cover all):

- **W1 hover/pick**: hover WMO, doodad-in-WMO, MDX, liquid (tooltips on/off, range limit on/off, dynamic
  range); click-select via the disambiguation list; wireframe reveal brush on/off; object wireframe overlay.
- **W2 taxi**: taxi paths draw and load on demand; select a node and a route; actors ride routes (speed and
  scale sliders); actor model override set/cleared; taxi-ride camera follows the route.
- **W3 filters**: UniqueId range filter (per map / camera tile), archaeology layers list, object-path filter
  add/remove/clear hides and shows objects; saved filters restore on reload.
- **W4 selection**: select WMO / MDX / WMO doodad; selection survives tiles streaming out and back in; move a
  placement and save; bounds of the selection draw.
- **W5 atmosphere**: LIT load/reload and automatic fallback message; LIT fog override; user fog range set and
  cleared (fog returns to the previous range); skybox on a map with a Light skybox; stars fallback.
- **W6 spawns / PM4**: SQL spawns load and stream (counts in the status); PM4 overlay MPRL position refs
  unchanged.
- **W7 terrain queries**: camera path playback with collision on; "hide terrain-occluded MDX" option.

Expected: identical behaviour to `450c181`. Report anything different against the step that moved it.
