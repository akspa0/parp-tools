# Receipt — U-01 step E1: PM4 overlay out of `WorldScene` (2026-09-25)

Epic 251 item U-01 (operator P1; E1→E4 order approved 2026-09-23). This receipt covers
**U01-T002** (verbatim move, delegation, build + tests, line counts). It does **not** claim any
runtime behaviour: the PM4 overlay, colours, selection and OBJ export smoke is **U01-T003**,
operator-owned. The viewer has no unit tests over `WorldScene`/PM4 overlay code, so the test run
below proves only that nothing else regressed.

## Result

| File | Before | After |
|---|---|---|
| `src/viewer/WoWViewer/Terrain/WorldScene.cs` | 17,175 | **8,326** |
| `src/viewer/WoWViewer/ViewerApp.cs` | 16,746 | 16,746 (receiver edits only) |

`git diff --diff-algorithm=histogram --stat -- Terrain/WorldScene.cs`: 174 insertions, 9,024 deletions.

New files under `src/viewer/WoWViewer/Terrain/Pm4/` (namespace unchanged, `WoWViewer.Terrain`):

| File | Lines | Contents |
|---|---|---|
| `Pm4OverlayScene.cs` | 1,993 | ctor, host bridge, all PM4 state + public properties, load / camera-window streaming / cache orchestration |
| `Pm4OverlayScene.Selection.cs` | 1,259 | hover, ray pick, selection, debug + research info, per-object transforms, render filters |
| `Pm4OverlayScene.Reports.cs` | 1,819 | interchange JSON, OBJ export, WMO correlation, match states, legend, summaries, scene facts, outline |
| `Pm4OverlayGeometry.cs` | 1,546 | static: tile-object build, MSLK / connectivity splits, CK24 lines/triangles, MSLK walls |
| `Pm4OverlayCoordinates.cs` | 521 | static: axis-convention detection, PM4→world/renderer conversion, placement solution |
| `Pm4OverlayMatching.cs` | 531 | static: asset-profile and placement-variant matching |
| `Pm4OverlayCacheCodec.cs` | 337 | static: cache ↔ tile-dictionary conversion, per-file disk cache read/write |
| `Pm4OverlayColors.cs` | 260 | static: HSV, type buckets, type-flag colours |
| `Pm4OverlayModels.cs` | 763 | public PM4 structs formerly at the top of `WorldScene.cs` |
| `Pm4OverlayObject.cs` | 396 | `Pm4OverlayObject`, `Pm4ResearchContext` (formerly after the class) |
| `Pm4OverlayReportModels.cs` | 431 | PM4 report/record types (formerly after the class) |
| `IPm4OverlayHost.cs` | 27 | the only world-scene state the overlay reads |

Every file is under the ~2,000-line budget (AGENTS.md §10).

## What changed, exactly

1. **Moved verbatim**: 387 `WorldScene` members (PM4-named members, helpers used only by them, and
   three dead PM4 helpers with no callers: `TryComputePlanarPrincipalYaw`, `BuildFallbackMeshLines`,
   `BuildFallbackMeshTriangles`) plus 42 top-level PM4 types. Member selection came from a Roslyn
   member map + reference closure, not by hand.
2. **Kept on `WorldScene`**: `Pm4OverlayWindowMs` / `Pm4OverlayWindowPeakMs` (render-region timing
   stats set in `Render()`), and the PM4 draw blocks inside `Render()` — moving those is E2.
3. **Delegation**: `WorldScene` holds `private readonly Pm4OverlayScene _pm4Overlay` (built where the
   cache service used to be assigned in both constructors) and exposes `public Pm4OverlayScene
   Pm4Overlay`. 181 references in the remaining `WorldScene` members (`Render`,
   `UpdateHoveredAssetInfo`, `Dispose`, `IsHoverPickPositionAllowed`, …) were rewritten to
   `_pm4Overlay.X` by a Roslyn pass that skips member-access names, initializer targets, named
   arguments and anything shadowed by a local or parameter (0 shadowed hits).
4. **Callers**: 278 lines in 7 `ViewerApp*.cs` files changed `_worldScene.X` → `_worldScene.Pm4Overlay.X`
   (and `WorldScene.Pm4ProfilingAccumulator` → `Pm4OverlayScene.Pm4ProfilingAccumulator`). Verified
   receiver-only: removing `.Pm4Overlay` from every added line reproduces every removed line.
5. **Host contract**: `IPm4OverlayHost` (internal), implemented explicitly by `WorldScene` (13
   one-line members). `Pm4OverlayScene` re-exposes each as a private bridge member with the old
   name, so no moved body was edited. Static helpers the overlay calls (`TransformBounds` ×2,
   `TryMeasureHoverInfoHit`, `ConvertMprlPositionToWorld`, `NoCullRadius`) went `private` → `internal`
   on `WorldScene` and are forwarded.
6. **Visibility only**: 84 declarations in the `Pm4OverlayScene` partials and every `private static`
   in the five static helper classes went `private` → `internal` so the scene, the helpers and
   `WorldScene.Render()` can still reach them. No body, signature or ordering change.

Verbatim audit (line multiset of old `WorldScene.cs` vs new `WorldScene.cs` + all `Pm4/` files,
after undoing only `_pm4Overlay.` and `private`/`internal`): 4 original lines absent — the class
declaration (gains `, IPm4OverlayHost`), the two constructor cache-service lines (now construct the
overlay), and `WorldScene.BuildPm4BaseTransform` → `Pm4OverlayScene.BuildPm4BaseTransform` in
`Pm4OverlayObject`. Everything else added is headers, `using` lines, the host bridge and docs.

## Design notes (spec-sync, AGENTS.md §9.4)

- The plan says "owned `Pm4OverlayScene` (+ smaller helpers)". The instance part is one class split
  over three partial files by responsibility; pure static code is in five separate static classes.
  A partial split was chosen for the instance code because it keeps every moved body verbatim; the
  static classes are real units with no shared state.
- Callers reach the overlay through `WorldScene.Pm4Overlay` instead of ~88 pass-through members on
  `WorldScene`. This is still "one field + delegation" and keeps `WorldScene` smaller; it touched
  only receiver text in `ViewerApp*` (no new members there, §10).
- Initialisation order: PM4 field initialisers now run when the overlay is constructed inside the
  `WorldScene` constructor, at the same point the cache service was created before. No `WorldScene`
  field initialiser references PM4 state (checked).

## Verification

Environment: Linux container, .NET SDK 10.0.112, `-p:EnableWindowsTargeting=true`.

| Command | Exit | Result |
|---|---|---|
| `dotnet build WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true` | 0 | 0 errors |
| `dotnet build src/viewer/WoWViewer/WoWViewer.csproj -c Debug -p:EnableWindowsTargeting=true --no-incremental`, before (clean `HEAD` worktree) vs after | 0 / 0 | identical compiler-warning set for `WorldScene.cs` + `Pm4/*` (location-normalised) |
| `dotnet test WowViewer.slnx -c Debug -p:EnableWindowsTargeting=true`, before vs after | 1 / 1 | identical 26 failing tests, same names. All 26 are environmental and fail on `HEAD` too (missing `test_data/development/…`, Windows-path expectations) |

Test totals (after = before): Core.Tests 1,578 passed / 18 failed / 1 skipped; Core.PM4.Tests 95 / 8;
Core.Editor.Tests 91 / 0; Core.Curation.Tests 39 / 0.

## Criterion → evidence

| U01-T002 criterion | Evidence |
|---|---|
| Verbatim move | line-multiset audit above: 4 intended line changes, no body edits |
| Delegation | one field + `Pm4Overlay` property + `IPm4OverlayHost`; 181 qualified references; 278 receiver-only caller lines |
| Build | solution build exit 0; no new compiler warnings in touched files |
| Tests | identical pre-existing failure set; no new failures |
| Line-count receipt | table above (17,175 → 8,326; every new file < 2,000) |
| Runtime (overlay, colours, selection, OBJ export) | **not claimed** — U01-T003 operator smoke |

## Operator smoke for U01-T003 (PowerShell 7)

```powershell
dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
dotnet run --project I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug
```

On a map with PM4 files: enable the PM4 overlay and check it loads and streams as the camera moves;
cycle colour modes and the legend; hover and click-select a PM4 object (debug, graph, research and
region panels populate); toggle object / CK24 bounds, placement-Z plane and MSCN/MSPV nodes; run
the PM4 OBJ export, interchange JSON and WMO correlation JSON; reload the overlay.
