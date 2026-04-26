# wow-viewer MdxViewer Cut-Away Reset Plan

## Status

- status: active reset plan
- intent: stop iterating on the current fragmented [`WowViewer.App`](../../wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs) shell and instead port the working interaction model from [`MdxViewer`](../src/MdxViewer/ViewerApp_Sidebars.cs) into `wow-viewer`
- design rule: treat [`MdxViewer`](../src/MdxViewer) as the UI and interaction reference, but keep `wow-viewer` as the long-term code owner for runtime, rendering, and shared file I/O
- porting rule: when `MdxViewer` already has working world behavior, use that exact code path as the functional reference and port it into `wow-viewer`; do not replace it with new guessed behavior just because the target renderer is newer or more GPU-oriented
- viewer-first rule: `wow-viewer` must act as a world viewer first and a diagnostics/tooling surface second; diagnostic panels are subordinate to the composed world image
- aesthetic target: prioritize the 0.5.3 client feel for this project, because the long-term purpose is exploratory data tooling and low-resolution visual restoration over early-world data, not a generic modern asset inspector
- Apr 24 correction: a one-tile diagnostic terrain preview is no longer an acceptable definition of "World Session"; the reset must produce a navigable multi-tile world viewport with minimap, layer controls, terrain texturing, and object rendering on the critical path
- Apr 24 camera/source correction: world viewport vertical FOV is `45` degrees, and ADT terrain is the primary loaded world-view source; WDL is only far-terrain/reference data and must not be described or treated as the loaded world surface
- Apr 25 live-path correction: the World Session now forces terrain on and WDL off for live loads, the spawn/tile picker no longer reads WDL as its source surface, and keyboard movement polls Silk.NET input in addition to ImGui key state; this is still not a multi-tile/textured/object-rendering viewer
- Apr 25 FOV correction: all viewer defaults should be `45` degrees, not `60`; remaining `60` values in viewer paths must be non-camera layout constants or explicit historical capture data, not projection defaults
- Apr 25 hardening correction: WDL is now disabled in `WowViewerWorldRuntimeBridge` itself, not only in the UI/session layer, and the world preview camera now uses position/yaw/pitch state like `MdxViewer` instead of a persistent look-at target
- Apr 25 multi-tile correction: the World Session runtime now builds a bounded `3x3` active ADT terrain window around the selected tile and the GPU preview renders that ADT quilt; terrain existence, not placement-catalog existence, is the authority for loading a tile
- Apr 25 terrain renderer correction: the world GPU preview no longer uses the CPU-sampled one-color-per-vertex terrain shortcut as the active texturing path; it now ports the working `MdxViewer` terrain GPU contract more directly with per-tile diffuse arrays, per-tile `64x64x256` alpha arrays, per-vertex layer indices, and shader-side terrain blending. Live GUI proof is still required.

## Apr 24, 2026 Course Correction - Stop Treating The Preview As The Viewer

The current `wow-viewer` World Session still fails the user's live-use test:

- minimaps do not reliably load in the active shell
- `WASD` and `Q/E` are not functioning as dependable viewport movement
- mouse wheel input can scroll the containing panel instead of belonging to the 3D viewport
- the app still loads and renders one selected ADT tile at a time
- terrain is height-shaded only; it does not render `MCLY`/`MCAL` texture layers
- UI copy and defaults still over-emphasize WDL-backed spawn/reference data even though the viewer needs to be ADT-first
- WMO, MDX, and M2 placements are markers and inspector rows, not rendered in-world objects
- world-layer controls live in the navigator/sidebar rather than as a viewer/editor options bar
- debug views still compete with viewer space instead of being strictly secondary diagnostics
- fixed side lanes make the app feel cramped and unlike either `MdxViewer` or the original `WoWEdit` working surface

This plan therefore changes priority. The next work should not be more explanatory UI around the same bounded frame. The next work must turn the center area into an owned viewport with editor-style controls.

### New Proof Rule

A `wow-viewer` world-viewer slice is not complete unless all relevant proof is true:

- the viewport owns mouse-wheel, drag, and keyboard focus while hovered or focused
- `WASD` movement and `Q/E` vertical movement work in the live app, with visible camera-speed control
- the containing panel does not scroll when the user is trying to fly or zoom the world camera
- minimap tiles load for the active client root and the user can choose or reload world tiles from the map
- the center view is free of debug/software-preview clutter by default
- the proof says whether it is still single-tile, multi-tile, textured terrain, or object-rendering capable

## Why This Reset Exists

- the current `wow-viewer` shell drifted into too many overlapping windows and duplicated information surfaces
- world discovery, spawn, minimap, and world-open paths became hard to reason about because each feature assembled its own partial workflow
- the current viewer-side archive and loose-file access patterns are slower and less coherent than the proven dataset-tooling path in [`CreateArchiveCatalog()`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs:2729)
- the user explicitly wants the feel and usability of [`MdxViewer`](../src/MdxViewer/ViewerApp_Sidebars.cs) preserved, not reinvented
- `MdxViewer` failed to fully comprehend skyboxes and backdrop composition; the reset must not repeat that mistake by treating terrain as the whole world
- WoW-like worlds should be modeled as layered backdrops around the camera plus a rigid Z-axis terrain quilt: sky spheres or domes, skybox/backdrop models, fog and haze, far WDL, detailed ADT terrain, liquid, WMOs, doodads, overlays, and diagnostics

## Source Of Truth

### UI and interaction source of truth

- [`DrawNavigatorPanelContent()`](../src/MdxViewer/ViewerApp_Sidebars.cs:461)
- [`DrawWorldOverviewContent()`](../src/MdxViewer/ViewerApp_Sidebars.cs:487)
- [`DrawMapDiscoveryContent()`](../src/MdxViewer/ViewerApp_Sidebars.cs:558)
- [`DrawUnifiedToolSidebar()`](../src/MdxViewer/ViewerApp_Sidebars.cs:788)
- [`DrawEditorWorkspaceNavigator()`](../src/MdxViewer/ViewerApp_Workspaces.cs:89)
- [`DrawEditorWorkspaceInspector()`](../src/MdxViewer/ViewerApp_Workspaces.cs:138)

### Runtime and rendering source of truth

- [`WowViewerWorldRuntimeBridge`](../../wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs)
- [`WorldGpuPreviewRenderer`](../../wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs)
- existing `wow-viewer` M2, WMO, and MDX GPU preview paths under [`wow-viewer/src/viewer/WowViewer.App`](../../wow-viewer/src/viewer/WowViewer.App)
- for terrain-family behavior, use the working `MdxViewer` path as the behavior reference before changing `wow-viewer`:
  - split-ADT sourcing and chunk ownership in [`StandardTerrainAdapter`](../src/MdxViewer/Terrain/StandardTerrainAdapter.cs)
  - GPU terrain-layer and alpha-array binding in [`TerrainTileMeshBuilder`](../src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs)
  - terrain layer blend/material behavior in [`TerrainRenderer`](../src/MdxViewer/Terrain/TerrainRenderer.cs)
- old [`WorldScene`](../src/MdxViewer/Terrain/WorldScene.cs) sky-dome and skybox handling is reference material only; new design ownership belongs in `wow-viewer`

### World composition source of truth

- the renderer should treat the world as ordered layers, not as one terrain mesh:
  - camera-centered spherical sky or dome backdrop
  - one or more skybox/backdrop model layers, selected by placement, zone, lighting, or explicit world metadata
  - fog and horizon haze that bridge the sky/terrain seam
  - WDL or other low-detail far terrain
  - detailed ADT terrain as a rigid Z-axis quilt
  - liquids
  - WMO and doodad geometry
  - editor/debug overlays
- early slices may use procedural colors or simplified gradients, but the architecture should keep room for decoded client skybox assets and shader-specific behavior later
- ADT terrain is the authoritative near-world surface for the viewer. WDL can support far terrain, coarse spawn/reference hints, or diagnostics, but it is not a substitute for loaded ADT data.

### Fast archive and loose-file source of truth

- dataset/converter archive path centered on [`CreateArchiveCatalog()`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs:2729)
- the key properties to preserve are:
  - bootstrap once per client root
  - prefer build-aware listfile/bootstrap inputs
  - include the legacy search-root behavior over client root plus `Data`
  - scan map MPQ archives once and reuse the catalog

## Reset Target Shape

The new shell should default to one dominant viewport plus supporting surfaces. The old three-lane idea is still useful, but the center viewport must behave more like `WoWEdit` or a game/editor viewport than an ImGui report window.

### Viewport and tool frame

- center viewport takes priority over panels
- default world camera vertical FOV is `45` degrees
- top or bottom tool strip owns common viewer controls:
  - camera mode and speed
  - sky/WDL/terrain/liquid/WMO/doodad/grid/overlay toggles
  - terrain texture visibility/debug modes
  - minimap and object-pick modes
  - reset camera and reload actions
- the viewport owns input:
  - click or hover to focus
  - right-drag free-look or left-drag orbit depending on active camera mode
  - wheel dolly/speed/zoom without scrolling the parent panel
  - `WASD` movement
  - `Q/E` vertical movement
  - `Shift` acceleration

### Supporting surfaces

1. **Navigator lane**
   - left side
   - source attach
   - map list
   - spawn chooser and minimap
   - load actions
   - runtime summary entrypoints

2. **Preview lane**
   - center
   - one active viewer surface for the current workspace
   - no duplicate preview-adjacent status windows
   - the world preview must compose sky/backdrop/terrain layers first, with technical proof text moved out of the normal view

3. **Inspector and diagnostics lane**
   - right side
   - selection details
   - object and runtime details
   - category-based diagnostics
   - no extra floating world detail windows by default
   - debug views live here or behind explicit debug tabs, never in the main viewport by default

## Non-Goals

- do not keep iterating on the current many-window `WowViewer.App` shell as the main direction
- do not port `MdxViewer` code wholesale into `wow-viewer` without refactoring ownership boundaries
- do not move new design ownership back into [`MdxViewer`](../src/MdxViewer)
- do not claim world-scene parity or editor parity before the shell and I/O cutover are stable
- do not call the current one-tile marker-only terrain frame a world viewer
- do not add more diagnostics as a substitute for rendered terrain textures, minimaps, objects, and working camera controls

## Required Architecture Change

Before more shell work, introduce a single viewer-facing I/O seam in `wow-viewer`.

### Apr 25, 2026 implementation status update

- landed app-side convergence in `wow-viewer/src/viewer/WowViewer.App/`:
  - `ViewerIoService` is now the thread-safe app-owned archive-catalog cache for viewer source signatures
  - world map discovery and spawn-picker flows were already using that seam; `LoadWorldSession()` now also acquires the active catalog from `ViewerIoService` and calls a shared-catalog `WowViewerWorldRuntimeBridge.Build(...)` overload
  - `WorldMinimapRenderer` now reads minimap tiles and `md5translate` candidates through the same `ViewerIoService` source key instead of a separate raw archive bootstrap path
- proof so far:
  - app build passed with `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/build-validation/wowviewer-viewerio-worldpath/`
- still required:
  - live GUI proof that minimap loading/reload behavior is improved on the active world-session surface
  - this seam convergence does not replace the remaining Recovery Slice 2 minimap UX/error-state work or the later textured-terrain and object-rendering slices

### New seam

Create one reusable viewer I/O service in `wow-viewer`, consumed by:

- world map discovery
- world bootstrap
- spawn and WDL lookup
- minimap tile loading
- placement ADT reads
- asset browsing
- standalone archive-backed M2, WMO, and MDX loads

### Service requirements

- one archive catalog per client root
- reuse across the app session instead of per-feature bootstrap
- legacy search roots matching [`BuildLegacySearchRoots()`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs:2742)
- build-aware bootstrap options and listfile resolution
- map MPQ scan reuse matching [`CreateArchiveCatalog()`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs:2729)
- overlay-first reads where loose files are present
- explicit invalidation on client root, build label, or loose overlay change

## Ordered Implementation Slices

The original slices below remain historical context, but the active route is now the recovery track here. Do these in order unless the user explicitly redirects.

### Recovery Slice 0 - Plan and checkpoint hygiene

- goal:
  - stop mixing shell polish, renderer work, and long-range world-runtime work in one vague thread
- scope:
  - record current failures as active plan blockers
  - keep any already-started input/UI patches clearly labeled until built and live-tested
  - avoid claiming visual or runtime closure from build-only proof
- proof:
  - plan names the real blockers and the next slice can be chosen without guessing

### Recovery Slice 1 - Viewport-owned input and WoWEdit-style tool strip

- goal:
  - make the center world surface behave like a real viewer viewport
- scope:
  - wrap the GPU world image in a no-scroll viewport child or equivalent owned surface
  - set the world viewport projection and visibility cone to a `45` degree vertical FOV
  - ensure wheel input does not scroll the parent panel when the viewport is hovered/focused
  - make `WASD` and `Q/E` movement work reliably in the live app
  - add a visible camera speed control and reset-camera command to a compact tool/options strip near the viewport, not buried in diagnostics
  - move software terrain preview and marker-canvas debug views out of the world viewport and into Inspector/Diagnostics
  - make side lanes resizable or move to a dockable/splitter layout
- proof:
  - live app run with the user's current world session proves `W`, `A`, `S`, `D`, `Q`, `E`, right-drag look, wheel dolly, camera speed, and parent-scroll isolation
  - screenshot shows the world viewport is not competing with debug views

#### Apr 25, 2026 camera-control correction

- the free-camera control reference is now explicitly the old `MdxViewer` contract:
  - right mouse button adjusts the free camera
  - yaw follows `yaw -= dx`
  - pitch follows `pitch -= dy`
  - planar strafe-right follows the same camera basis as `MdxViewer.Rendering.Camera.Move(...)`
- landed code changes in `wow-viewer`:
  - world GPU viewport now uses right-drag look instead of left-drag look
  - world `A/D` now map to negative/positive right over the corrected camera basis
  - model-output fly mode now also uses right-drag free-look with corrected vertical look sign
- proof so far:
  - app build passed with `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/build-validation/wowviewer-mdxviewer-camera-controls/`
- still required:
  - live GUI confirmation that right-drag look, `WASD`, `Q/E`, wheel dolly, camera speed, and parent-scroll isolation feel correct in the active world session

#### Apr 25, 2026 ADT/input recovery correction

- landed code changes in `wow-viewer`:
  - app startup and shared client-root changes normalize World Session layers to terrain on and WDL off
  - `LoadWorldSession()` forces `ShowTerrain=true` and `ShowWdl=false` before building the runtime frame request, so saved `ShowWdl=true` settings do not keep the main viewer on a WDL/far-reference path
  - the World Layers UI no longer exposes WDL as a normal live surface toggle and instead shows `Far WDL off`
  - the spawn/tile picker no longer opens, samples, or colorizes from WDL; it now reports and selects occupied ADT/WDT tiles
  - viewport movement now checks Silk.NET keyboard state as well as ImGui key state for `WASD`, `Q/E`, and `Shift`
- proof so far:
  - app build passed with `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/build-validation/wowviewer-adt-input-recovery/`
  - fixed local `H:\053-client`, `Kalidar`, tile `(27,34)` CLI proof from that build reports `wdl=False`, `Wdl:off`, `Terrain:256/256`, and terrain source `H:\053-client\Data\World\Maps\Kalidar\Kalidar.wdt#alpha-tile(27,34)`
- still required:
  - live GUI confirmation that `WASD`, `Q/E`, right-drag look, wheel dolly, and parent-scroll isolation work in the active app
  - Recovery Slice 3 remains open for the actual multi-tile terrain quilt; this patch only prevents WDL from masquerading as the live World Session surface

#### Apr 25, 2026 WDL/runtime hardening and free-camera correction

- landed code changes in `wow-viewer`:
  - `WowViewerWorldRuntimeBridge.Build(...)` forces `wdlVisible: false` regardless of incoming request flags
  - the bridge no longer has a private `ReadMapWdlTileData(...)` path for World Session frames
  - frame output now reports `WDL disabled for World Session; ADT terrain is the authoritative surface.` instead of a `.wdl` source path
  - `WorldFramePassOptions` defaults WDL visibility to false
  - `WorldGpuPreviewRenderer.WorldPreviewCameraState` now mirrors the legacy free-camera model: position plus yaw/pitch, with the target derived from `Position + Forward`
- proof so far:
  - app build passed with `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/build-validation/wowviewer-adt-camera-hardening/`
  - fixed local `H:\053-client`, `Kalidar`, tile `(27,34)`, even with `--show-wdl`, reports `wdl=False`, `Wdl:off`, `wdlTiles=0/0`, disabled WDL source text, and `Terrain:256/256` from `Kalidar.wdt#alpha-tile(27,34)`
- still required:
  - live GUI confirmation of camera feel
  - multi-tile ADT terrain loading remains unimplemented; single-tile ADT proof is not enough to close the World Session reset

### Recovery Slice 2 - Minimap loading recovery

- goal:
  - make the minimap a dependable navigation surface again
- scope:
  - trace why minimap tiles do not load in the current shell
  - use the shared viewer I/O/cache path for loose and archive minimap reads
  - show explicit loading/error state per minimap source instead of an empty panel
  - allow click/double-click tile selection and reload from the minimap
- proof:
  - fixed local `H:\053-client` with `Kalidar` and `Azeroth` shows minimap tiles in the live shell
  - the selected tile changes from the minimap without touching raw tile inputs

### Recovery Slice 3 - Multi-tile world frame

- goal:
  - stop treating one selected ADT tile as a world viewer
- scope:
  - keep ADT root terrain data as the primary loaded surface; WDL remains optional far/reference data only
  - build a small active tile window around the camera or selected spawn, starting with `3x3`
  - load adjacent terrain tiles through the same runtime frame contract or a new `WorldSceneFrame` contract
  - keep per-tile stage summaries and diagnostics available, but do not make them the user-facing model
  - avoid full infinite streaming until the bounded multi-tile path is stable
- proof:
  - center viewport renders at least a `3x3` terrain quilt on `H:\053-client` `Kalidar`
  - camera movement can cross a tile boundary without the terrain ending at a hard square

#### Apr 25, 2026 implementation status update

- landed code changes in `wow-viewer`:
  - `WowViewerWorldRuntimeFrameResult` now carries `ActiveTerrainTiles`, each with tile coordinates, ADT stage summary, terrain, liquid, and placement catalog data
  - `WowViewerWorldRuntimeBridge.Build(...)` now loads a bounded `3x3` ADT window around the selected tile instead of only the selected tile
  - tile-window loading is ADT-root driven; missing placement data produces an empty placement catalog instead of preventing terrain from loading
  - object placement inventory now aggregates WMO and MDX/M2 placements across the active tile window
  - terrain, liquid, and composition source counts now report the aggregate active window rather than only the selected tile
  - `WorldGpuPreviewRenderer` now builds terrain and hole-overlay buffers across all active ADT tiles
  - `Program.cs world-frame` and the desktop Inspector now report active ADT tile count and sample coordinates
- proof so far:
  - app build passed with `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -p:OutDir=i:/parp/parp-tools/output/build-validation/wowviewer-multitile-adt-quilt/`
  - fixed local `H:\053-client`, `Kalidar`, tile `(27,34)`, even with `--show-wdl`, reports `wdl=False`, `Wdl:off`, `active-adt-tiles: count=9`, `Terrain:2304/2304`, and terrain source `3x3 ADT window centered on (27,34); loaded 9 terrain tiles`
- still required:
  - live GUI confirmation that the camera can cross tile boundaries without hitting a terrain edge
  - terrain is still height-shaded debug geometry, not `MCLY`/`MCAL` textured terrain
  - WMO and MDX/M2 placements are still marker/runtime inventory, not rendered in-world geometry

### Recovery Slice 4 - Terrain texture layers

- goal:
  - render ADT terrain as terrain, not height-colored clay
- scope:
  - decode and bind `MCLY` texture layers and `MCAL` alpha data through `wow-viewer` shared I/O/runtime ownership
  - support alpha-era and later-client texture lookup differences explicitly
  - add viewer tool-strip toggles for textured terrain, layer debug, alpha debug, and grid/wire overlays
  - keep height-shaded terrain as a debug mode only
- proof:
  - `H:\053-client` `Kalidar` renders textured ground layers in the live viewport
  - a debug mode can isolate layer/alpha coverage without replacing the default viewer presentation

### Recovery Slice 5 - Liquids as visible world layers

- goal:
  - promote liquid data from stats to visible geometry
- scope:
  - render alpha and later-client liquid surfaces where the tile data reports water/liquid chunks
  - expose liquid visibility in the tool strip
  - keep liquid debug metrics in Inspector/Diagnostics
- proof:
  - a known watery fixed-root map shows liquid in the world viewport with the layer toggle on

### Recovery Slice 6 - In-world WMO rendering

- goal:
  - stop representing WMO placements only as markers
- scope:
  - reuse or extract existing WMO GPU preview rendering into placed world instances
  - apply placement transforms and rough culling first
  - keep portal/interior correctness as later parity work, but render visible exterior geometry now
- proof:
  - fixed root world session shows at least one placed WMO in-world at the correct terrain-relative location
  - marker-only fallback is no longer the default WMO presentation

### Recovery Slice 7 - In-world MDX/M2 rendering

- goal:
  - stop representing doodads only as markers
- scope:
  - reuse existing M2/MDX GPU preview paths for placed world doodads
  - apply placement transforms, scale, and basic material state
  - keep animation/material parity as follow-up, but render static placed objects now
- proof:
  - fixed root world session shows placed doodads in-world at correct positions
  - object navigator selection highlights or focuses an actual object, not just a marker

### Recovery Slice 7b - Model-output asset and PM4 overlay parity

- goal:
  - make model-output scenes useful for reconstruction review, not just terrain-preview islands
- scope:
  - place real M2/MDX and WMO assets into model-output scenes when the source data contains valid placements
  - layer PM4 overlay geometry or markers into the same camera space so missing or reconstructed world pieces are visible beside the client data
  - share the same placement transform, culling, and selection contracts used by the world viewer where possible
  - keep PM4 overlay rendering clearly labeled as overlay or reconstruction aid, not as decoded ADT/WMO ground truth
- proof:
  - a fixed-root model-output scene displays terrain plus at least one placed WMO or doodad asset
  - PM4 overlay data can be toggled independently and aligns with the same scene coordinates
  - object selection can distinguish client placements from PM4 overlay items

### Recovery Slice 8 - Skybox and light-selected backdrops

- goal:
  - replace procedural sky placeholders with real source selection where possible
- scope:
  - source 0.5.3 sky/backdrop selection from the appropriate sky/light metadata and known alpha assets
  - support later-client `LightSkybox.dbc` and WMO `MOSB` as separate source families
  - keep layered sphere/dome/backdrop ordering explicit
- proof:
  - at least one fixed-root map activates a real decoded or selected backdrop layer instead of only the procedural gradient

### Recovery Slice 9 - WoWEdit-style workspace finish

- goal:
  - make the app feel like a useful world viewer/editor surface rather than a diagnostics app
- scope:
  - top tool strip for high-frequency actions
  - optional vertical tool palette for mode selection
  - resizable/dockable navigator and inspector lanes
  - status bar for coordinates, tile, fps, selection, and load state
  - diagnostics as an opt-in secondary surface
- proof:
  - screenshot-level review against the original `WoWEdit` reference shows the viewport dominates, tools are discoverable, and the sidebars no longer choke the scene

### Slice 1 - Shared viewer I/O service

- goal:
  - replace repeated archive bootstrap and per-feature read patterns with one viewer-owned shared service
- scope:
  - extract a service from the dataset-tooling archive setup in [`CreateArchiveCatalog()`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs:2729)
  - wire it into world discovery, world bootstrap, spawn/WDL, minimap, and asset browse flows
- proof:
  - world map list loads immediately on known roots
  - minimap tile requests no longer perform independent archive bootstrap churn

### Slice 2 - Shell reset to MdxViewer layout

- goal:
  - replace the current floating-window shell with one navigator lane, one preview lane, one inspector and diagnostics lane
- scope:
  - make [`DrawNavigatorPanelContent()`](../src/MdxViewer/ViewerApp_Sidebars.cs:461) the structural reference
  - make [`DrawUnifiedToolSidebar()`](../src/MdxViewer/ViewerApp_Sidebars.cs:788) the right-lane reference
  - stop exposing duplicated world detail windows by default
- proof:
  - the app opens with one coherent world workflow instead of panel hunting

### Slice 3 - World map list and spawn interaction port

- goal:
  - port the working world-map and spawn workflow from `MdxViewer`
- scope:
  - copy the interaction model from [`DrawMapDiscoveryContent()`](../src/MdxViewer/ViewerApp_Sidebars.cs:558)
  - copy the overview and minimap behavior from [`DrawWorldOverviewContent()`](../src/MdxViewer/ViewerApp_Sidebars.cs:487)
  - make load and spawn selection obvious and one-click or two-click predictable
- proof:
  - the user can load a map and choose a start tile without touching raw input fields unless they want to

### Slice 4 - World session load path cleanup

- goal:
  - make world bootstrap and world frame loading dependable on fixed roots
- scope:
  - keep the existing runtime bridge but put it behind the new shared I/O seam
  - remove redundant per-feature task churn and stale-result races
  - preserve the readable auto-tile fallback logic already added in [`ResolveTileAndPlacements()`](../../wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs:515)
- proof:
  - fixed roots like `H:\053-client` and the Wrath roots load reliably through the live shell

### Slice 5 - Preview and inspector reattachment

- goal:
  - keep the current GPU/runtime consumers but attach them behind the reset shell
- scope:
  - leave existing GPU consumers in place
  - re-home status, selection, and runtime detail into the right-lane inspector and diagnostics surface
  - make the world preview a viewer-first composed image instead of a diagnostic report: sky/backdrop before terrain, terrain as a Z-axis quilt, diagnostics collapsed unless requested
- proof:
  - no duplicate preview-adjacent state windows remain necessary

### Slice 5b - Sky and backdrop composition foundation

- goal:
  - establish sky/backdrop as first-class world-render layers, not a flat clear color behind terrain
- scope:
  - add a camera-centered spherical sky pass to the `wow-viewer` world preview
  - keep it controlled by the existing world `ShowSky` option
  - define a future seam for multiple skybox/backdrop layers, decoded client skybox assets, fog/haze coupling, and shader-specific material behavior
  - target a 0.5.3-leaning color and atmosphere until real asset-backed sky selection lands
- proof:
  - opening a world frame renders sky before terrain in the center preview
  - disabling `Sky` removes the backdrop layer
  - this is documented as foundational atmosphere work, not final WoW skybox parity

#### Apr 24, 2026 implementation status update

- landed in `wow-viewer`:
  - `WorldGpuPreviewRenderer` now has a camera-centered spherical sky pass
  - `WorldRenderCompositionFrame` and `WorldRenderCompositionBuilder` now expose ordered sky/backdrop/far-terrain/terrain/liquid/object/overlay layer state
  - `WowViewerWorldRuntimeFrameResult` now carries classified `SkyboxBackdropInstances`
  - `WorldSkyboxBackdropClassifier` covers obvious later-client M2/MDX backdrop paths and the alpha-era `.mdl` case such as `Environments\Stars\stars.mdl`
  - classified backdrop placements now feed a procedural second spherical shell in the GPU sky shader, with deterministic tint/seed/strength derived from the source paths
- current proof:
  - focused composition tests pass
  - app build passes
  - fixed local `H:\053-client` proofs on `Shadowfang` and `Kalidar` report no classified backdrop placements on the tested tiles, so live visual activation of the procedural shell still needs a source tile or DBC/light-driven skybox record
- remaining work:
  - real 0.5.3 sky selection should come from Light/Skybox-era metadata, WMO `MOSB`, or decoded backdrop model assets rather than only ADT placement-path classification
  - procedural shell shading is a foundation placeholder, not native-client skybox parity

### Slice 6 - Standalone consumer regrouping

- goal:
  - make standalone M2, WMO, and MDX flows use the same shell grammar as the world path
- scope:
  - shared navigator lane for client/file source and asset browse
  - preview in center
  - selection/info/diagnostics in right lane
- proof:
  - all main workspaces follow one shell pattern

### Slice 7 - Cutover documentation and cleanup

- goal:
  - record the reset so future implementation does not drift back into the broken shell model
- scope:
  - update continuity docs and cutover notes
  - clearly mark `MdxViewer` as UI reference input only, not the new code owner
- proof:
  - future chats route new shell work into `wow-viewer` with the reset plan as the baseline

## Validation Gates

Each slice should be checked against the fixed real roots already documented in [`data-paths.md`](../memory-bank/data-paths.md):

- `H:\053-client` with `Kalidar`
- `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft`
- the current saved-state world-session path in [`wowviewer_app_settings.json`](../../wow-viewer/src/viewer/WowViewer.App/bin/Debug/net10.0/output/settings/wowviewer_app_settings.json)

Validation should distinguish:

- map discovery works
- spawn preview works
- world bootstrap works
- world frame works
- minimap tiles visibly load in the live shell
- viewport input works in the live shell
- the world frame is single-tile or multi-tile
- terrain is height-debug-only or texture-layered
- WMO and doodads are markers-only or rendered geometry
- UI is understandable without panel hunting
- the parent panel does not scroll when the user is flying or zooming the viewport
- the proof used build/test only, CLI real-data proof, or live GUI proof

## Acceptance Criteria For The Reset

The reset is successful when:

- the shell feels recognizably like `MdxViewer`
- the center world viewport feels closer to `WoWEdit`: dominant, navigable, and tool-driven
- `wow-viewer` remains the code owner
- viewer I/O is routed through one fast shared service
- map discovery, spawn, minimap, and world load behave consistently on fixed real roots
- `WASD`, `Q/E`, mouse look, wheel dolly, and camera speed controls work reliably
- the active world is not limited to a visible single-tile island
- terrain renders `MCLY`/`MCAL` texture layers by default, with height shading demoted to debug mode
- WMO, MDX, and M2 placements render in-world instead of only as markers
- sky/backdrop remains layered and can graduate from procedural placeholders to real sky/light-selected sources
- the number of normal-use windows is reduced to a stable and comprehensible minimum
- the center world surface reads as a viewer scene with sky/backdrop/terrain composition, not as a diagnostics dump
