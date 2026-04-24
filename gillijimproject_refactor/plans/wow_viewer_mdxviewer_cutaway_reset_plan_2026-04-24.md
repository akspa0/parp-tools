# wow-viewer MdxViewer Cut-Away Reset Plan

## Status

- status: active reset plan
- intent: stop iterating on the current fragmented [`WowViewer.App`](../../wow-viewer/src/viewer/WowViewer.App/WowViewerDesktopApp.cs) shell and instead port the working interaction model from [`MdxViewer`](../src/MdxViewer/ViewerApp_Sidebars.cs) into `wow-viewer`
- design rule: treat [`MdxViewer`](../src/MdxViewer) as the UI and interaction reference, but keep `wow-viewer` as the long-term code owner for runtime, rendering, and shared file I/O
- viewer-first rule: `wow-viewer` must act as a world viewer first and a diagnostics/tooling surface second; diagnostic panels are subordinate to the composed world image
- aesthetic target: prioritize the 0.5.3 client feel for this project, because the long-term purpose is exploratory data tooling and low-resolution visual restoration over early-world data, not a generic modern asset inspector

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

### Fast archive and loose-file source of truth

- dataset/converter archive path centered on [`CreateArchiveCatalog()`](../../wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs:2729)
- the key properties to preserve are:
  - bootstrap once per client root
  - prefer build-aware listfile/bootstrap inputs
  - include the legacy search-root behavior over client root plus `Data`
  - scan map MPQ archives once and reuse the catalog

## Reset Target Shape

The new shell should default to three durable surfaces only:

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

## Non-Goals

- do not keep iterating on the current many-window `WowViewer.App` shell as the main direction
- do not port `MdxViewer` code wholesale into `wow-viewer` without refactoring ownership boundaries
- do not move new design ownership back into [`MdxViewer`](../src/MdxViewer)
- do not claim world-scene parity or editor parity before the shell and I/O cutover are stable

## Required Architecture Change

Before more shell work, introduce a single viewer-facing I/O seam in `wow-viewer`.

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
- UI is understandable without panel hunting

## Acceptance Criteria For The Reset

The reset is successful when:

- the shell feels recognizably like `MdxViewer`
- `wow-viewer` remains the code owner
- viewer I/O is routed through one fast shared service
- map discovery, spawn, minimap, and world load behave consistently on fixed real roots
- the number of normal-use windows is reduced to a stable and comprehensible minimum
- the center world surface reads as a viewer scene with sky/backdrop/terrain composition, not as a diagnostics dump
