# wow-viewer Viewer App Cutover Plan

## Status

- status: active
- intent: replace `gillijimproject_refactor/src/MdxViewer/ViewerApp*` as the design owner with a real viewer app in `wow-viewer`
- current verified floor:
  - `wow-viewer/src/viewer/WowViewer.App` now has a real Silk.NET + ImGui desktop shell
  - the desktop shell and `m2-frame` share one `M2PreviewLoader` path over `wow-viewer` runtime code only
  - real fixed-root proof exists through `WowViewer.App m2-frame` on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2`
  - real fixed-root proof now also exists through `WowViewer.App world-bootstrap` on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Azeroth`

## Why This Plan Exists

- the user explicitly wants the old `ViewerApp` cut away, not treated as the permanent owner of viewer behavior
- the first wow-viewer desktop shell is real now, but it is still only an M2 preview host
- without a staged plan, the next steps risk collapsing back into another large `MdxViewer`-anchored rewrite

## Current Boundary

- `wow-viewer` now owns:
  - desktop app entrypoint and shell
  - shared M2 preview loading path
  - deterministic software visual preview for M2 runtime proof
  - runtime hash and submission diagnostics inside the new app
  - fixed-root world-session bootstrap over shared `Map.dbc` and WDT readers
- `wow-viewer` does not yet own:
  - a full GPU-backed viewer renderer
  - app-owned world runtime consumption over extracted services
  - a replacement for the broad `ViewerApp` panel set from `MdxViewer`

## Non-Goals For This Plan

- do not restart terrain editing, PM4 workbench, dataset-builder, or world editor work inside `MdxViewer`
- do not claim world-scene parity before a wow-viewer-owned world consumer exists
- do not merge all remaining `ViewerApp` concerns into one cutover step

## Ordered Slices

### Slice 01 - App State And Settings Persistence

- target problem:
  - the new wow-viewer app shell still behaves like a sessionless probe window
  - source selection, preview parameters, and app toggles are not yet persisted in the new repo
- implementation scope:
  - add a wow-viewer-owned app settings file under the app output tree
  - persist current source mode, archive root, virtual path, input path, profile index, sequence index, time, preview size, and basic window visibility toggles
  - keep this app-local and do not pull in `MdxViewer` settings infrastructure
- proof goal:
  - build passes
  - app shell loads persisted state on startup and saves it on close
- out of scope:
  - no world loading
  - no new renderer path
  - no recent-files UX or asset browser yet

### Slice 02 - Viewer Session Boundary

- target problem:
  - the app still treats every preview request as ad hoc fields on the desktop host
- implementation scope:
  - introduce a typed wow-viewer viewer-session contract for active asset source, build label, and current workspace mode
  - keep it app-owned for now, but shaped so later world/runtime consumers can share it
- proof goal:
  - the desktop host no longer owns raw source fields directly
  - one typed session object drives load requests and settings
- out of scope:
  - no world runtime extraction yet

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer/src/viewer/WowViewer.App/`:
  - `WowViewerSession.cs` now owns the typed app-local viewer session contract
  - the session currently carries workspace mode, typed asset source kind, build label, profile index, sequence index, time, and preview size
  - `WowViewerDesktopApp` now uses that session object instead of keeping raw source and preview fields directly on the host
  - `WowViewerAppSettings` now persists the session object instead of a flat source-field blob
  - `Program.cs` now parses `viewer` app bootstrap requests into a typed session object, while `m2-frame` still uses the narrower direct request path
- proof completed:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data loader proof still succeeds through the same app/runtime seam with `--build-label 3.3.5.12340` on `Creature/Wolf/Wolf.m2`
- current boundary:
  - this closes the app-side session boundary only; it does not yet create standalone workspaces, a GPU preview path, or world-session bootstrap

### Slice 03 - Standalone Asset Workspaces

- target problem:
  - the app currently only has one M2 preview surface and no explicit workspace split
- implementation scope:
  - turn the current M2 preview into a dedicated standalone workspace
  - add bounded placeholders and contracts for future WMO or MDX standalone consumers without claiming those slices are implemented
- proof goal:
  - the app shell is organized around explicit workspaces instead of one monolithic control window
- out of scope:
  - no world map viewer yet

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer/src/viewer/WowViewer.App/`:
  - `WowViewerSession.cs` now carries explicit standalone workspace modes for `M2`, `WMO`, and `MDX`
  - `WowViewerDesktopApp` now has a dedicated `Workspaces` window and view toggle so the shell is structured around explicit standalone workspace selection instead of only one generic M2 control surface
  - the M2 workspace remains the only implemented consumer in this slice
  - WMO and MDX now exist as honest placeholders with their own control, preview, and diagnostics surfaces stating that no live consumer is implemented yet
  - `WowViewerAppSettings` now persists workspace-window visibility, and `Program.cs` now accepts `--workspace m2|wmo|mdx` for desktop-session bootstrap
- proof completed:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- --help`
  - real-data loader proof still succeeds through the unchanged shared M2 runtime path on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2`, with the same runtime/render/visual hashes as slice 02
- current boundary:
  - this closes the shell-side standalone workspace split only
  - WMO and MDX are not implemented consumers yet
  - there is still no GPU preview renderer or world-session bootstrap in the app

### Slice 04 - GPU M2 Preview Consumer

- target problem:
  - the current preview is still the software visual snapshot, which is useful for proof but not a real viewer consumer
- implementation scope:
  - add a wow-viewer-owned GPU M2 preview renderer inside `WowViewer.App` or a supporting app-local layer
  - keep it standalone-first and do not tie it to world placement yet
- proof goal:
  - bounded real-data screenshots from the new app shell for fixed M2 assets
- out of scope:
  - no world scene, no map loading, no old ViewerApp parity claims

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer/src/viewer/WowViewer.App/`:
  - `M2GpuPreviewRenderer.cs` now owns an app-local GL consumer over `M2RenderFrame.DrawCommands` instead of building a second renderer contract outside the runtime frame
  - the GPU preview path uses runtime draw-command vertices, indices, texture bindings, and resolved effect flags to render a bounded standalone M2 preview into an offscreen framebuffer texture shown in the desktop shell
  - the existing software visual snapshot stays loaded as an explicit fallback and diagnostic reference instead of being deleted
  - `M2GpuPreviewCaptureRunner.cs` plus `Program.cs` now expose `m2-gpu-frame` for hidden-window BMP capture proof over the same app-local GPU renderer
  - `WowViewer.App.csproj` now references the vendored `SereniaBLPLib` decoder so archive-backed BLP textures can be sampled by the new preview path
  - `M2RenderFrame.cs` now carries the per-command material/effect state a real consumer needs, including diffuse or emissive color, alpha, blend mode, depth-write, alpha-test, transparency, additive state, and lighting flags
- proof completed:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data GPU proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- m2-gpu-frame --archive-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --virtual-path "Creature/Wolf/Wolf.m2" --build-label 3.3.5.12340 --sequence-index 0 --time-ms 0 --visual-size 512 --output "i:/parp/parp-tools/output/build-validation/wow-viewer-app-gpu-preview/wolf_335_gpu.bmp"`
  - the GPU proof artifact now exists at `output/build-validation/wow-viewer-app-gpu-preview/wolf_335_gpu.bmp` (`1048630` bytes)
  - the earlier `m2-frame` proof still preserved the existing Wolf hashes after this slice: runtime `9e9586068a443468ccec1abd62b3d717c0455e08999bd03beb21427a9df4ec30`, render-frame `177155d088dc8502be5b115b6b3d1a0fa67e75549cfe87c981bff6a8f8ac4122`, visual `b2fabb6da814c393ea149fb7321cbd3e05d24db8852f59dca35c755c29bfb177`
- current boundary:
  - this closes a bounded standalone GPU M2 consumer only
  - it is still a first-pass preview path over one primary texture stream and current runtime draw-command state, not native material parity or world-scene ownership
  - camera-only overlays, WMO, MDX, and world-session bootstrap remain later slices

### Slice 05 - World Session Bootstrap

- target problem:
  - the new app has no concept of game-root attach, build selection, or map session bootstrap yet
- implementation scope:
  - add wow-viewer-owned source attach flow for fixed local client roots first
  - add typed world session bootstrap state and map/open request flow
- proof goal:
  - the new app can open a bounded world session over a fixed client root and selected map input
- out of scope:
  - no full panel parity
  - no terrain editor or PM4 workbench yet

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer/src/viewer/WowViewer.App/`:
  - `WowViewerSession.cs` now carries typed `WorldSession` state for fixed client root, selected map input, and build label alongside the existing standalone asset session state
  - `WowViewerWorldSessionBootstrapper.cs` now owns the bounded attach/open flow over shared `MapDirectoryLookup`, `ArchiveCatalogBootstrapper`, `MapFileSummaryReader`, `WdtSummaryReader`, and `WdtTileIndexReader`
  - `WowViewerDesktopApp.cs` now exposes `World Session` as an implemented workspace with its own controls, summary surface, diagnostics, and honest boundary text stating that the slice stops at WDT/bootstrap proof rather than rendering
  - `Program.cs` now supports `--workspace world`, world-session viewer bootstrap arguments, and a direct `world-bootstrap` proof command
- proof completed:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data world bootstrap proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-bootstrap --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340`
  - that proof resolved `Azeroth -> Azeroth` via `Map.dbc`, opened `World\Maps\Azeroth\Azeroth.wdt` from archive data, reported `687/4096` occupied tiles, and summarized `MAIN` flags as `0x1:687`
- current boundary:
  - this closes bounded client-root attach plus map bootstrap only
  - there is still no world renderer, terrain/WMO/MDX placement consumer, or old `ViewerApp` panel parity in the new app

### Slice 06 - World Runtime Consumer Bridge

- target problem:
  - extracted runtime seams exist, but the new app does not consume them
- implementation scope:
  - consume the existing `WowViewer.Core.Runtime.World` visibility/pass seams from the new app host
  - keep `MdxViewer` as a reference input only when a missing algorithm still has to be extracted
- proof goal:
  - one bounded wow-viewer world view renders through runtime-owned seams
- out of scope:
  - no blanket replacement for all old overlay/editing tools

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer/src/viewer/WowViewer.App/`:
  - `WowViewerWorldRuntimeBridge.cs` now owns a bounded app-local world runtime frame builder over one selected ADT tile, consuming shared `WorldObjectVisibilityCollector`, `WorldObjectPassCoordinator`, `WorldFramePassCoordinator`, `WorldRenderFrameStats`, and `WorldRenderOptimizationAdvisor`
  - the bridge reuses the slice-05 attach/open flow, loads real ADT placement catalogs, resolves real WMO bounds plus real M2/MDX bounds from client data, and feeds those placements through the extracted runtime visibility/pass seams instead of inventing app-local fake object buckets
  - `WowViewerSession.cs` now persists optional world tile selection, `Program.cs` now exposes `world-frame`, and `WowViewerDesktopApp.cs` now shows a bounded top-down tile view plus runtime diagnostics for visible WMO/MDX counts, pending assets, pass routes, and optimization hints
- proof completed:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data runtime bridge proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340`
  - that proof auto-selected tile `(39,32)`, loaded `World\Maps\Azeroth\Azeroth_39_32.adt`, found `24` WMO placements plus `2991` MDX placements, admitted `24` visible WMO plus `2464` visible MDX through the shared runtime visibility collector, and executed the shared object/pass coordinators with `objectPhase=True`
- current boundary:
  - this is a bounded top-down world frame and runtime-summary consumer, not the final 3D world renderer or full `WorldScene` replacement
  - terrain, liquid, WDL, renderer batching backends, and broader panel rebuilds remain later slices

### Slice 07 - Shell Surface Expansion

- target problem:
  - even after a world consumer exists, the app still needs structured navigation, inspector, diagnostics, and asset panels
- implementation scope:
  - add shell panels in wow-viewer for navigator, selection, diagnostics, and current-world status
  - rebuild only the surfaces needed by the new runtime consumer instead of copying old `ViewerApp` wholesale
- proof goal:
  - the new app is usable for bounded world inspection without falling back to old `ViewerApp`

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer/src/viewer/WowViewer.App/`:
  - `WowViewerDesktopApp.cs` now carries app-local world selection state, interactive canvas picking, a `World Status` panel, a `World Navigator` panel, and a `World Inspector` panel over the bounded runtime frame instead of leaving the world workspace at top-down-canvas-only inspection
  - the navigator can filter WMO vs MDX, visible-only vs full placement inventory, and model-name or model-key text; selection can come from either the navigator list or direct canvas clicks
  - the inspector reports actual runtime-backed placement, bounds, visibility, and MDX pass-routing state from the shared world frame rather than inventing a second object-inspection contract
  - `WowViewerAppSettings.cs` now persists the new world-panel visibility toggles so the shell surface itself remains stable across app restarts
- proof completed:
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data runtime proof still succeeded via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340`
  - that proof still auto-selected tile `(39,32)`, loaded `World\Maps\Azeroth\Azeroth_39_32.adt`, found `24` WMO placements plus `2991` MDX placements, admitted `24` visible WMO plus `2464` visible MDX, and executed the shared object/pass phase with `objectPhase=True`
- current boundary:
  - this slice proves bounded shell usability for one-tile world inspection only; it does not prove the interactive desktop renderer, terrain/liquid submission, or broader `ViewerApp` parity
  - the next slice should review and state the remaining legacy boundary explicitly instead of conflating shell usability with renderer replacement

### Slice 08 - Legacy Cutover Review

- target problem:
  - once the new app owns real viewer behavior, the old viewer boundary must be stated explicitly
- implementation scope:
  - document which remaining `MdxViewer` surfaces are compatibility-only or editor-only
  - stop treating `ViewerApp` as the default future home for viewer design changes
- proof goal:
  - continuity docs and repo guidance point future viewer work at wow-viewer first

#### Apr 17, 2026 implementation status update

- landed in documentation and workflow guidance:
  - added `wow-viewer/docs/architecture/viewer-legacy-cutover-boundary-2026-04-17.md` as the explicit viewer ownership boundary, naming what `WowViewer.App` owns now and which remaining `MdxViewer` surfaces are compatibility-only versus legacy editor or archaeology work
  - `wow-viewer/README.md` now links to that boundary note so repo-local discovery no longer depends on memory-bank recovery alone
  - `.github/copilot-instructions.md` and `AGENTS.md` now require that boundary note plus this cutover plan before new viewer-app shell or cutover work, and they now include explicit viewer-app guardrails that keep long-range shell design out of `MdxViewer`
- proof completed:
  - this slice is documentation and workflow guidance only; no code or runtime behavior changed
  - the new proof is that the canonical repo instructions, README, cutover plan, and continuity files now all point future viewer work at `wow-viewer` first and classify remaining `MdxViewer` viewer surfaces explicitly
- current boundary:
  - this closes the documentation and workflow-routing side of the cutover review only
  - it does not claim renderer parity, terrain-runtime ownership, or editor-feature migration closure

### Slice 09 - World Frame Runtime Options

- target problem:
  - the bounded world app consumer still treated non-object passes as anonymous callbacks and could not drive runtime-owned stage or family gating
- implementation scope:
  - expand the runtime world-frame options contract so the current app consumer can control WMO/MDX family visibility and sky/WDL/terrain/liquid/overlay stage gating through runtime-owned options instead of host-local loose booleans
  - keep the slice bounded to pass-option ownership and current app-consumer proof; do not claim terrain or liquid renderer extraction yet
- proof goal:
  - the `world-frame` proof path changes real frame results when runtime pass options are toggled

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer`:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs` now exposes richer `WorldFramePassOptions` with runtime-owned flags for WMO/MDX family gating and sky/WDL/terrain/liquid/overlay stage gating while preserving the existing ordered pass contract
  - `wow-viewer/src/viewer/WowViewer.App/WowViewerSession.cs`, `Program.cs`, `WowViewerWorldRuntimeBridge.cs`, and `WowViewerDesktopApp.cs` now thread those options through persisted world-session state, CLI `world-frame --hide-*` flags, the bounded runtime frame request/result, and the current desktop controls or diagnostics

### Slice 10 - World Tile Stage Summary

- target problem:
  - the bounded world app consumer still zero-filled WDL, terrain, and liquid stage counts even after pass-option ownership moved into runtime
- implementation scope:
  - add a runtime-owned root-ADT stage-summary seam for WDL, terrain, and liquid counts over shared ADT readers
  - thread that summary through the bounded world-frame bridge and shell surfaces so non-object stage counts are real and option-sensitive
  - keep the slice bounded to summary ownership and current one-tile proof; do not claim terrain or liquid rendering extraction yet
- proof goal:
  - the `world-frame` proof path reports non-zero source terrain-side counts on a real tile and drops the active counts when `--hide-wdl`, `--hide-terrain`, or `--hide-liquid` disables those stages

#### Apr 17, 2026 implementation status update

- landed in `wow-viewer`:
  - `wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldTileStageSummary.cs` and `WorldTileStageSummaryBuilder.cs` now own the bounded root-ADT summary seam for WDL tile presence, terrain chunk counts, hole-bearing terrain chunks, liquid chunk counts, liquid layer counts, and visible liquid tile counts over shared `AdtSummaryReader`, `AdtMcnkSummaryReader`, and `AdtLiquidReader`
  - `wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs` now resolves the selected tile's root ADT through the same archive or loose-file path, carries the runtime-owned tile-stage summary in the bounded frame result, and uses it to populate active WDL or terrain or liquid stage counts instead of hard-coded zeros
  - `wow-viewer/src/viewer/WowViewer.App/Program.cs` and `WowViewerDesktopApp.cs` now report active-versus-source terrain-side counts for the bounded frame so CLI proof and desktop diagnostics expose the new runtime-owned summary directly
  - `wow-viewer/tests/WowViewer.Core.Tests/WorldTileStageSummaryBuilderTests.cs` now covers both the fixed development root ADT and a synthetic MH2O-bearing root ADT so the new runtime summary seam has focused terrain and liquid regression coverage
- proof completed:
  - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldTileStageSummaryBuilderTests`
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data runtime proof via `WowViewer.App world-frame` on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Azeroth`, showing non-zero terrain-side source counts and then zero active WDL or terrain or liquid counts when the corresponding `--hide-*` flags are applied
- current boundary:
  - this closes bounded non-object stage-summary ownership only
  - true terrain or WDL or liquid renderer extraction and overlay-stage ownership remain later slices
  - `wow-viewer/tests/WowViewer.Core.Tests/WorldFramePassCoordinatorTests.cs` now proves both the legacy ordered flow and the new disabled-layer behavior
- proof completed:
  - `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter WorldFramePassCoordinatorTests`
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
  - real-data option proof via `dotnet run --project i:/parp/parp-tools/wow-viewer/src/viewer/WowViewer.App/WowViewer.App.csproj -c Debug -- world-frame --client-root "H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft" --map Azeroth --build-label 3.3.5.12340 --hide-doodads`
  - that proof still auto-selected tile `(39,32)` and `World\Maps\Azeroth\Azeroth_39_32.adt`, but now reported `mdx=False`, `visibleMdx=0`, `updatedMdx=0`, `mdxOpaque=0`, and `mdxTransparent=0` while WMO counts remained active, proving the runtime-owned family gating changed the bounded frame result on fixed real data
- current boundary:
  - this closes runtime-owned pass-option control for the bounded world frame only
  - it still does not claim terrain/WDL/liquid renderer extraction, active desktop renderer parity, or full `WorldScene` host thinning

## Implementation Rule

- each slice should land with one small proof and one honest boundary
- if a slice only proves build behavior, say so
- if a slice has real-data proof, state exactly what asset or client root it used
- do not mix world-runtime extraction, panel rebuilds, and renderer replacement in the same step

## Immediate Next Slice

- next viewer-facing slice: a terrain/WDL/liquid or overlay runtime-service vertical slice in `wow-viewer`
- reason:
  - the bounded app consumer now owns runtime pass options instead of loose host-only toggles
  - the remaining meaningful gap is no longer option routing; it is explicit runtime-owned non-object stage or renderer service ownership for terrain, WDL, liquid, or overlay work