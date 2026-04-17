# wow-viewer Viewer App Cutover Plan

## Status

- status: active
- intent: replace `gillijimproject_refactor/src/MdxViewer/ViewerApp*` as the design owner with a real viewer app in `wow-viewer`
- current verified floor:
  - `wow-viewer/src/viewer/WowViewer.App` now has a real Silk.NET + ImGui desktop shell
  - the desktop shell and `m2-frame` share one `M2PreviewLoader` path over `wow-viewer` runtime code only
  - real fixed-root proof exists through `WowViewer.App m2-frame` on `H:/CLIENTS/WoW335/3.X_Retail_Windows_enUS_3.3.5.12340/World of Warcraft`, `Creature/Wolf/Wolf.m2`

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
- `wow-viewer` does not yet own:
  - a full GPU-backed viewer renderer
  - world session bootstrap and map loading
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

### Slice 07 - Shell Surface Expansion

- target problem:
  - even after a world consumer exists, the app still needs structured navigation, inspector, diagnostics, and asset panels
- implementation scope:
  - add shell panels in wow-viewer for navigator, selection, diagnostics, and current-world status
  - rebuild only the surfaces needed by the new runtime consumer instead of copying old `ViewerApp` wholesale
- proof goal:
  - the new app is usable for bounded world inspection without falling back to old `ViewerApp`

### Slice 08 - Legacy Cutover Review

- target problem:
  - once the new app owns real viewer behavior, the old viewer boundary must be stated explicitly
- implementation scope:
  - document which remaining `MdxViewer` surfaces are compatibility-only or editor-only
  - stop treating `ViewerApp` as the default future home for viewer design changes
- proof goal:
  - continuity docs and repo guidance point future viewer work at wow-viewer first

## Implementation Rule

- each slice should land with one small proof and one honest boundary
- if a slice only proves build behavior, say so
- if a slice has real-data proof, state exactly what asset or client root it used
- do not mix world-runtime extraction, panel rebuilds, and renderer replacement in the same step

## Immediate Next Slice

- slice 03: standalone asset workspaces
- reason:
  - the app now has both persisted state and a typed session boundary, so the next clean step is reorganizing the shell around explicit workspaces instead of one monolithic control surface
  - that gives the M2 path a stable home before any GPU preview or world bootstrap work starts