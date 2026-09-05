# Implementation Plan: Viewer UI Consolidation Audit (Spec 223)

**Branch**: `223-ui-consolidation-audit`
**Spec**: [spec.md](spec.md) · **Inventory**: [surface-inventory.md](surface-inventory.md)
**Created**: 2026-09-04

## Technical Context

- Shell: ImGui via Silk.NET, `ViewerApp` partial classes in `src/viewer/WoWViewer/` (~35k lines
  across 25 partials). Tab UI (`_useTabUi=true`, default) + legacy dockspace UI (kept per operator).
- Profile machinery already exists: `WorkspaceMode` (top bar) → `WorkbenchTab` destinations →
  per-destination page selectors (`WorkbenchNavigator`). Consolidation reorganizes content, not
  the shell.
- Selection pipeline: `CollectSceneObjectPickHits` (Spec 211/156) feeds `SelectedObjectType` +
  indices in `WorldScene`; the Inspector consumes this — no new selection machinery.
- Doodad-set data: MODS/MODD parsing exists (`WmoDoodadSetSummaryReader*`, `WmoDoodadSetRange*`);
  render path supports active set. Only the control is missing.

## Constitution Check

- **Core-first**: no test project references the viewer, so Inspector content models that need
  unit tests (e.g., doodad-set resolution) live in `WowViewer.Core*`; the viewer keeps only
  rendering of the data. UI-only glue stays in the viewer (untestable by policy).
- **No working-reader changes**: this spec touches UI only; format readers frozen.
- **Single-action rule** (212 FR-032 → FR-6): Quick mirrors invoke the same methods as full panels.

## Phases

### Phase 1 — Unified Object Inspector (US2a, US3)

Build the new right-sidebar **Inspector** tool: one host, per-type sections **ADT / MDX / M2 /
WMO / PM4 / WL\***. Content model is a plain data payload (sections of label/value rows, actions,
and sub-blocks) so the same payload can later render as the Spec 212 HUD object.

- 223-T101: Inspector host in the right sidebar (new `WorkbenchTab` destination or a page under
  Inspect — plan decision: **new destination "Inspector"** replacing the legacy `##InspectorTabs`),
  consuming the selection pipeline. HUD-ready content model in `WowViewer.Core.Runtime.World.Inspection`.
- 223-T102: ADT section — absorb MCNK Explorer + ADT Chunk Investigation + TerrainChunkHoverOverlay
  + Terrain Lab MCNK page content (capability checklist from inventory §5).
- 223-T103: WMO section — absorb WMO group/doodad details; **add doodad-set switching** (MODS
  list → active set → re-render without map reload; selection and camera preserved).
- 223-T104: MDX/M2 section — absorb Model Info (Workspaces:286) + Inspect>Animations/Context.
- 223-T105: PM4 section — absorb PM4 "Selected PM4"/"Match details"/"Scene measurements".
- 223-T106: WL\* liquid section — absorb WL Liquid Investigation (Investigation:310).
- 223-T107: Hover overlays (SceneHoverAssetOverlay, TerrainChunkHoverOverlay) become compact
  summaries that deep-link into the Inspector; no duplicated detail blocks.
- **Gate A**: every absorbed surface's capabilities verified present (inventory checklist); old
  surfaces removed in the same change (FR-2/FR-3).

### Phase 2 — Duplicate retirement (US2)

- 223-T201: Retire floating windows absorbed by the Inspector (MCNK Explorer, Terrain Analysis
  tile pages, PM4 detail floaters, Log/RenderQuality/Perf floaters → Utilities pages).
- 223-T202: Merge the ×2 implementations (World Overview, World Maps, Chunk Clipboard) into one
  each; legacy UI keeps working via the same shared draw methods.
- 223-T203: Retire Terrain Workbench floating window after 222-T111 parity checklist.

### Phase 3 — Editor + Archaeology merge (US4 step 1)

- 223-T301: Move Editor-only content (converters, ML dataset, imports, editor task nav) into the
  Archaeology destination as an "Editor" page group; top-bar Editor mode routes to Archaeology.
- 223-T302: Cartography (222) lands under Archaeology: layer panel right-sidebar, synthesized
  minimap tab, save-merged-output (222-T108/T109/T109a/T110).
- **Gate B**: feature-parity walkthrough against the inventory — nothing lost.

### Phase 4 — True-editor split (US4 step 2)

- 223-T401: Split authoring/writing features into a dedicated Editor profile; Archaeology keeps
  analysis/inspection. Old separate tabs removed.

### Phase 5 — Quick per profile (US5)

- 223-T501: Quick content becomes a shared mirror component; each profile's Quick lists that
  profile's top controls (derived from the inventory), Fog End first in every profile.
### Phase 6 — Transport Fixes, MCNK Deep-Link, Shared UI Library & Documentation (US6)

- 223-T601: Transport Fixes — Unconditionally simulate and update active taxi ride route poses in `WorldScene.UpdateTaxiActorInstances()` during rides, regardless of UI selection filter; relax strict map/build string matching and pending preloads in `UpdateCameraPathPlayback()`.
- 223-T602: Terrain Inspector Fix & Chunk Pinning — Fix standalone model check in `BuildInspectorContent()` (`_worldScene == null && _terrainManager == null && ...`); add click-to-pin terrain chunk target (`_selectedTerrainChunk`); enrich chunk inspection with Area Table names, liquid info, and tile doodad counts; provide active World Overview fallback.
- 223-T603: Scene Tab Absorption — Merge 'Scene' pages (`Placements` and `LOD & Budget`) directly into the Inspector tab as first-class subtabs; remove top-level 'Scene' tab.
- 223-T604: Utilities Tab Dispersion — Disperse Render Quality, Lighting, Audio, Capture, and Performance into the Quick tab, Inspector, View menu, and Tools menu; remove top-level 'Utilities' tab.
- 223-T605: MCNK Flags Integration — Integrate MCNK Flag Overlays and Chunk Flags into the main Inspector ADT section; eliminate duplicate redundant ADT explorer panels.
- 223-T606: Central Shared UI Library — Create `WoWViewer.UI.SharedUiWidgets` to standardize section headers, `[?]` help pop-up ready widgets, action groups, and compact status readouts across the workbench.
- 223-T607: Universal Quick Tab & Profile State Preservation — Retain active `Quick` tab and all viewer settings when switching top-level profiles (`Viewer`, `Editor`, `Archaeology`).
- 223-T608: User Guide & Documentation — Update `docs/WoWViewer/USERGUIDE.md` with complete documentation for the streamlined 4-tab workbench, 3D Object Library, Imports/Exports dashboard, and updated controls.

## File Changes (expected)

- New: `src/viewer/WoWViewer/UI/SharedUiWidgets.cs`.
- Modified: `src/viewer/WoWViewer/ViewerApp_InspectorPayloads.cs`, `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`,
  `src/viewer/WoWViewer/ViewerApp_Workspaces.cs`, `src/viewer/WoWViewer/ViewerApp_ClickSelection.cs`,
  `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs`, `src/viewer/WoWViewer/ViewerApp_CameraPaths.cs`,
  `src/viewer/WoWViewer/Terrain/WorldScene.cs`, `src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs`,
  `src/viewer/WoWViewer/ViewerApp.cs`, `docs/WoWViewer/USERGUIDE.md`.

## Validation

- Build: `dotnet build wow-viewer/WowViewer.slnx -c Debug` per phase.
- Focused tests: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug`.
- Operator walkthrough: verify taxi ride follows path, camera path plays smoothly, Inspector shows ADT/MCNK flags without selection, and Quick tab stays consistent across profiles.