# UI Surface Inventory v2 — Spec 227

**Started**: 2026-09-06
**Source baseline**: current `223-ui-consolidation-audit` worktree
**Scope of this task**: source-audited workbench roots and operator-named duplicate families.
**Not yet evidenced**: screenshots, visual styling, interaction count, minimap teleport, and any
runtime reachability. A `screenshot pending` entry is not proof that the surface looks or behaves
as the source suggests.

## Inventory rules

- An item may be changed only after its row has a named replacement ID.
- `Source-audited` establishes the current code host only. `Operator-verified` is required for
  visual, input, or three-interaction claims.
- A route/link row retains the target's data/action authority; it must not reproduce its data.
- Legacy-shell rows remain in scope because Spec 223 explicitly retains that shell pending Spec 212.

## A. Workbench roots and current page surfaces

| ID | Profile | Surface / source host | Current authority | Duplicate family | Disposition and replacement | Evidence |
|---|---|---|---|---|---|---|
| UIV2-SHELL-001 | All | Four top buttons: Quick, Inspector, Editor, Archaeology — `ViewerApp_Sidebars.cs::DrawRightSidebar` | `WorkbenchNavigator` labels and `OpenWorkbenchTab` route | None | Keep; this is the root route for all rows below. | Source-audited; screenshot pending |
| UIV2-QUICK-001 | Viewer | Quick controls — `ViewerApp_Sidebars.cs::DrawQuickControlsContent` | Existing Quick control actions | Inspector/settings overlap to audit | Keep pending T002; no replacement selected. | Source-audited; screenshot pending |
| UIV2-INSP-001 | Viewer | Inspector Context — `ViewerApp_Sidebars.cs::DrawInspectorContextPage` and `DrawUnifiedInspectorContent` | `ViewerApp_InspectorPayloads.cs` payload builder | Inspector | Keep as candidate authoritative object-details host; final per-type mapping pending UIV2-INSP-002. | Source-audited; screenshot pending |
| UIV2-INSP-002 | Viewer | Inspector Placements and LOD & Budget — `ViewerApp_Sidebars.cs::DrawInspectorWorkbenchSubTabContent` | Existing placements / LOD actions | Inspector | Keep; map per-object duplication in T013 before source changes. | Source-audited; screenshot pending |
| UIV2-EDITOR-001 | Editor | Tasks & Workspace — `ViewerApp_Sidebars.cs::DrawEditorWorkbenchSubTabContent`, `ViewerApp_Editor.cs::DrawArchaeologyEditorTasksSubTab` | Existing editor task host | Inspector | Keep pending v2 comparison; no replacement selected. | Source-audited; screenshot pending |
| UIV2-EDITOR-002 | Editor | Converters — `ViewerApp_Sidebars.cs::DrawConvertersSubTabContent` | Existing dialog launch actions | None | Keep; no movement before T002 verifies the old paths. | Source-audited; screenshot pending |
| UIV2-EDITOR-003 | Editor | 3D Object Library — `ViewerApp_Editor.cs::DrawRosettaObjectLibrarySubTab` | Existing Rosetta/object-library action | Inspector | Keep pending T013 per-object mapping. | Source-audited; screenshot pending |
| UIV2-EDITOR-004 | Editor | Imports & Exports — `ViewerApp_Editor.cs::DrawArchaeologyEditorImportsSubTab` | Existing import/export dialogs | Minimap | Keep pending T002; its synthesized-minimap launcher must not be removed by inference. | Source-audited; screenshot pending |
| UIV2-EDITOR-005 | Editor | Terrain Lab — `ViewerApp_Sidebars.cs::DrawTerrainLabSubTab` | Existing terrain-lab controls | Weak-signal; Inspector | Route/link candidate only after UIV2-ARCH-001 and UIV2-INSP-001 are operator-verified. | Source-audited; screenshot pending |
| UIV2-EDITOR-006 | Editor | Population — `ViewerApp_Sidebars.cs::DrawPopulationSubTabContent` | Existing population controls | None | Keep pending T002. | Source-audited; screenshot pending |
| UIV2-ARCH-001 | Archaeology | Weak Signal & Stratigraphy — `ViewerApp_Sidebars.cs::DrawArchaeologyWorkbenchSubTabContent` → `DrawTerrainControlsAdjustmentWeakSignalContent` | Spec 194 weak-signal owner | Weak-signal | Candidate authority for UIV2-EDITOR-005; confirm every control and route before conversion. | Source-audited; screenshot pending |
| UIV2-ARCH-002 | Archaeology | UniqueId Timeline — `ViewerApp_Sidebars.cs::DrawArcheologyRangeSubTab` | Existing UniqueId archaeology state/action | Inspector | Keep pending T002/T013. | Source-audited; screenshot pending |
| UIV2-ARCH-003 | Archaeology | Layers & Provenance — `ViewerApp_Sidebars.cs::DrawArcheologyLayersSubTab` | Existing archaeology layer data | None | Keep. | Source-audited; screenshot pending |
| UIV2-ARCH-004 | Archaeology | Playback & Capture — `ViewerApp_Sidebars.cs::DrawArcheologyPlaybackSubTab`, `DrawArcheologyCaptureSubTab`, `DrawCapturePanelContent` | Existing capture / path actions | Capture | Keep; Phase 6 runtime result remains separately operator-owned. | Source-audited; screenshot pending |
| UIV2-ARCH-005 | Archaeology | PM4 Analysis — `ViewerApp_Sidebars.cs::DrawPm4SubTabContent` | Existing PM4 tooling | Inspector | Keep pending T013. | Source-audited; screenshot pending |
| UIV2-ARCH-006 | Archaeology | Cartography — `ViewerApp_Sidebars.cs::DrawArchaeologyCartographyContent` | Spec 222 cartography owner | Minimap | Keep; no minimap authority decision until T011. | Source-audited; screenshot pending |

## B. Duplicate-family matrix

| ID | Family | Present hosts | Proposed authority / replacement | Source-edit status | Required operator observation |
|---|---|---|---|---|---|
| UIV2-WEAK-001 | Weak-signal amplifiers | UIV2-ARCH-001 and UIV2-EDITOR-005; legacy weak-signal delegate `ViewerApp_Sidebars.cs::DrawWeakSignalContent` | UIV2-ARCH-001 is the Spec 194 candidate. It is not yet an approved removal target for the other hosts. | Blocked on T009 source comparison and owner walkthrough. | Verify the owner exposes the required weak-signal actions and every retained route reaches it. |
| UIV2-MINIMAP-001 | Minimap | Window `ViewerApp_MinimapAndStatus.cs::DrawMinimapWindow`; utilities sidebar `ViewerApp_Sidebars.cs::DrawUtilitiesMinimap`; legacy paths identified in Spec 223 inventory v1 | **Pending runtime decision.** Shared `MinimapHelpers.RenderMinimapContent` is rendering reuse, not a teleport-authority decision. | Blocked on T011 click/teleport matrix. | For each host: click location, expected tile, actual camera position, and drag/pan behavior. |
| UIV2-INSP-003 | Inspector repetition | UIV2-INSP-001; Terrain Lab UIV2-EDITOR-005; archaeology/editor task hosts; investigation/hover compatibility views | **Pending per-object mapping.** UIV2-INSP-001 is the candidate host, not yet the final authority for all types. | Blocked on T013. | For ADT, MDX, M2, WMO, PM4, and WL*: identify the one page that is usable and the former duplicate's route/link. |

## C. Legacy and Spec 223 reconciliation queue

These rows are deliberately open until T002 compares the older inventory with current source and
the operator captures the surface. Their presence prevents a v1 disposition from being misread as
completed or as authorization to remove a legacy route.

| ID | Previous Spec 223 disposition to re-check | Current source anchors | Replacement state | Evidence |
|---|---|---|---|---|
| UIV2-LEGACY-001 | Legacy dockspace/left-sidebar surfaces retained, not retired | `ViewerApp_Sidebars.cs::DrawLegacyLeftSidebar` and legacy shell callers | Retained pending Spec 212; enumerate child panels in T002. | Source-audited root; screenshot pending |
| UIV2-LEGACY-002 | World Overview / World Maps duplicate merge | `ViewerApp_Sidebars.cs` world sidebar sections; `ViewerApp_Workspaces.cs` workspace map surface | Pending source comparison; do not delete either path. | Source-audited anchors; screenshot pending |
| UIV2-LEGACY-003 | Floating Minimap and fullscreen variants merge | `ViewerApp_MinimapAndStatus.cs`; existing old minimap entry points | Replacement is UIV2-MINIMAP-001 after T011. | Source-audited anchors; screenshot pending |
| UIV2-LEGACY-004 | MCNK Explorer, ADT investigation, hover overlays, and Terrain Lab detail surfaces consolidate | `ViewerApp_Investigation.cs`, `ViewerApp_Sidebars.cs`, `ViewerApp.cs` hover overlay | Replacement is UIV2-INSP-003 after T013. | Source-audited anchors; screenshot pending |
| UIV2-LEGACY-005 | Floating weak-signal / UniqueId archaeology entries route to Archaeology | `ViewerApp_Sidebars.cs::DrawWeakSignalContent`, `DrawUniqueIdArchaeologyWindow` | Weak-signal target pending UIV2-WEAK-001; UniqueId target UIV2-ARCH-002. | Source-audited anchors; screenshot pending |

## D. Screenshot and interaction matrix

The operator creates one file per row under `evidence/screenshots/` using the suggested names
below. Each record must state the configured client root, viewer build, map/asset if applicable,
route taken, and result. Image assets stay in the same directory; this table links to neither an
image nor a claimed result until supplied.

| Record file | Rows covered | Required check | Status |
|---|---|---|---|
| `viewer-roots.md` | UIV2-SHELL-001, UIV2-QUICK-001, UIV2-INSP-001, UIV2-INSP-002 | Find Load, fog, wireframe, capture, and inspect; record interaction count. | Pending operator capture |
| `editor-pages.md` | UIV2-EDITOR-001 through UIV2-EDITOR-006 | Open every editor page and note its visible section/action areas. | Pending operator capture |
| `archaeology-pages.md` | UIV2-ARCH-001 through UIV2-ARCH-006 | Open every archaeology page; especially observe Playback & Capture. | Pending operator capture |
| `weak-signal.md` | UIV2-WEAK-001 | Compare owner and duplicate entry points without changing settings. | Pending operator capture |
| `minimap-input.md` | UIV2-MINIMAP-001, UIV2-LEGACY-003 | Click-to-teleport and pan/drag result for each host. | Pending operator capture |
| `inspector-mapping.md` | UIV2-INSP-003, UIV2-LEGACY-004 | One authorative surface for each ADT/MDX/M2/WMO/PM4/WL* object type. | Pending operator capture |
| `legacy-shell.md` | UIV2-LEGACY-001 through UIV2-LEGACY-005 | Retained legacy surface inventory and visible duplicates. | Pending operator capture |

## Gate result after T001

The source baseline exists, but the full inventory gate is **OPEN**. T002 must reconcile every
Spec 223 row, and T003 must add current-build evidence before any source consolidation starts.

## E. Spec 223 retire/merge disposition reconciliation (T002)

This table compares every retire, merge, keep, or move disposition recorded by Spec 223 inventory
v1 to the current source. `Reopened` is intentional: a source route or shared body does not prove
that a duplicate visual surface is gone or that its replacement is usable.

| V1 item / intended disposition | Current source route or shared body | V2 status and next authority decision |
|---|---|---|
| Quick — keep and upgrade | UIV2-QUICK-001 remains the visible Quick root. | Reopened for profile action ordering (T015); not a duplicate removal claim. |
| Inspect — authoritative object inspector | `DrawInspectorWorkbenchSubTabContent` routes Context, Placements, and LOD; `DrawUnifiedInspectorContent` supplies context. | Reopened as UIV2-INSP-001/002; per-object authority waits for T013. |
| Scene — merge with Inspect | `OpenWorkbenchTab(WorkbenchTab.Scene)` adapts to Inspector Placements or LOD. | Source route present; operator must confirm no visible Scene surface remains (T003). |
| Utilities — keep utility pages | `OpenWorkbenchTab(WorkbenchTab.Utilities)` retains its page index and opens Quick; `DrawUtilitiesSubTabContent` retains the content hosts. | Reopened: Quick expansion and utility discoverability require screenshots (T003) and action-order review (T015). |
| Experimental — reorganize Terrain Lab / PM4 / converters | Legacy adapter and `DrawEditorWorkbenchSubTabContent` retain Terrain Lab, Population, and converter routes; PM4 is also UIV2-ARCH-005. | Reopened: Editor Terrain Lab carries weak-signal and Inspector duplicates (T009/T013). |
| Editor — merged then true editor split | UIV2-EDITOR-001 through UIV2-EDITOR-006 remain a visible top-level Editor destination. | Reopened: current source has an Editor root, but its pages must be audited before any reorganization. |
| Archaeology — merged home / Cartography | UIV2-ARCH-001 through UIV2-ARCH-006 are the declared Archaeology pages. | Reopened for standardization and screenshot evidence; Cartography remains its Spec 222 owner. |
| World Overview x2 — merge implementations | `DrawSharedWorldOverviewSection` and `DrawWorldOverviewContent` are shared by the current hosts. | Source-body merge observed; two host surfaces remain under the retained legacy shell. T003 determines whether both still duplicate visible information. |
| File Browser — keep | `DrawNavigatorPanelContent` calls `DrawFileBrowserContent`. | Keep; record its reachability in `viewer-roots.md`. |
| World Maps x2 — merge implementations | `DrawSharedWorldMapsSection` and `DrawMapDiscoveryContent` are shared. | Source-body merge observed; duplicate-host status remains open under the retained legacy shell. |
| Phase Map Layers — move to Cartography | Previous phase-layer sidebar anchor remains outside the primary v2 roots; Cartography is UIV2-ARCH-006. | Reopened: do not remove any entry until T003 finds its visible route and Spec 222 supplies required parity. |
| Editor workspace task nav — merge with Archaeology | `DrawEditorWorkbenchSubTabContent` still hosts `DrawArchaeologyEditorTasksSubTab`. | Reopened: name and host remain Editor; any movement needs an inventory-approved replacement. |
| Chunk Clipboard x2 — one home | `DrawChunkClipboardContent` has multiple current call sites in `ViewerApp_Sidebars.cs`. | Reopened; exact visible-home choice awaits T002 child-surface screenshots and an inventory row before a source edit. |
| Model Info — candidate Inspector duplicate | `DrawModelInfoCoreContent` and `DrawModelInfoContent` remain separate helpers/call sites. | Reopened as Inspector mapping work (T013). |
| Camera — Quick authoritative | Existing `DrawCameraControlsContent` remains callable from multiple surfaces. | Reopened for action-order audit (T015); no data/action moved. |
| Minimap + fullscreen variants — one implementation | Window path `DrawMinimapWindow`, utilities path `DrawUtilitiesMinimap`, and shared `MinimapHelpers.RenderMinimapContent` remain. | UIV2-MINIMAP-001 remains pending T011; shared rendering is not interaction proof. |
| Settings — keep | `ViewerApp_Settings.cs` remains the settings host. | Keep; fog/render-quality overlap is reviewed with UIV2-QUICK-001 and Utilities, not removed now. |
| MCNK Explorer / ADT Chunk Investigation / TerrainChunkHoverOverlay — Inspector | `DrawMcnkExplorerContent` delegates to terrain investigation; `TryDrawTerrainChunkHoverOverlay` remains callable; Inspector Context exposes MCNK controls. | Reopened as UIV2-INSP-003; object/type mapping required before route/link conversion. |
| Terrain Workbench — retire after Cartography parity | Terrain Lab/selection code remains reachable from Editor; Spec 222 parity was not evidenced here. | Blocked by the existing Spec 222 parity task; do not retire. |
| Floating Chunk Clipboard — merged home | Legacy clipboard window calls the same `DrawChunkClipboardContent` body. | Reopened with the multi-host Clipboard row above. |
| UniqueId Archaeology / Weak Signal floating windows — Archaeology pages | `DrawUniqueIdArchaeologyWindow` and `DrawWeakSignalWindow` call existing content delegates; UIV2-ARCH-001/002 are current candidates. | UniqueId target is UIV2-ARCH-002; weak-signal conversion waits for T009 owner comparison. |
| Terrain Analysis — Archaeology | `DrawTerrainAnalysisSubTab` and `DrawTerrainAnalysisContent` remain reachable through Terrain Lab paths. | Reopened: v2 must select its final profile after duplicate mapping; no retirement is authorized. |
| Synthesized Terrain Minimap Export — Cartography | `DrawSynthesizedMinimapExportDialog` remains a dialog; UIV2-EDITOR-004 retains its launcher and UIV2-ARCH-006 owns Cartography. | Reopened: target relationship is documented, but workflow parity and user route need T003. |
| Camera Path / Capture Automation — Capture page | `DrawCameraPathWindow` and `DrawCaptureAutomationWindow` retain window hosts; UIV2-ARCH-004 renders capture/path content. | Reopened: current canonical visible location must be confirmed by the separate Spec 223 live gate and T003. |
| Perf / Render Quality floating windows — Utilities | `DrawPerfContent` and `DrawRenderQualityContent` have workbench hosts; Settings calls render-quality content too. | Reopened: render-quality duplicate remains unresolved; source does not choose the user-facing authority. |
| PM4 alignment/correlation/object-match floating entries — workbench pages | `DrawPm4SubTabContent` dispatches workbench PM4 content; legacy utility methods remain. | Reopened: UIV2-ARCH-005 is the current candidate, but final Inspector relationships wait for T013. |
| Log Viewer — Utilities Log page | `DrawLogViewer` delegates to `DrawLogViewerContent`; Utilities hosts that content. | Source delegation observed; T003 confirms whether the floating entry remains visible. |
| ML training / map/WMO converters / texture-transfer launchers — Editor | Dialog/window draw methods remain; Editor UIV2-EDITOR-002/004 provide current related hosts. | Reopened: retain launchers until T003 records the actual entry points and safe replacement rows. |
| Alpha masks / MCCV / heightmaps — Editor | Existing editor/import action paths remain. | Reopened with UIV2-EDITOR-004; no source change. |
| WDL Preview / Map Preview — Utilities or Archaeology | `DrawWdlPreviewDialog` remains a dialog route. | Unresolved by v1 itself; T003 must identify current user-visible route before a placement decision. |
| Hover overlays — concise pointer to Inspector | `DrawSceneHoverAssetOverlay` and terrain-hover helper remain; Inspector Content remains separate. | Reopened as UIV2-INSP-003; do not remove visual hover information without an accepted pointer route. |
| Client Build / Rosetta / Map Converter dialogs — keep | Modal dialog hosts remain distinct from sidebar sections. | Keep; no duplicate-data action identified. |
| WMO doodad-set switching — add to authoritative WMO surface | Source has WMO model controls, but v2 has not witnessed a selected-world-WMO route. | Unresolved; include the WMO check in `inspector-mapping.md` before new UI work. |
| Quick Fog and Settings Fog Defaults — one implementation, mirror as needed | Quick and Settings code paths remain distinct presentation hosts over current fog controls. | Reopened for T015; prior source repair does not prove discoverability or duplicate-free UX. |

**T002 result**: every v1 disposition now has a current source route/body or an explicit unresolved
state. The inventory gate remains **OPEN** because the matrix has no current-build screenshots or
interaction results.
