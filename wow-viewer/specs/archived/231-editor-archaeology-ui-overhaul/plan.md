# Plan — Spec 231 Editor & Archaeology Workspace UI Overhaul

Status: **Planned** (2026-09-07). Implementation runs in a fresh session; this plan is the
complete contract for that session. Follows AGENTS.md §7 (Spec Kit), §10 (god-class freeze —
owned service classes), §11 (SharedUiWidgets, single authoritative homes).

## Target information architecture

### Editor tab (4 pages — replaces today's 6)

| New page | Content (single home) | Today's scattered locations |
|---|---|---|
| **Placement & Objects** | PM4 object transform nudge/rotate/scale, alignment, match suggestions, reconcile, collection/graph tools, placement authoring entry | ViewerApp_Pm4Utilities selection/transform/correlation panels; ViewerApp_Sidebars selection panel pieces |
| **Terrain Tools** | Terrain template brushes, chunk clipboard + save (one copy), stratigraphy/weak-signal *editing* entry points, terrain lab | ViewerApp_Editor tasks page; Sidebars "Clipboard + Save" ×2 |
| **Data I/O** | Every export/import: PM4 JSON/OBJ set, correlation JSON, GLB scene/collision/map tiles, terrain layer export + import, synthesized minimap, VLM/ML harvest launchers | ViewerApp_Editor imports page; ViewerApp_Pm4Utilities export buttons ×5 |
| **Converters** | Map/WMO/M2-MDX/ADT converters + round-trip validation + converter regression harness (Spec 221) | ViewerApp_Sidebars ~6073–6110; Spec 221 surfaces |

### Archaeology tab (analysis only)

Keep: Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback & Capture,
PM4 Analysis (read-only outliner/research), Cartography (Spec 222). Remove: "Tasks & Workspace"
and "Imports & Exports" hosting (their content moves per the table above). The duplicate PM4/WMO
correlation panel in `ViewerApp_Sidebars.cs` dies; the survivor lives behind the Placement &
Objects page and the Inspector cross-links to it.

### Left sidebar (Viewer profile — touch only where shared)

No structural change beyond the 2026-09-07 de-congestion. Shared widgets only.

## Structural strategy (Spec 228 compliance)

God-class freeze applies: no new `ViewerApp_*.cs` partials, no new feature members on
`ViewerApp`/`WorldScene`. Extraction pattern per move:

1. Create one owned workbench service class per new page, e.g.
   `src/viewer/WoWViewer/Workbench/Pages/EditorDataIoPage.cs`,
   `EditorPlacementObjectsPage.cs`, `EditorTerrainToolsPage.cs`,
   `EditorConvertersPage.cs` (each ~<800 lines, receiving `ViewerAppContext` via constructor —
   the existing `Workbench/` folder already hosts `WorkbenchNavigator`; follow its style).
2. The service class receives a narrow context object (scene refs, status callback, path-picker,
   settings access) — never the god-class instance. Where a callback into god-class state is
   unavoidable, define a small interface implemented by the existing partial and inject it.
3. The god-class tab switch (`DrawEditorWorkbenchSubTabContent`,
   `DrawArchaeologyWorkbenchSubTabContent`) becomes a 4-way/6-way delegation to page objects —
   one line per page, no logic.
4. Moved methods are deleted from the partials in the same change that adds the page class
   (no dead code, no file growth). `ViewerApp_Sidebars.cs` (6,173 lines) and
   `ViewerApp_Pm4Utilities.cs` (4,451 lines) must **shrink** this spec.
5. `WorkbenchNavigator.GetEditorWorkbenchLabels()` returns the 4 new page labels; update
   `DrawEditorWorkbenchSubTabContent`'s switch and the migration
   (`NormalizeWorkbenchStateAfterLoad`) so saved bottom-tab indices remap: old 0→Placement &
   Objects, 1→Converters, 2→Placement & Objects (library), 3→Data I/O, 4→Terrain Tools, 5→
   Population (Population folds into Placement & Objects as a section unless it deserves its own
   page — decide at implementation with the inventory open; default: fold).

## Dedupe moves (FR-1) — each is a named change

| # | Move | From | To |
|---|---|---|---|
| D1 | PM4 JSON/OBJ export buttons (5 sites) | ViewerApp_Pm4Utilities ~378/448/1201/1782/3900 | Data I/O page (one draw site); other sites become `Open Data I/O` links carrying current selection |
| D2 | PM4/WMO correlation panel (2 copies) | Sidebars ~801–901 (Workbench) + Pm4Utilities ~1863–2007 | One `Pm4CorrelationPage` section under Placement & Objects; both former sites link |
| D3 | Clipboard + Save (2 copies) | Sidebars ~1868, ~5581 | Terrain Tools page (one copy) |
| D4 | Doodad-set combos (4) | Sidebars ~2281/3039/5283 + toolbar | Object page owns the full combo; toolbar keeps the hovered-WMO quick combo (FR-6) — document both in inventory v3 |
| D5 | Editor pages hosted in Archaeology | `DrawArchaeologyEditorTasksSubTab` / `DrawArchaeologyEditorImportsSubTab` | Deleted; content lands in Placement & Objects / Data I/O |
| D6 | Background-execution audit | any sync export/report still on the render thread | Apply the `ExportPm4ObjectsObjSet` pattern (Task.Run + status + re-entrancy guard); minimum set: correlation JSON, object-match report build, PM4 JSON dump |

## Background-execution contract (FR-7)

Pattern to copy: `ViewerApp_Pm4Utilities.ExportPm4ObjectsObjSet` (2026-09-07) —
`Task.Run` + `_pm4ObjExportRunning` guard + immediate/completion status. Audit list at
implementation: every `File.WriteAllText`/directory-walk triggered from a button on a
reorganized page. MPQ data source reads are safe off-thread (terrain streaming precedent,
`TerrainManager.MaxConcurrentMpqReads`).

## Phases and gates

| Phase | Content | Gate (receipt required) |
|---|---|---|
| **P0 — Inventory v3 baseline** | Run the Spec 227 T003/T004-style screenshot matrix for Editor + Archaeology tabs as they exist today; write `evidence/inventory-v3-baseline.md`; fold Spec 227's Editor-tab item into this spec (record supersession in 227's tasks.md with a dated note) | Baseline committed; operator acknowledges scope freeze on the 4-page IA |
| **P1 — Page skeletons + migration map** | Create the 4 page classes; switch `WorkbenchNavigator` labels; route old sub-pages to new pages (content still in old partials, called from page classes) | `dotnet build` 0 errors; old content reachable from new pages; no behavior change |
| **P2 — Data I/O consolidation (D1, D6)** | Move all export/import content into `EditorDataIoPage`; delete the 5 export sites + imports page from Editor partial; apply background contract | SC-1 grep receipt for D1; build/test clean; export smoke (small map) |
| **P3 — Placement & Objects (D2, D4, D5a)** | Consolidate transform/match/reconcile/collection + surviving correlation panel; delete duplicates | SC-1 receipts D2/D4; SC-5 code-block dedupe proof; build/test |
| **P4 — Terrain Tools + Converters (D3, D5b) & Archaeology de-hosting** | Clipboard+Save single copy; converters page; remove Archaeology editor hosting | SC-1 receipts D3/D5; build/test |
| **P5 — Navigation test + inventory v3 final** | Operator task test (US1–US3); inventory v3 rows for every moved/removed surface; STATUS/epic updates; final build/test | Operator smoke receipt; spec marked Implemented with user gates (→ Complete after operator acceptance) |

## Risks / mitigations

- **Saved-settings remap**: old bottom-tab indices persist in settings; the migration in
  `NormalizeWorkbenchStateAfterLoad` must map all six old indices (plan table above) — test with
  a pre-change settings file.
- **Cross-link selection preservation**: navigation links must carry the active PM4/instance
  selection; page classes receive selection state via the context object, not globals.
- **Hidden behavioral coupling**: the two correlation panels differ subtly (Workbench version
  binds `_selectedPm4WmoCorrelationMatchIndex`); diff them before deleting one and fold any
  unique behavior into the survivor (Phase P3 begins with that diff as its first task).
- **File budget**: `ViewerApp_Pm4Utilities.cs` shrinking below 2,000 lines is a stated outcome;
  if any page class would exceed ~2,000 lines, split it in the same change.
