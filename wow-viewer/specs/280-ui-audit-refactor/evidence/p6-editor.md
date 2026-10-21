# P6 bounded editor / PM4 receipt — 2026-10-21

## Result and remaining gate

Editor owner labels and duplicate-section deletion are implemented. PM4 now has five inspection
tabs and one inspection export command home; matching/alignment/reconciliation UI and the manual
store are deleted within the authorized scope. **Parent integration and layout reset remain
required. This is not a passing P6 build/runtime gate.** No shell, host interface, app, navigator,
panels, settings, core matcher/offline service, rendering, tasks/status/memory, or commits were
changed by this slice. Unrelated dirty work was preserved.

## Changed source files

Paths are relative to `wow-viewer/src/viewer/WoWViewer/`:

| File | Change |
|---|---|
| `Workbench/Pages/EditorPlacementObjectsPage.cs` | “Tasks & Workspace” → “Archaeology Tasks”; “3D Object Library” → “Rosetta Object Library”; removed the PM4 Placement Tools section. Population and the real content delegates remain. |
| `Workbench/Pages/EditorDataIoPage.cs` | “Imports & Exports” → “Archaeology Imports”; deleted the second PM4 export home. New Map Creator and archaeology import delegate remain. |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.cs` | Service-owned five-page inspection dispatch: Overlay, Selection, Info, Outliner, Object Groups. Removed inline WMO/M2 matching/search/confirmation, match details, manual curation/save, alignment buttons/readouts, offline tab dispatch, and Editor Data I/O links. Selection owns the PM4 Exports section even without a selected object. |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Alignment.cs` → `Pm4WorkbenchService.Exports.cs` | Deleted alignment/correlation UI and their private workflows, old saved-match helpers/key builder, and correlation JSON export. Retained objects JSON/OBJ writers, camera framing, existing saved-overlay restoration, and cache invalidation required by existing callers. Added the four-command inspection export set and setup-failure reporting. |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.App.cs` | Deleted alignment-step, correlation, WMO/M2-match, and manual-curation UI state. Kept inspection/collection state. |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Host.cs` | Removed unused matching/store/settings/workspace/compatibility-navigation bridges; retained the host state still used for inspection and unchanged saved-overlay restoration/cache invalidation. |
| `Workbench/Services/Pm4Workbench/Pm4WorkbenchService.SceneGraph.cs` | Replaced Data I/O redirects with PM4 Selection & Exports navigation. Overlay report uses the existing managed `ExportDir`, reports the full path, and flushes the writer before success. |
| `Pm4ObjectMatchStore.cs` | Deleted the store and its store-specific entry type. No substitute shim. |
| `Workbench/Services/Archaeology/ArchaeologyPanelService.Editor.cs` and `.Host.cs` | Removed PM4 reconciliation panel, transient UI state, private preview/apply workflow, and now-unused editor-host bridge. **Kept `MaterializeReconciliationOutput()` unchanged** because `ViewerApp_Editor.cs:127` still calls it for existing operation materialization/undo/redo. |

Other archaeology changes and their file list are in [P5](p5-archaeology.md). The PM4 Collections
file and its existing graph/evidence/collection serialization logic are unchanged.

### T065 — deliberately gated, not silently completed

`Workbench/Pages/EditorWorkbenchPages.cs:32` still declares `MigrateLegacyEditorPageIndex()`.
The parent has **not yet reset persisted navigation**: the sole caller is still
`Workbench/Services/Settings/ViewerSettingsService.cs:287`, gated by the old navigation version.
Per the operator's “once parent resets layout” condition, neither the migration nor that
out-of-scope caller was prematurely deleted. After the reset lands, remove the caller's obsolete
remap block and then delete this method/documentation. Keep the four-page draw dispatcher.

## Exact parent integration / caller migrations

All paths below are relative to `wow-viewer/src/viewer/WoWViewer/`; line numbers are handoff anchors.

| Location | Required parent action |
|---|---|
| `ViewerApp.cs:514–515` | Delete `_pm4ObjectMatchStore` and `_pm4ObjectMatchEntries` fields, whose types were deleted. |
| `Workbench/Services/IViewerAppHost.cs:344–347` | Remove `Pm4ObjectMatchStore`, `Pm4ObjectMatchEntries`, `SavedPm4ObjectMatches`, and `TrySavePm4ObjectMatch` host members after their consumers below are removed. |
| `ViewerApp_Host.cs:337–341` | Remove the corresponding store/entries/saved-selections bridges and `TrySavePm4ObjectMatch()` forwarder; its target was deleted. |
| `Workbench/Services/Settings/ViewerSettingsService.cs:102–103,365–366` | Remove the manual-store bridges and load initialization. |
| `Workbench/Services/Settings/ViewerSettingsService.cs:108,391–408,577–584,749` | Remove obsolete saved-match selection persistence/loading and settings property. The loader still calls the deleted `BuildSavedPm4ObjectMatchKey()` at line 405; remove that entire selection-load path rather than recreating the helper. Preserve unrelated taxi/object-filter settings and overlay render-transform restoration. |
| `ViewerApp.cs:513,1732` and `ViewerApp_Host.cs:339` | Retire the old saved-match dictionary/selection DTO and bridge once settings references are gone. Do not remove unrelated scene object matching primitives or rendering state. |
| `Workbench/Services/EditorPanels/EditorPanelsService.cs:1022–1029` | Delete the **“Match to selected PM4 object”** button and its immediately preceding `SameLine()` only. Its action calls `TrySavePm4ObjectMatch()`. Keep Copy Path, Set as Active Placement Model, active-model state, and all unrelated Rosetta library actions. |
| `Workbench/Pages/ViewerAppContext.cs:22,28` and `ViewerApp_Editor.cs:59–60` | Remove the unused editor-only `DrawPm4Exports()` and parameterless `OpenPm4Workbench()` page-host declarations/implementations. The live export command set is called only by PM4 Selection; do not keep the editor bridge as a second draw route. |
| `ViewerApp.cs:55` | Remove the obsolete Correlation value from the app-owned `Pm4WorkbenchTab` request enum after its callers are migrated. Scoped service uses only Overlay/Selection requests; unknown old requests resolve to Selection, never to matching UI. |
| `Workbench/WorkbenchNavigator.cs:90,139` | Remove compatibility `Pm4BottomTab` and its old eight-label helper with parent P4. Scoped PM4 dispatch no longer depends on them. Its new indices are 0 Overlay, 1 Selection, 2 Info, 3 Outliner, 4 Object Groups; reset persisted `_activePm4TabIndex` rather than treating old eight-tab indices as compatible. |
| `Pm4WorkbenchService.OpenPm4Workbench():40–45` and parent inspector routing | Existing route is `WorkbenchTab.Inspect` plus current `FocusInspectorTab(4)` and legacy drawer/pending-section state. Parent must coordinate this with Viewer naming/tab order and retire the legacy shell routing during its migration. No host API or god-class member was added here. |
| `Workbench/Services/Settings/ViewerSettingsService.cs:287` / `Workbench/Pages/EditorWorkbenchPages.cs:32` | Perform layout/navigation reset first, then remove the old editor index remap caller and migration function (T065). |

There are **no `UI/RosettaObjectLibrary*` source files** in this tree. The actual Rosetta match
consumer is the EditorPanels location above, outside this slice's write scope, and was not edited.
The four pre-existing chrome callers below were also deliberately not repaired.

Saved overlay transform application remains unchanged because settings/world loading still call
it and the operator forbids changing render behavior. Existing hover-match cache invalidation
also remains; no matching UI or new matching job is provided by this service.

## Verification — commands, statuses, real output

### Baseline viewer build (before these edits)

```powershell
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore --nologo -v:q
```

**Exit 1**, 373 warnings / **4 errors** (command artifact `cmd-1792546686375.txt`):

| Existing caller | Missing chrome member |
|---|---|
| `ViewerApp.cs:1404` | `DrawFixedSidebarSplitters` |
| `ViewerApp.cs:1407` | `DrawToolbar` |
| `ViewerApp.cs:1409` | `DrawBottomBar` |
| `ViewerApp_Host.cs:163` | `DrawToolbarPopupButton` |

### Post-removal viewer build

```powershell
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore --nologo -v:q -clp:ErrorsOnly
```

**Exit 1**, 141 warnings / **8 CS0246 errors**; all are parent-owned declarations referencing the
deleted manual-store types:

```text
Settings/ViewerSettingsService.cs(102,17): Pm4ObjectMatchStore not found
Settings/ViewerSettingsService.cs(103,36): Pm4ObjectMatchEntry not found
Workbench/Services/IViewerAppHost.cs(344,9): Pm4ObjectMatchStore not found
ViewerApp_Host.cs(337,9): Pm4ObjectMatchStore not found
Workbench/Services/IViewerAppHost.cs(345,28): Pm4ObjectMatchEntry not found
ViewerApp_Host.cs(338,28): Pm4ObjectMatchEntry not found
ViewerApp.cs(514,13): Pm4ObjectMatchStore not found
ViewerApp.cs(515,32): Pm4ObjectMatchEntry not found
```

These missing-type errors mask later semantic diagnostics. This is **not** evidence that the
four baseline chrome errors or the removed-member callsites are resolved. Full semantic build
and solution tests must be rerun after parent integration. No temporary stub was installed.

### Source verification / whitespace

The exact runnable source-check command and script are in [P5](p5-archaeology.md): **exit 0**,
15 parsed files, correct bounded structure, 21 preserved methods token-identical to HEAD, and
unchanged Collections. That includes both existing JSON/OBJ writer bodies and saved-overlay
restoration, plus archaeology playback/composition helpers and operation materialization.

```powershell
git diff --check -- wow-viewer/src/viewer/WoWViewer/Workbench/Services/Archaeology wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench wow-viewer/src/viewer/WoWViewer/Workbench/Pages/EditorPlacementObjectsPage.cs wow-viewer/src/viewer/WoWViewer/Workbench/Pages/EditorDataIoPage.cs
```

**Exit 0**; only git line-ending normalization notices. No full solution test run or real-client
run was attempted while parent integration is incomplete.

## Criterion → evidence

| Contract | Actual evidence | Unproved / parent-owned |
|---|---|---|
| T060–T062 real editor owner names | Source assertions pass for Archaeology Tasks, Rosetta Object Library, Population, Archaeology Imports, New Map Creator | Operator UI inspection |
| T063 delete PM4 placement section | Source assertion excludes the old label; original section is deleted | No runtime claim |
| T064 one PM4 export home | One live scoped draw caller in PM4 Selection; Data I/O no longer calls it | Remove unused editor page-host bridges |
| T065 remove old editor migration after reset | Reset not present; migration intentionally retained and exact caller reported | Parent layout reset, then method deletion |
| T066 inspection only / delete manual store | Five local tab labels; structural assertions exclude matching/alignment/reconciliation UI/helpers; store file absent | Remove parent-owned store declarations/consumers and stale navigation |
| Preserve unrelated actions/data | 21 method token comparisons pass; Collections unchanged; retained materialization callback | Real map, rendering, interaction, undo/redo proof operator-owned |
| Export result/failure reporting | Existing JSON/OBJ/bundle writers retained; four actions use setup-failure reporter; overlay writer uses managed output root and flushes before path report | Actual files, cancellation/error UX, background completion, and responsiveness require operator execution |

## Export/operator proof boundary

The sole command set is **Viewer → PM4 → Selection → PM4 Exports**: Objects JSON, OBJ Set,
LLM Evidence Bundle, Visible Overlay Report. Correlation JSON is removed with matching.
JSON/OBJ retain existing background guards; bundle/report retain existing serialization logic.
Picker-selected destinations are operator-supplied paths; the report uses the existing managed
export root. Success reports output paths; writer failures and setup failures report explicitly.

After integration the operator must exercise the four exports and failure cases, inspect the
written artifacts and paths, visit all five PM4 and five archaeology pages, and confirm editor
actions and saved-state reset on a real map. Compilation/source checks do not prove those results.
