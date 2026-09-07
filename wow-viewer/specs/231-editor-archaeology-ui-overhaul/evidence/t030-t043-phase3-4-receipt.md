# Receipt — Phases 3 & 4 (T030, T031, T032, T033, T040, T042, T043) + operator glossary fix

**Date**: 2026-09-07 · **Spec**: [231 Editor/Archaeology UI Overhaul](../spec.md) · Phases 3–4 (D2, D3, D4, D5) + operator-directed PM4 glossary correction

## Files changed

| File | Change |
|---|---|
| [ViewerApp_Pm4Utilities.cs](../../../src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs) | **D2**: deleted `DrawPm4CorrelationInspectorContent` (149-line workbench subset); the workbench Correlation tab now draws the canonical `DrawPm4WmoCorrelationContent`. **Operator-directed**: `DrawPm4GlossarySummary` rewritten to current measured truths (see below) |
| [ViewerApp_Sidebars.cs](../../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs) | **D4**: removed the `##DoodadSet` combo (Model Info) and `##ActionsDoodadSet` combo (Actions page). **D3**: deleted dead `DrawTerrainToolsWindow` (unreachable — `_showTerrainToolsWindow` was never set true) and dead `DrawTerrainToolsSubTab` (no callers), each of which carried a duplicate "Clipboard + Save" header |
| [ViewerApp.cs](../../../src/viewer/WoWViewer/ViewerApp.cs) | Removed the dead floating-window call block and the `_showTerrainToolsWindow` field |
| [ViewerApp_Editor.cs](../../../src/viewer/WoWViewer/ViewerApp_Editor.cs) | **D5**: deleted the dead `DrawArchaeologyEditorContent` dispatcher and `_archaeologyEditorSubTab` field (no callers — the Archaeology tab hosts analysis pages only). Added `IEditorPageHost.DrawPm4Workbench()` delegate |
| [Workbench/Pages/ViewerAppContext.cs](../../../src/viewer/WoWViewer/Workbench/Pages/ViewerAppContext.cs) | `IEditorPageHost` gains `DrawPm4Workbench()` |
| [Workbench/Pages/EditorPlacementObjectsPage.cs](../../../src/viewer/WoWViewer/Workbench/Pages/EditorPlacementObjectsPage.cs) | **T031/de-clutter**: Tasks & Workspace open; new collapsed "PM4 Placement Tools" section (the PM4 workbench: overlay/selection/correlation tabs); 3D Object Library and Population now collapsed by default |

## Operator-directed fix: PM4 Glossary / Evidence

`DrawPm4GlossarySummary` previously presented outdated readings. Rewritten against
[pm4-chunk-semantics.md](../../../docs/architecture/pm4-chunk-semantics.md) (2026-06-09) and the
measured [pm4-pd4-draft.md](../../../docs/wowdev-wiki/pm4-pd4-draft.md) (2026-08-24):

- **MSUR.0x18 indexes MSLK** (adjacency window, 100.0000% fit) — the old "MscnRefIndex / indexes
  MSCN" claim is eliminated; the debug "MscnRef" label is documented as a legacy alias.
- **CK24 (MSUR.0x1C) is an IEEE float = placement Z** (93.58% bit-exact vs MODF/MDDF); "type" byte
  = exponent band; 0x1C==0 is the unattributed bucket. The old "packed key: type=high byte,
  objId=low 16 bits" text is gone.
- **MSHD**: 0x00/0x08 = clamped world-unit spans, 0x04==1 = empty tile, 0x0C–0x1C reserved zeros.
  The viewer's "MSHD Region" bucket is documented as a tile-level hint, not a grouping semantic.
- **MSCN**: world-frame connector/boundary node network shared across objects (64.4%); no in-file
  index consumer. **MSLK**: undirected adjacency graph, 98.76% reciprocity, ~47% wall quads.
- Retained: TypeFlags walkable/structural classifier, MSVI/MSVT walk, MPRL, viewer-generated
  `part`, cyan:magenta fingerprint.

## SC-1 / SC-5 grep receipts

| Check | Command basis | Result |
|---|---|---|
| D2: workbench correlation ids gone | `Pm4WmoPlacementListWorkbench|Pm4WmoMatchWorkbench|Pm4WmoCorrelationFilterWorkbench` | **0** |
| D2: correlation panel copies | `DrawPm4WmoCorrelationContent` definitions | **1** (2 intentional route sites: PM4 Analysis Correlation tab + Placement workbench tab) |
| D4: doodad-set combos | `BeginCombo("##…DoodadSet"` in Sidebars | **2**: toolbar hovered-WMO quick combo (FR-6) + Selected WMO Controls (object-inspector home) |
| D3: "Clipboard + Save" headers | text match in Sidebars | **1** live header (`DrawSharedChunkClipboardSection` inside `DrawTerrainLabSubTab`, Terrain Tools page) + 1 tooltip text |
| D5: Archaeology editor hosting | `DrawArchaeologyEditorContent` | **0** references |

## Verification

- `dotnet build wow-viewer/WowViewer.slnx -c Debug` — **0 Errors** (twice: after the consolidation
  edits and after the dead-code deletions; artifacts `cmd-1788758803260.txt`, final run exit 0).
- Test suite: unchanged from the Phase 2 full run (11 pre-existing Core/PM4-corpus failures) —
  this phase's diff touches only the viewer UI project, which no test project references.
- UI smoke captures: [screenshots/phase3-editor-0-placement-decluttered/](screenshots/phase3-editor-0-placement-decluttered/)
  and [screenshots/phase3-editor-1-terrain-tools/](screenshots/phase3-editor-1-terrain-tools/).
- File budget (T043): `ViewerApp_Sidebars.cs` **6,173 → 6,013**; `ViewerApp_Pm4Utilities.cs`
  **4,451 → 4,368**; `ViewerApp_Editor.cs` 1,953 → 1,930. All shrink.

## Known remaining work (recorded, not hidden)

1. **Tasks panel interior**: `DrawArchaeologyEditorTasksSubTab`'s task inspector still embeds
   Chunk Clipboard and Terrain Import/Export blocks (visible in the phase-3 capture). Those belong
   on Terrain Tools / Data I/O pages per the IA — this is the next visible de-clutter step (open).
2. `DrawChunkClipboardContent` is now referenced only via `DrawSharedChunkClipboardSection` (fine);
   naming of `DrawArchaeologyEditor*` content methods is legacy (routing is dead) — rename in a
   later pass.
3. T041 (Converters page: Spec 221 harness surfaces) still open.
