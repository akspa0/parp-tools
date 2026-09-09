# Receipt — Spec 232 T054: Archaeology Map Layers default

**Date**: 2026-09-08 · **Spec**: [232](../spec.md) · **Task**: T054 / FR-4

## Files changed

| File | Change |
|---|---|
| [ViewerApp_Sidebars.cs](../../../src/viewer/WoWViewer/ViewerApp_Sidebars.cs) | Makes a no-page navigation to Archaeology select workbench page 5, Cartography; Cartography itself initializes on its existing Layers sub-tab. Explicit Range/UniqueId requests and remembered per-tab page choices are left unchanged. |
| [tasks.md](../tasks.md), [activeContext.md](../../../memory-bank/activeContext.md), [progress.md](../../../memory-bank/progress.md) | Records implementation and the remaining UI witness. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**; 543 pre-existing warnings, including NU1903 advisories. |
| `git diff --check` | 0 | No whitespace errors. Git emitted unrelated user-config access and CRLF notices. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| Fresh/no-page Archaeology navigation defaults to Map Layers | `OpenWorkbenchTab(WorkbenchTab.Archaeology)` uses a sentinel default that resolves to page 5, Cartography, whose initial sub-tab is Layers. This covers the Archaeology workspace action and F4. | Implemented (solution build) |
| First top-tab switch to Archaeology defaults to Map Layers | `DrawTopTabButton` assigns page 5 when no remembered Archaeology page exists. | Implemented (solution build) |
| Explicit and remembered choices remain valid | An explicit page index remains authoritative; per-top-tab remembered indices still win over the default. | Implemented (source audit + solution build) |
| Operator sees Map Layers rather than UniqueId on default entry | Requires one interactive viewer witness; a build cannot establish a rendered/default UI state. | Pending operator witness |

## Remaining operator witness

From a workbench session with no remembered Archaeology page, press F4 or switch to Archaeology
and capture Cartography with its `Layers` sub-tab selected. Then open `UniqueId Timeline`, leave
and return to Archaeology to confirm that explicit/remembered selection is still honored. T054
remains unchecked until this UI witness is recorded.
