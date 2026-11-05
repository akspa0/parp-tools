# Spec 293 — Tasks (Restoration Workbench)

**Status:** Implementation in progress on operator-originated scope (AGENTS.md §9.1).
The spec remains DRAFT for the full workbench; the operator explicitly requested the
zoomable donor tile picker and per-tile settings, so that slice is authorized.

## Phase 1 — Zoomable donor tile picker + per-tile settings (operator request 2026-11-04/05)

### T001 — Zoom + pan on the donor tile picker map surface
- [x] Wheel zoom anchored on the tile under the cursor (1×–16×).
- [x] Middle/right-drag pan with clamping to the 64×64 grid.
- [x] `Fit` button resets to 1× and top-left.
- [x] Visible tile-range readout (`XX_YY .. XX_YY`).
- [x] Hover tooltip reports the tile under the cursor at any zoom.
- [x] Drag-box selection maps screen coordinates through the zoom window.
- [x] Selection settings tree (Z offset / Z scale) applied to every tile added from the selection.

### T002 — Cartography "Placed tiles" text clarity
- [x] Checkbox label names the target tile (`Include target XX_YY`).
- [x] Tooltip explains the target is where the donor tile is placed over.
- [x] Secondary text shows `donor XX_YY → placed over target XX_YY`.

### T003 — Build verification
- [x] `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --nologo -v minimal` → exit 0, 0 errors.

## Receipt — 2026-11-05

### Files changed
| File | Change |
|---|---|
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Archaeology/DonorTilePickerService.cs` | Added `_zoom`/`_pan`/`_panDragStart`/`_panAtDragStart`/`_selectionZOffset`/`_selectionZScale`; zoom/pan controls; zoom-aware texture UV crop; zoom-aware hover/grid/selection/drag-box mapping; `SetZoom`, `ClampPan`, `VisibleTileRangeText`; selection settings tree; placements now carry `ZOffset`/`ZScale`. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Archaeology/ArchaeologyPanelService.PhaseLayers.cs` | Placed-tile checkbox label now names the target; tooltip explains donor→target; secondary text shows `donor XX_YY → placed over target XX_YY`. |

### Verification commands
| Command | Result |
|---|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --nologo -v minimal` | Exit 0; 0 errors; 229 warnings (pre-existing). |

### Criterion → evidence
| Criterion | Evidence |
|---|---|
| **AC-007** (zoomable picker + XX_YY tooltips) | Implemented in `DonorTilePickerService.DrawDialog`: wheel zoom anchored on cursor, pan, Fit, visible-range readout, hover tooltip. **Runtime/visual proof owed** — operator must run the viewer on a real map. |
| **AC-008** (include/exclude per placed tile) | Implemented in `ArchaeologyPanelService.PhaseLayers.cs`; `PhaseTilePlacement.Included` honored by `ResolveTileSource`/`IsTargetLockedByLayer`; persisted in `PhaseTilePlacementDto`. **Runtime proof owed** — toggle a placed tile and save/load. |
| **AC-008b** (per-tile Z offset/scale) | Implemented in `ArchaeologyPanelService.PhaseLayers.cs` and `DonorTilePickerService` selection settings; `PhaseLayerZ.Apply` combines layer + per-tile transforms; persisted. **Runtime proof owed** — adjust a per-tile Z offset and confirm only that tile moves. |
| **AC-015** (SharedUiWidgets + inventory registration) | New picker controls are inside the modal donor-tile dialog (not a sidebar surface); the Cartography placed-tile controls use `SharedUiWidgets.SetNextItemWidth`. Inventory registration for the modal is tracked under the full workbench phase. |
| **AC-016** (≤2,000-line budget) | `DonorTilePickerService.cs` ≈ 490 lines; `ArchaeologyPanelService.PhaseLayers.cs` 1,127 lines. Both under budget. |

### Owed
- Operator visual check: zoom/pan the donor tile picker, place tiles, adjust selection Z, save/load the layer project.
- Operator review/approval of the full Spec 293 workbench scope (AGENTS.md §9.1).
