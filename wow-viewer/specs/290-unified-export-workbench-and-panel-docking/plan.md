# Implementation Plan: Spec 290 — Unified Export Workbench & Panel Docking

## 1. Architectural Strategy & Boundaries

### 1.1 Dedicated Service Class (`ExportWorkbenchService`)
In accordance with AGENTS.md §10 (God-Class Freeze: no new members in `WorldScene` or `ViewerApp`), the new Exports panel will be implemented in a dedicated owned service class:

```
wow-viewer/src/viewer/WoWViewer/Workbench/Services/Export/
└── ExportWorkbenchService.cs       # Owned service for all export operations (<800 lines)
```

The service receives `IViewerAppHost` via constructor and delegates export operations to existing backend exporters:
- PM4 exports: `_host.Pm4Workbench` / `_host.Pm4DoodadDuplication` / `_worldScene.Pm4Overlay`
- World exports: `_host.MainMenuBar.DrawWorldExportContent()` / `_host.NavigatorPanel`
- Model exports: `_host.MainMenuBar.DrawModelExportContent()`
- Minimap / Dataset exports: `_host.DrawSynthesizedMinimapExportContent()` / `_host.DatasetExportDialogs`

### 1.2 Right Sidebar Navigation Integration
1. Extend `ViewerPanel` enum in `WorkbenchTab.cs`:
   ```csharp
   public enum ViewerPanel { Inspector, World, Gossip, PM4, Exports }
   ```
2. In `ViewerActionBarsService.cs`:
   - Add `IconKind.Export` (using an intuitive export glyph or icon).
   - In `DrawRightBar()`, add:
     ```csharp
     DrawActionBarIcon("Exports", "ActionExports", IconKind.Export, size, "Export PM4 OBJ/JSON models, placement ledgers, and reports", () => _host.UiManager.ToggleRightDock(ViewerPanel.Exports), dockOpen && active == ViewerPanel.Exports);
     ```
3. In `WorkbenchPanelsService.cs`:
   - Wire `case ViewerPanel.Exports: _exportWorkbench.DrawExportPanelContent(); break;`

### 1.3 Bottom Destination Window Docking Repositioning
In `ViewerActionBarsService.DrawToolWindows()`:
- Previously:
  ```csharp
  float rightX = Math.Max(rail, display.X - rail - 430f * scale - 12f * scale);
  float rightY = top + mapHeight + 12f * scale;
  ```
- Updated anchor:
  Position the destination window in the lower-left area above the footer bar, adjacent to the chat region:
  ```csharp
  float destX = rail + 12f * scale;
  float destHeight = Math.Min(600f * scale, Math.Max(300f, display.Y - 180f * scale));
  float destY = Math.Max(top, display.Y - destHeight - 54f * scale);
  float destWidth = Math.Min(560f * scale, Math.Max(380f, display.X - destX - 440f * scale));
  ```
- This places `Editor` and `Archaeology` popups in the lower-left workspace area, leaving the entire right side of the screen (`UiRegion.RightDock`) completely visible and interactive.

---

## 2. Phased Implementation Roadmap

### Phase 1: Service Architecture & UI Manager Wiring
- Define `ViewerPanel.Exports` in `WorkbenchTab.cs`.
- Create `ExportWorkbenchService.cs` in `WoWViewer/Workbench/Services/Export/`.
- Wire `ExportWorkbenchService` into `IViewerAppHost` and `ViewerApp`.
- Add `Exports` icon button to `ViewerActionBarsService.DrawRightBar()`.
- Add dispatch in `WorkbenchPanelsService.cs`.

### Phase 2: Consolidating PM4, World, and Dataset Exports
- Implement `ExportWorkbenchService.DrawExportPanelContent()`:
  - PM4 Navigation & Collision Exports section (`SharedUiWidgets.SectionHeader`).
  - World & Terrain Exports section.
  - Model & Asset Exports section.
  - Archaeology & Datasets section.
- Add quick-jump button in PM4 Sidebar: "Open Export Workbench" (`_host.UiManager.OpenViewer(ViewerPanel.Exports)`).

### Phase 3: Bottom Destination Tool Window Repositioning
- Update `ViewerActionBarsService.DrawToolWindows()` to anchor `destination` in the lower-left workspace area.
- Verify resolution scaling and non-overlap with `UiRegion.RightDock`.

### Phase 4: UI Reference Catalog & Regression Verification
- Update `eng/ui/ui-controls.json`.
- Run `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1`.
- Verify compilation with 0 errors and 0 warnings.
