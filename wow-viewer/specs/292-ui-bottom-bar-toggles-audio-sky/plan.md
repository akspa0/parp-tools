# Plan — Spec 292

Constraints: AGENTS.md §10 (no new members in `WorldScene`/`ViewerApp`; new logic in owned service classes), §11 (`SharedUiWidgets`, one authoritative home per control, register in UI inventory).

## Phase 0 — Inventory (read-only, no behavior change)
- Enumerate every control on the bottom bar, Pages, Spawns, Options, Profile buttons, and every audio setting (grep audio services, `ViewerSettingsService`, `SettingsWindowService`).
- List every sub-tab/section that renders empty, and every collapsible that triggers scroll reset (ImGui cause: content-height change + `SetScrollY`/child recreated by changing ID — to be confirmed in code).
- Output: `evidence/inventory.md` table: control → current home → target home.

## Phase 1 — Bottom bar
- New `BottomBarService` in `Workbench/Services/Chrome/` (owned service): toggle checkboxes bound to existing settings properties, plus 4 mode switches calling existing dock-open APIs. Remove the bottom floating `destination` window and Profile buttons.

## Phase 2 — Dock re-homing
- Viewer, Options → `UiRegion.LeftDock`; Editor, Archaeology, Spawns → `UiRegion.RightDock`. Remove Pages (or re-home real content).

## Phase 3 — Audio panel
- New `AudioPanelService` in left dock consolidating all audio settings; delete duplicates elsewhere.

## Phase 4 — Sky controls
- New `SkyControlsService` (Workspace section): bind to `SceneAtmosphere`/`SkyDomeRenderer` via a small settings object (override time-of-day, lock, color overrides, stars/moon/cloud flags). Renderer receives flags through parameters; minimal additive properties on `SkyDomeRenderer` only.

## Phase 5 — Scroll/tab defects
- Fix scroll-reset root cause in shared section helper (`SharedUiWidgets`) once, not per panel; remove or fill empty sub-tabs.

## Phase 6 — Docs & verification
- Update `ui-controls.json`, regenerate `UI-REFERENCE.md`, run `Verify-UiReference.ps1`, build, targeted tests; operator screenshots for AC-1/2/5/6/7.

