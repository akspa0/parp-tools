# Progress — wow-viewer

## 2026-06-22 — 072: Sidebar resize + toolbar layout hotfix (complete)

### What landed

- Removed `DrawFixedSidebarWidthControl` sliders from inside tab-mode left/right sidebars.
- `DrawFixedSidebarSplitters` now draws left/right edge splitters in tab mode.
- `DrawToolbar` spans only the scene viewport width (`viewportX`..`viewportWidth`).
- `DrawToolbar` is called after sidebars in `DrawUI` so it stays on top if edges overlap.
- Build: 0 errors, 284 pre-existing warnings.
- Commit `bcdcb752` pushed to `071-left-right-sidebar-split`.

### Status
072 hotfix complete. Next: audit/dedup scope with user.

## 2026-06-22 — Spec 071 Phase H: Memory bank + spec sync + final build (complete)

### What landed

- Updated `specs/071-left-right-sidebar-split/spec.md` to match final implementation.
- Updated memory bank with full 8-phase history.
- Final build: 0 errors, 286 pre-existing warnings.
- Commit `8190fb65` pushed to `071-left-right-sidebar-split`.

### Status
Spec 071 complete.

## 2026-06-22 — Spec 071 Phases A-G (complete)

Summary of earlier 071 phases:
- **A**: Viewport subtracts left/right sidebars.
- **B**: Left sidebar with workspace bars, file browser, world maps.
- **C**: Right sidebar = workbench anchored to right edge.
- **D**: 3 top tabs (Model/World/Tools) with `WorkbenchNavigator` and typed `OpenWorkbenchTab` helpers.
- **E**: Model > Info sub-tab with path line.
- **F**: Model > Animations sub-tab with Play/Pause/Stop, loop, speed buttons, timeline slider; added `PlaybackSpeed`/`Loop` to `IAnimationController`.
- **G**: Model > Actions + LOD sub-tabs; selected world object auto-switches to Model > Info.

All phases built clean and pushed to `071-left-right-sidebar-split`.

## 2026-06-21 — Spec 071 drafted

- Two-side layout + Model Viewer mode, 8 phases, branch cut from `069-viewer-ui-overhaul`.

## 2026-06-21 — Spec 069: Viewer UI overhaul (tab system → workbench)

- Cells overlay, tab data model, archeology playback, sticky settings, headless content variants.
- Learned: top/bottom tab bars failed (debug overlay look), per-sub-tab popouts failed (window sprawl), single Workbench panel succeeded.
- 14 phases committed to `069-viewer-ui-overhaul`.

## Previous work
- Spec 068: fractal-aware height loss + curation hardening (V21c)
- Spec 067: V20 multi-modal terrain intent
- Spec 066: V19 minimal-signal height regressor
- PM4 surface correlation, PM4 simplification reverse-engineering

## Branch summary

- `071-left-right-sidebar-split` — 071 + 072, active, ready for audit/dedup or merge.
- `069-viewer-ui-overhaul` — legacy tab UI work, salvageable concepts extracted into 071.

## Out-of-Phase Work

- 070: Per-map workbench windows (deferred, large rewrite).
- Audit/dedup: toolbar vs left sidebar duplication, legacy shell-panel cleanup, file splitting.
