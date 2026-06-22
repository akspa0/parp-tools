# Plan: 071 Left/Right Sidebar Split + Model Viewer Mode

**Generated from**: `spec.md`
**Branch**: `071-left-right-sidebar-split` (cut from `069-viewer-ui-overhaul`)
**Phases**: 8 small phases (A-H), each independently-validatable.

## Phase Map

| Phase | Scope | Risk | Commit Size |
|-------|-------|------|-------------|
| A | 3D viewport math (subtract both sidebars) | Low | Small |
| B | DrawLeftSidebar (file browser + map list) | Low | Medium |
| C | DrawRightSidebar (rename workbench) | Low | Small |
| D | 3 top tabs (Model/World/Tools) + Tools menu integration | Medium | Large |
| E | Model Viewer — Info sub-tab | Low | Small |
| F | Model Viewer — Animations sub-tab (Play/Pause/Stop) | Medium | Medium |
| G | Model Viewer — Actions + LOD sub-tabs | Medium | Medium |
| H | Memory bank + spec sync | Low | Small |

## Order of execution

1. **A → B → C** (layout): establish two-sidebar frame. Each phase builds on previous. After A, B, C, the user has file browser + workbench side by side.
2. **D** (tabs): collapse 6 → 3 top tabs, route Tools menu items. After D, no more floating windows for the common case.
3. **E → F → G** (Model Viewer): add the missing model inspection panels. F is the most important (animation controls).
4. **H** (docs): update memory bank + spec.

## Phase dependencies

- A must precede B + C (need correct viewport math first)
- B and C can be parallel
- D requires B + C (tabs live in sidebars)
- E, F, G require D (workbench model sub-tabs live under new Model top tab)
- H is last

## Validation per phase

- A: build, verify 3D viewport size with both sidebars shown
- B: build, verify file browser renders in left sidebar
- C: build, verify workbench renders in right sidebar
- D: build, verify each Tools menu item switches to a tab (no floating window)
- E: build, load a model, check Info sub-tab shows vertex/triangle count
- F: build, load a model with animation, check Animations sub-tab has Play/Pause/Stop
- G: build, check Actions and LOD sub-tabs render
- H: memory bank + spec/plan/tasks all current

## Build state expectations

- 0 errors on every commit
- Phase A: 1 commit
- Phase B: 1 commit
- Phase C: 1 commit
- Phase D: 1-2 commits
- Phase E: 1 commit
- Phase F: 1 commit
- Phase G: 1 commit
- Phase H: 1 commit
- Total: 8-9 commits on `071-left-right-sidebar-split`

## Out-of-phase work (future)

- 070: per-map workbench windows (per-map state, native window per map)
- Custom model viewer themes / colors
- Animation timeline editor
- Per-vertex inspection (click vertex → see position/normal/UV)
- Skeleton/bone viewer (M2/MDX rig overlay)
