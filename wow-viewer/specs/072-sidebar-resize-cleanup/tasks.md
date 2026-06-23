# Tasks: 072 Sidebar Resize + Toolbar Layout Hotfix

## Goal

Fix two immediate UX regressions in the 071 tab-mode layout:

1. Sidebar width sliders inside the sidebars fight the mouse cursor because the hosting window moves as the value changes.
2. Toolbar is visually overwritten by the sidebars because it spans the full display width and is drawn before them.

Also prep for follow-up audit work by NOT adding new duplicated controls.

## Phase 1: Sidebar resize splitters

- [x] T001 Remove `DrawFixedSidebarWidthControl` calls from `DrawLeftSidebar` and `DrawRightSidebar`.
- [x] T002 Add tab-mode edge splitters: left splitter at `x = _leftSidebarWidth`, right splitter at `x = displayWidth - _rightSidebarWidth`.
- [x] T003 Compute new widths from absolute mouse delta, not from a slider inside the moving window.
- [x] T004 Clamp widths with existing `ClampFixedSidebarWidth`.

## Phase 2: Toolbar layout

- [x] T005 Change `DrawToolbar` main window to span only the scene viewport width (`viewportX` to `viewportWidth`).
- [x] T006 Move `DrawToolbar()` call in `DrawUI()` to after `DrawLeftSidebar`/`DrawRightSidebar` so toolbar renders on top if any overlap remains.
- [x] T007 Keep centered terrain toolbar spanning viewport only (already correct).

## Phase 3: Validation

- [x] T008 Build `WoWViewer.csproj` with 0 errors.
- [x] T009 Update this tasks.md and memory bank; commit + push.

## Notes

- DO NOT lose existing functionality.
- This is a hotfix slice; the broader tab/sidebar duplication audit is a separate follow-up slice.
