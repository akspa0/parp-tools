# 073a — Toolbar / Left Sidebar Dedup and Alignment

## Goal
Remove duplicated workspace controls between the toolbar and the left sidebar, and fix toolbar alignment/spacing.

## Constraints
- Do NOT delete functionality.
- Legacy mode (`View > Legacy UI`) must still work.
- Build `WoWViewer.csproj` clean after every phase.

## Phase 1: Audit and decide ownership

- [x] T001 Read `ViewerApp_Sidebars.cs` `DrawToolbar` and `DrawWorkspaceBarsPanelContent`.
- [x] T002 List every control that appears in both places.
- [x] T003 Decide: toolbar = viewport/scene chrome; left sidebar = source/workspace chrome.

## Phase 2: Toolbar cleanup

- [x] T004 Remove workspace-source buttons from toolbar (Open Game Folder, Open File, etc.).
- [x] T005 Keep scene-level buttons in toolbar: cells toggle, camera reset, frame model, screenshot, etc.
- [x] T006 Fix toolbar button spacing/alignment so buttons sit cleanly on the toolbar row.

## Phase 3: Left sidebar consolidation

- [x] T007 Ensure left sidebar has all source/workspace controls that were removed from toolbar.
- [x] T008 Add separators/grouping so left sidebar is scannable.
- [x] T009 Keep `DrawWorkspaceBarsPanelContent` usable by other callers if needed.

## Phase 4: Validation

- [x] T010 Build `WoWViewer.csproj` with 0 errors.
- [x] T011 Toggle `View > Legacy UI`; verify legacy toolbar/sidebars still render.
- [x] T012 Update this tasks.md; commit + push.
