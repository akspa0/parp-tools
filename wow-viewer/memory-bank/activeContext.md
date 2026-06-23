# Active Context — wow-viewer

**Last updated**: 2026-06-22 | **Focus**: 072 sidebar resize/toolbar hotfix done; next = audit/dedup scope

## Current State

- **Spec 071 complete** on branch `071-left-right-sidebar-split`: two-side layout (left = file/maps/workspace, right = workbench), 3 top tabs (Model/World/Tools), Model Viewer sub-tabs (Info/Animations/Actions/LOD).
- **072 complete**: fixed tab-mode sidebar resize sliders jumping with the moving window by moving resize to edge splitters; toolbar now spans viewport width only and draws after sidebars for correct z-order. Build clean, committed `bcdcb752`.
- **073a complete**: deduplicated toolbar / left sidebar controls; toolbar now scene-only, left sidebar owns source/workspace controls. Build clean, committed `b11dd518`.
- **073b ready**: Tools tab converter integration spec'd at `specs/073b-tools-tab-converters/tasks.md`. Implementation is a separate sub-plan; start in fresh chat.

## What Works
- Tab UI with left/right sidebars and workbench.
- Model Viewer info/animation/action/LOD tabs.
- Archeology playback, sticky settings, cells overlay.

## What's Rough / Needs Audit
- Duplicate controls: workspace bars exist in left sidebar AND toolbar.
- `ViewerApp.cs` / `ViewerApp_Sidebars.cs` are large and mixed-concern.
- Legacy shell-panel code still lives alongside tab-mode code.
- 069/071 tab/sub-tab enums may overlap or be redundant.

## Next
- Define scope of audit/dedup slice with user.
- Likely >10 tasks → split into sub-plans, each with own `tasks.md` and fresh chat.

## Open Questions
1. Should deduplication remove the toolbar workspace buttons or the left-sidebar workspace bars?
2. Should legacy shell-panel code be deleted or kept behind a compatibility flag?
3. Should the audit start with file splitting (ViewerApp_Sidebars, ViewerApp_Tabs) or with data-model cleanup?
4. Is 070 (per-map workbench windows) still the long-term target?

## Files Touched Recently
- `src/viewer/WoWViewer/ViewerApp.cs`
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`
- `specs/072-sidebar-resize-cleanup/tasks.md`
- `specs/071-left-right-sidebar-split/{spec,plan,tasks}.md`
