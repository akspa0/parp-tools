# Implementation Plan: WoWViewer UI Overhaul

**Branch**: `145-wow-ui-overhaul` (manual artifacts on current branch because branch helper was blocked) | **Date**: 2026-08-12 | **Spec**: [spec.md](spec.md)

## Summary

Deliver the first safe UI overhaul slice for v0.5.2: make keyboard ownership explicit by page context, add a persistent visual shortcut-help window, replace the right sidebar's horizontally scrolling main tabs with a vertical page rail, constrain the left world overview so maps remain reachable, make log entries wrap inside a vertical scroll region, align version metadata and READMEs, and document the remaining frame/window migration as follow-up phases.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: ImGui.NET, Silk.NET viewer shell, existing `WorkbenchNavigator`, existing viewer settings and `.ini` persistence

**Storage**: Existing `viewer_settings.json` and ImGui layout persistence; no new external store

**Testing**: Focused C# tests where practical, `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`, and user-run visual viewer checks

**Target Platform**: Windows desktop viewer; cross-platform project must retain compile compatibility

**Project Type**: Desktop 3D viewer application

**Performance Goals**: UI layout work must be constant-bounded per frame; no new client scans, render passes, or per-entry allocations proportional to the whole map

**Constraints**:

- New implementation remains in `wow-viewer`; legacy reference code remains read-only.
- Existing terrain/WMO/MDX/M2/capture routes and bottom bars remain available.
- UI source edits do not constitute real-client rendering or FPS proof.
- The current dirty worktree contains unrelated user changes; only intended UI/version/spec files may be edited in this slice.

**Scale/Scope**: Main right-sidebar pages, nested utility context, capture keyboard routing, Help menu, left navigator layout, log surface, version metadata, READMEs, and continuity docs.

## Constitution Check

- Repo independence: PASS. All source and docs remain under `wow-viewer` or the root README.
- Library-first / parser ownership: PASS. No format reader or renderer ownership changes.
- Real-data validation: N/A for source UI changes; runtime proof is explicitly user-run.
- One phase at a time: PASS. This plan implements only the foundational shell slice; persistent-frame migration is later.
- Documentation hygiene: PASS. Spec 145, research, plan, tasks, READMEs, and memory-bank updates are included.
- Branch workflow: EXCEPTION RECORDED. The feature helper failed at Git index-lock creation; manual artifacts remain traceable to 145.

## Phases

### Phase 1 — Keyboard context and visual help

Introduce a small viewer-owned binding catalog and active context state. Scope Capture authoring to the Capture context. Add Help > Keyboard Shortcuts as a persistent closeable window. Validate with focused source-level tests/build.

### Phase 2 — Right rail and bounded content

Replace only the Model/World/Tools workbench selector with a vertical rail and a separate content child. Remove horizontal scrolling from the main selector. Bound the left world overview and preserve map/file list access. Wrap log entries and disable horizontal scrolling in the log body.

### Phase 3 — Persistent utility window audit

Move selected utility pop-outs out of tab-only dispatch and give each an explicit visibility flag and title-bar close path. Audit LOD and other placeholder pages; hide dead controls or provide honest disabled states.

### Phase 4 — Release truth and continuity

Align Windows/cross-platform/shared version metadata, root and viewer READMEs, and memory-bank state. Record remaining visual/manual gates without claiming them complete.

## Project Structure

```text
wow-viewer/
├── specs/145-wow-ui-overhaul/
│   ├── spec.md
│   ├── research.md
│   ├── plan.md
│   ├── data-model.md
│   ├── quickstart.md
│   ├── contracts/keybinding-context.md
│   └── tasks.md
├── src/viewer/WoWViewer/
│   ├── ViewerApp.cs
│   ├── ViewerApp_Sidebars.cs
│   ├── ViewerApp_LogViewer.cs
│   ├── ViewerApp_CameraPaths.cs
│   └── Workbench/WorkbenchNavigator.cs
├── tests/WowViewer.Core.Tests/
├── README.md
└── memory-bank/
    ├── activeContext.md
    └── progress.md
```

**Structure Decision**: Keep the existing partial-class viewer shell and add only a small viewer-owned binding catalog. Do not introduce a new UI framework or move domain code into the shell.

## Validation Gates

1. Focused compile/build of the solution.
2. Shortcut help source/UI check: Help menu opens, active context is displayed, Capture keys are gated.
3. Narrow-width UI check: vertical rail is readable; left map list and log content remain reachable; no horizontal scrollbar is required for the main surfaces.
4. Persistent-window check: promoted windows remain after viewport clicks and close from their X.
5. User-run real-client check: load a configured client, inspect a world, test camera movement and Capture path authoring. This is not claimed by the assistant from build output alone.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Manual feature artifacts on current branch | Git index-lock permission prevented the required branch helper from creating 145 | Retrying branch creation would repeat the same workspace failure and risk the user's dirty tree |
