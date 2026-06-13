# Implementation Plan: Viewer UI Cleanup + ImGui Migration Notes

**Branch**: `060-ui-cleanup-and-migration-notes` | **Date**: 2026-06-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/060-ui-cleanup-and-migration-notes/spec.md`

## Summary

Five independent cleanup slices for the viewer UI: (1) Runtime Stats dedup, (2) status bar button removal, (3) capture UI-hide default, (4) SceneInspector dedup, (5) migration note doc. No architecture changes. Each slice ships in its own commit.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: ImGui.NET, Silk.NET.OpenGL, existing `ViewerApp.cs` partial classes

**Testing**: Manual viewer run + smoke test of capture pipeline (which shots have UI, which don't)

**Target Platform**: Windows desktop

**Constraints**: No new UI features. Only cleanup. The capture pipeline behavior change is opt-in (existing `with_ui: true` flag still works).

## Project Structure

All changes in existing files. No new projects.

```text
wow-viewer/src/viewer/WoWViewer/
├── ViewerApp_Sidebars.cs        # Runtime Stats dedup, tab bar dedup
├── ViewerApp_MinimapAndStatus.cs  # Status bar button removal
├── ViewerApp_CaptureAutomation.cs  # Capture UI-hide default
└── ViewerApp.cs                  # SceneInspector removal (if going that route)

wow-viewer/docs/architecture/
└── ui-migration-options-2026-06-12.md  # DONE (FR-014)
```

## Implementation Phases

### Phase 1: Migration Note (doc-only, ships first)

**Goal**: Capture the ImGui-vs-Avalonia question in writing.

**Approach**: Doc already written at `wow-viewer/docs/architecture/ui-migration-options-2026-06-12.md`. No code.

**Validation**: Doc exists, has the rationale, lists the options, says "no commitment."

### Phase 2: Runtime Stats Dedup

**Goal**: `DrawRuntimeStatsPanelContent()` has one (or few, documented) call sites.

**Approach**:
1. Grep all callers of `DrawRuntimeStatsPanelContent` (5 known: Navigator, Inspector tab, SceneInspector tab, RuntimeStats panel, TerrainControls trailing).
2. Remove from Navigator (it's not a navigation concept).
3. Remove "Stats" tab from Inspector tab bar.
4. Remove "Stats" tab from SceneInspector tab bar (or remove SceneInspector entirely — see Phase 5).
5. Remove trailing Runtime Stats block from Terrain Controls.
6. Keep the dedicated `ShellPanelId.RuntimeStats` panel.
7. Update memory bank.

**Validation**: grep shows ≤2 call sites, all intentional (the panel + maybe one tab).

### Phase 3: Status Bar Button Removal

**Goal**: Bottom status bar shows only status / FPS / coords. No buttons.

**Approach**:
1. In `DrawStatusBar()` at `ViewerApp_MinimapAndStatus.cs:237`, remove the "Actions" column with Copy/Log Scene buttons.
2. Reduce table to 2-3 columns.
3. Document that "Copy Scene" already exists in Capture Automation window.
4. "Log Scene" gets removed (it was a misnomer that did nothing useful).

**Validation**: status bar has no `ImGui.Button` calls. Visual confirmation.

### Phase 4: Capture UI-Hide Default

**Goal**: Single-shot and video capture hide UI chrome by default.

**Approach**:
1. In `PrepareNextCaptureRequest()` at `ViewerApp_CaptureAutomation.cs:564`, set `_hideUiChrome = true` when the dequeued request has `includeUi: false` (the default).
2. Restore `_hideUiChrome = false` after the capture frame completes.
3. Video recording: same toggle per frame.
4. Existing `includeUi: true` still opts in to UI-in-frame for users who want it.

**Validation**: Capture a video with default settings, verify no UI chrome in output. Capture with `with_ui` flag set, verify UI is present.

### Phase 5: SceneInspector Dedup

**Goal**: Resolve the two-near-identical-tab-bars problem.

**Approach**:
1. Compare the SceneInspector tab bar (`ViewerApp_Sidebars.cs:1051`) and the Inspector tab bar (`ViewerApp_Sidebars.cs:802`).
2. If overlap > 50%, remove SceneInspector (it's the newer of the two and Inspector has more content).
3. Remove `ShellPanelId.SceneInspector`, all related fields, all related dispatch cases.
4. Update quadrant groupings and `GetDockPanelStateRef` switch.
5. Update Tools menu.

**Validation**: One inspector tab bar. No SceneInspector in `ShellPanelId`.

### Phase 6: Polish

**Goal**: Final test pass + memory bank.

**Approach**:
1. `dotnet build` clean.
2. Launch viewer, verify all panels still render.
3. Memory bank update with 060 summary.
