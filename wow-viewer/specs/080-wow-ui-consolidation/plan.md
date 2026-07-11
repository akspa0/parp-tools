# Implementation Plan: Spec 080 WoW UI Consolidation

> **Release-convergence amendment (2026-07-11):** This earlier partial plan
> remains historical implementation context. The canonical completion plan is
> [`ui-release-convergence-plan.md`](ui-release-convergence-plan.md), which
> consolidates the remaining viewer UI work from Specs 049, 053, 056, 057,
> 060, 069, 070, 071, 073b, and 080. Do not start another competing sidebar or
> workbench redesign; follow its inventory-first release gates.

**Branch**: `v0.5.0-dev` | **Date**: 2026-07-05 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `wow-viewer/specs/080-wow-ui-consolidation/spec.md`

## Summary

Consolidate the viewer UI around one reliable bottom action bar plus named Model, World, Tools, and Settings information surfaces. The immediate correction is to keep all new work in `wow-viewer`, expose WMO group boxes and group names for standalone WMO inspection, split wireframe controls into independent terrain/object modes, and make the right sidebar audit actionable before broad panel migration.

## Technical Context

**Language/Version**: C#/.NET, ImGui.NET viewer shell

**Primary Dependencies**: `WoWViewer` viewer app, `WoWViewer.Terrain`, `WoWViewer.Rendering`, `WorkbenchNavigator`

**Storage**: Existing viewer settings and ImGui `.ini` window persistence

**Testing**: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; targeted manual UI checks in `WowViewer.App`

**Target Platform**: Windows desktop viewer

**Project Type**: Desktop viewer app

**Performance Goals**: Keep frame UI cheap enough for interactive 3D inspection; avoid per-frame heavy scans in bottom bar controls.

**Constraints**:
- `gillijimproject_refactor` is read-only reference.
- New UI work lands under `wow-viewer/src/viewer/WoWViewer`.
- Do not move reliable left-sidebar behavior until the right-sidebar/frame work is proven.
- Do not hide unfinished controls as working UI; disabled or planned states must be explicit.

**Scale/Scope**: Viewer shell tabs, bottom action bar, standalone WMO overlays, settings panel, and model/world info tabs.

## Constitution Check

- **Repo ownership**: Pass. Implementation target is `wow-viewer`; legacy MdxViewer remains reference-only.
- **One phase at a time**: Pass. The plan separates bottom bar/WMO inspection from the larger right-sidebar frame migration.
- **Documentation hygiene**: Pass. This plan and `tasks.md` define the missing Spec 080 execution path.
- **Training-script guardrail**: N/A. No ML training code is touched.
- **Client data guardrail**: N/A for UI source changes; any later real-data UI validation must use staged clients only.

## Audit Findings

1. `ViewerApp_WmoGroups.cs` already supports standalone WMO group bounding boxes, but the control is buried in model info and group names only draw for selected or highlighted groups.
2. `DrawBottomBar()` owns terrain layers, grids, liquids, world bounding boxes, PM4 overlay, and one monolithic wireframe checkbox.
3. `WorldScene` already has separate terrain renderer wireframe and object renderer wireframe internals, but the public UI path toggled them together.
4. The right sidebar has both legacy and workbench paths. The workbench bottom tabs exist for Model and World, but World has no LOD tab and Model LOD is a placeholder.
5. `ViewerApp_Settings.cs` exists, but Settings is not surfaced clearly enough; a bottom-bar gear/button and a stable settings frame remain required.
6. The left sidebar is operational and should only receive wording cleanup after the right-sidebar replacement has proof.

## Project Structure

```text
wow-viewer/
├── specs/080-wow-ui-consolidation/
│   ├── spec.md
│   ├── plan.md
│   └── tasks.md
├── src/viewer/WoWViewer/
│   ├── ViewerApp.cs
│   ├── ViewerApp_Sidebars.cs
│   ├── ViewerApp_Settings.cs
│   ├── ViewerApp_WmoGroups.cs
│   ├── Workbench/WorkbenchNavigator.cs
│   └── Terrain/WorldScene.cs
└── memory-bank/
    ├── activeContext.md
    └── progress.md
```

**Structure Decision**: Keep the current partial-class layout for the small fixes. For the larger migration, extract named frame draw methods from the right-sidebar content without moving left-sidebar loading logic in the same phase.

## Phased Delivery

### Phase A: Bottom Bar And WMO Inspection

Goal: Make current inspection controls discoverable without changing the overall panel model.

- Split world wireframe into `Terrain WF` and `M2/WMO WF`.
- Preserve standalone model wireframe on the bottom bar.
- Put standalone WMO group boxes and group names on the bottom bar.
- Default standalone WMO group names to visible for all groups.
- Build-validate `WowViewer.slnx`.

### Phase B: Settings Surface

Goal: Make Settings a real reachable surface.

- Add a persistent settings launcher in the bottom bar.
- Ensure File -> Settings and any Settings button use the same `_showSettingsWindow` route.
- Add missing Camera category before claiming FR-022 complete.
- Persist `_showSettingsWindow` only if the user expects settings to reopen across sessions; otherwise persist settings values, not the window open state.

### Phase C: Right Sidebar Audit Cleanup

Goal: Stop duplicated and placeholder controls from pretending to be final UI.

- Inventory every right-sidebar/workbench content method and classify it as Model info, World info, Tools, Settings, or remove.
- Disable or hide dead controls with clear tooltip text.
- Add World LOD as a real tab entry before moving LOD content.
- Keep left sidebar intact except text cleanup.

### Phase D: Named Frame Migration

Goal: Replace the right sidebar with stable named frames after the current content map is known.

- Extract Model, World, Tools, Settings frame methods from existing content.
- Route Tools menu items to frame booleans.
- Verify frames stay open after viewport clicks and restore through ImGui `.ini`.
- Remove duplicate right-sidebar dispatch only after the frames are independently validated.

## Validation Plan

1. Build: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
2. Standalone WMO manual check: load one WMO, verify bottom bar shows WMO wireframe, WMO group boxes, and group names.
3. World manual check: load a terrain-backed map, toggle `Terrain WF` and `M2/WMO WF` independently.
4. Settings manual check: File -> Settings and bottom-bar Settings open the same stable window.
5. Right-sidebar audit check: every visible placeholder is either implemented, disabled with tooltip, or listed in `tasks.md`.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
