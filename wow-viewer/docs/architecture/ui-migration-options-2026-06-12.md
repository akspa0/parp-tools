# UI Library Migration Options — Exploration Note

**Status**: Exploration, no commitment
**Date**: 2026-06-12
**Author**: viewer team
**Related**: spec 060 (UI cleanup), spec 044 (Viewer Shell Usability), spec 049 (Viewer UI Consolidation)

## TL;DR

The viewer UI is currently built on **ImGui.NET** (with Silk.NET.OpenGL). ImGui has been the right call for fast prototyping and a tooling/instrumentation-oriented app, but it has produced real, accumulating cost: every UI change cascades into layout refactors, scaling is manual, drag-and-drop docking is fragile, and content duplication is easy to introduce and hard to spot. This note captures the options for moving off ImGui, the rough cost of each, and the rationale for **not** migrating right now.

## Why ImGui was chosen

The viewer is a tool, not a product. It exists to inspect WoW terrain, M2/WMO models, PM4 navigation meshes, and related data. The use cases are:

- Real-time 3D navigation (camera, fly-mode, model placement)
- Data inspection (MPQ contents, ADT/M2/WMO headers, PM4 object graphs)
- Validation work (rendering parity checks against the native client)
- Dataset export (minimap captures, terrain tensor packs)

For this use case, ImGui's strengths line up:

- **No designer required.** The viewer is a single-developer, internal-tooling project. ImGui is code-only.
- **Per-frame cost is irrelevant for tool usage.** The viewer is a single window, not a 60fps game.
- **Immediate-mode composition fits the "data panel shows whatever is selected" model.** No MVVM, no view state synchronization.
- **Per-pixel control.** Overlays, debug visualizations, and 3D scene HUDs sit on top of the 3D viewport cleanly.

## What's hurting

ImGui's weaknesses, as experienced on this codebase:

### 1. Manual layout, all the time

Every panel, window, tab, and status bar needs hand-tuned pixel positions. `ViewerApp_MinimapAndStatus.cs:237` (status bar) is a 4-column `ImGui.BeginTable` with fixed-width columns. Adding a column means re-tuning every existing column's width. There's no "fill the remaining space and be reasonable" — only fixed widths or stretch.

### 2. Scaling is fragile

HiDPI / non-1.0 display scale factors cause text overlap, panel sizes wrong, and hit-boxes in the wrong place. The codebase works around this with a `_displayScale` field and per-component scale math, but it's manual and inconsistent.

### 3. Docking is rough

`ShellPanelId` + `_xxxDockState` fields + `_savedShellPanelLayouts` + `ResetShellLayoutToDefaults` is ~200 lines of plumbing for a feature that any native UI library gives you for free with drag-and-drop and persistence. The fix `624fabeb` (missing `_pm4SceneGraphDockState`) was the kind of low-grade bug ImGui docking invites.

### 4. Duplication traps

`DrawRuntimeStatsPanelContent` is currently called from 5 places. Nobody notices in a code review because ImGui's call-site pattern is just "call the function where you want it." A real UI library would make a `RuntimeStatsView` component that gets instantiated once and referenced from each tab, surfacing the duplication as a code review smell.

### 5. Cost per change

The user has called this out directly: "every time we try to do 1 simple edit, we end up in refactor hell for literally days." ImGui is genuinely hostile to incremental UI work after the codebase crosses some size threshold. This codebase is well past that threshold.

## Options

| Library | .NET binding | Rendering | Layout | Pros | Cons |
|---|---|---|---|---|---|
| **Avalonia 11+** | First-class .NET | Skia or native widgets | XAML + CSS-like selectors | Most-mature .NET-native UI; MVVM-friendly but immediate-mode also possible; XAML designer exists; community + tooling | Not a single-window OpenGL app — 3D viewport integration requires `Viewport3D` or embed-the-window; migration cost is large |
| **MAUI (.NET 8+)** | First-class .NET | Native per platform | XAML | Cross-platform native; ships with VS; backed by MS | Heavier than Avalonia; WinUI 3 backend has scaling issues; OpenGL/Silk.NET integration is non-trivial |
| **WinUI 3** | First-class .NET on Windows | Native (UWP-style) | XAML | Best Windows-native feel; good designer tooling | Windows-only; .NET 10 has rough edges; 3D viewport integration is bespoke |

For this project (Windows-first, .NET 10, Silk.NET.OpenGL already in the stack), **Avalonia 11** is the most viable target. The OpenGL viewport would need to become a hosted HWND with a `DxInterop` or external-window glue; this is the single hardest technical piece.

## Rough cost estimate

A clean Avalonia port of the current viewer surface area:

- **Surface area**: 12 shell panels, 15 floating windows, 1 dockspace layout, ~30 menu items, 1 status bar, 1 capture automation pipeline, 1 ImGui/OpenGL render loop
- **Effort estimate**: 2-3 months with 1 developer, focused
  - Week 1-2: Avalonia shell + host the existing OpenGL viewport as an embedded HWND
  - Week 3-4: 1:1 port of the 12 shell panels (XAML instead of ImGui)
  - Week 5-6: port floating windows and status bar
  - Week 7-8: port the capture pipeline (which is ImGui-state-coupled today)
  - Week 9-10: dock state persistence, menu wiring, settings save/load
  - Week 11-12: bug bash, test parity, performance
- **Risk**: capture pipeline is tightly coupled to ImGui frame semantics ("capture one frame, hide chrome during it"). The Avalonia equivalent will need explicit "begin/end capture" hooks into the main loop.

## What makes migration easier

The 060 UI cleanup (in this same spec) helps:

- **Runtime Stats as a single component** — the duplication is currently ImGui's "call-where-you-want" pattern. Cleaning it up means there's one source of truth regardless of UI library.
- **Capture UI-hide as a single toggle** — currently `_hideUiChrome` is a single bool, but it's only auto-toggled in the `ActiveMkHarvestViewerValidationBatch` path. The cleanup extends it to single-shot and video. That logic moves to any UI library 1:1.
- **Status bar cleanup** — at most 3 columns, no buttons. The "Copy/Log Scene" buttons are renamed/removed; the "real" home for those actions is the Capture Automation window, which becomes a real Avalonia window with a real toolbar.

## What makes migration harder

- **PM4 color modes and overlay rendering** are tightly coupled to ImGui. `_pm4ColorMode`, `_pm4Legend`, the per-object color computation — all use `ColorFromSeed`, `ImGui.ColorConvertU32ToFloat4`, etc. A migration needs to re-implement the color legend UI.
- **The 3D viewport** itself. The viewer draws terrain + models + PM4 overlay + skybox + minimap via `WorldScene.Render(...)`. Avalonia doesn't render this for you; you'd need to embed the OpenGL context as a child HWND and route the render loop. Not impossible but it's the one piece of real risk.
- **Save/load semantics**. ImGui has no persistence; we hand-rolled it. Avalonia has persistence; we'd need to map our custom save format to Avalonia's.

## Decision: do not migrate right now

We are **not** going to migrate off ImGui in the near term. The work is too large to do incrementally, and a full rewrite would itself be "refactor hell" — just in a different language.

What we **are** doing (in spec 060):
- Fix the specific user-visible issues: duplication, status bar, misnamed buttons, capture UI
- Make the code more refactor-friendly so future incremental work is cheaper
- Document this note so a future contributor can pick up the migration question with full context

## When to revisit

Trigger conditions for re-evaluating:

- A new contributor needs more than 2 days to add a simple new panel (currently typical)
- The viewer needs to ship on non-Windows platforms (ImGui supports this, but our docking + custom layout code does not)
- A new feature (e.g. real-time collaboration, undo/redo) becomes easier with a retained-state UI library
- Avalonia 12+ adds first-class 3D viewport support (or Skia 3D) that removes the OpenGL-in-HWND risk

If any of these trigger, re-read this note and start with a spike: port ONE panel (Runtime Stats, since it's small and self-contained) to Avalonia, measure the effort, extrapolate.
