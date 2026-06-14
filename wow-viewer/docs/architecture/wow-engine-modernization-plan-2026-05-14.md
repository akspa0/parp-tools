# wow-viewer Program Direction

**Status**: Replaced 2026-06-14 — this is a WoW viewer. The libraries we build for the viewer serve as the tooling that bridges to Unreal Engine.

## Core Philosophy

This is a WoW viewer. Not an engine program, not a multi-backend framework, not a Museum platform.

Everything in `wow-viewer/` exists to serve the viewer: rendering WoW worlds, inspecting WoW formats, and preserving WoW data truth across Alpha 0.5.3 through Cataclysm.

## Library Strategy

The format readers, converters, runtime contracts, and PM4 analysis code we build for the viewer are the same code that powers the Unreal Engine bridge. `WowViewer.Core`, `Core.IO`, `Core.PM4`, `Core.Runtime`, and `Core.Anim` are shared libraries with clean public surfaces — the viewer consumes them, and the UE bridge will consume them too.

No separate engine extraction. No Vulkan backend. No WebGL. No Museums profile.

- The viewer uses OpenGL/Silk.NET + ImGui — this is the primary diagnostic and inspection surface
- The libraries are built viewer-first, bridge-accessible
- The UE bridge (spec 055) consumes these same libraries from C++/C# interop

## What This Means For Existing Docs

- Remove references to "engine program", "museum-explorer", "Vulkan primary", "BASE repo extraction"
- Spec 055 (Unreal Engine Bridge) is the correct backend strategy — not Phases E0-E9
- The viewer remains OpenGL/Silk.NET for the foreseeable future
- Pipeline: format library → viewer proof → UE bridge consumption

## References Superseded

- `game-viewer-host-plan-2026-05-13.md` — remains active as the viewer migration sub-plan
- `wow-engine-editor-and-interop-plan-2026-05-14.md` — remains active as the editor/import-export sub-plan
- `wow-viewer-full-porting-roadmap.md` — remains active as the porting capability roadmap
- `game-viewer-plan-pack-2026-05-14/` — remains active as execution micro-plans for viewer work
- `specs/055-unreal-engine-bridge/` — correct backend strategy, supersedes Vulkan-first direction
- `specs/056-viewerapp-gpu-lod-modernization/` — OpenGL modernization within the viewer, not a multi-backend effort
