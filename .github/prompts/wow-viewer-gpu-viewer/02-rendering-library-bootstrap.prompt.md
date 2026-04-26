---
description: "Create the dedicated wow-viewer rendering library so GPU world/model rendering ownership moves out of WowViewer.App. Use when GPU renderers are still app-local preview code and the next slice is establishing library-owned rendering contracts and project shape."
name: "wow-viewer Rendering Library Bootstrap"
argument-hint: "Optional focus such as project shape, renderer service contracts, frame input contracts, or temporary app adapter wiring"
agent: "agent"
---

# wow-viewer Rendering Library Bootstrap

Target repo file: .github/prompts/wow-viewer-gpu-viewer/02-rendering-library-bootstrap.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs
4. wow-viewer/src/viewer/WowViewer.App/M2GpuPreviewRenderer.cs
5. wow-viewer/src/viewer/WowViewer.App/WmoGpuPreviewRenderer.cs
6. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
7. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs
8. wow-viewer/README.md

## Problem

GPU rendering still lives in app-local preview classes. That blocks a clean thin-host architecture and keeps renderer ownership in the wrong place.

## Goal

Add a dedicated wow-viewer rendering library and move foundational GPU renderer ownership into it.

- Rendering service contracts should be library-owned.
- App shell should become a consumer, not the renderer owner.
- Keep this slice narrow and architecture-first.

## Required Constraints

1. Do not touch gillijimproject_refactor implementation code.
2. Do not mix this slice with full feature-port work.
3. Keep final renderer ownership out of WowViewer.App.

## Concrete Scope

1. create rendering library project in wow-viewer
2. define base renderer contracts and frame/scene input contracts
3. move first shared shader/buffer plumbing out of app-local renderers
4. add temporary adapter wiring from app shell to new library seam

## Out Of Scope

1. no full terrain renderer port yet
2. no full WMO/MDX/M2 world rendering parity yet
3. no WorldScene service split yet
4. no thin host implementation yet

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused compile-level validation for new rendering contracts
3. explicit statement of what moved to library ownership and what remains temporary in app shell

## Deliverables

1. rendering library project and references
2. first library-owned renderer seam
3. temporary app consumer wiring
4. focused validation and ownership boundary summary
