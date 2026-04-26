---
description: "Port WMO/MDX/M2 world-object rendering ownership into wow-viewer rendering libraries so world-object drawing is fully GPU-owned and no longer trapped in app-local preview code. Use when terrain GPU ownership exists and object rendering is next critical parity gap."
name: "wow-viewer World Object Renderer Port"
argument-hint: "Optional focus such as WMO pass routing, MDX batching, M2 runtime draw integration, material/effect handling, or world-object renderer contracts"
agent: "agent"
---

# wow-viewer World Object Renderer Port

Target repo file: .github/prompts/wow-viewer-gpu-viewer/07-world-object-renderer-port.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs
4. gillijimproject_refactor/src/MdxViewer/Rendering/ModelRenderer.cs
5. gillijimproject_refactor/src/MdxViewer/Rendering/M2Renderer.cs
6. wow-viewer/src/viewer/WowViewer.App/WmoGpuPreviewRenderer.cs
7. wow-viewer/src/viewer/WowViewer.App/M2GpuPreviewRenderer.cs
8. wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs
9. wow-viewer/README.md

## Problem

World-object rendering still depends on temporary app-local consumers and mixed ownership, which blocks end-to-end GPU/runtime ownership for the world viewer.

## Goal

Land library-owned GPU world-object renderers for WMO, MDX, and M2 world consumption.

- keep renderer contracts in library ownership
- preserve working pass/batch behavior where possible
- use app shell only as temporary consumer

## Required Constraints

1. Do not modify gillijimproject_refactor implementation code.
2. Do not describe standalone preview parity as world parity.
3. Keep this slice renderer ownership-focused, not shell UI-focused.

## Concrete Scope

1. define/port WMO world-render consumer path in rendering library
2. define/port MDX world-render consumer path in rendering library
3. define/port M2 world-render consumer path in rendering library
4. connect object-pass inputs to library-owned render services
5. add temporary app adapter usage only where needed

## Out Of Scope

1. no full WorldScene service split yet
2. no thin host build yet
3. no editor-focused surfaces
4. no broad shell polish

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for pass/input contract generation where feasible
3. real-data proof that states which object families are visible in-world after this slice
4. explicit statement of unresolved material/ordering/perf gaps after this slice

## Deliverables

1. library-owned WMO/MDX/M2 world-object render seams
2. temporary app consumption path
3. focused validation and real-data object visibility proof
4. remaining gap summary before WorldScene split
