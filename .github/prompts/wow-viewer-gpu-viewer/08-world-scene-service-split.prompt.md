---
description: "Split WorldScene behavior into wow-viewer library services by staged ownership seams. Use when fast source path, asset streaming, and terrain/object GPU renderer seams already exist and the next slice is replacing synthetic app-local world orchestration with library-owned scene services."
name: "wow-viewer WorldScene Service Split"
argument-hint: "Optional focus such as scene bootstrap, active tile planning, update loop ownership, selection state, visibility orchestration, or pass sequencing boundaries"
agent: "agent"
---

# wow-viewer WorldScene Service Split

Target repo file: .github/prompts/wow-viewer-gpu-viewer/08-world-scene-service-split.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/plans/wow_viewer_world_runtime_service_plan_2026-03-31.md
4. gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs
5. wow-viewer/src/core/WowViewer.Core.Runtime/World/Visibility/WorldObjectVisibilityCollector.cs
6. wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldFramePassCoordinator.cs
7. wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/WorldObjectPassCoordinator.cs
8. wow-viewer/src/viewer/WowViewer.App/WowViewerWorldRuntimeBridge.cs
9. wow-viewer/README.md

## Problem

World orchestration is still partly synthetic/app-local and not yet fully library-owned. That keeps ownership fragmented and blocks a clean thin host.

## Goal

Split working WorldScene behavior into staged wow-viewer library services.

- move scene ownership to libraries in narrow slices
- reduce synthetic app-local bridge ownership
- keep pass/visibility seams integrated with new renderer/runtime services

## Required Constraints

1. Do not modify gillijimproject_refactor implementation code.
2. Do not attempt monolithic single-PR WorldScene copy.
3. Keep slice boundaries explicit and testable.
4. Keep proof language precise and incremental.

## Concrete Scope

1. define scene bootstrap/open service seam
2. define active tile planning/service seam
3. define update/streaming loop service seam
4. define selection/state ownership seam
5. align visibility/pass sequencing with existing runtime coordinators
6. shrink temporary app-local runtime bridge usage

## Out Of Scope

1. no thin host cutover yet
2. no broad shell redesign
3. no editor feature migration
4. no unrelated format work

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for each new scene service seam
3. real-data proof showing which world orchestration responsibilities moved to libraries
4. explicit statement of remaining temporary bridge ownership

## Deliverables

1. staged scene service seams in wow-viewer libraries
2. reduced app-local/synthetic orchestration surface
3. focused validation and real-data ownership proof
4. next unresolved seams before thin host cutover
