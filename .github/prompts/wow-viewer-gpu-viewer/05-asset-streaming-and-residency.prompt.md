---
description: "Port the useful non-UI core of MdxViewer WorldAssetManager into wow-viewer runtime ownership, including queue policy, residency, cache reuse, and negative lookup suppression. Use when the next slice is reducing deferred load stalls and making asset loading predictable and fast."
name: "wow-viewer Asset Streaming and Residency"
argument-hint: "Optional focus such as pending queue policy, residency state machine, negative lookup cache, background-ready loading, or per-tile manifest reuse"
agent: "agent"
---

# wow-viewer Asset Streaming and Residency

Target repo file: .github/prompts/wow-viewer-gpu-viewer/05-asset-streaming-and-residency.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs
4. wow-viewer/src/viewer/WowViewer.App/WowViewerWorldSceneHost.cs
5. wow-viewer/src/viewer/WowViewer.App/WowViewerWorldAssetInventory.cs
6. wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderFrameStats.cs
7. wow-viewer/src/core/WowViewer.Core.Runtime/World/WorldRenderOptimizationAdvisor.cs
8. wow-viewer/README.md

## Problem

Asset loading still incurs expensive deferred drains and shell-owned queue handling. Residency policy and load behavior are not fully library-owned yet.

## Goal

Create library-owned asset streaming and residency services in wow-viewer runtime.

- own pending/priority queues and residency state
- own ready/fail/missing caches
- suppress repeated failed lookups
- support fast-source/manifest reuse from prompt 01

## Required Constraints

1. Do not modify gillijimproject_refactor implementation code.
2. Keep this slice runtime/service ownership only.
3. Keep WowViewer.App as temporary consumer.
4. Keep proof language precise: no renderer parity claims yet.

## Concrete Scope

1. define shared runtime asset residency contracts
2. port useful queue/cache behavior from WorldAssetManager
3. add negative lookup suppression in shared runtime seam
4. align queue processing with source-cache/manifest path
5. expose measurable queue/drain metrics for runtime proof

## Out Of Scope

1. no terrain renderer port yet
2. no object renderer parity yet
3. no WorldScene service split yet
4. no shell UX work

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for queue ordering, residency transitions, cache hit behavior, and negative lookup suppression
3. real-data proof with before/after metrics for pending count and deferred-load drain cost
4. explicit statement of what remains app-local after this slice

## Deliverables

1. shared runtime asset-streaming/residency seam
2. temporary app consumption path
3. focused validation and performance deltas
4. remaining gap summary before renderer port slices
