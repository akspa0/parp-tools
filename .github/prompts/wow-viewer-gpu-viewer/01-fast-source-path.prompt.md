---
description: "Implement the first performance-critical wow-viewer viewer slice by replacing slow viewer-time archive bootstrap and repeated MPQ probing with a shared fast source-catalog or manifest path. Use when the pain is map-open latency, shell-owned archive bootstrap, repeated virtual-file lookups, or failure to reuse the faster cached-catalog/direct-pipeline seams in wow-viewer tooling."
name: "wow-viewer Fast Source Path"
argument-hint: "Optional focus such as world-open latency, cached archive catalogs, viewer manifests, invalidation rules, or temporary app consumer wiring"
agent: "agent"
---

# wow-viewer Fast Source Path

Target repo file: .github/prompts/wow-viewer-gpu-viewer/01-fast-source-path.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/plans/wow_viewer_mdxviewer_cutaway_reset_plan_2026-04-24.md
4. gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md
5. gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md
6. wow-viewer/src/viewer/WowViewer.App/ViewerIoService.cs
7. wow-viewer/src/viewer/WowViewer.App/WowViewerArchiveBootstrap.cs
8. wow-viewer/src/viewer/WowViewer.App/WowViewerWorldSessionBootstrapper.cs
9. wow-viewer/src/viewer/WowViewer.App/WowViewerWorldSceneHost.cs
10. wow-viewer/tools/converter/WowViewer.Tool.Converter/Program.cs
11. wow-viewer/README.md
12. .github/copilot-instructions.md

## Problem

The current viewer still opens worlds through app-local archive bootstrap and shell-owned virtual-file reads. That keeps world-open latency high and makes runtime browsing pay costs already solved by cached catalog/direct pipeline seams in wow-viewer tooling.

## Goal

Create a shared fast source-access seam in wow-viewer libraries that future viewer hosts can consume directly.

- WowViewer.App may consume it only as a temporary bridge.
- The new seam should reuse cached archive catalogs, staged roots, and precomputed manifests where possible.
- Full archive enumeration and repeated MPQ probing must leave the live world-open hot path.

## Required Constraints

1. Do not touch gillijimproject_refactor/MdxViewer code.
2. Do not deepen WowViewer.App architecture beyond minimum consumer wiring.
3. Keep wow-viewer as the code owner for source access and cache policy.
4. Treat converter/dataset cached-catalog behavior as source of truth for the fast path.
5. Keep proof language precise. Build success is not runtime signoff.

## Concrete Scope

1. Extract a shared source service out of app-local ViewerIoService.
2. Converge bootstrap behavior with CreateArchiveCatalog/BuildLegacySearchRoots and cache-aware tooling flow.
3. Add viewer-facing cache keys and/or manifests for world-open and asset discovery.
4. Define invalidation rules for client-root/build-label/overlay change.
5. Expose measurable metrics for world-open latency and source-cache hit behavior.

## Out Of Scope

1. No terrain renderer port.
2. No WorldScene service split.
3. No new thin viewer host.
4. No shell polish work.
5. No claims about final rendering parity.

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for cache reuse, manifest reuse, invalidation, and virtual-file resolution
3. real-data world-open proof on fixed roots with before/after latency numbers
4. explicit statement whether path is cached-catalog-backed, manifest-backed, or still bootstrap-only

## Deliverables

1. the shared source-access seam
2. first temporary consumer wiring in wow-viewer
3. focused validation
4. measured speed delta on a fixed real-data world-open path
5. short note on remaining unsolved latency after this slice

## First Output

Start with:

1. current slow-path boundary you are assuming
2. exact shared seam you will add first
3. smallest file set that can land this slice
4. narrow proof that shows world-open is materially faster
5. what you are explicitly not claiming yet
