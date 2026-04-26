---
description: "Port the working terrain adapter cluster from MdxViewer into WowViewer.Core.IO so ADT/WDT tile ownership, split-ADT rules, and era-specific terrain behavior become library-owned. Use when the next slice is tile-load correctness and shared terrain data ownership rather than renderer parity."
name: "wow-viewer Terrain Adapter Extraction"
argument-hint: "Optional focus such as Alpha adapter behavior, standard adapter behavior, split ADT file-family ownership, placement reads, or tile-load contracts"
agent: "agent"
---

# wow-viewer Terrain Adapter Extraction

Target repo file: .github/prompts/wow-viewer-gpu-viewer/04-terrain-adapter-extraction.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/src/MdxViewer/Terrain/StandardTerrainAdapter.cs
4. gillijimproject_refactor/src/MdxViewer/Terrain/AlphaTerrainAdapter.cs
5. wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTextureReader.cs
6. wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTextureChunkReader.cs
7. wow-viewer/README.md

## Problem

Terrain tile ownership is still split between temporary app paths and legacy behavior references, which blocks clean runtime and renderer migration.

## Goal

Land library-owned terrain adapter services in WowViewer.Core.IO that match working MdxViewer behavior for tile loading and split-family handling.

## Required Constraints

1. Do not modify gillijimproject_refactor implementation code.
2. Keep this slice in wow-viewer shared libraries.
3. Reuse existing shared readers where valid, but keep behavior parity with working adapters.

## Concrete Scope

1. define shared terrain adapter contracts
2. port standard-era adapter behavior
3. port Alpha-era adapter behavior
4. port split-family tile resolution and placement read rules
5. add focused real-data tests on fixed Alpha and standard roots

## Out Of Scope

1. no terrain GPU renderer port yet
2. no world-object rendering changes
3. no WorldScene service split yet
4. no shell-level viewer changes beyond temporary consumption

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for tile enumeration, split-family resolution, and placement loading
3. real-data proof on fixed roots with explicit era/build coverage
4. explicit statement of what terrain families are closed vs still partial

## Deliverables

1. terrain adapter contracts and implementations in WowViewer.Core.IO
2. focused regression tests
3. temporary consumer wiring (if needed)
4. precise proof summary and remaining gaps
