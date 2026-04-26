---
description: "Port terrain mesh and terrain texturing behavior into a dedicated wow-viewer rendering library as a fully GPU-owned terrain consumer. Use when the next slice is replacing temporary app-local terrain preview rendering and eliminating CPU-baked terrain color paths as architecture targets."
name: "wow-viewer Terrain GPU Renderer Port"
argument-hint: "Optional focus such as terrain mesh contract, layer blending, alpha array handling, texture upload strategy, or renderer input contracts"
agent: "agent"
---

# wow-viewer Terrain GPU Renderer Port

Target repo file: .github/prompts/wow-viewer-gpu-viewer/06-terrain-gpu-renderer-port.prompt.md

## Read First

1. gillijimproject_refactor/memory-bank/activeContext.md
2. gillijimproject_refactor/memory-bank/progress.md
3. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs
4. gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs
5. wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs
6. wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTextureChunkReader.cs
7. wow-viewer/src/core/WowViewer.Core.IO/Maps/AdtTextureReader.cs
8. wow-viewer/README.md

## Problem

Terrain rendering ownership is still tied to temporary app-local code, and existing preview paths include behavior that should not be final architecture.

## Goal

Port terrain mesh and texturing into the dedicated rendering library as the canonical GPU terrain path.

- preserve working terrain-layer blend behavior
- keep terrain input and draw contracts library-owned
- remove shell-owned terrain rendering ownership

## Required Constraints

1. Do not touch gillijimproject_refactor implementation code.
2. No CPU-baked terrain-color fallback as target behavior.
3. No final shell-owned texture loops.
4. Keep this slice focused on terrain GPU ownership.

## Concrete Scope

1. port terrain mesh build contracts into rendering library
2. port terrain shader/layer blend behavior
3. define texture/alpha upload contracts compatible with fast-source path
4. add temporary app consumer adapter to new library renderer
5. instrument first-textured-frame timing for proof

## Out Of Scope

1. no full world-object renderer port yet
2. no WorldScene service split yet
3. no thin host implementation yet
4. no shell redesign work

## Required Validation

1. dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug
2. focused tests for mesh/input contracts and layer/alpha handling where feasible
3. real-data proof that explicitly states textured terrain capability and timing
4. explicit statement of what temporary terrain path remains and why

## Deliverables

1. library-owned terrain GPU renderer seam
2. temporary app consumption path
3. focused validation with timing metrics
4. clear remaining gaps before world-object rendering slice
