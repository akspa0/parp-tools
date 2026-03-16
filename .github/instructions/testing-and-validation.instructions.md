---
description: "Use when adding tests, claiming a fix is verified, or touching regression-prone viewer or converter code in gillijimproject_refactor. Covers the current testing reality, practical test seams, and required validation language."
name: "Testing And Validation Reality"
applyTo: "gillijimproject_refactor/src/**/*.cs, gillijimproject_refactor/src/**/*.csproj"
---
# Testing And Validation Reality

- The active `gillijimproject_refactor/src` path does not currently have meaningful first-party automated coverage for terrain alpha-mask regressions.
- Do not describe tests under `lib/*`, archived folders, or `gillijimproject_refactor/next` as coverage for the active MdxViewer and WoWMapConverter terrain pipeline.
- Do not mark terrain work complete based only on synthetic fixtures. This repo has a history of code that looked correct in isolation but failed on real game data.

## Preferred Test Strategy

- Put parser and decode tests in a first-party test project next to the active code, for example `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core.Tests`.
- Put viewer-side mesh packing and adapter integration tests in a first-party viewer test project, for example `gillijimproject_refactor/src/MdxViewer.Tests`.
- Favor small deterministic unit tests for pure decode and packing math.
- Pair those with at least one real-data validation step using the fixed development data paths.

## High-Value Regression Seams

- `Mcal` decoding for 4-bit, big-alpha, and compressed-alpha variants.
- `StandardTerrainAdapter.ExtractAlphaMaps()` behavior for LK tiles and split `*_tex0.adt` sourcing.
- `TerrainTileMeshBuilder` alpha-plus-shadow texture-array packing.
- Viewer debug and export paths that expose alpha masks, especially `TerrainImageIo` and `ViewerApp` menu actions.

## Required Language

- If no automated tests were added or run, say that explicitly.
- If only a build was run, say that explicitly.
- If real-data validation was not performed, say that explicitly.
- When proposing tests, prefer the smallest high-value seam instead of broad placeholder test suites.