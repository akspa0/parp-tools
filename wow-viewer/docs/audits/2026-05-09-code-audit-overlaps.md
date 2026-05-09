# Code Audit: Overlaps in ADT Parsing Logic

## Audit Date
2026-05-09

## Overview
This audit identifies significant code duplication in ADT file parsing logic across multiple components of the wow-viewer codebase. The primary overlap exists between terrain data extraction implementations in different modules, leading to maintenance challenges and potential consistency issues.

## Identified Overlaps

### 1. AdtTensorPackBuilder vs WorldTerrainTileBuilder

Both classes implement nearly identical ADT parsing functionality:

- **AdtTensorPackBuilder** (`WowViewer.Core.IO.Maps.AdtTensorPackBuilder`)
  - Primary purpose: Build terrain tensor packs for ML training
  - Extracts: Heightmaps (MCVT), normals (MCNR), vertex colors (MCCV), lighting (MCLV), texture data (MCLY/MCAL), liquid data (MH2O/MCLQ)
  - Uses: Converter tool for dataset generation

- **WorldTerrainTileBuilder** (`WowViewer.Core.Runtime.World.Terrain.WorldTerrainTileBuilder`)
  - Primary purpose: Build terrain data for viewer runtime rendering
  - Extracts: Heightmaps (MCVT), normals (MCNR), vertex colors (MCCV), texture data
  - Uses: Viewer application for real-time rendering

**Identical implementations:**
- Chunk location resolution logic
- MCVT height extraction with base height offset
- MCNR normal extraction
- MCCV vertex color extraction
- Heightmap filling algorithms (`FillMixedParityGaps`, `FillRemainingGaps`)
- Vertex position mapping (`GetVertexPosition`)
- Chunk header parsing and offset calculation

### 2. AlphaEmbeddedAdtReader

- **Location**: `WowViewer.App.AlphaEmbeddedAdtReader`
- **Purpose**: Parse Alpha-era ADT files from embedded WDT archives
- **Overlap**: Implements nearly identical ADT parsing logic as the other two components, with minor variations for Alpha-specific formats
- **Duplicated functionality**: 
  - MCVT height extraction
  - Texture layer parsing
  - Liquid chunk extraction
  - Heightmap building
  - Vertex position mapping

### 3. Converter Tool Dependency

- **WowViewer.Tool.Converter** directly uses `AdtTensorPackBuilder` to generate tensor packs
- This creates a circular dependency where the converter tool depends on the core library that should be the source of truth
- The converter should be a consumer of the core library, not a direct caller of its internal parsing logic

## Recommendations

### 1. Consolidate ADT Parsing Logic

- Make `AdtTensorPackBuilder` the single source of truth for ADT file parsing
- Refactor `WorldTerrainTileBuilder` to use `AdtTensorPackBuilder` as a dependency rather than duplicating parsing logic
- Create a new `AdtTerrainData` class that encapsulates the parsed terrain data structure
- `WorldTerrainTileBuilder` should consume `AdtTerrainData` objects rather than parsing files directly

### 2. Replace AlphaEmbeddedAdtReader

- Remove `AlphaEmbeddedAdtReader` entirely
- Extend `AdtTensorPackBuilder` to handle both Alpha and LK-era ADT formats
- Add format detection and appropriate parsing logic within the consolidated parser
- Maintain backward compatibility through format-specific parsing branches

### 3. Refactor Converter Tool

- Update `WowViewer.Tool.Converter` to use the consolidated `AdtTensorPackBuilder` as intended
- Remove any direct file parsing logic from the converter tool
- Ensure the converter tool only calls the public API of `AdtTensorPackBuilder`

### 4. Create Unified ADT Reader Interface

- Define an `IAdtReader` interface that declares the contract for ADT file parsing
- Implement `AdtTensorPackBuilder` as the primary implementation
- Allow for future extensions (e.g., optimized readers for specific use cases)
- Use dependency injection to provide the reader to consumers

## Benefits of Implementation

- **Reduced code duplication**: Eliminates ~1,500+ lines of duplicated code
- **Improved maintainability**: Single source of truth for ADT parsing logic
- **Consistent behavior**: Ensures identical parsing results across all components
- **Easier debugging**: Issues can be isolated to one implementation
- **Simplified testing**: Single test suite for ADT parsing functionality
- **Better performance**: Eliminates redundant parsing operations

## Implementation Plan

1. Create `AdtTerrainData` class to encapsulate parsed terrain data
2. Refactor `AdtTensorPackBuilder` to return `AdtTerrainData` objects
3. Refactor `WorldTerrainTileBuilder` to consume `AdtTerrainData`
4. Extend `AdtTensorPackBuilder` to handle Alpha-era formats
5. Remove `AlphaEmbeddedAdtReader`
6. Update `WowViewer.Tool.Converter` to use new API
7. Create `IAdtReader` interface and dependency injection setup
8. Update all tests to use new architecture

## Verification

After implementation, verify consistency by:
- Running identical ADT files through all components
- Comparing output data structures
- Validating rendering results in viewer app
- Ensuring tensor packs maintain compatibility with training pipeline

This consolidation will significantly improve the codebase's maintainability and reliability while reducing technical debt.