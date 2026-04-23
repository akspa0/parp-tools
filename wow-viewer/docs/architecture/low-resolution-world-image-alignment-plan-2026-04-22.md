# Low-Resolution World Image Alignment And Verification Plan

## Why This Note Exists

- there are two different problems getting mixed together:
  - preprocessing historical low-resolution world images into a tile-aligned dataset surface
  - interactively verifying outputs in `WowViewer.App`
- the second problem matters, but it should not block the first
- this note separates the ownership boundary so the tool and viewer work can progress in parallel instead of waiting on a full GUI parity jump

## Problem Statement

- older game eras include very low-resolution world images that may need to become continent-sized `64x64` tile-aligned supervision inputs
- the current `WowViewer.App` shell is not yet a reliable full-map verification surface:
  - it is still a bounded viewer shell rather than full legacy parity
  - loading broad map content can freeze or exhaust memory
- the legacy `MdxViewer` can still serve as reference evidence, but the forward ownership for both preprocessing and viewer parity belongs in `wow-viewer`

## Separate The Work Into Two Tracks

### Track A - Tool Or Shared-Core Ownership

- canonical home:
  - `wow-viewer/src/core/WowViewer.Core`
  - `wow-viewer/src/core/WowViewer.Core.IO`
  - `wow-viewer/tools/converter/WowViewer.Tool.Converter`
- purpose:
  - define the preprocessing contract for world-image alignment without requiring the desktop app first

### Track B - Viewer Verification Ownership

- canonical home:
  - `wow-viewer/src/viewer/WowViewer.App`
- purpose:
  - inspect aligned inputs and terrain-model outputs with streamed, bounded world loading instead of all-at-once map loads

## Track A - Alignment Pipeline

### Target Capabilities

- accept a low-resolution source image for a known world or continent map
- align it to the expected world orientation and extents
- scale it to a `64x64` tile grid contract
- cut or index it into per-tile products with reproducible metadata
- carry enough provenance so the ML pipeline knows where each tile came from and what transformations were applied

### Recommended Shared Contract

- define a world-image alignment manifest containing:
  - source image path
  - source build or era label
  - target world or continent name
  - transform parameters used for alignment
  - output tile-grid dimensions
  - tile coverage and missing-data markers
  - optional confidence or manual-override notes
- the converter should then emit per-tile products or references that can feed dataset or training workflows without inventing app-local state

### Recommended Converter Commands

- likely future command families:
  - `world-image-align`
  - `world-image-preview-transform`
  - `world-image-cut-tiles`
  - `world-image-export-manifest`
- exact naming can change, but the workflow boundary should stay tool-owned

### Important Non-Goals

- do not bury this logic in the desktop app first
- do not make ML training depend on manual GUI-only alignment work
- do not tie the preprocessing contract to the current terrain model shape

## Track B - Viewer Verification

### Real Viewer Requirement

- output verification needs a world viewer that behaves more like the legacy tile-streamed path and less like an all-content load attempt
- the immediate blocker is not missing shell chrome; it is missing streamed world consumption and memory-bounded loading

### Required Viewer Slice

- the next useful viewer parity slice for ML verification is:
  - streamed or bounded tile loading
  - explicit selection of a tile window or working set
  - loading only the terrain, placements, and overlays needed for that window
  - displaying project outputs beside or over real source data

### Verification Surfaces That Matter

- for ML verification, the viewer should be able to show:
  - source minimap or aligned image tiles
  - generated terrain outputs
  - optional WDL or prior overlays
  - object, PM4, liquid, or mask overlays where they affect interpretation
  - tile coordinates and dataset provenance

### What Does Not Need To Wait

- the preprocessing and alignment pipeline can be implemented before the viewer reaches full parity
- early proof can come from:
  - converter outputs
  - deterministic preview artifacts
  - bounded CLI diagnostics
  - legacy viewer reference captures where needed

## Ordered Implementation Path

1. define the alignment manifest and transform contract in shared or tool code
2. add converter proof commands for transform preview and `64x64` tile-grid cutting
3. wire those outputs into the dataset-building and training workflow as a new low-resolution-source-derived input family
4. add streamed tile-window loading to `WowViewer.App`
5. add side-by-side or overlay verification panels for aligned source imagery and generated outputs

## Success Criteria

- low-resolution world imagery can be aligned and cut reproducibly without using the desktop app as the only execution surface
- the ML pipeline can ingest those aligned tiles as a named input family with provenance
- `WowViewer.App` can inspect outputs over bounded tile windows without freezing or exhausting memory
- verification no longer depends on loading an entire development map in one shot