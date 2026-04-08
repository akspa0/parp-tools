---
description: "Implement or plan the next wow-viewer minimap-generation slice in the active compatibility host. Use when building deterministic one-PNG-per-ADT capture, capture presets, queue state, filter-aware map harvest flows, or repeatable tile framing in MdxViewer before the wow-viewer CLI cutover is real."
name: "wow-viewer Minimap 01 Deterministic ADT Capture Queue"
argument-hint: "Optional map, capture preset, or queue-seam hotspot to prioritize"
agent: "agent"
---

Implement or plan the deterministic active-viewer capture queue that can harvest real minimap tiles as one PNG per ADT.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
4. `gillijimproject_refactor/plans/wow_viewer_minimap_generation_plan_2026-04-08.md`
5. `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
6. `gillijimproject_refactor/src/MdxViewer/ViewerApp_Sidebars.cs`
7. `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
8. `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
9. `wow-viewer/src/core/WowViewer.Core.Runtime/World/ObjectPathFilterEntry.cs`

## Goal

Produce a deterministic viewer-hosted minimap capture queue that emits exactly one output PNG per ADT tile and can already use path-family filtering when hiding buildings or doodad families for the capture pass.

## Current Working Boundary

- fog distance and detailed ADT residency are already large enough to support much bigger aerial views than the old `5000` cap
- `MdxViewer` now has per-map object path-family filters and selected-object quick-add prefix buttons
- `ObjectPathFilterEntry` already lives in shared `wow-viewer` runtime, but the rest of minimap capture ownership is still inside the active viewer host
- there is still no deterministic tile queue, tile manifest, or one-PNG-per-ADT proof path

## Non-Negotiable Constraints

- each exported image must represent exactly one ADT tile worth of capture intent
- keep the tile naming and coordinate mapping deterministic and machine-consumable
- do not replace real scene capture with a fake WDL-only or minimap-texture-only shortcut
- reuse the existing path-family filter model instead of inventing one viewer-only object suppression format
- keep this slice narrow enough that the active viewer remains the compatibility host; the CLI cutover belongs to the next slice
- do not claim global minimap correctness or runtime signoff from build-only proof

## What The Work Must Produce

1. the exact tile-framing contract for one ADT output
2. the exact queue or preset state that should exist in the active viewer after this slice
3. the exact files to change
4. the narrowest real-data proof that shows the queue is deterministic
5. what should stay out of scope until the CLI slice

## Deliverables

Return all items:

1. the exact tile-capture seam to implement
2. why it is the right first minimap slice
3. exact files to change
4. exact validation to run
5. what should stay out of scope
6. what capture artifacts should exist afterward

## First Output

Start with:

1. the tile-framing boundary you are assuming today
2. the single next queue seam to extract or implement
3. the narrow proof that would show one-PNG-per-ADT capture is real
4. what you are explicitly not claiming yet