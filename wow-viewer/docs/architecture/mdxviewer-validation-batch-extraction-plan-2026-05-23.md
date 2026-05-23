# MdxViewer Validation Batch Extraction Plan

## Why This Note Exists

- the real renderer-truth path for dataset validation already exists, but it lives inside the legacy `MdxViewer` app shell
- the recent `wow-viewer` preview-only detour proved the failure mode clearly:
  - a bounded terrain preview is not the real validation renderer
  - a fake top-down preview path is not a substitute for the capture stack that already produces the primary, `noobjects`, `noliquids`, and `objectsonly` variants
- this note defines the smallest extraction seam that preserves the real renderer behavior without dragging the whole `ViewerApp` UI into `wow-viewer`

## Current Status

- the shared validation-batch contracts, readiness logic, camera math, and artifact builder are landed in `wow-viewer`
- the dedicated host tool `WowViewer.Tool.ValidationCapture` is landed
- `ValidationWorldSceneAdapter` now owns the bounded hidden-window GPU render/readback path behind `IValidationWorldSceneAdapter`
- bounded real-data proof now exists on both staged anchors:
  - `0_5_3_3368 / Azeroth_30_48`
  - `3_3_5_12340 / Azeroth_30_48`
- each current bounded proof completes the four capture families:
  - primary
  - `noliquids`
  - `noobjects`
  - `objectsonly`
- the remaining open step is not Phase 4 proof anymore; it is Phase 5 dataset handoff for `object_visibility_mask` and `no_object_minimap`, followed by longer-range replacement of the temporary `WorldGpuPreviewRenderer` backend reuse

## Problem Statement

- current renderer-truth capture for V16 and related dataset work depends on the legacy validation batch in:
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp_StartupAutomation.cs`
  - `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
- the actual rendered scene is owned by the legacy world stack:
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
  - `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
  - `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs`
  - `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`
  - `gillijimproject_refactor/src/MdxViewer/Rendering/Camera.cs`
- the forward ownership still belongs in `wow-viewer`, but the correct target is the legacy validation batch plus real world renderer behavior, not the existing `WorldGpuPreviewRenderer` preview surface in `WowViewer.App`

## What The Legacy Validation Batch Actually Owns

### Real Capture-Orchestration Responsibilities

- enqueue per-tile capture work for four concrete families:
  - primary
  - `noliquids`
  - `noobjects`
  - `objectsonly`
- apply deterministic scene policy overrides for the whole batch:
  - hide UI chrome
  - resize the window to the requested square resolution
  - force validation lighting
  - widen object streaming
  - disable object path filters
  - suppress very tall MDX clutter
  - disable WMO runtime group visibility gating
- apply per-request variant overrides:
  - terrain on or off
  - terrain liquids on or off
  - world objects on or off
- replace the normal free camera with a deterministic top-down orthographic capture matrix for validation requests
- wait for the scene to settle before capture:
  - scene content exists
  - framebuffer is ready and large enough
  - target tile is loaded
  - terrain streaming has stopped
  - world-object pending-load count is zero
  - required settled-frame count is reached
- read back the scene framebuffer and write PNG output
- stitch full-map outputs and derive `object_visibility_mask` plus `no_object_minimap`

### What Is Not Core To The Extraction

- the ImGui capture-automation window
- camera-shot bookmark persistence
- filtered shot-list browsing
- generic with-UI screenshot features
- taxi-ride video capture
- startup argument parsing as the long-term API surface
- the rest of `ViewerApp` panel state and shell chrome

## Minimal Dependency Cut

The first extraction should port only the behavior needed to reproduce the validation batch faithfully.

### Slice-A Critical Contracts

1. `ValidationCaptureBatchPlan`

- map identity
- build label
- dataset root
- output roots for each variant
- requested square resolution
- list of tile requests and variant flags

1. `ValidationCaptureScenePolicy`

- deterministic batch-wide render overrides
- deterministic variant-specific visibility overrides
- explicit constants now hard-coded in legacy capture automation

1. `ValidationCaptureCameraSolver`

- tile center computation
- ground-height sampling fallback order
- top-down orthographic matrix construction
- exact span and near/far policy for square and non-square outputs

1. `ValidationCaptureReadinessEvaluator`

- tile loaded
- terrain streaming idle
- framebuffer ready
- framebuffer large enough
- pending world-object loads drained
- settled-frame counting and timeout accounting

1. `ValidationCapturePostProcess`

- `object_visibility_mask` from either direct `objectsonly` silhouette or primary-vs-noobjects diff
- `no_object_minimap`
- stitched full-map outputs

### Slice-A Required Renderer Inputs

The first real port must consume a real world renderer that can already do these things:

- terrain and liquid rendering from actual ADT data
- WMO rendering through real `WmoRenderer`-equivalent behavior
- doodad and MDX/M2 rendering through real world-object placement paths
- scene-level visibility toggles for terrain, liquids, WMOs, doodads, WDL, sky
- readiness counters for tile streaming and deferred world-object loads

If those renderer inputs are missing, the extraction is not ready for batch capture parity yet.

## What Must Stay Out Of The First Port

- do not route the extraction through `wow-viewer/src/viewer/WowViewer.App/WorldGpuPreviewRenderer.cs`
- do not rebuild another fake CPU-composed world frame and call it capture parity
- do not port the entire `ViewerApp` class into `wow-viewer`
- do not make the first slice depend on ImGui, shell panels, or bookmark UX
- do not start with stitched-map polish before single-tile parity exists

## Recommended Ownership In wow-viewer

### Shared Runtime Or Core Ownership

Canonical home for shared validation-batch logic:

- `wow-viewer/src/core/WowViewer.Core.Runtime`

Recommended new area:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

This area should own:

- batch-plan models
- scene-policy constants and request-to-variant mapping
- tile-center and top-down camera math
- readiness-evaluator contracts
- post-process contracts for mask derivation metadata

### Headless Host Ownership

The first executable host should not be the desktop viewer shell.

Recommended home:

- a dedicated tool under `wow-viewer/tools/`

Recommended shape:

- `WowViewer.Tool.ValidationCapture`

Purpose:

- create the hidden GL window or offscreen context
- open the real world session
- drive the real world renderer
- execute the shared validation batch contract
- write capture artifacts for the harvester or dataset pipeline

### Harvester Integration Ownership

The dataset lane should consume capture outputs, not own the renderer.

Current downstream homes:

- `wow-viewer/data-harvester/`
- existing dataset-build scripts that patch renderer-truth artifacts into stores

## Ordered Extraction Path

### Phase 1 - Contract And Boundary Lock

1. add a `wow-viewer` architecture note or spec that states the real source of truth is the legacy validation batch, not `WorldGpuPreviewRenderer`
2. define `ValidationCaptureBatchPlan`, tile-request, variant, and result models in shared runtime code
3. port the deterministic top-down camera and scene-policy constants into shared runtime code with no renderer host yet

### Phase 2 - Real Headless Single-Tile Parity

1. build a dedicated headless tool host that opens a real world session and drives the real renderer without desktop-shell UI
2. port the readiness evaluator so single-tile capture waits for the same conditions the legacy batch requires
3. reproduce one real single-tile capture family set for `primary`, `noobjects`, `noliquids`, and `objectsonly`

Status: landed for the bounded `0_5_3_3368` and `3_3_5_12340` `Azeroth_30_48` proof anchors.

### Phase 3 - Artifact Parity

1. port the object-artifact post-process so `object_visibility_mask` and `no_object_minimap` are generated from the captured families with the same build-policy branch
2. validate parity on the existing bounded proof anchor before expanding scope:

- staged `0_5_3_3368`
- staged `3_3_5_12340`
- the established bounded `Azeroth_30_48` tile roots

Status: proof anchors now exist for the raw capture families, but artifact generation is still the next open work item.

### Phase 4 - Batch And Pipeline Ownership

1. add manifest-driven batch execution so the tool can consume dataset-generated tile plans directly
2. wire the existing `wow-viewer` dataset scripts to call the new tool instead of the legacy app for renderer-truth batches

Status: not started. Current next slice should stay smaller: finish Phase 3 artifact parity first.

## First Implementation Slice Recommendation

The safest first slice is not "headless everything." It is:

- define the shared validation-batch contract in `wow-viewer`
- port only the deterministic policy math and readiness contract first
- prove that the headless host can reproduce one single-tile four-variant capture family against the real renderer

That slice is small enough to validate independently and strict enough to reject any fallback into the fake preview path.

## Success Criteria

- `wow-viewer` has a headless renderer-truth capture path that uses the real world renderer rather than a preview substitute
- one bounded tile can reproduce the four real validation families:
  - primary
  - `noliquids`
  - `noobjects`
  - `objectsonly`
- readiness and settle behavior match the legacy batch closely enough that captures stop depending on manual delays
- the dataset pipeline can consume `object_visibility_mask` and `no_object_minimap` from the new tool without invoking `MdxViewer`
- the new ownership boundary is explicit enough that future work does not drift back into preview-only world capture again
