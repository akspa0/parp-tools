# Implementation Plan: Real Validation Batch Extraction

**Spec**: `012-real-validation-batch-extraction/spec.md`

**Public Contract Shapes**: `012-real-validation-batch-extraction/contracts.md`

**Pre-Port Checklist**: `012-real-validation-batch-extraction/checklists/pre-implementation.md`

**Created**: 2026-05-23

## Port Discipline

- Phase order is strict. Do not start Phase 3 work until Phase 2 shared tests are green.
- Do not let the tool-host work backfill missing shared-runtime contracts. If a host need appears first, add the shared contract first and return to the host.
- Do not bind to `WorldGpuPreviewRenderer` or any preview-only world path for interim progress.
- Keep one bounded tile and one bounded four-variant proof as the controlling validation target until Phase 4 succeeds.
- Keep the legacy method-to-new-file mapping in `checklists/pre-implementation.md` current if the plan or contract split changes.

## Phase 1: Shared Validation Contract

**Goal**: Move the deterministic validation-batch contract into shared runtime code before building any new executable host.

Phase 1 file signatures are frozen in `contracts.md` for the initial implementation pass.

### Step 1.1 — Add batch models under shared runtime

Target project and folder:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

Add these files:

- `ValidationCaptureVariant.cs`
- `ValidationCaptureTileRequest.cs`
- `ValidationCaptureBatchPlan.cs`
- `ValidationCaptureBatchResult.cs`
- `ValidationCaptureVariantResult.cs`

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `MkHarvestViewerValidationCaptureTile`
  - `MkHarvestViewerValidationCapturePlan`
  - `PendingCaptureRequest`

**Validation**: shared runtime builds cleanly and the new models are usable from tests without pulling in viewer-shell state.

### Step 1.2 — Add deterministic validation scene policy types

Target project and folder:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

Add these files:

- `ValidationCaptureScenePolicy.cs`
- `ValidationCaptureVariantPolicy.cs`
- `ValidationCaptureArtifactPolicy.cs`

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `StartMkHarvestViewerValidationBatch(...)`
  - `ApplyCaptureRequestSceneOverrides(...)`
  - `ShouldPreferDirectObjectsOnlyMask(...)`

**Validation**: shared policy tests can assert the intended per-variant visibility and build-policy behavior.

### Step 1.3 — Add deterministic top-down camera math

Target project and folder:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

Add these files:

- `ValidationCaptureCameraInput.cs`
- `ValidationCaptureCameraFrame.cs`
- `ValidationCaptureCameraSolver.cs`

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `BuildMkHarvestViewerValidationShot(...)`
  - `TryGetMkHarvestViewerValidationSceneMatrices(...)`

**Validation**: tests cover tile-center placement, top-down eye or target generation, and orthographic span behavior for square and non-square outputs.

## Phase 2: Readiness And Artifact Logic

**Goal**: Port the settle logic and artifact derivation rules before wiring a headless renderer host.

### Step 2.1 — Add readiness snapshot and evaluator types

Target project and folder:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

Add these files:

- `ValidationCaptureReadinessSnapshot.cs`
- `ValidationCaptureReadinessState.cs`
- `ValidationCaptureReadinessEvaluator.cs`

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `IsCaptureRequestReady(...)`
  - `HasCaptureSceneContent()`
  - `HasCaptureFramebufferReady(...)`

**Validation**: tests cover pending-object loads, tile streaming, framebuffer-size gating, settled-frame accumulation, and timeout outcomes.

### Step 2.2 — Add derived-artifact builders in shared runtime

Target project and folder:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`

Add these files:

- `ValidationCaptureArtifactInputs.cs`
- `ValidationCaptureArtifactOutputs.cs`
- `ValidationCaptureArtifactBuilder.cs`

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `GenerateMkHarvestViewerValidationObjectArtifacts(...)`
  - `TryBuildDirectObjectVisibilityMask(...)`
  - `BuildObjectVisibilityMaskFromObjectsOnlyCapture(...)`
  - `BuildObjectVisibilityDiffMask(...)`

**Validation**: tests cover early-build direct silhouette policy and later-build primary-vs-noobjects diff policy.

### Step 2.3 — Add focused shared tests

Target test project:

- `wow-viewer/tests/WowViewer.Core.Tests/`

Add these files:

- `ValidationCaptureCameraSolverTests.cs`
- `ValidationCaptureReadinessEvaluatorTests.cs`
- `ValidationCaptureArtifactBuilderTests.cs`
- `ValidationCaptureScenePolicyTests.cs`

**Validation**: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter ValidationCapture`

## Phase 3: Headless Tool Host

**Goal**: Create a dedicated executable host for the replacement lane instead of hanging the new capture path off the desktop viewer shell.

Entry gate:

- Phase 1 and Phase 2 files exist and the focused shared tests pass.

### Step 3.1 — Create a dedicated validation-capture tool project

Target project root:

- `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`

Add these files:

- `WowViewer.Tool.ValidationCapture.csproj`
- `Program.cs`
- `ValidationCaptureCommand.cs`

Repository updates:

- add the new tool project to `wow-viewer/WowViewer.slnx`

Reference inputs:

- CLI shape and logging patterns from:
  - `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/`
  - `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/`

**Validation**: the new tool builds and exposes a bounded help or argument surface without invoking the desktop UI.

### Step 3.2 — Add the headless run loop and framebuffer export host

Target project and folder:

- `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`

Add these files:

- `HeadlessValidationCaptureRunner.cs`
- `HeadlessValidationFramebufferExporter.cs`
- `HeadlessValidationCaptureSession.cs`

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/ViewerApp_CaptureAutomation.cs`
  - `CompleteCaptureIfReady(...)`
  - `TryCaptureFramebufferToPng(...)`
- `gillijimproject_refactor/src/MdxViewer/ViewerApp.cs`
  - render-loop timing around validation matrix override use

**Validation**: the headless host can create a bounded renderable session and write a single PNG frame when the scene is ready.

## Phase 4: Real Renderer Adapter

**Goal**: Wire the headless tool to the real world renderer path rather than the existing preview-only app surface.

Entry gate:

- the tool project builds
- the runner shell can execute the shared contract without taking a dependency on preview-only world capture code

### Step 4.1 — Add a real validation-scene adapter layer

Target project and folder:

- `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`

Add these files:

- `ValidationWorldSceneAdapter.cs`
- `ValidationWorldSceneSnapshot.cs`
- `ValidationWorldScenePolicyApplier.cs`

Required inputs the adapter must surface:

- terrain visibility control
- liquid visibility control
- WMO visibility control
- doodad or MDX visibility control
- framebuffer size state
- target tile loaded state
- terrain streaming state
- pending world-object load count
- real frame render invocation

Legacy source reference:

- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldScene.cs`
- `gillijimproject_refactor/src/MdxViewer/Terrain/WorldAssetManager.cs`
- `gillijimproject_refactor/src/MdxViewer/Terrain/TerrainRenderer.cs`
- `gillijimproject_refactor/src/MdxViewer/Rendering/WmoRenderer.cs`

**Validation**: the adapter exposes the state needed by readiness evaluation without inventing a second fake world-render contract.

### Step 4.2 — Reproduce the bounded four-variant tile proof

Target output behavior:

- primary
- `noliquids`
- `noobjects`
- `objectsonly`

Bounded proof anchors:

- staged `0_5_3_3368`
- staged `3_3_5_12340`
- `Azeroth_30_48`

**Validation**: one headless run per bounded proof anchor produces the four expected capture families over real terrain plus real world-object rendering.

## Phase 5: Dataset Handoff

**Goal**: Replace the legacy app dependency for bounded renderer-truth artifact generation.

Entry gate:

- Phase 4 produced the four expected capture variants on the staged proof anchors
- artifact derivation behavior matches the build-policy branch captured in shared runtime

### Step 5.1 — Emit derived validation artifacts from the new tool path

Target project and folder:

- `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`

Add these files:

- `ValidationCaptureArtifactCommand.cs`
- or extend `ValidationCaptureCommand.cs` if the command shape remains small

Shared dependency:

- `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/ValidationCaptureArtifactBuilder.cs`

**Validation**: bounded headless runs can emit `object_visibility_mask` and `no_object_minimap` artifacts compatible with the current dataset lane.

### Step 5.2 — Document the cutover point

Target docs:

- `wow-viewer/README.md`
- relevant V16 renderer-truth workflow docs if the operator path changes
- continuity memory and active-context docs if the proof boundary moves

**Validation**: docs say exactly what is replaced, what proof exists, and what remains legacy-only.
