# Tasks: Real Validation Batch Extraction

**Plan**: `012-real-validation-batch-extraction/plan.md`

**Phase 1 API Sketch**: `012-real-validation-batch-extraction/contracts.md`

**Pre-Port Checklist**: `012-real-validation-batch-extraction/checklists/pre-implementation.md`

---

Execution rule:

- do not start a later phase until the current phase validation is complete and the checklist still matches the planned file split

## Phase 1: Shared Validation Contract

- [x] **1.1** Add shared validation batch models under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureVariant.cs`
  - create `ValidationCaptureTileRequest.cs`
  - create `ValidationCaptureBatchPlan.cs`
  - create `ValidationCaptureBatchResult.cs`
  - create `ValidationCaptureVariantResult.cs`
  - create `ValidationCaptureReadinessState.cs` because the result model already depends on it
- [x] **1.2** Add deterministic validation scene policy types under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureScenePolicy.cs`
  - create `ValidationCaptureVariantPolicy.cs`
  - create `ValidationCaptureArtifactPolicy.cs`
- [x] **1.3** Add deterministic top-down camera math under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureCameraInput.cs`
  - create `ValidationCaptureCameraFrame.cs`
  - create `ValidationCaptureCameraSolver.cs`

## Phase 2: Readiness And Artifact Logic

- [x] **2.1** Add shared readiness types under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureReadinessSnapshot.cs`
  - create `ValidationCaptureReadinessEvaluator.cs`
- [x] **2.2** Add shared derived-artifact builders under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureArtifactInputs.cs`
  - create `ValidationCaptureArtifactOutputs.cs`
  - create `ValidationCaptureArtifactBuilder.cs`
- [x] **2.3** Add focused shared tests under `wow-viewer/tests/WowViewer.Core.Tests/`
  - create `ValidationCaptureCameraSolverTests.cs`
  - create `ValidationCaptureReadinessEvaluatorTests.cs`
  - create `ValidationCaptureArtifactBuilderTests.cs`
  - create `ValidationCaptureScenePolicyTests.cs`

## Phase 3: Headless Tool Host

- [x] **3.0** Confirm the Phase 1 and Phase 2 gates are actually closed before adding a tool project
  - shared tests pass for `ValidationCapture` slices
  - no host requirement is still hiding a missing shared contract
  - `checklists/pre-implementation.md` still matches the file split

- [x] **3.1** Create the new tool project under `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`
  - create `WowViewer.Tool.ValidationCapture.csproj`
  - create `Program.cs`
  - create `ValidationCaptureCommand.cs`
  - create shared runtime `HeadlessValidationCaptureSession.cs` under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - add the project to `wow-viewer/WowViewer.slnx`
- [x] **3.2** Add the headless run loop and framebuffer export host under `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`
  - create shared runtime `ValidationWorldSceneSnapshot.cs`
  - create shared runtime `IValidationWorldSceneAdapter.cs`
  - create `HeadlessValidationCaptureRunner.cs`
  - create `HeadlessValidationFramebufferExporter.cs`

## Phase 4: Real Renderer Adapter

- [x] **4.0** Confirm the tool shell exists without preview-only renderer coupling
  - tool project builds cleanly
  - runner shell uses shared contracts rather than app-shell state bags
  - no dependency on `WorldGpuPreviewRenderer`

- [x] **4.1** Add the real validation-scene adapter under `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`
  - `ValidationWorldScenePolicyApplier.cs` landed first as a pure tested slice
  - `ValidationWorldSceneAdapter.cs` now binds real runtime-frame snapshots through `WowViewerWorldRuntimeBridge.Build(...)`
  - `WorldGpuPreviewRenderer.cs` now has an explicit `ValidationCaptureCameraFrame` render path so the GPU backend can consume viewer-style validation matrices instead of only the preview camera
  - `ValidationWorldSceneAdapter` now owns the bounded Phase 4 GL-host slice itself: hidden-window OpenGL capture, direct `ValidationCaptureCameraFrame` use, framebuffer readback, and PNG export through `IValidationWorldSceneAdapter`
  - the current implementation bypasses `WowViewerWorldScenePlanner` preview framing, but it still reuses `WorldGpuPreviewRenderer` as a temporary backend rather than a final wow-viewer-owned renderer seam
- [x] **4.2** Reproduce the bounded four-variant proof on the real renderer
  - [x] prove primary output on staged `0_5_3_3368` `Azeroth_30_48`
  - [x] prove `noliquids` output on staged `0_5_3_3368` `Azeroth_30_48`
  - [x] prove `noobjects` output on staged `0_5_3_3368` `Azeroth_30_48`
  - [x] prove `objectsonly` output on staged `0_5_3_3368` `Azeroth_30_48`
  - [x] repeat the same bounded proof on staged `3_3_5_12340` `Azeroth_30_48`

## Phase 5: Dataset Handoff

- [ ] **5.1** Emit replacement downstream artifacts from the new tool path
  - extend `ValidationCaptureCommand.cs` or add `ValidationCaptureArtifactCommand.cs`
  - use shared `ValidationCaptureArtifactBuilder.cs`
  - write compatible `object_visibility_mask` output
  - write compatible `no_object_minimap` output
- [ ] **5.2** Document the cutover point and proof boundary
  - [x] update `wow-viewer/README.md`
  - [x] update continuity docs or memory when the bounded proof exists
  - [x] update the architecture note and Speckit status to reflect completed Phase 4 proof on both bounded anchors
  - state what remains legacy-only after the first slice
