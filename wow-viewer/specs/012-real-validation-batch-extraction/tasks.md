# Tasks: Real Validation Batch Extraction

**Plan**: `012-real-validation-batch-extraction/plan.md`

**Phase 1 API Sketch**: `012-real-validation-batch-extraction/contracts.md`

**Pre-Port Checklist**: `012-real-validation-batch-extraction/checklists/pre-implementation.md`

---

Execution rule:

- do not start a later phase until the current phase validation is complete and the checklist still matches the planned file split

## Phase 1: Shared Validation Contract

- [ ] **1.1** Add shared validation batch models under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureVariant.cs`
  - create `ValidationCaptureTileRequest.cs`
  - create `ValidationCaptureBatchPlan.cs`
  - create `ValidationCaptureBatchResult.cs`
  - create `ValidationCaptureVariantResult.cs`
- [ ] **1.2** Add deterministic validation scene policy types under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureScenePolicy.cs`
  - create `ValidationCaptureVariantPolicy.cs`
  - create `ValidationCaptureArtifactPolicy.cs`
- [ ] **1.3** Add deterministic top-down camera math under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureCameraInput.cs`
  - create `ValidationCaptureCameraFrame.cs`
  - create `ValidationCaptureCameraSolver.cs`

## Phase 2: Readiness And Artifact Logic

- [ ] **2.1** Add shared readiness types under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureReadinessSnapshot.cs`
  - create `ValidationCaptureReadinessState.cs`
  - create `ValidationCaptureReadinessEvaluator.cs`
- [ ] **2.2** Add shared derived-artifact builders under `wow-viewer/src/core/WowViewer.Core.Runtime/World/Validation/`
  - create `ValidationCaptureArtifactInputs.cs`
  - create `ValidationCaptureArtifactOutputs.cs`
  - create `ValidationCaptureArtifactBuilder.cs`
- [ ] **2.3** Add focused shared tests under `wow-viewer/tests/WowViewer.Core.Tests/`
  - create `ValidationCaptureCameraSolverTests.cs`
  - create `ValidationCaptureReadinessEvaluatorTests.cs`
  - create `ValidationCaptureArtifactBuilderTests.cs`
  - create `ValidationCaptureScenePolicyTests.cs`

## Phase 3: Headless Tool Host

- [ ] **3.0** Confirm the Phase 1 and Phase 2 gates are actually closed before adding a tool project
  - shared tests pass for `ValidationCapture` slices
  - no host requirement is still hiding a missing shared contract
  - `checklists/pre-implementation.md` still matches the file split

- [ ] **3.1** Create the new tool project under `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`
  - create `WowViewer.Tool.ValidationCapture.csproj`
  - create `Program.cs`
  - create `ValidationCaptureCommand.cs`
  - add the project to `wow-viewer/WowViewer.slnx`
- [ ] **3.2** Add the headless run loop and framebuffer export host under `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`
  - create `HeadlessValidationCaptureRunner.cs`
  - create `HeadlessValidationFramebufferExporter.cs`
  - create `HeadlessValidationCaptureSession.cs`

## Phase 4: Real Renderer Adapter

- [ ] **4.0** Confirm the tool shell exists without preview-only renderer coupling
  - tool project builds cleanly
  - runner shell uses shared contracts rather than app-shell state bags
  - no dependency on `WorldGpuPreviewRenderer`

- [ ] **4.1** Add the real validation-scene adapter under `wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/`
  - create `ValidationWorldSceneAdapter.cs`
  - create `ValidationWorldSceneSnapshot.cs`
  - create `ValidationWorldScenePolicyApplier.cs`
- [ ] **4.2** Reproduce the bounded four-variant proof on the real renderer
  - prove primary output on staged `0_5_3_3368` `Azeroth_30_48`
  - prove `noliquids` output on staged `0_5_3_3368` `Azeroth_30_48`
  - prove `noobjects` output on staged `0_5_3_3368` `Azeroth_30_48`
  - prove `objectsonly` output on staged `0_5_3_3368` `Azeroth_30_48`
  - repeat the same bounded proof on staged `3_3_5_12340` `Azeroth_30_48`

## Phase 5: Dataset Handoff

- [ ] **5.1** Emit replacement downstream artifacts from the new tool path
  - extend `ValidationCaptureCommand.cs` or add `ValidationCaptureArtifactCommand.cs`
  - use shared `ValidationCaptureArtifactBuilder.cs`
  - write compatible `object_visibility_mask` output
  - write compatible `no_object_minimap` output
- [ ] **5.2** Document the cutover point and proof boundary
  - update `wow-viewer/README.md`
  - update continuity docs or memory when the bounded proof exists
  - state what remains legacy-only after the first slice
