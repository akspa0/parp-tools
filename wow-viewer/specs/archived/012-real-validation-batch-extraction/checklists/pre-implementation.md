# Pre-Implementation Checklist: Real Validation Batch Extraction

**Purpose**: lock the first port to the real legacy validation seam and prevent drift during implementation

**Spec**: [../spec.md](../spec.md)

**Plan**: [../plan.md](../plan.md)

**Contracts**: [../contracts.md](../contracts.md)

## Guardrails

- [x] No planned file depends on `WowViewerWorldScenePlanner` or the default preview-camera path; the current temporary `WorldGpuPreviewRenderer` reuse stays isolated behind `IValidationWorldSceneAdapter`
- [ ] No new shared-runtime type depends on ImGui, desktop-window state, or preview-only image objects
- [ ] No work item requires editing `gillijimproject_refactor`; it remains reference-only for this feature
- [ ] The current implementation target is one bounded tile and four variants, not stitched-map parity first
- [ ] Validation language still distinguishes build proof from renderer parity proof

## Legacy To New Ownership Map

- [x] `StartMkHarvestViewerValidationBatch(...)` is split into `ValidationCaptureBatchPlan`, `ValidationCaptureScenePolicy`, and `ValidationCaptureCommand`
- [x] `PendingCaptureRequest` readiness outcome fields map cleanly into `ValidationCaptureReadinessState` and `ValidationCaptureVariantResult`
- [x] `ApplyCaptureRequestSceneOverrides(...)` is split into `ValidationCaptureScenePolicy` and `ValidationWorldScenePolicyApplier`
- [x] `BuildMkHarvestViewerValidationShot(...)` is split into `ValidationCaptureCameraInput`, `ValidationCaptureCameraFrame`, and `ValidationCaptureCameraSolver`
- [x] `TryGetMkHarvestViewerValidationSceneMatrices(...)` stays under `ValidationCaptureCameraSolver` plus adapter-provided ground-height resolution
- [x] `IsCaptureRequestReady(...)` is split into `ValidationWorldSceneSnapshot`, `ValidationCaptureReadinessSnapshot`, and `ValidationCaptureReadinessEvaluator`
- [x] `CompleteCaptureIfReady(...)` is split into `HeadlessValidationCaptureRunner` and `HeadlessValidationFramebufferExporter`
- [x] `GenerateMkHarvestViewerValidationObjectArtifacts(...)` is split into `ValidationCaptureArtifactInputs`, `ValidationCaptureArtifactOutputs`, and `ValidationCaptureArtifactBuilder`
- [x] `TryBuildDirectObjectVisibilityMask(...)` maps to the early-build branch in `ValidationCaptureArtifactBuilder`
- [x] `BuildObjectVisibilityMaskFromObjectsOnlyCapture(...)` maps to the direct silhouette branch in `ValidationCaptureArtifactBuilder`
- [x] `BuildObjectVisibilityDiffMask(...)` maps to the later-build diff branch in `ValidationCaptureArtifactBuilder`

## Phase Exit Gates

### Phase 1 Exit

- [x] Shared batch models compile under `WowViewer.Core.Runtime`
- [x] `ValidationCaptureReadinessState` is available early enough for `ValidationCaptureVariantResult` to compile without a fake placeholder type
- [x] Scene-policy types compile under `WowViewer.Core.Runtime`
- [x] Camera math compiles under `WowViewer.Core.Runtime`
- [x] No tool-project file was added to compensate for missing shared types

### Phase 2 Exit

- [x] Readiness types compile under `WowViewer.Core.Runtime`
- [x] Artifact types compile under `WowViewer.Core.Runtime`
- [x] Focused tests exist in `wow-viewer/tests/WowViewer.Core.Tests/`
- [x] `dotnet test i:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter ValidationCapture` passes

### Phase 3 Exit

- [x] `WowViewer.Tool.ValidationCapture` builds without opening the desktop viewer shell
- [x] The tool shell constructs the shared-runtime `HeadlessValidationCaptureSession` and routes through shared contracts
- [x] The runner and exporter do not introduce preview-only rendering dependencies

### Phase 4 Exit

- [x] A concrete scene adapter implements `IValidationWorldSceneAdapter`
- [x] Readiness evaluation consumes adapter snapshots instead of tool-local state bags
- [x] A bounded run produces `primary`, `noliquids`, `noobjects`, and `objectsonly`
- [x] The bounded proof exists on staged `0_5_3_3368` and staged `3_3_5_12340` for `Azeroth_30_48`

### Phase 5 Exit

- [x] The shared artifact builder emits `object_visibility_mask` and `no_object_minimap`
- [x] The build-policy branch is still explicit for early versus later builds
- [x] The bounded dataset workflow can consume the new artifacts without invoking `MdxViewer`
- [x] Continuity docs state what remains legacy-only after the bounded cutover

## Reference Proof Surface

Existing legacy proof command for the first bounded anchor:

```powershell
& "I:\parp\parp-tools\gillijimproject_refactor\src\MdxViewer\bin\Debug\net10.0-windows\ParpToolsWoWViewer.exe" --game-path "I:\parp\parp-tools\output\tmp\wowarchive-clients\0_5_3_3368\World of Warcraft" --build 0.5.3.3368 --listfile "I:\parp\parp-tools\gillijimproject_refactor\test_data\community-listfile-withcapitals.csv" --world "World\Maps\Azeroth\Azeroth.wdt" --validation-dataset-root "I:\parp\parp-tools\output\tmp\mdxviewer_validation_smoke\0_5_3_3368" --validation-output "I:\parp\parp-tools\output\tmp\mdxviewer_validation_smoke\0_5_3_3368" --validation-resolution 512 --force-validation-regeneration --exit-after-validation
```

Use that command's output family as the reference truth for the first `0_5_3_3368` parity checks. Do not treat it as proof for broader build coverage by itself.
