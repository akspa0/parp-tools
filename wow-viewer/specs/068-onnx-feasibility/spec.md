# Feature Specification: ONNX Export Feasibility for Terrain Reconstruction Models

**Feature Branch**: `068-onnx-feasibility`

**Created**: 2026-06-29

**Status**: Feasibility / Research

**Input**: User request — assess how feasible it would be for terrain reconstruction models (V14/V19/V20 chain) to target ONNX, enabling execution on any hardware and generalizing the reconstruction pipeline.

---

## Problem Statement

The current terrain reconstruction pipeline (V14 D1 → R1, V19 height regressor, V20 chained models) trains and runs inference exclusively in PyTorch. This ties the entire pipeline to:

1. **Hardware lock-in**: CUDA GPUs only for performant inference. No AMD ROCm, Apple Metal, Intel Arc, or CPU backends without manual work.
2. **Framework lock-in**: Every inference consumer must install PyTorch, which carries a multi-GB footprint and complex platform-specific dependencies.
3. **Deployment friction**: Embedding terrain reconstruction in tools (e.g., the C# viewer app, Unreal Engine bridge, web service) requires interop layers or subprocess calls to Python.

ONNX (Open Neural Network eXchange) is an open standard for representing trained models. ONNX Runtime (ORT) provides a lightweight cross-platform inference engine with backends for CUDA, DirectML, ROCm, OpenVINO, CoreML, CPU, and WebGPU.

This spec assesses whether each model in the terrain reconstruction pipeline can be exported from PyTorch to ONNX, what constraints apply, and what architectural changes (if any) would be required.

---

## Models Under Analysis

### V14 D1 — Tileset Decomposition (~3M params)

| Aspect | Detail |
|--------|--------|
| Architecture | 4-layer U-Net: `Conv2d → BN → ReLU` double-conv blocks, MaxPool down, bilinear upsample + skip, 4× 1×1 conv heads (2× sigmoid, 2× identity output) |
| Input | `[B, 3, 256, 256]` — minimap RGB |
| Outputs | `[B, 3, 256, 256]` tileset_1, `[B, 3, 256, 256]` tileset_2, `[B, 1, 256, 256]` alpha_1, `[B, 1, 256, 256]` alpha_2 |
| Custom ops | None |

### V14 R1 — Terrain Reconstruction (~5M params)

| Aspect | Detail |
|--------|--------|
| Architecture | 5-layer U-Net: same block style, 3 output heads with different spatial sizes (257×257, 16×16, 256×256) |
| Input | `[B, 3, 256, 256]` — residual image |
| Outputs | `[B, 1, 257, 257]` height, `[B, 1, 16, 16]` holes, `[B, 1, 256, 256]` liquid |
| Special features | `AdaptiveAvgPool2d((16, 16))` in head_holes; `Upsample(size=(257,257))` in head_height; `nn.functional.pad` for odd-sized skip alignment |
| Custom ops | None |

### V19 — Height Regressor (~20M params)

| Aspect | Detail |
|--------|--------|
| Architecture | ResConvBlock (residual + reflect padding + GroupNorm/BatchNorm) + BilinearUp U-Net, global pooling + FC for height bounds |
| Input | `[B, 3, 256, 256]` — minimap RGB (or more with normals) |
| Outputs | `[B, 1, 257, 257]` global height, `[B, 1, 257, 257]` local height, `[B, 4]` bounds |
| Special features | `padding_mode="reflect"` in Conv2d; `GroupNorm`; AdaptiveAvgPool2d for bounds; `BilinearUp` uses `F.interpolate(scale_factor=2)` |
| Custom ops | None |

### V20-MSS — Semantic Segmentor

| Aspect | Detail |
|--------|--------|
| Architecture | ResConvBlock + GroupNorm + BilinearUp U-Net, 3 output heads (5ch logits, 1ch sigmoid, 4ch sigmoid) |
| Input | `[B, 3, 256, 256]` |
| Outputs | `[B, 5, 256, 256]` liquid logits, `[B, 1, 256, 256]` object mask, `[B, 4, 256, 256]` alpha weights |
| Custom ops | None |

### V20-TII — Terrain Intent Inpainter

| Aspect | Detail |
|--------|--------|
| Architecture | Same backbone as MSS, 10 input channels, single output head with `F.interpolate(size=(257,257))` |
| Input | `[B, 10, 256, 256]` — minimap + object mask + liquid map + brush prior |
| Output | `[B, 1, 257, 257]` heightmap |
| Custom ops | None |

### V20-TFC — Fingerprint Classifier

| Aspect | Detail |
|--------|--------|
| Architecture | Simple 6-layer CNN: Conv2d → ReLU → MaxPool → Conv2d → BN → ReLU → MaxPool × 2, then AdaptiveAvgPool1d + FC classifier + FC regressor |
| Input | `[B, 7, 64, 64]` or similar crop |
| Outputs | `[B, 200]` class logits, `[B, 4]` regression params |
| Custom ops | None |

### V20-OPR — Placement Restorer

| Aspect | Detail |
|--------|--------|
| Architecture | Same pattern as TFC, 4 input channels, FC classifier + regressor |
| Input | `[B, 4, 64, 64]` RGB crop + object mask |
| Outputs | `[B, 500]` model logits, `[B, 5]` placement params |
| Custom ops | None |

---

## Feasibility Analysis

### 1. Operator Coverage

Every operator used across all models maps cleanly to ONNX ops:

| PyTorch Op | ONNX Opset | Status |
|-----------|-----------|--------|
| `nn.Conv2d` | `Conv` (opset 1+) | ✅ |
| `nn.BatchNorm2d` | `BatchNormalization` (opset 1+) | ✅ |
| `nn.GroupNorm` | `GroupNormalization` (opset 17+) | ✅ (needs opset ≥ 17) |
| `nn.MaxPool2d` | `MaxPool` (opset 1+) | ✅ |
| `nn.AdaptiveAvgPool2d` | `GlobalAveragePool` or `AveragePool` (opset 7+) | ✅ |
| `F.avg_pool2d` | `AveragePool` (opset 7+) | ✅ |
| `nn.ReLU` | `Relu` (opset 1+) | ✅ |
| `F.sigmoid` / `nn.Sigmoid` | `Sigmoid` (opset 1+) | ✅ |
| `F.interpolate` (bilinear) | `Resize` with coordinate transform (opset 10+) | ✅ |
| `nn.Upsample` (bilinear) | `Resize` (opset 10+) | ✅ |
| `torch.cat` | `Concat` (opset 1+) | ✅ |
| `F.pad` (reflect) | `Pad` mode `reflect` (opset 18+) | ✅ (needs opset ≥ 18) |
| `F.pad` (zeros/replicate) | `Pad` (opset 2+) | ✅ |
| `nn.Linear` | `Gemm` / `MatMul` (opset 1+) | ✅ |
| `padding_mode="reflect"` in Conv2d | Decomposed to `Pad` + `Conv` (with `torch.onnx.export`) | ✅ |
| Element-wise add (residual) | `Add` (opset 1+) | ✅ |
| `torch.clamp` | `Clip` (opset 6+) | ✅ |

**Minimum required ONNX opset**: **18** (to cover GroupNorm and reflect padding without workarounds).

### 2. Multi-Output Model Export

All models with multiple outputs (D1: 4, R1: 3, MSS: 3, TFC: 2, OPR: 2) export correctly. ONNX `torch.onnx.export` naturally maps tuple return values to multiple output tensors.

**Pattern**: `model(inputs)` → `torch.onnx.export(model, inputs, "model.onnx", output_names=["out1", "out2", ...])`

**Risk**: Low.

### 3. Static vs Dynamic Shapes

All models use fixed spatial dimensions (256×256 input, specific output sizes). The only variable dimension is batch size. Using `dynamic_axes={"input": {0: "batch"}, "output_*": {0: "batch"}}` works for all models.

**Risk**: Low. All Resize/Interpolate ops use static target sizes.

### 4. Model Chaining (Inference Pipeline)

The V14 pipeline chains: `Minimap → D1 → compositor (arithmetic) → R1 → outputs`. The V20 pipeline chains: `Minimap → MSS → TFC → TII → OPR`.

For ONNX deployment, two strategies exist:

#### Strategy A: Separate ONNX models (Recommended)
Export each model as its own `.onnx` file. The orchestrator (C# app, UE plugin, etc.) runs each model sequentially via ONNX Runtime, passing outputs as inputs to the next model.

- **Pros**: Independent versioning, per-model optimization, no coupling
- **Cons**: Serialization/deserialization overhead between models (~negligible for 256×256 tensors)
- **Feasibility**: ✅ Trivial

#### Strategy B: Combined ONNX model
Wrap D1 + compositor + R1 (or the V20 chain) into a single `nn.Sequential`-style module. This collapses the chain into one `.onnx` file.

- **Pros**: Single file, no intermediate buffer plumbing
- **Cons**: The compositor (D2) is arithmetic subtraction — exportable as pure tensor ops; forces fixed architecture coupling
- **Feasibility**: ✅ Possible but not recommended for long-term flexibility

### 5. Post-Export Verification

Standard workflow:
```
torch.onnx.export(model, dummy, "model.onnx", ...)
onnx.check("model.onnx")                    # shape/type correctness
onnxruntime.InferenceSession("model.onnx")  # runtime correctness
np.testing.assert_allclose(ort_output, torch_output, atol=1e-5)  # numerical
```

### 6. Framework Compatibility and Versioning

| Component | Required Version | Notes |
|-----------|----------------|-------|
| PyTorch | ≥ 2.5 (current) | ONNX export stable; `dynamo_export` available as alternative to `torch.onnx.export` |
| `torch.onnx` | Export API | `torch.onnx.export` (classic) works; `torch.onnx.dynamo_export` (newer, better op coverage) available in 2.5+ |
| ONNX Runtime | ≥ 1.18 | Stable; supports opset 18+; backends for CUDA, DirectML, CPU, etc. |
| onnx | ≥ 1.16 | For model checker |
| onnxruntime | ≥ 1.18 | For runtime inference |

### 7. C# Integration via ONNX Runtime

ONNX Runtime has a first-class NuGet package: `Microsoft.ML.OnnxRuntime` and `Microsoft.ML.OnnxRuntime.Gpu` (CUDA) / `Microsoft.ML.OnnxRuntime.DirectML` / `Microsoft.ML.OnnxRuntime.OpenVino`.

This means the C# viewer app (`WowViewer.App`) can consume ONNX models without any Python dependency:

```csharp
using var session = new InferenceSession("d1.onnx");
var input = OrtTensor.FromArray(minimapData, new long[] { 1, 3, 256, 256 });
var outputs = session.Run(new[] { "input" }, new[] { input });
var tileset1 = outputs[0].AsTensor<float>();
```

**Feasibility**: ✅ This is a well-tested, documented production path.

### 8. Unreal Engine Integration via ONNX Runtime

Unreal Engine plugins exist (Microsoft's official `onnxruntime-unreal`, community `UONNX`). ONNX models load directly into UE inference sessions without Python.

**Feasibility**: ✅ Matches existing UE bridge strategy (spec 055).

### 9. Web/Edge Deployment

- **WebGPU backend**: ONNX Runtime Web with WebGPU backend is available. Models under 10M params (all of ours) are viable for browser inference.
- **ONNX Runtime WebAssembly**: CPU inference in browser. 20M-param models are feasible but slower.
- **Mobile**: ONNX Runtime for Android/iOS; CoreML backend for Apple devices.

---

## Constraint Summary

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| GroupNorm requires opset ≥ 17 | Must set `opset_version=18` in export | Trivial |
| Reflect padding requires opset ≥ 18 | Same; set opset 18 | Trivial |
| `AdaptiveAvgPool2d` to specific size | Exports correctly; ONNX uses `AveragePool` with kernel matching input size | Verify with onnxruntime after export |
| Multi-output dynamic batches | All models export with `dynamic_axes` | Test with batch=1, 2, 4, 8 |
| Sigmoid in forward produces floating-point mask | Exports as standalone `Sigmoid` op | Matches PyTorch behavior exactly |
| Sobel edge filter (R1 loss) | **Loss-only, not part of forward pass** | ❌ Not needed in ONNX; only for training |
| BCEWithLogitsLoss | **Loss-only** | ❌ Not needed in ONNX |

---

## Architectural Changes Required

### None required for export

Every model in the pipeline (V14 D1, R1; V19; V20 MSS, TFC, TII, OPR) uses **only standard PyTorch ops that have direct ONNX equivalents**. No custom autograd functions, no third-party ops, no JIT-incompatible control flow.

### Changes to consider (not required, but recommended for clean ONNX)

1. **Normalization outside the model**: If normalization (mean/std scaling) is currently embedded in the training script's dataset pipeline, it should be either:
   - (a) Moved into the model as the first layer (`nn.Identity` that records mean/std), or
   - (b) Documented as a pre-processing step the caller must perform before ONNX inference.
   
   Current code does normalization in the dataset, so (b) applies — document the expected input domain.

2. **Output post-processing**: If post-processing (denormalizing height values, argmax for liquid class) is done in training scripts after model outputs, it should be:
   - (a) Added to the ONNX model as extra ops, or
   - (b) Documented as caller responsibility.
   
   Current pipeline: R1 outputs raw height (not denormalized). Argmax for liquid is done outside. Option (b) is simplest.

---

## Risks and Limitations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| PyTorch ONNX export may produce different numerics for GroupNorm | Low | Test with `onnxruntime` and compare to PyTorch output at 1e-5 tolerance |
| `torch.onnx.dynamo_export` (new path) may have coverage gaps | Low | Use `torch.onnx.export` (classic path) which is battle-tested |
| ONNX Runtime CPU inference for 20M-param V19 model may be slow | Medium | Estimate: ~100-500ms per 256×256 tile on CPU. Use DirectML/CUDA backends for production |
| Model update cadence (training new checkpoints) requires re-export | Low | Automate export as a post-training step in the training script |
| ONNX opset version creep (opset 18 may become legacy) | Low | Update export opset as needed; models are simple enough to port forward |

---

## Recommendation

**HIGH FEASIBILITY** — all terrain reconstruction models are clean candidates for ONNX export with zero architectural changes. The models use only standard Conv2d/BatchNorm/ReLU/MaxPool/Interpolate/Concat operations that have been ONNX-supported since early opsets. The two special cases (GroupNorm, reflect padding) require opset ≥ 18, which is well-supported in current ONNX Runtime 1.18+.

The recommended strategy is:
1. Export each model as a separate `.onnx` file via `torch.onnx.export(opset_version=18, dynamic_axes={"input": {0: "batch"}})`
2. Chain models in the calling application (C# viewer, UE plugin, web service) via ONNX Runtime sessions
3. Automate re-export as part of the training checkpoint pipeline
4. Add ONNX Runtime NuGet package to `WowViewer.App` for direct viewer integration

**Estimated effort for a full ONNX enablement pass**: ~2-4 days (one-time export scripts + validation) per model family, ~1-2 weeks total for all models.

---

## Non-Goals

- Writing an ONNX export implementation or training pipeline changes
- Benchmarking ONNX Runtime performance across backends
- Building the C#/UE inference integration (these are downstream consumers)
- Converting the deterministic compositor (D2 subtraction) — this is pure arithmetic, not a model

---

## User Stories

### US-001 (P1) — Export V14 D1 to ONNX
As a pipeline operator, I can export a trained D1 checkpoint to a `.onnx` file with opset 18, and verify that ONNX Runtime produces numerically identical outputs to PyTorch (within 1e-5 tolerance).

### US-002 (P1) — Export V14 R1 to ONNX
As a pipeline operator, I can export a trained R1 checkpoint to `.onnx`, including its multi-scale outputs (height 257×257, holes 16×16, liquid 256×256), and verify correctness.

### US-003 (P1) — Export V19 height regressor to ONNX
As a pipeline operator, I can export a trained V19 checkpoint (with reflect padding + GroupNorm) to `.onnx` and verify.

### US-004 (P2) — Export V20 MSS to ONNX
As a pipeline operator, I can export a trained V20 semantic segmentor to `.onnx`.

### US-005 (P2) — Export V20 TII to ONNX
As a pipeline operator, I can export a trained V20 terrain inpainter to `.onnx`.

### US-006 (P2) — Export V20 TFC and OPR to ONNX
As a pipeline operator, I can export the V20 classifier and placement restorer models to `.onnx`.

### US-007 (P2) — ONNX chain validation
As a pipeline operator, I can run V14's D1 → compositor → R1 chain entirely through ONNX Runtime sessions and verify that the full-chain output matches the PyTorch equivalent.

### US-008 (P3) — Automated re-export
As a developer, I can trigger ONNX re-export automatically from a training checkpoint's final save event, producing `<model>_<epoch>.onnx` alongside `<model>_final.pt`.

---

## Success Criteria

- **SC-001**: All models (D1, R1, V19, V20-MSS, V20-TII, V20-TFC, V20-OPR) export to ONNX without errors
- **SC-002**: ONNX Runtime outputs match PyTorch outputs within 1e-5 absolute tolerance for all exported models
- **SC-003**: Multi-output models produce correct number and shape of outputs in ONNX Runtime
- **SC-004**: Full D1 → D2 → R1 chain in ONNX Runtime matches PyTorch chain output
- **SC-005**: ONNX models support dynamic batch sizes (1, 2, 4, 8)

---

## Assumptions

- PyTorch ONNX export (`torch.onnx.export` / `torch.onnx.dynamo_export`) remains stable in PyTorch ≥ 2.5
- ONNX Runtime NuGet packages (`Microsoft.ML.OnnxRuntime`) remain available and maintained
- The models' forward passes do not acquire any training-only ops (AMP, dropout) that would break export — current inspection confirms this
- ONNX opset 18 is sufficient for all model operations (confirmed above)

---

## Key Terms

- **ONNX**: Open Neural Network eXchange — open standard format for ML models
- **ONNX Runtime**: Cross-platform inference engine for ONNX models
- **ORT**: Shorthand for ONNX Runtime
- **opset**: ONNX operator set version — defines which ops are available
- **dynamic_axes**: ONNX export parameter allowing variable batch size
