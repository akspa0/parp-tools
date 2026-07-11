# Implementation Plan: V25 SegFormer Decompiler and Terrain-Texture Convergence Network (Spec 102)

**Referenced Specification**: [`spec.md`](file:///i:/parp/parp-tools/wow-viewer/specs/102-v25-terrain-convergence/spec.md)

**Status**: Draft

---

## Summary

The V25 model (`V25SegformerDecompiler`) is a completely universal neural network that processes a single raw RGB minimap tile to predict:
1. **Low-res WDL height prior** ($33\times33$ mesh) via downsampling the predicted high-res mesh.
2. **High-res terrain heightmap** ($257\times257$ edge-aligned mesh) via the progressive deep unfolded Sylvester solver.
3. **Object segment footprints** and 3D placements/rotations.
4. **Terrain texturing layers** (MTEX selections, MCLY layer indices, and MCAL alpha blend maps).

Decoupled components:
- **Differentiable Fractal Noise Generator**: Predicts parameters (translation seeds, frequency, scale) to generate MCAL alpha maps, ensuring 100% out-of-distribution generalization.
- **PM4-Guided Post-Processing Handler (`V25Pm4GuideHandler`)**: A modular post-processing utility. It is **completely separate** from the main neural network architecture. When `--pm4` is passed at inference, the handler snaps predicted object coordinates to PM4 segment centroids and matches WMO/M2 names via the `pm4_asset_matching` library.
- **8 GB VRAM Constraint**: Optimized training pipeline (gradient checkpointing, 8-bit AdamW, and Zarr slice preloading) ensuring compatibilty with consumer GPUs.
- **Zarr Output Datastore**: All inputs and outputs must pass through Zarr. The inference CLI writes the predicted terrain geometry, objects, and textures directly into a structured Zarr group store with lightweight Blosc LZ4 level 1 compression (no random array files on disk). This aligns with the repository's standard data structure, allowing downstream tooling to easily slice, query, and process the model outputs.

---

## Technical Context

- **Platform**: Python 3.11 / PyTorch 2.x
- **Segmentation Frontend**: Pretrained `SegformerForSemanticSegmentation` (`nvidia/mit-b0` or `nvidia/mit-b1`).
- **PM4 Handler**: Completely separate post-processing class (`V25Pm4GuideHandler`) leveraging `harvester.pm4_asset_matching` modules.
- **Fractal Generator**: Implements multi-octave Simplex or Perlin noise on PyTorch GPU.
- **VRAM Optimizations**: Peak memory target $< 7.0$ GB on GPU. Sequential preloading maps contiguously.

---

## Project Structure

We will add the new modular files under `data-harvester/`:

```
wow-viewer/
├── data-harvester/
│   ├── src/
│   │   └── harvester/
│   │       └── v25/
│   │           ├── __init__.py           # Library entrypoint
│   │           ├── segformer.py          # SegFormer wrapper, inpainting, and object placement heads
│   │           ├── pm4_guide.py          # Decoupled PM4 handler alignment and matching modules
│   │           ├── fractal.py            # DifferentiableFractalGenerator & FractalParameterHead
│   │           ├── texture.py            # MtexPredictor & MclyDecoder
│   │           ├── solver.py             # Sylvester solver & tridiagonal math
│   │           ├── prior.py              # WdlDownsampler
│   │           ├── lapnet.py             # Progressive height solver and model routing
│   │           └── losses.py             # Multi-task loss functions
│   ├── scripts/
│   │   ├── train_v25_decompiler.py       # Unified training script
│   │   └── infer_v25_decompiler.py       # Single-image deployment inference script
│   └── tests/
│       └── v25/
│           ├── test_segformer.py         # SegFormer frontend and head tests
│           ├── test_pm4_guide.py         # PM4 handler tests (verifies decoupled logic)
│           ├── test_fractal.py           # Fractal generator & parameter tests
│           ├── test_solver.py            # Mathematical solver tests
│           ├── test_lapnet.py            # Progressive upsampling tests
│           └── test_losses.py            # Multi-task loss tests
```

---

## Implementation Phases

### Phase 1 — SegFormer Frontend and Decompiler Decoders

- **Goal**: Implement SegFormer feature extraction and decoders.
- **Approach**:
  - Load `nvidia/mit-b0` using Hugging Face's `transformers` package.
  - Implement `TerrainInpaintHead` and `ObjectPlacementHead`.

---

### Phase 2 — Height Predictor & Sylvester Math

- **Goal**: Implement $257\times257$ progressive height solver and Sylvester solver.
- **Approach**:
  - `V25StageBPredictor` scales heights progressively ($33 \rightarrow 65 \rightarrow 129 \rightarrow 257$) using the Sylvester solver.
  - `BatchedSylvesterSolver` solves tridiagonal row/column Laplacians on the GPU.
  - `WdlDownsampler`: An average-pooling layer mapping $(257, 257) \rightarrow (33, 33)$ quincunx lattice.

---

### Phase 3 — Decoupled PM4 Post-Processing

- **Goal**: Support PM4 alignment as an optional post-processing step.
- **Approach**:
  - `V25Pm4GuideHandler` operates only at inference. It is never invoked during network training or loss evaluations.
  - Snaps predicted object translations to nearby PM4 segment centroids.
  - Resolves asset names by running `harvester.pm4_asset_matching.scorer` on the PM4 bounds.

---

### Phase 4 — Differentiable Fractal Generator and Parameter Head

- **Goal**: Implement fractal brush simulation.
- **Approach**:
  - `DifferentiableFractalGenerator`: Implements vectorized Perlin/Simplex noise in PyTorch.
  - `FractalParameterHead`: Estimates translation seeds $(S_x, S_y)$, frequencies $f$, amplitude $A$, persistence $p$, and coarse boundary mask $M$ ($256\times256$).
  - Evaluates the final alpha map: $\text{Alpha} = M \cdot \text{Noise}(x \cdot f + S_x, y \cdot f + S_y)$.

---

### Phase 5 — Texture Decoders & Losses

- **Goal**: Implement texture layer decoders and multi-task losses.
- **Approach**:
  - `MtexPredictor` and `MclyDecoder`.
  - `V25UnifiedLoss`: Computes combined losses (SegFormer CE, height L1, progressive height L1/SiLog, fractal parameter MSE, MCLY CE, MTEX CE, object placements).

---

### Phase 6 — Training and Zarr Dataset Integration

- **Goal**: Build trainer, validation, and CLI inference.
- **Approach**:
  - `train_v25_decompiler.py` trains the decompiler.
  - Integrates `--gradient-checkpointing`, `--8bit-optimizer`, and `TileSource.preload()` from our training codebase.
  - `infer_v25_decompiler.py` runs inference. If `--pm4` is passed, runs `V25Pm4GuideHandler` on the predictions, and outputs predicted heights ($257\times257$ & $33\times33$), objects, and textures directly to a structured Zarr group store with Blosc LZ4 level 1 compression.
