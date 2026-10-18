# Technical Design Plan: Minimap Hillshade Correlation & Domain Translation

**Spec ID**: 274  
**Status**: In Progress  
**Owning Epics**: Epic 254 (Datasets, Client Datastore & Terrain ML) / Epic 250 (Map Reconstruction & Editor Platform)  

---

## 1. Technical Architecture & End-to-End Flow

```mermaid
graph TD
    subgraph Data Stores
        Z1["Zarr v3 Maps (3_0_1_8303-*.zarr)<br/>Azeroth, Kalimdor, Northrend (4,400+ tiles)"]
    end

    subgraph Batch Export Pipeline (parp-tools)
        E1["batch_export_clean_pairs.py<br/>Headless Scanner & DXT1 Codec Bridge"]
        P1["Clean Paired Store (clean_pairs.zarr / index.parquet)<br/>1. real_minimap_rgb (256x256x3)<br/>2. synthetic_hillshade_dxt1 (256x256x3)<br/>3. wdl_lattice (33x33)<br/>4. adt_heightmap (257x257)"]
    end

    subgraph Stage 1: Domain Translation
        M1["Domain Translation Network (UNet < 10M params)<br/>Learns Object Stripping & Texture Normalization"]
        L1["Losses: L1 + Sobel Gradient + SSIM"]
        O1["Predicted Clean Hillshade"]
    end

    subgraph Stage 2: Terrain Implication Correlation
        C1["Terrain Implication Model<br/>Dual-Head Macro WDL + Micro ADT"]
        O2["Reconstructed 3D Elevation (Yards)<br/>WDL 33x33 Anchor + ADT 257x257 Mesh"]
    end

    Z1 --> E1
    E1 --> P1
    P1 -- "Train Inputs" --> M1
    M1 --> L1
    P1 -- "Train Targets" --> L1
    M1 --> O1
    O1 --> C1
    P1 -- "Elevation Targets" --> C1
    C1 --> O2
```

---

## 2. Component Specifications

### 2.1 Batch-Export Clean Pairs (`batch_export_clean_pairs.py`)
- **Input Stores**: Scans all 37 Zarr v3 map stores located at `output/datasets/v60/3_0_1_8303-*.zarr`.
- **Signal Extraction & Validation**:
  - `minimap_rgb`: shape `(256, 256, 3)` uint8. Must have visible contrast (not pure black/clear water).
  - `terrain_shadow_256`: shape `(256, 256)` float32 in $[0, 1]$.
  - `dxt1_parity_cycle`: Grayscale hillshade is expanded to RGB `(256, 256, 3)` and quantized via 4x4 block DXT1 / BC1 block fitting (`Dxt1TileCodec` or fast Python block quantizer). This ensures the target exhibits the exact block-quantization noise floor as the real minimap input, eliminating the codec confound.
  - `wdl_lattice`: Combined $33 \times 33$ lattice from `wdl_outer_17` ($17 \times 17$) and `wdl_inner_16` ($16 \times 16$), matching the authentic Blizzard WDL format.
  - `adt_heightmap`: $257 \times 257$ float32 clean elevation from `ground_intent_height_257` (preferring clean object-stripped ground) with fallback to `height_257`.
- **Output Dataset Layout**:
  - Destination: `output/datasets/v60_clean_pairs.zarr/` (or sharded storage) with `index.parquet`.
  - Partitioning: Strict map-based holdout split (e.g. `Kalimdor` / `Azeroth` / `Northrend` partitioned into 80% train, 20% validation).

### 2.2 Domain Translation Network (`DomainTranslationUNet`)
- **Model Topology**:
  - Input: $3 \times 256 \times 256$ normalized RGB float32 $\in [0, 1]$.
  - Encoder: 4 stages with strided depthwise-separable residual convolutions:
    - Layer 1: $3 \to 32$ filters, kernel 3x3, stride 1 ($256 \times 256$)
    - Layer 2: $32 \to 64$ filters, kernel 3x3, stride 2 ($128 \times 128$)
    - Layer 3: $64 \to 128$ filters, kernel 3x3, stride 2 ($64 \times 64$)
    - Layer 4: $128 \to 256$ filters, kernel 3x3, stride 2 ($32 \times 32$)
  - Bottleneck: 2 residual blocks at 256 channels with dilation ($32 \times 32$).
  - Decoder: 4 stages with bilinear upsampling, concatenation of skip connections, and depthwise-separable convs:
    - Up 1: $256 + 128 \to 128$ filters ($64 \times 64$)
    - Up 2: $128 + 64 \to 64$ filters ($128 \times 128$)
    - Up 3: $64 + 32 \to 32$ filters ($256 \times 256$)
    - Final Head: $32 \to 3$ channels ($256 \times 256$) with Sigmoid activation.
  - Total Parameter Count: $\approx 2.45\text{M}$ parameters (lightweight, easily fits in $< 10\text{ MB}$ checkpoint).
- **Objective Function**:
  $$\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{L_1} + 0.5 \cdot \mathcal{L}_{\text{grad}} + 0.3 \cdot \mathcal{L}_{\text{ssim}}$$
  - $\mathcal{L}_{L_1}$: Pixel-wise absolute difference against synthetic DXT1 hillshade.
  - $\mathcal{L}_{\text{grad}}$: Horizontal and vertical Sobel gradient difference to ensure ridge crests and cliff drop-offs are crisp.
  - $\mathcal{L}_{\text{ssim}}$: Structural similarity index matching illumination gradients.

### 2.3 Terrain Implication Correlation Pipeline
- Ingests predicted clean hillshade $Y \in [0, 1]^{256 \times 256}$.
- Predicts low-resolution WDL lattice ($33 \times 33$) and high-resolution ADT elevation mesh ($257 \times 257$).
- Because the input is stripped of texture noise and object clutter, the correlation between Lambert shading and 3D terrain slopes is direct, achieving high Pearson $r \ge 0.85$ and low metric error $\le 8.0$ yards.

---

## 3. Implementation Phases

- **Phase 1**: Headless Batch Export Pipeline
  - Implement `batch_export_clean_pairs.py` and CLI wrapper `v60_batch_export_clean_pairs.py`.
  - Implement fast DXT1 block quantization parity codec in Python (`dxt1_parity.py`) aligning with `Dxt1TileCodec.cs`.
  - Add test suite `tests/v60/test_batch_export_clean_pairs.py`.
- **Phase 2**: PyTorch Clean Pair Dataset & DataLoader
  - Implement `CleanPairDataset` in `data-harvester/src/harvester/v60/clean_pair_dataset.py`.
  - Add data augmentations (random horizontal/vertical flips, subtle exposure variation).
  - Add test suite `tests/v60/test_clean_pair_dataset.py`.
- **Phase 3**: Domain Translation Model & Training Script
  - Implement `DomainTranslationUNet` and multi-scale composite losses in `data-harvester/src/harvester/v60/domain_translation_model.py`.
  - Implement training harness `v60_train_domain_translation.py` with dry-run support, checkpointing, and validation metrics (SSIM, L1, PSNR).
  - Add test suite `tests/v60/test_domain_translation_model.py`.
- **Phase 4**: Terrain Implication Correlation & Verification
  - Bridge translated hillshade to elevation inference via `ZarrElevationDataset` and `TrestleElevationUNet`.
  - Measure end-to-end metrics on held-out maps (Pearson $r$, MAE in yards, ridge coincidence).
  - Provide PowerShell 7 commands for operator-owned dataset generation and GPU training.
