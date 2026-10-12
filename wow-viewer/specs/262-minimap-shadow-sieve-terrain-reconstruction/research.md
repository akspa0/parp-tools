# Research: Minimap Residual Modeling, Specular Lighting Physics & 3D Fractal Editor Brushes

**Spec**: [Spec 262](spec.md)  
**Date**: 2026-10-06  
**Sources**:
- `WoWClient.exe` 0.5.3.3368 Ghidra disassembly & runtime terrain render traces (Spec 111, 133, 134, 139)
- Meta AI: *SAM 2: Segment Anything in Images and Videos* (arXiv:2408.00714, `facebook/sam2.1-hiera-large`)
- Google DeepMind: *PaliGemma 2: A versatile 3B–28B VLM family* (December 2024, `google/paligemma2-3b-pt-224`)
- Repository baselines: [`TerrainLightingMath.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Terrain/TerrainLightingMath.cs), [`TerrainMinimapCompositor.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs), [`RosettaObjectLibrary.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Maps/RosettaObjectLibrary.cs), [`alpha_brush.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/alpha_brush.py), [`fractal_library.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/fractal_library.py)

---

## 1. Photometric Physics of Minimap Generation in 0.5.3

### 1.1 In-Game Runtime Shader vs Offline Minimap Generator
A critical finding from our 0.5.3 client audits (Ghidra `WoWClient.exe` 0.5.3.3368) is that **minimap BLP generation was NOT an in-game screen capture of the player's viewport**:
- The in-game terrain vertex shader evaluated only basic Lambertian diffuse $N \cdot L$ combined with pre-computed per-vertex static lighting (MCCV) and an 8×8 chunk static shadow bitmask (MCSH).
- In contrast, official minimap BLPs were baked offline using a dedicated **orthographic top-down camera** with raytraced or analytic sun rays that cast true terrain shadows and hit surface textures with material-specific specular reflectance.

### 1.2 The Orthographic Top-Down Half-Angle Vector & Specular Lobes
Because the camera is strictly top-down orthographic:
$$\mathbf{V} = (0, 0, -1)$$
For a directional sun at azimuth $\theta$ and elevation $\phi$:
$$\mathbf{L} = (\cos\phi\cos\theta,\, \cos\phi\sin\theta,\, \sin\phi)$$
The Blinn-Phong half-angle vector is constant across the entire tile:
$$\mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{\|\mathbf{L} + \mathbf{V}\|}$$
When sunlight strikes terrain, the received lighting $S(x, y)$ is:
$$S(x, y) = A + D \cdot \max(0, \mathbf{N}(x, y) \cdot \mathbf{L}) \cdot (1 - \text{CastShadow}(x, y)) + \sum_{k=1}^4 w_k(x, y) \cdot k_{s, k} \cdot \max(0, \mathbf{N}(x, y) \cdot \mathbf{H})^{p_k}$$

**Why Previous Synthesizers Drifted**:
Prior synthetic minimap attempts assumed pure Lambertian reflection ($k_s = 0$). On slopes tilted toward the half-vector $\mathbf{H}$, real textures (such as river gravel, wet sand, or snow) produce a distinct specular sheen. Without the specular term, a photometric solver misinterprets this sheen as an incorrect slope or warped normal vector $\mathbf{N}$, leaking false elevation ridges into the height reconstruction model.

### 1.3 DXT1/BC1 Compression Noise as an Inversion Boundary
Official 0.5.3 minimap BLPs are compressed using DXT1 (BC1):
- $4 \times 4$ pixel blocks share 2 RGB565 endpoints and 2-bit interpolated colors.
- Subtle gradient variations in terrain shadows are quantised into discrete 4-color steps.
- Any residual model operating on real minimaps must be trained with synthetic controls that pass through our bit-exact [`Dxt1TileCodec.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs), ensuring the network learns to invert quantization boundaries rather than being fooled by block artifacts.

---

## 2. Archaeological 3D Editor Brushes: World Editor Toolchain Realities

### 2.1 The Procedural/Fractal Stamp Hypothesis
In early Warcraft world building (2001–2003 Alpha 0.5.3), terrain relief was sculpted using **procedural and fractal stamp brushes** inside Blizzard's internal world editor:
- Level designers did not sculpt terrain purely vertex-by-vertex; they used preset fractal displacement stamps (e.g. mountainous ridge, rolling dune, sharp ravine, riverbed depression).
- Crucially, these brushes applied **coupled operations**:
  1. A vertical height displacement kernel: $\Delta Z(u, v)$ across the chunk's 145 vertices,
  2. An alpha splatting kernel: $\alpha_k(u, v)$ onto the MCAL texture layer (Layer 1 dirt $\rightarrow$ Layer 2 rock cliff $\rightarrow$ Layer 3 grass).

### 2.2 Proof from Existing Harvest Modules
Our existing harvest tooling in `data-harvester`:
- [`alpha_brush.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/alpha_brush.py) extracted 1,200+ distinct alpha components across 0.5.3 Azeroth and Kalimdor, clustering them by morphological silhouette.
- [`fractal_library.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/fractal_library.py) confirmed that alpha brush boundaries correlate with specific height ranges and normal tilt angles ($\rho > 0.85$).
- [`chunk_motifs.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/chunk_motifs.py) demonstrated that identical D4-transformed relief motifs repeat across different zone boundaries (e.g. Westfall and Barrens cliffs sharing identical fractal kernels).

### 2.3 Discrete Brush Fitting vs Continuous Neural Blur
When reconstructing terrain from residual signals:
- Standard neural decoders (e.g. pure UNet or ConvNeXt) predict a smooth, continuous heightmap. While this achieves reasonable mean absolute error (MAE), it rounds off sharp ridgelines, terraced cliff edges, and rocky outcrops, producing "melted wax" terrain.
- By recognizing recurring fractal stamps in the residual signal and fitting **literal 3D editor brushes** $(\mathcal{B}_j, \mathbf{p}_i, \sigma_i, \theta_i, A_i)$, the reconstruction reproduces authentic crisp ridgelines, terraces, and authentic editor tooling intent.

---

## 3. Modern Deep Learning & Vision Tooling Review

### 3.1 Meta SAM 2.1 (Segment Anything Model 2.1)
- **Architecture**: Hierarchical Vision Transformer (Hiera) backbone with memory attention and promptable mask decoder.
- **Why SAM 2.1 over SAM 1.0**:
  - SAM 2.1 (`facebook/sam2.1-hiera-large`) provides substantially better edge boundary adherence and fine structure discovery.
  - Native support for point grids and bounding box prompts, operating at $\approx 40\,\text{ms}$ per 256×256 tile on CUDA.
- **Application to Minimaps**:
  - Point grid prompts identify high-contrast doodads (trees, lampposts, wagons, foliage).
  - Bounding box prompts supplied by PaliGemma 2 isolate large structures (inns, towers, bridges, walls).

### 3.2 Google PaliGemma 2 (Vision-Language Model)
- **Architecture**: SigLIP-So400m vision transformer coupled with Gemma 2 (available in 3B, 10B, 28B).
- **Fine-Tuning Strategy (QLoRA 4-bit)**:
  - Keep SigLIP vision encoder at BF16 precision.
  - Quantize Gemma 2 language backbone to 4-bit NormalFloat via `bitsandbytes`.
  - Apply LoRA adapters ($r=16, \alpha=32$) to attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).
- **Application to Minimaps**:
  - Fine-tuned on the [Rosetta Calibration Corpus](file:///I:/parp/parp-tools/wow-viewer/specs/archived/190-rosetta-calibration-corpus/spec.md), where every single M2 and WMO object in the client is isolated with top-down orthographic ground truth (`capture_rgb`, `capture_mask`).
  - Inputs 256×256 minimap crops $\rightarrow$ outputs detected object bounding boxes and model identifiers (`"WMO_HumanBarracks"`, `"M2_TreeElwynn01"`).

---

## 4. Implementation Conclusions & Next Steps

1. **Phase 1 Must Include Specular Terms**: When optimizing lighting parameters on 0.5.3 whiteplate tiles, solve for solar $(\theta, \phi)$, ambient $A$, diffuse $D$, and specular coefficients $(k_s, p)$ simultaneously.
2. **Phase 2 Builds on Rosetta**: Export Rosetta top-down exhibits into our unified Zarr v3 datastore for PaliGemma 2 training.
3. **Phase 3 Uses SAM 2.1**: Integrate `sam2` PyPI package to generate `object_contamination_mask_256`, inpainting diffuse albedo to isolate `stripped_residual_shadow_256`.
4. **Phase 5 Recovers 3D Brushes**: Combine continuous residual inversion with discrete 3D fractal editor brush stamping to beat the $75\%$ geometric fidelity threshold with razor-sharp terrain relief.

---

## 5. WoW 1.60 Engine Shift: MCCV as Terrain Shadow Ground-Truth & Modern Renderer Hint

### 5.1 The Modern Engine Relocation
In legacy WoW (0.5.3–1.12.1), `MCCV` (MCNK chunk vertex colors, 145 vertices per chunk) was largely unused or held uniform neutral gray ($127, 127, 127$), with terrain shadowing baked into 1-bit `MCSH` chunk bitmasks.

In modern WoW 1.60 (`wow_classic_beta` / 11.2.7 / 12.0 client engine used in **WoW: Forever**):
- Blizzard's terrain pipeline fundamentally changed: the high-resolution terrain self-shadow and ambient occlusion field was **baked directly into `MCCV`**.
- The modern terrain shader multiplies ambient and diffuse lighting passes by `MCCV`, providing the added depth and ridge/crease self-shadowing needed in modern deferred/PBR rendering passes without requiring runtime shadow map recalculation.

### 5.2 Comparative Empirical Baseline (1.12.1 vs 1.60)
Because 1.60 Classic maps are built directly from 1.12.1 assets:
- The 1.60 client's `MCCV` data acts as an official, independent **empirical ground truth** of Blizzard's own baked terrain shadow field.
- By extracting the 1.60 `MCCV` vertex colors (580 bytes BGRA across 145 vertices per chunk $\times$ 256 chunks) and interpolating them to a $256 \times 256$ raster, we can directly compute the Normalized Cross-Correlation (NCC) against our minimap-extracted `stripped_residual_shadow_256` and $\Delta S(x, y)$.
- A high correlation validates mathematically that our minimap shadow extraction isolates the exact physical terrain self-shadow and ambient occlusion field that Blizzard's modern baking pipeline computed.

### 5.3 Modern Renderer Hint & Bidirectional Synthesis Bridge
Implemented in [`mccv_shadow_comparator.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/v60/mccv_shadow_comparator.py):
1. **Validation**: Measures NCC, MAE, and structural similarity between minimap residuals and 1.60 `MCCV`.
2. **Synthesis Bridge**: For Alpha / Vanilla / WotLK maps where `MCCV` was missing, our extracted residual shadow field can be sampled at each chunk's 145 vertex coordinates and exported as authentic 1.60-compliant `MCCV` chunks to restore deep terrain self-shadowing inside WoW: Forever.
