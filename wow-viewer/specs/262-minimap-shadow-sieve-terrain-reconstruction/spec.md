# Spec 262: Minimap Residual Model, Automated Lighting Calibration & 3D Fractal Editor Brush Reconstruction

**Owner**: Epic 254 (Datasets, Client Datastore & Terrain ML) & Epic 250 (Map Reconstruction, Composition & Editor Platform)  
**Origin**: Operator prompt (2026-10-06) regarding terrain reconstruction from minimaps. The entire pipeline is fundamentally a **residual image model** operating on the photometric and geometric signals left behind after albedo normalization. It leverages DXT1-compressed synthetic terrain shadow controls, real 0.5.3 whiteplate terrain shadow data, automated time-of-day/minimap lighting calibration, SAM 3.1 visual segmentation for object stripping, Rosetta overhead object library vision inference (QLoRA / Gemma4-E2B), ridge/residual signal extraction, and the extraction/re-application of **literal 3D fractal editor brushes** (recovering both texture alpha layers and spatial terrain mesh displacement used by the original world editors) to achieve $\ge 75\%$ geometric fidelity.  
**Status**: Draft / Planned  

---

## 1. Executive Summary

Reconstructing 3D terrain heightmaps from 2D minimaps is fundamentally a **residual image modeling problem**:
1. A minimap tile is a composite rendering where diffuse surface albedo (MCAL splatted texture layers) overlays terrain relief, dynamic time-of-day lighting, cast shadows, doodads (M2 models), buildings (WMO structures), roads, and water bodies, compressed via DXT1/BC1.
2. When albedo is stripped or normalized, what remains is the **residual image signal**: the pure physical shading, self-shadowing, ridge relief, and micro-topography.
3. Crucially, the early terrain of World of Warcraft (0.5.3 and early Alpha 2001–2003) was sculpted and painted using **procedural and fractal stamp brushes** inside Blizzard's internal world editor (`WoWEdit`). These brushes were not arbitrary continuous noise: they were discrete, parameterized 3D tools that simultaneously stamped **3D mesh height displacement** ($\Delta Z(u, v)$) and **texture alpha splatting** ($\alpha_k(u, v)$).
4. By discovering, cataloging, and fitting these **literal 3D editor brushes** from residual signals, the reconstruction engine avoids muddy neural blurring, achieving authentic, razor-sharp archaeological relief and exact spatial-texture relationships.

Because our engine already features:
1. High-fidelity terrain shadow synthesis with DXT1 compression parity ([`Dxt1TileCodec.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs), [`TerrainMinimapCompositor.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs), [`TerrainTileTensorPack.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Maps/TerrainTileTensorPack.cs)),
2. Real Alpha 0.5.3 data containing untextured "whiteplate" tiles with raw terrain shadow signals ([`workstream-terrain-ml.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/workstream-terrain-ml.md), [`weak-signal-tile-archaeology.md`](file:///I:/parp/parp-tools/wow-viewer/memory-bank/weak-signal-tile-archaeology.md)),
3. The **Rosetta Calibration Corpus** ([Spec 190](file:///I:/parp/parp-tools/wow-viewer/specs/archived/190-rosetta-calibration-corpus/spec.md), [`RosettaObjectLibrary.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core/Maps/RosettaObjectLibrary.cs)) containing isolated overhead captures (`capture_rgb`, `capture_mask`) of every single object in the game,
4. Foundations for clean-signal reconstruction, object sieving, and fractal libraries ([`v60/clean_signal_*`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/v60/), [`v60/object_library_sieve.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/v60/object_library_sieve.py), [`alpha_brush.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/alpha_brush.py), [`fractal_library.py`](file:///I:/parp/parp-tools/wow-viewer/data-harvester/src/harvester/fractal_library.py)),

we have $\ge 90\%$ of the prerequisites already in place.

---

## 2. User Stories & Acceptance Criteria

### User Stories

- **US1: Automated Solar & Minimap Lighting Calibration**: As an operator, I want a tool that compares synthetic DXT1-compressed terrain shadow renders against real 0.5.3 client whiteplate/minimap tiles to automatically calibrate solar direction, ambient light ratios, and shading response, eliminating hand-tuned lighting discrepancies.
- **US2: Rosetta Overhead Object Asset Library**: As a researcher, I want an exportable overhead image dictionary of all M2 doodads and WMO structures generated from the Rosetta calibration maps, providing visual ground-truth templates and bounding envelopes for overhead visual matching.
- **US3: SAM 3.1 Minimap Sieve & Mothership VLM Orchestration**: As a developer, I want to route minimap crops through the local ComfyUI instance (`http://127.0.0.1:8199`) to leverage authenticated **SAM 3.1** and **Mothership VLM** custom nodes, segmenting doodads, structures, and roads, and inpainting masked regions to expose the pure underlying terrain shadow residual signal.
- **US4: Closed-Loop Shadow Difference & Ridge Residual Synthesis**: As an ML engineer, I want to compute the pixel-level difference between real 0.5.3 terrain shadow signals and synthetic control renders, isolating high-frequency ridge and terrain-break features to iteratively refine our synthetic shadow generator.
- **US5: Archaeological 3D Fractal Brush Recovery & Cataloging**: As an engine architect, I want recurring fractal and procedural terrain patterns in the residual signal to be isolated and saved as **literal 3D editor brushes**, coupling 3D mesh displacement ($\Delta Z$) with multi-layer texture splatting ($\alpha_k$) to faithfully reproduce the original authoring tools.
- **US6: End-to-End Terrain Mesh Reconstruction ($\ge 75\%$ Parity)**: As an operator, I want to input an arbitrary minimap tile and generate a reconstructed 3D terrain mesh (`height_257` / MCVT 145 points per chunk) that achieves $\ge 75\%$ geometric accuracy against real 0.5.3 ground truth, utilizing both continuous residual inversion and 3D fractal brush fitting.

### Acceptance Criteria

- **AC-001 (Lighting Calibration Convergence)**: The automated optimizer achieves $< 5\%$ photometric MAE on unoccluded whiteplate tiles between synthetic DXT1 shadows and real 0.5.3 minimap shadow observations, converging on optimal solar azimuth $\theta$, elevation $\phi$, and ambient baseline $A$.
- **AC-002 (Rosetta Overhead Catalog Completeness)**: The Rosetta overhead asset library exports 100% of placed M2 and WMO objects with registered 2D top-down silhouettes, bounding extents, and normalized orthographic captures (`capture_rgb`, `capture_mask`).
- **AC-003 (SAM 3.1 ComfyUI Orchestration & IoU)**: The loopback orchestrator client communicates with `http://127.0.0.1:8199`, executing SAM 3.1 and Mothership VLM workflows to achieve $\ge 0.85$ IoU against control contamination masks, cleanly isolating object boundaries without eroding terrain ridge shadows.
- **AC-004 (Bare Shadow Residual Extraction)**: Stripped minimap tiles produce an isolated `stripped_residual_shadow_256` tensor where object footprint energy is attenuated by $\ge 90\%$ compared to raw minimap inputs.
- **AC-005 (Ridge/Residual Fidelity)**: Ridge and crest lines extracted from the terrain shadow difference match ground-truth terrain gradient extrema with $\ge 80\%$ precision and recall on synthetic validation pairs.
- **AC-006 (3D Fractal Brush Catalog & Spatial Coupling)**: The pipeline extracts a catalog of discrete 3D editor brushes where:
  - Each brush specifies a spatial mesh displacement kernel $\Delta Z(u, v)$ and associated texture alpha footprint $\alpha(u, v)$ over a normalized footprint (e.g. $16\text{m} \times 16\text{m}$ to $64\text{m} \times 64\text{m}$),
  - Recurring fractal motifs achieve $\ge 85\%$ cross-correlation with authentic 0.5.3 chunk alpha/height signatures.
- **AC-007 (75% Geometric Accuracy Target)**: On held-out 0.5.3 evaluation tiles (Kalimdor/Azeroth test splits), the reconstructed terrain mesh achieves:
  - Height MAE $\le 0.25 \times \text{terrain\_amplitude}$ ($\ge 75\%$ height fidelity),
  - Vertex normal cosine similarity $\ge 0.88$,
  - Ridge/slope contour F1 score $\ge 0.75$.
- **AC-008 (Architecture & Governance Conformance)**:
  - Strict compliance with `AGENTS.md`: §4 Core library architectural boundaries, §5 Python environment (`wow-viewer/data-harvester` with `uv`), §9 Governance receipts, and §10 God-Class freeze (zero new members in `WorldScene.cs` or `ViewerApp.cs`).
  - All heavy training runs and real-client data extractions remain operator-owned with reproducible commands.

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Minimap Photometric Model & Specular Lighting Physics

A minimap pixel $I(x, y)$ in client era 0.5.3 is modeled as:
$$I(x, y) = \text{DXT1}\Big( \alpha(x, y) \cdot S(x, y) + O(x, y) \Big)$$
where:
- $\alpha(x, y) = \sum_{k=1}^4 w_k(x, y) \cdot T_k(x, y)$ is the composite diffuse terrain albedo from up to 4 MCAL splat layers and base textures $T_k$,
- $O(x, y)$ is object contamination (doodads, roofs, roads, water overlays),
- $\text{DXT1}(\cdot)$ represents 4×4 block endpoint quantization and interpolation artifacts,
- $S(x, y)$ is the **shading and lighting signal** received by the terrain surface.

Crucially, real terrain surfaces are not strictly Lambertian. Sunlight reflects off terrain materials (rock, wet riverbeds, sand, snow, polished cobblestone) according to texture-dependent specular reflectance:
$$S(x, y) = A + D \cdot \max(0, \mathbf{N}(x, y) \cdot \mathbf{L}) \cdot (1 - \text{CastShadow}(x, y)) + \sum_{k=1}^4 w_k(x, y) \cdot k_{s, k} \cdot \max(0, \mathbf{N}(x, y) \cdot \mathbf{H})^{p_k}$$
where:
- $\mathbf{N}(x, y)$ is the terrain surface normal,
- $\mathbf{L} = (\cos\phi\cos\theta, \cos\phi\sin\theta, \sin\phi)$ is the sun direction vector,
- $\mathbf{V} = (0, 0, -1)$ is the orthographic top-down minimap camera vector,
- $\mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{\|\mathbf{L} + \mathbf{V}\|}$ is the Blinn-Phong half-angle vector,
- $k_{s, k} \in [0, 1]$ and $p_k \ge 1$ are texture-specific specular reflectance coefficients and roughness powers.

When $O(x, y)$ is excised by the SAM 2.1 / Rosetta sieve and diffuse albedo $\alpha(x, y)$ is normalized, what remains is the **pure residual image signal**:
$$R(x, y) \approx S(x, y) + \epsilon_{\text{DXT1}}$$

### 3.2 Unified Zarr v3 & Parquet Datastore Contract

To prevent schema rot across models, all pipeline artifacts are stored in a canonical, versioned **Zarr v3 Datastore** (`residual-datastore.zarr/`) indexed by Apache Parquet (`catalog.parquet`):
- `real_minimap_rgb_256`: 256×256 uint8 raw minimap RGB tiles from 0.5.3,
- `synth_control_shadow_256`: 256×256 float32 DXT1-quantized synthetic shadow controls,
- `object_contamination_mask_256`: 256×256 float32 SAM 2.1 / Rosetta object masks,
- `stripped_residual_shadow_256`: 256×256 float32 infilled, albedo-normalized residual signals,
- `shadow_difference_delta_256`: 256×256 float32 $\Delta S = S_{\text{real}} - S_{\text{synth}}$,
- `fractal_brushes_3d/`: Nested array group storing discrete extracted 3D brushes $(\Delta Z, \alpha_k)$ and spatial metadata.

### 3.3 Modern Open-Source Tooling Stack (HuggingFace & GitHub)
- **Zero-Shot Object Sieve**: Meta's **SAM 2.1** (`facebook/sam2.1-hiera-large`, `sam2` PyPI package) for promptable segmentation of minimap doodads, structures, and roads.
- **Overhead Asset Classifier**: Google's **PaliGemma 2** (`google/paligemma2-3b-pt-224` on HuggingFace), fine-tuned via QLoRA 4-bit (`peft` + `bitsandbytes`) on Rosetta overhead exhibits to identify and prompt candidate objects.

### 3.4 3D Fractal Editor Brush Formulation

Early terrain authoring utilized fractal stamping operations:
$$Z_{\text{mesh}}(x, y) = Z_{\text{base}}(x, y) + \sum_{i=1}^M A_i \cdot \mathcal{B}_Z\left( \mathbf{R}_{\theta_i} \left( \frac{(x, y) - \mathbf{p}_i}{\sigma_i} \right) \right)$$
$$\alpha_k(x, y) = \sum_{i=1}^M \omega_{i, k} \cdot \mathcal{B}_\alpha\left( \mathbf{R}_{\theta_i} \left( \frac{(x, y) - \mathbf{p}_i}{\sigma_i} \right) \right)$$
where:
- $\mathcal{B} = (\mathcal{B}_Z, \mathcal{B}_\alpha)$ is an authentic **3D Editor Brush**, with coupled height displacement kernel $\mathcal{B}_Z(u, v)$ and texture weight kernel $\mathcal{B}_\alpha(u, v)$,
- $\mathbf{p}_i = (x_i, y_i)$ is the world stamp origin,
- $\sigma_i$ is spatial scale / radius,
- $\mathbf{R}_{\theta_i}$ is 2D planar rotation,
- $A_i$ is vertical displacement amplitude,
- $\omega_{i, k}$ is the texture layer mixing weight.

When the residual model identifies recurring fractal motifs in $R(x, y)$, it parameterizes them into brush stamp instances $(\mathcal{B}_j, \mathbf{p}_i, \sigma_i, \theta_i, A_i)$ rather than fitting unconstrained smooth noise.

### 3.5 Reconstruction Evaluation Metrics ($\ge 75\%$ Parity)

Let $H_{\text{gt}}$ be ground-truth 0.5.3 height and $H_{\text{pred}}$ be reconstructed height over tile domain $\Omega$:
1. **Normalized Relative Height Error**:
   $$\text{RelMAE} = \frac{\frac{1}{|\Omega|} \sum_{p \in \Omega} |H_{\text{pred}}(p) - H_{\text{gt}}(p)|}{\max(H_{\text{gt}}) - \min(H_{\text{gt}}) + \epsilon} \le 0.25 \quad (\ge 75\%\text{ accuracy})$$
2. **Normal Cosine Alignment**:
   $$\text{NormalSim} = \frac{1}{|\Omega|} \sum_{p \in \Omega} (\mathbf{N}_{\text{pred}}(p) \cdot \mathbf{N}_{\text{gt}}(p)) \ge 0.88$$
3. **Ridge Contour F1 Score**:
   $$F_1(\text{Canny}(\nabla H_{\text{pred}}), \text{Canny}(\nabla H_{\text{gt}})) \ge 0.75$$
