# Plan: Spec 268 Multi-Tile Quilt Canvas Terrain Reconstruction & MCAL Deciphering

## 1. Technical Architecture & System Overview

Spec 268 unifies the terrain reconstruction pipeline into a cohesive, multi-stage architecture operating across continuous multi-tile **quilt canvases**. Rather than processing tiles in isolation, the system treats maps as an uninterrupted virtual canvas.

```
                      +---------------------------------------+
                      | Raw Minimap Quilt (N x M Tiles)       |
                      |   - Stitched Virtual Canvas           |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 1: Quilt Canvas Assembler       |
                      |   - Global Coord (u_glob, v_glob)     |
                      |   - Seam Boundary Solver (C0/C1)      |
                      +-------------------+-------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
+---------------------------------------+ +---------------------------------------+
| Stage 2: Minimap Albedo De-Mixing     | | Stage 3: Bare Shadow Sieve & WDL      |
|   - Texture Palette Matching (BLPs)   | |   - Object / Doodad Albedo Stripping  |
|   - Non-Negative Matrix Factorization | |   - Photometric Sun Inversion         |
|   - Dynamic Chunk Layer Stacker       | |   - TrestleElevationUNet (Spec 266)   |
|   => MTEX / MCLY / MCAL Multi-Layers  | |   => Seamless Continent WDL Lattice   |
+-------------------+-------------------+ +-------------------+-------------------+
                    |                                       |
                    +-------------------+-------------------+
                                        |
                                        v
                      +---------------------------------------+
                      | Stage 4: Inches-Scale Refiner         |
                      |   - 36x Sub-Cell Resolution (Inches)  |
                      |   - 3D Fractal Editor Brushes         |
                      |   - v7 Prefab Pastes & Brush Scars    |
                      |   - Deterministic Yards Downsampling  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | Stage 5: Monolithic ADT / WDL / Meshes|
                      |   - Monolithic 3.3.5 ADTs (100% Chunks|
                      |   - Upright Watertight OBJ / GLB      |
                      |   - Verified Clean Load in WoWViewer  |
                      +---------------------------------------+
```

---

## 2. Component Design

### 2.1 Stage 1: Quilt Canvas Assembler (`quilt_canvas_assembler.py`)
- **Global Quilt Coordinate Mapping**:
  For an arbitrary bounding box of tiles $[(T_{x,\min}, T_{y,\min}) \dots (T_{x,\max}, T_{y,\max})]$, maps local tile pixels $(u, v) \in [0, 255]^2$ to global coordinates:
  $$U_{\text{global}} = (T_x - T_{x,\min}) \times 256 + u, \quad V_{\text{global}} = (T_y - T_{y,\min}) \times 256 + v$$
- **Seam Boundary Relaxation**:
  Boundary rows and columns between adjacent tiles ($u = 255$ on tile $T_x$ and $u = 0$ on tile $T_x + 1$) share a single continuous elevation value. The boundary solver enforces:
  $$Z_{\text{border}}(T_x, T_x + 1) = \frac{1}{2} \left( Z_{\text{pred}}(T_x, 256) + Z_{\text{pred}}(T_x + 1, 0) \right)$$
  accompanied by 1D Laplacian smoothing over a 2-pixel margin to eliminate gradient shear.

### 2.2 Stage 2: Minimap Albedo De-Mixing & MCAL Decipherer (`mcal_layer_decipherer.py`)
- **Tileset Texture Palette Extraction**:
  Loads authentic tileset texture assets (BLP) referenced in `development.wdt` or extracted via `NativeMpqService`. Computes normalized mean RGB chromaticity and texture covariance for each candidate texture $k$.
- **Albedo De-Mixing (Constrained NMF)**:
  At each canvas pixel, decomposes minimap color $\mathbf{C}(u, v)$ into a convex combination of texture albedos:
  $$\mathbf{C}(u, v) \approx S(u, v) \sum_{k=1}^{K} \alpha_k(u, v) \mathbf{T}_k, \quad \sum \alpha_k = 1, \quad \alpha_k \ge 0$$
  where $S(u, v)$ is the scalar surface shading (diffuse illumination + shadow).
- **Dynamic Chunk Layer Stacker**:
  - Each ADT chunk ($33.33 \times 33.33$ yards) contains an $8 \times 8$ or $64 \times 64$ alpha grid and supports at most 4 active layers (L0 to L3).
  - The system treats the tile as an artist's canvas: as new textures are painted across chunk borders, the stacker dynamically allocates layer indices to maximize spatial continuity across chunk boundaries, avoiding jarring layer flips.
  - Outputs authentic `MTEX` chunks, `MCLY` chunk descriptors, and 8-bit uncompressed or 4-bit packed `MCAL` alpha buffers.

### 2.3 Stage 3: Bare-Terrain Shadow Sieve & WDL Macro Trestle (`wdl_quilt_synthesizer.py`)
- **Shadow Residual Field Extraction**:
  Removes the estimated texture albedo $\sum \alpha_k \mathbf{T}_k$ and masked building/doodad footprints (OBB sieve from Spec 266), yielding the bare normalized photometric shadow signal $S(u, v)$.
- **Macro-Trestle Elevation Inference**:
  Invokes `TrestleElevationUNet` (Spec 266, 4.7M parameters) across the quilt. Because the model operates on coarse WDL lattices ($17 \times 17$ vertices per tile = 533.33 yards span), the quilt synthesizer stitches overlapping tile contexts to predict a continuous $64 \times 64$ continent WDL lattice, guaranteeing $>250\text{--}425$ yards of authentic relief.

### 2.4 Stage 4: Inches-Resolution Refiner & 3D Fractal Engine (`quilt_fractal_refiner.py`)
- **Inches Resolution**:
  As proven by DAT v22/v23/v26 project files, authentic authoring geometry was stored in inches ($36\text{ inches} = 1\text{ yard}$). The refiner upsamples the macro surface by $36\times$ (sub-cell grid) to simulate the artist's sculpting canvas.
- **v7-Era Prefab Pastes & Scars**:
  - Scans the quilt canvas for recurring 3D fractal brush stamps $(\Delta Z, \alpha_k)$ and multi-tile prefab pastes (hills, dunes, ravines, ramps).
  - Detects historical "brush scars" (regions where heightmap fractal stamps exist but alpha layers were modified during later re-texturing).
  - Fits matching pursuit brush stamps to sculpt micro-relief and sharp ridge spines.
- **Deterministic Downsampling to Client Yards**:
  The sculpted inches-scale canvas is deterministically integrated onto the client's 145-vertex MCNK lattice ($9 \times 9$ outer + $8 \times 8$ inner vertices per chunk) using area-weighted bicubic filtering, preventing high-frequency spatial aliasing.

### 2.5 Stage 5: Monolithic ADT / WDL / 3D Mesh Materializer (`quilt_adt_materializer.py`)
- Patches or constructs monolithic 3.3.5 ADTs preserving 100% of authentic chunks:
  - `MCVT`: 145 vertex heights in client yards.
  - `MCNR`: 145 surface normal vectors (packed signed bytes).
  - `MCLY` / `MCAL`: Deciphered multi-layer texture splats.
  - `MCCV`: 145 vertex shading colors (preserving authentic vertex shadows from Spec 263).
  - `MMDX`, `MWMO`, `MDDF`, `MODF`: Authentic object and building placements.
- Exports continuous multi-tile OBJ/GLB meshes with counter-clockwise face winding (+Z / +Y upward normals).

---

## 3. Directory Layout & File Roadmap

```
wow-viewer/
├── specs/
│   └── 268-quilt-canvas-terrain-and-mcal-reconstruction/
│       ├── spec.md
│       ├── plan.md
│       ├── tasks.md
│       └── evidence/
└── data-harvester/
    ├── src/harvester/v60/
    │   ├── quilt_canvas_assembler.py       (Stage 1: Quilt canvas stitching & seam relaxation)
    │   ├── mcal_layer_decipherer.py        (Stage 2: Albedo de-mixing & dynamic layer stacker)
    │   ├── wdl_quilt_synthesizer.py        (Stage 3: Bare shadow extraction & WDL lattice quilt)
    │   ├── quilt_fractal_refiner.py        (Stage 4: 36x inches-scale refiner, pastes & scars)
    │   └── quilt_adt_materializer.py       (Stage 5: Monolithic ADT patcher & multi-tile exporter)
    ├── scripts/
    │   └── v60_reconstruct_quilt.py        (CLI runner: End-to-end multi-tile quilt reconstruction)
    └── tests/v60/
        ├── test_quilt_canvas_assembler.py  (Tests for quilt stitching and seam boundary continuity)
        ├── test_mcal_layer_decipherer.py   (Tests for albedo de-mixing and layer stacking)
        ├── test_quilt_fractal_refiner.py   (Tests for inches-resolution scaling and paste fitting)
        └── test_quilt_adt_materializer.py  (Tests for monolithic ADT export and chunk integrity)
```
