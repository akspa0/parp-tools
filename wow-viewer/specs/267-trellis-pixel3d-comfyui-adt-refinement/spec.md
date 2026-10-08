# Spec 267: Generative 3D Foundation Model Refinement (Trellis.2 / Pixel3D) & parp-tools ComfyUI Nodes

## 1. Overview & Problem Statement

In Spec 266, the pipeline established an accurate metric elevation reconstruction using the coarse WDL macro-trestle anchor (`TrestleElevationUNet`), achieving a validation MAE of 5.20 yards and restoring high-relief alpine topography up to 425 yards.

As a downstream enhancement, high-resolution generative 3D foundation models—specifically **Microsoft Trellis.2** (structured latent 3D representation / SLAT with sparse voxel and 3D Gaussian / mesh generation) and **Pixel3D**—offer accelerated geometric synthesis. However, raw generative 3D foundation models cannot be used out-of-the-box for WoW map reconstruction:
1. **Unconstrained Metric Space**: Generative models produce arbitrary unit-cube geometries ($[-1, 1]^3$) without real world-space metric scaling (yards), flattening mountain massifs or distorting global coordinates.
2. **ADT Feature Conflation**: Without domain-aware decomposition, generative models mistake 2D painted texture splats (MCAL), roads, and painted developer annotations for extruded 3D geometry, while hallucinating unnatural deformations on authentic terrain.
3. **Siloed Tooling & Reinvented Wheels**: The local environment already hosts a running ComfyUI service at `http://127.0.0.1:8199` containing the custom mothership nodes for VLM inference. Running ad-hoc standalone scripts for every 3D experiment reinvents asset loading, coordinate conversions, and batch visualizers.

### The Solution
Spec 267 designs a modular 3D refinement harness:
1. **"Hard Z" Metric Elevation Anchor**: Conditions generative 3D backbones with exact metric bounds ($Z_{\min}, Z_{\max}$) and the WDL trestle lattice ($17 \times 17 \rightarrow 257 \times 257$). The generative surface is normalized, scaled, and warped to respect Blizzard world-space elevation.
2. **ADT Semantic Dissection**: Decouples minimap imagery into semantic ADT layers prior to 3D inference:
   - Authentic WMO buildings & M2 doodad footprints (from `_obj0.adt`) sieved out to prevent double-geometry generation.
   - Alpha texture splats and developer painted annotations strictly clamped to terrain surfaces.
3. **`parp-tools` Custom Nodes for ComfyUI**: Native nodes installed in the ComfyUI instance (`http://127.0.0.1:8199`) to expose ADT/WDL I/O, Hard-Z calibration, and Blizzard-conforming upright mesh export directly into visual node workflows.

---

## 2. User Stories

### User Story 1 - Hard-Z Metric Conditioning (Priority: P1)
As an environment artist, I want generative 3D foundation outputs (Trellis.2 / Pixel3D) to be strictly anchored to an authentic WDL elevation lattice, so that generated high-resolution terrain matches authentic WoW world-space elevation in yards rather than floating in an arbitrary unit scale.

### User Story 2 - ADT Aspect Dissection (Priority: P1)
As a terrain reconstruction engineer, I want the ComfyUI workflow to distinguish between terrain base relief, authentic object placements (`_obj0.adt`), and painted surface annotations, so that developer handwriting and grass textures remain 2D splats rather than extruding into deformed 3D geometry.

### User Story 3 - parp-tools ComfyUI Custom Node Suite (Priority: P2)
As a pipeline developer, I want custom ComfyUI nodes for WoW data structures (`WoW_AdtLoader`, `WoW_HardZCalibrator`, `WoW_ObjectSieveConditioner`, `WoW_MeshExporter`), so that 3D elevation experiments can be assembled visually in ComfyUI alongside our existing VLM mothership nodes without rewriting Python plumbing.

---

## 3. Acceptance Criteria

| ID | Criterion | Requirement | Target Metric |
|---|---|---|---|
| **AC-001** | ComfyUI Service Discovery | Connect to local ComfyUI instance at `http://127.0.0.1:8199` and verify model availability (Trellis.2 / Pixel3D / VLM mothership nodes) | Live HTTP ping and node registry dump verified |
| **AC-002** | Hard-Z Heightfield Calibrator | Scale and warp unit-space generative elevation maps to match WDL trestle lattice bounds | Vertical span within $\pm 2.0\%$ of WDL anchor bounds ($Z_{\min}, Z_{\max}$) |
| **AC-003** | Semantic ADT Conditioning | Sieve authentic `_obj0.adt` placements and mask painted developer text | 0% of painted text classified as extruded 3D geometry; 100% of WMOs sieved |
| **AC-004** | parp-tools Node Package Structure | Standalone ComfyUI custom node package under `wow-viewer/data-harvester/comfyui_parp_nodes/` | Functional node definitions for `WoW_AdtLoader`, `WoW_HardZCalibrator`, `WoW_MeshExporter` |
| **AC-005** | Upright Watertight Mesh Output | ComfyUI mesh export produces CCW-wound OBJ/GLB meshes | 100% upward normal vectors (+Z in OBJ, +Y in GLB) matching Spec 266 convention |

---

## 4. Constraints

- Zero hardcoded machine-local client paths in source code (AGENTS.md §5). ComfyUI service endpoint configurable via environment or CLI (`http://127.0.0.1:8199`).
- PowerShell 7 syntax for all operator commands (AGENTS.md §5).
- God-class freeze: Zero member additions to `WorldScene.cs` or `ViewerApp.cs` (AGENTS.md §10).
- ComfyUI nodes must be modular, self-contained, and run within the Python `uv` environment under `wow-viewer/data-harvester/`.
