# Technical Plan: Spec 267 Generative 3D Foundation Refinement & ComfyUI Custom Nodes

## 1. System Architecture

```
                       [ Input Minimap Image (256x256 RGB) ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
      [ Authentic ADT Placements ]                    [ ComfyUI Running Service ]
      - WMO BBoxes from _obj0.adt                     (http://127.0.0.1:8199)
      - M2 Footprints from _obj0.adt                  - VLM Mothership Nodes
      - Texture Splats from _tex0.adt                 - Trellis.2 / Pixel3D Model Loader
                  │                                               │
                  ▼                                               ▼
    [ Semantic ADT Dissection ]                       [ Generative 3D Backbone ]
    - Sieve structures from terrain                   - High-res surface mesh / latent
    - Protect developer annotations                   - Output: Unit-space geometry [-1, 1]
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                             [ WoW_HardZCalibrator Node ]
                             - Anchors to WDL Trestle Lattice (Spec 266)
                             - Enforces Blizzard metric elevation span (yards)
                             - Eliminates unit-cube scale distortions
                                          │
                                          ▼
                               [ WoW_MeshExporter Node ]
                               - North-up Cartesian (wy) alignment
                               - CCW face winding (100% upward normals)
                               - Dual OBJ & GLB export + 16-bit ADT heightfield
```

---

## 2. Key Modules & parp-tools ComfyUI Custom Nodes

### 2.1 ComfyUI Custom Nodes Directory (`data-harvester/comfyui_parp_nodes/`)
A dedicated, lightweight custom node package loadable by the running ComfyUI instance at `http://127.0.0.1:8199`:

1. **`WoW_AdtLoader`**:
   - Inputs: `map_dir`, `tile_x`, `tile_y`.
   - Outputs: `minimap_image` (IMAGE), `wdl_trestle` (IMAGE/TENSOR), `building_mask` (MASK), `alpha_mask` (MASK).
   - Logic: Direct ingestion of authentic ADT, `_obj0.adt`, and WDL binary chunks without re-implementing loaders in Python scripts.

2. **`WoW_HardZCalibrator`**:
   - Inputs: `generative_depth_or_mesh` (IMAGE/MESH), `wdl_trestle` (IMAGE), `min_z` (FLOAT), `max_z` (FLOAT).
   - Outputs: `calibrated_heightfield` (IMAGE/TENSOR), `world_scale_mesh` (MESH).
   - Logic: Affine and non-linear metric warp aligning unconstrained Trellis.2/Pixel3D depth channels to the exact Blizzard elevation bounds in yards.

3. **`WoW_ObjectSieveConditioner`**:
   - Inputs: `minimap_image` (IMAGE), `building_mask` (MASK).
   - Outputs: `sieved_image` (IMAGE), `inpaint_mask` (MASK).
   - Logic: Replaces buildings and doodads with multi-scale Laplacian background inpainting, strictly isolating terrain while preserving painted developer annotations and road splats.

4. **`WoW_MeshExporter`**:
   - Inputs: `calibrated_heightfield` (IMAGE), `output_path` (STRING), `format` (OBJ / GLB / BOTH).
   - Outputs: `mesh_file_path` (STRING).
   - Logic: Employs `harvester.v60.mesh_exporter` with North-up Cartesian coordinates and CCW winding.

---

## 3. ComfyUI Workflow Integration (`trellis_adt_refinement_workflow.json`)

- Graph wiring:
  - `WoW_AdtLoader` loads minimap, WDL trestle, and authentic object masks.
  - Minimap RGB passes through `WoW_ObjectSieveConditioner` into the Trellis.2 / Pixel3D inference node.
  - Generative 3D surface/depth output passes to `WoW_HardZCalibrator` with the WDL trestle as reference.
  - Calibrated heightfield is exported via `WoW_MeshExporter`.
- Integration with existing VLM mothership nodes:
  - Mothership VLM nodes can inspect the bare minimap vs sieved terrain for zero-shot prompt enrichment.

---

## 4. Phase Roadmap

- **Phase 1: ComfyUI & Foundation Model Inventory**:
  - Audit live ComfyUI instance at `http://127.0.0.1:8199`.
  - Enumerate Trellis.2, Pixel3D, and VLM mothership node signatures.
- **Phase 2: Custom Node Implementation**:
  - Implement `comfyui_parp_nodes/` package.
  - Implement unit tests mocking ComfyUI node tensors and verify Blizzard format parity.
- **Phase 3: Hard-Z Metric Calibration Engine**:
  - Implement `WoW_HardZCalibrator` algorithm.
  - Validate that Trellis.2 depth outputs conform to authentic WDL bounds within $\pm 2.0\%$.
- **Phase 4: End-to-End Visual Workflow & Verification**:
  - Export and run `trellis_adt_refinement_workflow.json` on sample development tiles (`development_0_0`, `development_16_33`).
  - Compare mesh fidelity and relief span against Spec 266 baseline.
