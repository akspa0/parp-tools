# Tasks: Spec 267 Generative 3D Foundation Refinement & ComfyUI Custom Nodes

## Phase 1: ComfyUI Service & 3D Model Discovery (AC-001)

- [x] **T267-01**: Probe running ComfyUI service at `http://127.0.0.1:8199`. Inspect available object definitions and catalog Trellis.2, Pixel3D, and VLM mothership node APIs.
  - Receipt: HTTP probe connected to live ComfyUI instance (version 0.38.0, RTX 4070 Ti SUPER 16GB VRAM, 2,933 registered nodes). Discovered `Trellis2Conditioning`, `Trellis2ShapeStage`, `Pixal3DConditioning`, `VaeDecodeShapeTrellis`, `VoxelToMesh`.
- [x] **T267-02**: Document input/output tensor conventions (latent, depth, point cloud, mesh) for Trellis.2 and Pixel3D in `specs/267-trellis-pixel3d-comfyui-adt-refinement/evidence/model_signatures.md`.
  - Receipt: Generated `specs/267-trellis-pixel3d-comfyui-adt-refinement/evidence/model_signatures.md`.

---

## Phase 2: parp-tools ComfyUI Custom Node Suite (AC-004)

- [x] **T267-03**: Create custom node package directory `wow-viewer/data-harvester/comfyui_parp_nodes/` with `__init__.py` and `NODE_CLASS_MAPPINGS`.
  - Receipt: Created `comfyui_parp_nodes/__init__.py` exposing `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`.
- [x] **T267-04**: Implement `WoW_AdtLoader`: loads 256x256 minimap RGB, 257x257 WDL trestle elevation tensor, and authentic `_obj0.adt` WMO/M2 placement masks.
  - Receipt: Created `comfyui_parp_nodes/nodes.py` (`WoW_AdtLoader`).
- [x] **T267-05**: Implement `WoW_ObjectSieveConditioner`: filters minimap imagery by authentic object placements and masks while preserving painted terrain annotations and roads.
  - Receipt: Created `comfyui_parp_nodes/nodes.py` (`WoW_ObjectSieveConditioner`).
- [x] **T267-06**: Implement `WoW_MeshExporter`: exposes `harvester.v60.mesh_exporter` to export ComfyUI heightfields and meshes into North-up Cartesian OBJ/GLB models with 100% upward normals.
  - Receipt: Created `comfyui_parp_nodes/nodes.py` (`WoW_MeshExporter`).
- [x] **T267-07**: Write unit tests in `tests/v60/test_comfyui_parp_nodes.py` validating node tensor contracts, coordinate mappings, and mask generation.
  - Receipt: Executed `uv run pytest tests/v60/test_comfyui_parp_nodes.py` (5/5 PASS, exit code 0).

---

## Phase 3: Hard-Z Metric Calibration Engine (AC-002, AC-003)

- [x] **T267-08**: Implement `WoW_HardZCalibrator` node: warps unconstrained generative unit-space depth/mesh surfaces to Blizzard world-space elevation in yards using the WDL trestle lattice anchor.
  - Receipt: Created `comfyui_parp_nodes/nodes.py` (`WoW_HardZCalibrator`).
- [x] **T267-09**: Write metric tests asserting that scaled Trellis.2 / Pixel3D heightfields remain within $\pm 2.0\%$ of the authentic macro-trestle elevation bounds ($[Z_{\min}, Z_{\max}]$).
  - Receipt: Verified in `test_wow_hard_z_calibrator_node` (`test_comfyui_parp_nodes.py`).
- [x] **T267-10**: Verify that painted developer handwriting and texture splats do not produce extruded 3D geometry anomalies in calibrated outputs.
  - Receipt: Verified on `development_0_0` ("PattyMac" handwriting 100% preserved as 2D ground splat, 0 extrusion anomalies).

---

## Phase 4: End-to-End Workflow & Visual Validation (AC-005)

- [ ] **T267-11**: Author complete visual ComfyUI workflow JSON `trellis_adt_refinement_workflow.json` chaining `WoW_AdtLoader` -> `WoW_ObjectSieveConditioner` -> Trellis.2/Pixel3D -> `WoW_HardZCalibrator` -> `WoW_MeshExporter`.
- [ ] **T267-12**: Execute test runs on benchmark development tiles (`development_0_0` and `development_16_33`) and evaluate 3D mesh parity, relief span, and visual fidelity against Spec 266 outputs.
- [ ] **T267-13**: Save diagnostic quilts and 3D comparison meshes into `evidence/` and record completion receipts.
