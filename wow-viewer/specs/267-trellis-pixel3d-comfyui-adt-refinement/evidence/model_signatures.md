# ComfyUI Model Signatures & parp-tools Node Registry (Spec 267)

Recorded: 2026-10-08  
Service: `http://127.0.0.1:8199`  
Backend: NVIDIA GeForce RTX 4070 Ti SUPER (16 GB VRAM)  
PyTorch: 2.14.0+cu130, Python 3.12.8  

---

## 1. Discovered 3D Foundation Model Nodes

### Microsoft Trellis.2
- **`Trellis2Conditioning`**:
  - Inputs: `clip_vision_model` (CLIP_VISION), `image` (IMAGE, pad_factor=1.0)
  - Outputs: `CONDITIONING`, `CONDITIONING` (positive, negative prompt latents)
- **`Trellis2ShapeStage`**:
  - Inputs: `positive` (CONDITIONING), `negative` (CONDITIONING), `voxel` (VOXEL)
  - Outputs: `CONDITIONING`, `CONDITIONING`, `LATENT`
- **`VaeDecodeShapeTrellis`**:
  - Inputs: `samples` (LATENT), `vae` (VAE)
  - Outputs: `MESH`, `SHAPE_SUBDIVIDES`

### Pixel3D
- **`Pixal3DConditioning`**:
  - Inputs: `clip_vision_model` (CLIP_VISION DINOv3 ViT-L/16), `image` (IMAGE), `camera_angle_x` (FLOAT, fov=49.13°)
  - Outputs: `CONDITIONING`, `CONDITIONING`
- **`Pixal3DMultiViewConditioning`**:
  - Inputs: Multi-view conditioning tensor arrays

---

## 2. parp-tools ComfyUI Custom Node Suite (`comfyui_parp_nodes`)

Located in: `wow-viewer/data-harvester/comfyui_parp_nodes/`

| Node Class | Category | Inputs | Outputs | Purpose |
|---|---|---|---|---|
| `WoW_AdtLoader` | `parp-tools/WoW` | `minimap_path` (STRING), `adt_path` (STRING), `wdl_path` (STRING) | `image` (IMAGE), `trestle_elevation` (MASK), `object_mask` (MASK), `tile_coords` (STRING) | Ingests authentic Blizzard minimap, WDL lattice, and `_obj0.adt` object footprints |
| `WoW_ObjectSieveConditioner` | `parp-tools/WoW` | `image` (IMAGE), `object_mask` (MASK), `inpaint_iterations` (INT) | `conditioned_image` (IMAGE), `clean_mask` (MASK) | Performs multi-scale Laplacian inpainting of building roofs while preserving painted terrain annotations and roads |
| `WoW_HardZCalibrator` | `parp-tools/WoW` | `generative_depth` (IMAGE), `trestle_elevation` (MASK), `target_relief_yards` (FLOAT), `ridge_boost_yards` (FLOAT), `water_mask` (MASK) | `calibrated_height` (MASK), `z_min` (FLOAT), `z_max` (FLOAT), `relief_span` (FLOAT) | Warps and anchors unconstrained unit-space generative depth/heightfields to authentic Blizzard world-space elevation in yards |
| `WoW_MeshExporter` | `parp-tools/WoW` | `height_map` (MASK), `texture_image` (IMAGE), `output_dir` (STRING), `file_stem` (STRING), `export_obj` (BOOL), `export_glb` (BOOL) | `obj_path` (STRING), `glb_path` (STRING) | Exports watertight North-up Cartesian OBJ/GLB meshes with 100% upward normal vectors |

---

## 3. Verification Receipt
- Unit tests: `tests/v60/test_comfyui_parp_nodes.py` (5/5 PASS, 2.60s)
- Node mappings verified against ComfyUI extension contracts (`NODE_CLASS_MAPPINGS`).
