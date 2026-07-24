# Research Decisions: Minimap Object Detection & Sidecar Generation (Spec 120)

**Date**: 2026-07-23 | **Refactored**: 2026-07-24 | **Spec**: [spec.md](spec.md)

## D-01: Two-Stage Hybrid Architecture (OBB Detector + Identity Retrieval Sidecar)

**Decision**: Instead of relying on pure contrastive crop retrieval given pre-existing placement points, Spec 120 is re-architected as a **Two-Stage Hybrid Detector & Metadata Sidecar Generator**:
- **Stage 1**: Single-pass **Oriented Bounding Box (OBB)** detector (YOLOv11-OBB / RT-DETR-v2) trained on `minimap_rgb_authored` with labels derived from `placements.parquet`. Continuous sub-pixel prediction for center $(x, y)$, width $w$, height $h$ (scale), orientation angle $\theta$, confidence, and coarse class (`wmo` vs `mdx/m2`).
- **Stage 2**: Identity retrieval over extracted OBB crops matched against the Object Library (`embeddings.parquet`), producing the metadata sidecar (`.json`/`.parquet`).

**Rationale**: Contrastive retrieval alone cannot discover object locations or extract bounding boxes/scales on raw minimap tiles where placements are unknown. OBB detectors run in $\sim 10\text{ ms per tile}$ with continuous sub-pixel spatial precision ($<2\text{px}$ error), whereas VLMs suffer from spatial quantization noise and token generation latency bottlenecks on full tiles.

---

## D-02: World-to-Tile Coordinate & OBB Label Conversion

**Decision**: For a placement at world position $(x, y, z)$ on tile $(t_x, t_y)$ with asset bounding box $(d_x, d_y, d_z)$ and orientation angle $\theta_{yaw}$, convert to tile pixel space (256×256):
- $f_x = ((32 - t_x) \times \text{TILE\_SIZE} - x) / \text{TILE\_SIZE}$
- $f_y = ((32 - t_y) \times \text{TILE\_SIZE} - y) / \text{TILE\_SIZE}$
- $p_x = f_x \times 256.0$, $p_y = f_y \times 256.0$
- $w_{px} = d_x / \text{YARDS\_PER\_PIXEL}$, $h_{px} = d_y / \text{YARDS\_PER\_PIXEL}$ ($\text{YARDS\_PER\_PIXEL} \approx 2.0833\text{ yd/px}$)
- Normalized OBB Target: `[class_id, cx_norm, cy_norm, w_norm, h_norm, angle_deg]`

**Rationale**: Leverages exact geometric relationship between game world space and minimap tile space to auto-generate thousands of labeled OBB training examples directly from `placements.parquet`.

---

## D-03: Metadata Sidecar Format Contract

**Decision**: The inference CLI exports metadata sidecar files in `.json` and `.parquet` formats matching the following schema per detected object:

```json
{
  "instance_id": 1042,
  "position_px": [128.45, 96.12],
  "world_position": [1845.20, -432.10, 65.40],
  "scale_px": [32.40, 28.10],
  "scale_factor": 1.05,
  "rotation_deg": 45.0,
  "coarse_class": "wmo",
  "retrieved_asset": "World/wmo/Azeroth/Buildings/Castle01.wmo",
  "confidence": 0.941
}
```

---

## D-04: Role of Unsloth & VLM (Gemma 4 / Qwen2.5-VL)

**Decision**: VLMs fine-tuned via Unsloth are scoped to **Crop-Level Annotation & Hard-Negative Auditing** (operating on $64\times 64$ cropped OBB patches) rather than full minimap tile object detection.

**Rationale**: Prevents VLM spatial quantization loss on tiny $5\text{–}25\text{px}$ objects and eliminates the $\sim 50\text{ sec/tile}$ text token generation latency bottleneck on 256×256 tiles, while still allowing natural language metadata enrichment for individual detected crops.