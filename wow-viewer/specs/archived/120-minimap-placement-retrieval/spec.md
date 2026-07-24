# Feature Specification: Minimap Object Detection, Scale/Position Extraction & Metadata Sidecar Output

**Feature Branch**: `120-minimap-placement-retrieval`

**Created**: 2026-07-23 | **Refactored**: 2026-07-24 (Architectural Pivot to OBB Hybrid Detector)

**Status**: Active / Refactored

**Input**: User request: "let's pursue that [Oriented Bounding Box detector + retrieval sidecar model]. use speckit before doing any implementation work, and update our memory bank files".

The Spec 119 classifier/segmenter proved that clean 128px studio renders can be segmented, but the minimap-retrieval PoC proved that studio-render embeddings do NOT transfer to real minimap crops ($p50=10\text{px}$, $\max=29\text{px}$ at $2\text{ yd/px}$) due to the large domain and resolution gap. Furthermore, pure contrastive crop retrieval cannot discover object locations or extract bounding boxes, continuous positions $(x, y)$, scale $(w, h)$, or orientation ($\theta$) on raw minimap tiles where placements are unknown.

This feature implements a **Two-Stage Hybrid Detector & Metadata Sidecar Generator**:
1. **Stage 1 (Localization, Scale, Orientation & Coarse Class)**: Trains a single-pass **Oriented Bounding Box (OBB)** detector (YOLOv11-OBB / RT-DETR-v2) on `minimap_rgb_authored` (256×256 tile imagery), supervised by object bounds derived from `placements.parquet`.
2. **Stage 2 (Identity Retrieval & Sidecar Generation)**: Scale-matched identity retrieval over extracted OBB crops matched against the Object Library (`embeddings.parquet`), emitting structured sidecar metadata (`.json`/`.parquet`).
3. **VLM Integration**: Uses Unsloth + Gemma 4 / Qwen2.5-VL on $64\times 64$ detected crops for fine annotation and auditing without incurring token generation latency bottlenecks on full tiles.

---

## User Scenarios & Testing

### User Story 1 - OBB Object Detector for Minimap Tiles (Priority: P1)

Train an Oriented Bounding Box (OBB) detector on `minimap_rgb_authored` tiles. The model accepts a minimap tile (256×256) and predicts oriented bounding boxes: center $(x, y)$, width $w$, height $h$ (scale), rotation angle $\theta$, confidence $c$, and coarse class (`wmo` vs `mdx/m2`). Labels are converted from `placements.parquet` world positions and asset bounding boxes.

**Why this priority**: Object detection and continuous position/scale extraction must work before any sidecar file can be populated.

**Independent Test**: Convert `placements.parquet` into an OBB dataset. Split tiles by spatial regions or map blocks. Train the OBB detector; report held-out mAP@50, mAP@50-95 (for rotated boxes), positional MAE (px), and scale MAE (px).

**Acceptance Scenarios**:
1. **Given** a held-out minimap tile containing WMO buildings, **When** the OBB detector runs, **Then** it predicts oriented bounding boxes covering the buildings with positional error $<2.0\text{px}$ and scale error $<10\%$.
2. **Given** a minimap tile with tiny M2 doodads ($5\text{–}15\text{px}$), **When** detected, **Then** the detector identifies them as oriented boxes rather than missing them or merging adjacent doodads.
3. **Given** a terrain-only minimap tile (no objects), **When** detected, **Then** it outputs zero bounding boxes (no false positives above confidence threshold 0.25).

---

### User Story 2 - Identity Retrieval & Metadata Sidecar Exporter (Priority: P2)

Given detected OBB bounding boxes on a minimap tile, extract scale-normalized crops, embed them via a feature extractor, perform nearest-neighbor matching against `embeddings.parquet` (from Spec 119), and emit a sidecar metadata file (`.json` or `.parquet`).

**Why this priority**: Delivers the user-requested output format: position data, scale data, and asset identity sidecar file alongside the minimap tile.

**Independent Test**: Run the detection + sidecar exporter pipeline on held-out tiles; verify the output sidecar file contains valid continuous coordinates, scale, rotation, class, and retrieved asset path.

**Acceptance Scenarios**:
1. **Given** a minimap tile, **When** the sidecar exporter runs, **Then** it writes a valid JSON/Parquet sidecar containing `{instance_id, position_px, world_position, scale_px, scale_factor, rotation_deg, coarse_class, retrieved_asset, confidence}`.
2. **Given** a loose minimap PNG with no ground truth store present, **When** the inference CLI runs, **Then** it outputs the sidecar file cleanly.

---

### User Story 3 - Crop-Level VLM Annotation & Hard-Negative Audit (Priority: P3)

Use Unsloth + Gemma 4 / Qwen2.5-VL to fine-tune a LoRA model on $64\times 64$ cropped OBB patches to generate rich natural language descriptions and audit low-confidence or candidate misclassified detections.

**Why this priority**: Leverages VLM capabilities on cropped patches where spatial token resolution is preserved, enriching the metadata sidecar without slowing down full-tile detection.

---

## Requirements

### Functional Requirements

- **FR-001**: The detector MUST accept a single minimap tile (`minimap_rgb_authored`, 256×256) and output Oriented Bounding Boxes (OBB): center $(x, y)$, width $w$, height $h$, rotation angle $\theta$, confidence, and coarse class.
- **FR-002**: Label preparation MUST convert `placements.parquet` world positions, asset paths, and asset dimensions into normalized tile OBB coordinates `[class_id, x_center, y_center, width, height, angle_deg]`.
- **FR-003**: The dataset split MUST isolate spatially by map chunk / region to prevent spatial leakage across train and held-out sets.
- **FR-004**: The detection inference CLI MUST accept a loose minimap PNG or tile Zarr array and produce a structured metadata sidecar file (`.json` or `.parquet`).
- **FR-005**: All training, dataset conversion, and evaluation scripts MUST be user-run (Rule 0); dry-run-first CLIs MUST refuse to train without `--confirm-run`.
- **FR-006**: Code MUST reside entirely within `wow-viewer/data-harvester/` (Rule 2/4/5) and run within the `uv` environment.

---

## Success Criteria

- **SC-001**: OBB detector achieves held-out mAP@50 $\ge 0.65$ and positional center MAE $< 2.0\text{px}$ on `minimap_rgb_authored` tiles.
- **SC-002**: Scale extraction error is within $\le 15\%$ of ground truth placement bounding box dimensions.
- **SC-003**: Sidecar exporter outputs schema-valid JSON and Parquet metadata files for any input tile.
- **SC-004**: All CLIs run dry-runs cleanly without mutating source map stores.
