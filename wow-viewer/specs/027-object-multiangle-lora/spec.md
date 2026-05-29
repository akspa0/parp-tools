# Feature Specification: Object Multi-Angle LoRA Dataset (from Existing Roof Captures)

**Feature Branch**: `027-object-multiangle-lora`

**Created**: 2026-05-29

**Status**: Draft

**Input**: Build a multi-angle object reference image dataset from the existing roof capture pipeline (spec `025`) for fine-tuning the `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`. Use the 7 existing captures per asset (6 perspective angles + 1 orthographic roof top-down, already 2-8GB per client at 2K resolution) instead of rendering 96 new poses. The LoRA improves the base model's ability to generate plausible WoW-style object views — a fallback when visual identification models can't match an unknown object to a known asset.

## Problem Statement

The existing roof capture pipeline already produces high-fidelity 2K renders of known WMO/M2 assets at 6 perspective angles (front, back, left, right, top, three-quarter) plus an orthographic roof top-down. These captures total 2-8GB per client build and represent a uniquely valuable dataset — many of these assets are from 2003 with no proper high-resolution renders available anywhere.

The `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` already understands the multi-angle camera control task (trained on 3000+ Gaussian Splatting renders across 96 poses). Fine-tuning it on WoW object renders teaches it what WoW assets look like, while preserving the base model's multi-angle generalization.

The mapping from our 7 captures to the LoRA's 96-pose prompt system is a best-fit: each of our 6 perspective angles maps to the nearest `[azimuth] [elevation] [distance]` prompt. The model already knows the remaining 90 poses from the base LoRA; we just add WoW-specific visual knowledge.

## User Scenarios & Testing

### User Story 1 — Existing roof captures are mapped to LoRA prompt format (Priority: P1)

A dataset builder can run a reformatting pass over an existing roof capture output directory and produce LoRA-compatible training pairs, mapping each of the 6 perspective angles to the closest `<sks> [azimuth] [elevation] [distance]` prompt.

**Why this priority**: The roof captures are already on disk. The bottleneck is just reformatting and labeling. This story validates the entire data pipeline.

**Independent Test**: Run the reformatter on one object's captures and verify 7 output mappings (6 perspective + 1 top-down) with correct prompts and image pairs.

**Acceptance Scenarios**:

1. **Given** a roof capture directory with `front.jpg`, `back.jpg`, `left.jpg`, `right.jpg`, `top.jpg`, `three_quarter.jpg`, and `roof_topdown.png`, **When** the LoRA pair-builder runs, **Then** it produces 7 training (input, prompt, target) tuples per object
2. **Given** a `front.jpg` at 0° azimuth, 15° elevation, **When** mapped, **Then** the prompt is `<sks> front view elevated shot close-up`
3. **Given** a `three_quarter.jpg` at 35° azimuth, 25° elevation, **When** mapped, **Then** the prompt is `<sks> front-right quarter view elevated shot close-up`
4. **Given** a `roof_topdown.png` orthographic capture, **When** mapped, **Then** the prompt is `<sks> front view high-angle shot wide shot` (best approximation for top-down)

---

### User Story 2 — Multi-object LoRA fine-tuning produces WoW-aware checkpoint (Priority: P2)

A trainer can fine-tune the existing `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` on 100+ WoW objects using the mapped roof captures, producing a checkpoint that generates plausible WoW-style object views from novel angles given a front-view input.

**Why this priority**: Validates that our limited-angle captures (6 poses vs the LoRA's 96) are sufficient to teach WoW visual style. The base LoRA already knows how to do view control — we just add WoW-specific appearance.

**Independent Test**: Fine-tune on 100 objects, then run inference on a held-out object's front view with an angle prompt not in our 6 (e.g. `right side view low-angle shot wide shot`) and visually assess plausibility.

**Acceptance Scenarios**:

1. **Given** a LoRA checkpoint fine-tuned on 100+ WoW objects, **When** inference runs on a held-out object's front view with prompt `<sks> right side view low-angle shot wide shot` (not in our training set), **Then** the output is a plausible WoW-style object render from that angle
2. **Given** inference outputs for 8 azimuths at eye-level medium shot, **When** viewed in sequence, **Then** they form a consistent object rotation (the base LoRA's view control is preserved)
3. **Given** a non-WoW input image (e.g. a real-world photo), **When** inference runs, **Then** the output retains input structure but adopts WoW-style rendering (texture quality, color palette, edge sharpness)
4. **Given** the fine-tuned checkpoint, **When** LoRA strength is varied from 0.0 to 1.0, **Then** strength 0.0 produces the base model's real-world style and strength 1.0 produces WoW-style rendering

---

### User Story 3 — High-resolution asset renders serve dual purposes (Priority: P2)

A researcher can use the same roof capture output directory for both the existing roof-mask pipeline (spec 025, top-down orthographic for roof silhouettes) and the LoRA training pipeline (6 perspective angles + top-down for generative training) without duplicating capture work.

**Why this priority**: This validates the design choice to reuse existing captures rather than building a separate multi-angle renderer.

**Independent Test**: Run `build_v18_object_catalog_pipeline.py` once, then verify the output directory serves both `patch_v18_object_roof_masks.py` and the new LoRA pair-builder.

**Acceptance Scenarios**:

1. **Given** a single roof capture run, **When** both the roof-mask pipeline and the LoRA pair-builder read from the same directory, **Then** both produce valid outputs
2. **Given** the roof capture at 2K resolution, **When** the LoRA pair-builder runs, **Then** it produces 512×512 downsampled training images (configurable resolution) without modifying the originals

---

### User Story 4 — Generative fallback for unrecognized objects (Priority: P3)

An inference pipeline can use the fine-tuned LoRA to generate a plausible reference image for an object that the visual identification system cannot match to a known asset.

**Why this priority**: This is the ultimate motivation — generating plausible stand-in images when identification fails. It's P3 because it requires the LoRA checkpoint first.

**Independent Test**: Feed a cropped front-view image of an object region from a novel (not in training) map tile into the pipeline, and verify the output is a plausible WoW-style object from the requested angle.

**Acceptance Scenarios**:

1. **Given** a cropped input image of an unrecognized object region, **When** the generative fallback runs, **Then** it outputs a plausible WoW-style object from the requested angle
2. **Given** the same input, **When** different angle prompts are supplied, **Then** the outputs show consistent object identity across angles
3. **Given** a clearly non-object input (empty terrain region), **When** the fallback runs, **Then** output degrades gracefully rather than hallucinating a detailed object

---

### Edge Cases

- **2003-era WMO quirks**: Early Alpha WMOs may have missing textures, degenerate geometry, or incorrect bounding boxes. The capture pipeline already handles these (black pixels in renders). The LoRA reformatter must detect and skip assets where >50% of pixels are background-colored.
- **Alpha vs LK renders**: Alpha (0.5.3) textures are lower resolution and different color palette. Mixing builds in training data is fine for diversity, but the metadata must record which build each capture came from.
- **Interior-only WMOs**: Some WMO groups have no exterior mesh. The capture pipeline already marks these by low mask_coverage. Skip them for LoRA training.
- **M2 with alpha-cutout textures**: Trees, fences, signs with transparent regions. The existing JPEG capture loses alpha; the roof top-down PNG retains it. For LoRA training, we need to handle background removal or use white-background JPEGs as-is (the LoRA should learn that WoW objects are rendered on white backgrounds).
- **Per-object distance variation**: Our captures don't have a distance variable (only one distance per angle). The reformatter maps all perspective angles to `close-up` since the object fills most of the frame at our standard capture distance.

## Requirements

### Functional Requirements

- **FR-001**: The LoRA pair-builder MUST consume the existing roof capture directory structure produced by `build_v18_object_catalog_pipeline.py` (spec 025 output).
- **FR-002**: The pair-builder MUST map each of the 6 perspective captures to the nearest LoRA prompt:

  | Capture | Azimuth | Elevation | Prompt |
  |---------|---------|-----------|--------|
  | `front.jpg` | 0° | 15° | `<sks> front view elevated shot close-up` |
  | `back.jpg` | 180° | 15° | `<sks> back view elevated shot close-up` |
  | `left.jpg` | 90° | 15° | `<sks> left side view elevated shot close-up` |
  | `right.jpg` | 270° | 15° | `<sks> right side view elevated shot close-up` |
  | `top.jpg` | 0° | 80° | `<sks> front view high-angle shot close-up` |
  | `three_quarter.jpg` | 35° | 25° | `<sks> front-right quarter view elevated shot close-up` |
  | `roof_topdown.png` | ortho | ortho | `<sks> front view high-angle shot wide shot` |

- **FR-003**: All distances map to `close-up` (×0.6 factor) since the object fills the frame at our standard capture distance.
- **FR-004**: The pair-builder MUST use `<sks>` as the trigger token, matching the reference LoRA convention.
- **FR-005**: The canonical reference view (input for all pairs of one object) MUST be the `front.jpg` capture mapped to `<sks> front view elevated shot close-up`. The input image for all 7 pairs of that object is the same `front.jpg`; the target varies.
- **FR-006**: Output images MUST be 512×512 by default (configurable), downsampled from the original capture resolution.
- **FR-007**: Assets with >50% background-colored pixels MUST be skipped and logged.
- **FR-008**: The output MUST be structured as a directory of training pairs consumable by the fal.ai Qwen Image Edit 2511 Trainer or local diffusers `load_dataset()`.
- **FR-009**: Each training pair MUST carry metadata: source asset path, build label, source capture angle, mapped prompt, and the original capture resolution.
- **FR-010**: The pair-builder MUST detect and skip identity pairs (where input == target) — the front→front pair is trivial.
- **FR-011**: The training pipeline MUST support both fal.ai cloud training and local diffusers fine-tuning as interchangeable backends.
- **FR-012**: The local training path MUST use the existing `wow-viewer/data-harvester/` Python environment and `uv` with `diffusers` and `transformers`.
- **FR-013**: The fine-tuning MUST start from the existing `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` checkpoint (not from the base Qwen model alone), preserving multi-angle generalization.
- **FR-014**: All code MUST live inside `wow-viewer/` with no references outside the repo. The pair-builder goes in `wow-viewer/data-harvester/scripts/` or `wow-viewer/data-harvester/src/harvester/`.
- **FR-015**: The dataset output MUST NOT contain any proprietary game client data beyond what is lawfully accessible (Bring Your Own Data policy).

### Key Entities

- **Roof Capture Output**: The per-asset directory containing 7 images + metadata from spec 025's pipeline.
- **LoRA Training Pair**: (input_image=front.jpg, prompt_string, target_image=angle-specific render) — 6 usable pairs per object (front→front is identity and skipped).
- **Angle-to-Prompt Map**: Static mapping from capture filenames to the nearest LoRA prompt string.
- **LoRA Checkpoint**: The `.safetensors` weights file, loadable by `pipe.load_lora_weights()`, fine-tuned on top of `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`.
- **Generative Fallback**: Inference-time logic taking a cropped object image + angle prompt → synthesized reference view.

## Success Criteria

- **SC-001**: The pair-builder produces valid LoRA training pairs from one roof capture directory (6+1 pairs per object, prompts match the reference LoRA convention).
- **SC-002**: A LoRA fine-tuning run on 100+ objects completes successfully on fal.ai or local diffusers.
- **SC-003**: Inference with the fine-tuned LoRA on a held-out object produces plausible multi-angle outputs that look like WoW-style renders.
- **SC-004**: The fine-tuned LoRA preserves the base model's view-control capability (smooth rotation across 8 azimuths at eye level).
- **SC-005**: The pair-builder correctly filters out degenerate assets (interior-only, missing textures).
- **SC-006**: The pair-builder runs in under 5 minutes for a full 6-build roof capture corpus.
- **SC-007**: Python environment setup works with `uv sync` in `wow-viewer/data-harvester/` after adding `diffusers` and `transformers`.

## Assumptions

- The existing roof capture pipeline (spec 025) already produces valid captures for hundreds of WMO/M2 assets across 6 client builds.
- The reference LoRA's multi-angle generalization is robust enough that fine-tuning on only 6 angles per object (vs 96 in the original training set) transfers to novel angles.
- 100+ objects is a sufficient training set size for a meaningful LoRA. Scaling to 1000+ would improve results but is not required for the first pass.
- The front.jpg capture is a reasonable canonical input for all training pairs (the model learns: "given this front view, render from angle X").
- White-background JPEG captures are acceptable training images (the LoRA learns to expect white backgrounds for WoW objects).
- The existing roof capture resolution already exceeds the 512×512 training resolution, so no upscaling artifacts are introduced.
- Degenerate captures (all-black interior WMOs, missing-texture M2s) are a small fraction (<5%) of the total corpus.

## Relationship to Other Specs

- **Consumes**: `025-object-roof-mask-library-and-minimap-sieve` — uses the roof capture output directory and per-asset renders. No changes needed to spec 025's pipeline.
- **Depends on**: `013-object-mask-rendering-fix` (object renderer correctness for capture quality)
- **Depends on**: `020-renderer-culling-and-tile-capture` (culling fixes for correct object visibility in captures)
- **Complements**: `025`'s roof-identification goal (025 handles known-object identification via roof silhouettes; this handles unknown-object generation via LoRA)
- **Independent from**: `017-mdxviewer-port-headless-capture` (spec 025 already routes through MdxViewer for captures; no new renderer work needed here)