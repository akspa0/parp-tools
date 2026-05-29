# Feature Specification: Object Multi-Angle LoRA Dataset

**Feature Branch**: `027-object-multiangle-lora`

**Created**: 2026-05-29

**Status**: Draft

**Input**: Build a multi-angle object reference image dataset from WoW WMO/M2 renders, structured for LoRA fine-tuning of Qwen-Image-Edit-2511 using the prompt schema from `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` (96 camera poses: 4 elevations × 8 azimuths × 3 distances). The eventual goal is to generate plausible reference images for unrecognized objects — a fallback when visual identification models can't match an object to a known asset.

## Problem Statement

The V18 terrain pipeline and object-roof library (spec `025`) can identify known objects from placement metadata and roof silhouettes. But there will always be edge cases: damaged or partial M2/WMO assets, objects from client builds not in the corpus, or novel combinations. When visual identification fails, the system needs a way to generate a plausible reference image for the unrecognized object — something that looks like what a WoW object would look like from that angle.

Fine-tuning a multi-angle LoRA on real WoW object renders gives us a generative prior that "knows" what WoW objects look like at any angle. We can then use that LoRA to synthesize missing views or even entire objects.

The reference LoRA (`fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`) was trained on 3000+ Gaussian Splatting renders of real-world objects. We can extend or retrain it on WoW-specific object renders, using the same 96-pose camera system and the exact same `<sks> [azimuth] [elevation] [distance]` prompt format.

## User Scenarios & Testing

### User Story 1 — WoW objects can be rendered from 96 standard camera poses (Priority: P1)

A dataset builder can take any known WMO or M2 object from a staged game client and produce 96 rendered views matching the reference LoRA's camera system (4 elevations × 8 azimuths × 3 distances).

**Why this priority**: Everything else depends on having the multi-angle renders. Without this, there is no dataset.

**Independent Test**: Render one WMO (e.g. `STORMWIND.WMO`) from all 96 poses and verify each output is a non-black, non-trivial image with the object clearly visible.

**Acceptance Scenarios**:

1. **Given** a staged game client and a known WMO path, **When** the multi-angle renderer runs, **Then** it produces 96 output images (PNG, 512×512 or configurable)
2. **Given** a single-pose render, **When** the camera is at `front view eye-level shot medium shot` (0° azimuth, 0° elevation, ×1.0 distance), **Then** the object is centered and fully visible
3. **Given** all 96 renders for one object, **When** they are inspected, **Then** each file's metadata records the azimuth, elevation, distance, and source asset path

---

### User Story 2 — Rendered views are formatted as LoRA training pairs (Priority: P1)

A trainer can take the 96 renders for one object and produce 96 training examples, each being (input_image, prompt_string, target_image) where input_image is the canonical front-view eye-level medium shot, prompt contains `<sks> [azimuth] [elevation] [distance]`, and target_image is the render from that angle.

**Why this priority**: The LoRA requires paired training data. The prompt format must exactly match the reference LoRA's expectations.

**Independent Test**: Format one object's 96 renders into training pairs and verify one pair: `input=front_eye_medium.png`, `prompt="<sks> right side view elevated shot close-up"`, `target=right_elevated_closeup.png`.

**Acceptance Scenarios**:

1. **Given** 96 renders for one object, **When** the pair-builder runs, **Then** it emits 96 (input, prompt, target) tuples
2. **Given** a training tuple, **When** the prompt is inspected, **Then** it matches the reference LoRA schema exactly (`<sks> {azimuth} {elevation} {distance}`)
3. **Given** all tuples, **When** the canonical front-view (0° az, 0° elev, ×1.0 dist) tuple is checked, **Then** input == target (identity pair — standard practice for viewpoint control LoRAs)

---

### User Story 3 — Multi-object dataset is usable for LoRA fine-tuning (Priority: P2)

A trainer can run LoRA fine-tuning of `Qwen/Qwen-Image-Edit-2511` using the rendered WoW object dataset, producing a checkpoint that generates WoW-style objects from arbitrary angles given a front-view input.

**Why this priority**: This validates the dataset quality and demonstrates the generative capability.

**Independent Test**: Fine-tune on 10+ objects with full 96-pose sets, then run inference on a held-out object's front view with a novel-angle prompt and visually assess the output.

**Acceptance Scenarios**:

1. **Given** a dataset of 10+ objects with 96 poses each, **When** LoRA training runs on `fal.ai` or local diffusers, **Then** training completes without errors
2. **Given** a trained LoRA checkpoint, **When** inference runs on a held-out object's front view with prompt `<sks> back view low-angle shot close-up`, **Then** the output is a plausible WoW-style object render from that angle
3. **Given** inference outputs for 8 azimuths at eye-level medium shot, **When** they are viewed in sequence, **Then** they form a smooth rotation around the object

---

### User Story 4 — Generative fallback for unrecognized objects (Priority: P3)

An inference pipeline can use the trained LoRA to generate a plausible reference image for an object that the visual identification system cannot match to a known asset.

**Why this priority**: This is the ultimate goal, but it depends on having a working LoRA first. The pipeline itself is thin wrapper logic.

**Independent Test**: Feed a cropped top-down or front-view image of an unknown object-like blob into the pipeline and verify the output is a plausible multi-angle WoW object image (even if not faithful to the "true" unknown object).

**Acceptance Scenarios**:

1. **Given** an input image of an unrecognized object region, **When** the generative fallback runs, **Then** it outputs a plausible WoW-style object from the requested angle
2. **Given** the same input, **When** different angle prompts are supplied, **Then** the outputs are consistent (same object style, different viewpoints)
3. **Given** a clearly non-object input (empty terrain), **When** the fallback runs, **Then** the output quality degrades gracefully rather than producing a hallucinated object

---

### Edge Cases

- **Empty or transparent renders**: Some WMOs are fully underground or invisible from certain angles. The renderer must detect near-empty frames and either skip them or mark them as low-quality.
- **Very large/small objects**: Camera framing must adapt to object bounding box size. A mailbox and a cathedral should both fit the output frame.
- **Objects with interior-only geometry**: Some WMOs are interiors with no exterior mesh. The renderer must detect this and not produce blank frames.
- **Partial objects**: M2 doodads may have alpha-cutout textures (trees, fences). The renderer must handle alpha blending correctly.
- **Mixed-build objects**: Objects from different client builds (0.5.3 Alpha vs 3.3.5 LK) have different formats. The renderer must handle both.

## Requirements

### Functional Requirements

- **FR-001**: The multi-angle renderer MUST consume a WMO or M2 path from a staged game client and produce 96 renders at the defined 4×8×3 camera poses.
- **FR-002**: The camera system MUST match the reference LoRA's conventions: 8 azimuths (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°), 4 elevations (-30°, 0°, 30°, 60°), 3 distances (×0.6, ×1.0, ×1.8).
- **FR-003**: The output image size MUST be configurable (default 512×512).
- **FR-004**: Each render output MUST carry metadata: source asset path, azimuth, elevation, distance, client build, and render timestamp.
- **FR-005**: The pair-builder MUST use `<sks>` as the trigger token in every prompt, matching the reference LoRA convention.
- **FR-006**: The pair-builder MUST use the reference LoRA's exact prompt vocabulary: `{front/back/left/right/etc} view`, `{low-angle/eye-level/elevated/high-angle} shot`, `{close-up/medium shot/wide shot}`.
- **FR-007**: The canonical reference view (input for all pairs of one object) MUST be `front view eye-level shot medium shot` (0° az, 0° elev, ×1.0 dist).
- **FR-008**: The renderer MUST detect and skip or flag near-empty renders (<1% non-background pixels).
- **FR-009**: The renderer MUST handle both Alpha (0.5.3) and LK (3.3.5) WMO/M2 formats through the existing runtime data types.
- **FR-010**: The dataset MUST be stored in a structured format consumable by the fal.ai Qwen Image Edit 2511 Trainer or local diffusers fine-tuning scripts.
- **FR-011**: All game file I/O MUST use existing `IArchiveCatalog` (via `NativeMpqService` or `MpqArchiveCatalog`). No direct file path assumptions.
- **FR-012**: The entire pipeline MUST live inside `wow-viewer/` with no references outside the repo.
- **FR-013**: The dataset output MUST NOT contain any proprietary game client data beyond what is lawfully accessible. The spec follows Bring Your Own Data policy.
- **FR-014**: The camera MUST orbit the object's bounding-box center, not the world origin.
- **FR-015**: Distance values (×0.6, ×1.0, ×1.8) MUST be relative to the object's bounding sphere radius, not absolute world units.
- **FR-016**: The training pipeline MUST support both fal.ai cloud training and local diffusers fine-tuning as interchangeable backends.
- **FR-017**: The local training path MUST use the existing `wow-viewer/data-harvester/` Python environment and `uv` for dependency management, adding `diffusers` and `transformers` if not already present.

### Key Entities

- **Camera Pose**: A tuple of (azimuth: 0-315°, elevation: -30/0/30/60°, distance: 0.6/1.0/1.8) defining one viewpoint.
- **Multi-Angle Render Set**: 96 images for one object, one per camera pose.
- **Training Pair**: (input_image, prompt_string, target_image) — one per non-reference pose, totaling 95 per object (the reference pose is identity).
- **Object Catalog**: A registry of known WMO/M2 assets to render, optionally filtered by placement frequency, object family, or client build.
- **Render Manifest**: JSON/CSV file listing all renders produced with full metadata per row.
- **LoRA Checkpoint**: The `.safetensors` LoRA weights file produced by fine-tuning, loadable by `pipe.load_lora_weights()`.
- **Generative Fallback**: Inference-time logic that takes a cropped input image + angle prompt and produces a synthesized reference view.

## Success Criteria

- **SC-001**: At least one WMO (e.g. `STORMWIND.WMO`) produces 96 non-empty renders covering all poses.
- **SC-002**: At least one M2 (e.g. a tree doodad) produces 96 non-empty renders with correct alpha handling.
- **SC-003**: The pair-builder produces training tuples consumable by `fal.ai` trainer without manual reformatting.
- **SC-004**: A LoRA fine-tuning run on 10+ objects completes successfully (either on fal.ai or local diffusers).
- **SC-005**: Inference with the trained LoRA on a held-out object produces visually plausible multi-angle outputs (validated by manual inspection).
- **SC-006**: The full pipeline — render → pair-build → train → infer — is documented in a single script or README.
- **SC-007**: Build succeeds with `dotnet build WowViewer.slnx -c Debug` for any C# components.
- **SC-008**: Python environment setup works with `uv sync` in `wow-viewer/data-harvester/` after adding new dependencies.

## Assumptions

- The existing headless capture infrastructure (spec `017`) and object renderer (spec `013`, `020`) can load individual WMO/M2 assets in isolation, not just full-tile scenes.
- The headless renderer supports user-controlled camera positioning (not just tile-centered cameras).
- We can load a single WMO or M2 file (with its dependencies) and render it in a minimal scene with no terrain, no lighting from terrain, just the object itself.
- The object bounding box is available from the WMO/M2 reader (WMO groups have bounding boxes; M2 has a bounding radius).
- The fal.ai Qwen Image Edit 2511 Trainer accepts the same prompt format as the reference LoRA and can be pointed at a custom dataset.
- Local diffusers fine-tuning of LoRA on Qwen-Image-Edit-2511 is feasible on available hardware (RTX 4090 with 24GB VRAM for small runs, A100 for full runs).
- Most WMO objects have meaningful exterior geometry visible from multiple angles (not just interiors).
- Objects with zero placement instances in the corpus are not worth rendering for the initial dataset.
- The reference LoRA's 96-pose system is the right choice; we don't need to invent a new camera system.

## Relationship to Other Specs

- **Depends on**: `013-object-mask-rendering-fix` (object renderer works correctly)
- **Depends on**: `017-mdxviewer-port-headless-capture` (headless renderer infrastructure)
- **Depends on**: `020-renderer-culling-and-tile-capture` (culling fixes for correct object visibility)
- **Shares concern with**: `025-object-roof-mask-library-and-minimap-sieve` (both need per-object rendering with pose metadata, but for different purposes — 025 needs top-down roof views, this needs 96-angle sphere)
- **Extends**: the renderer infrastructure to support standalone object rendering (not tile-anchored)
- **Enables**: future generative object synthesis for unknown-asset fallback in the terrain pipeline