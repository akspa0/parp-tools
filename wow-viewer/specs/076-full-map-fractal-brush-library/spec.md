# Feature Specification: Full-Map Fractal Brush Library

**Feature Branch**: `076-full-map-fractal-brush-library`

**Created**: 2026-06-23

**Status**: Draft

**Input**: User request: replace tile-local alpha-brush and coarse scar-mask plans with a full-map, provenance-preserving decomposition pipeline that identifies real reusable terrain brush/paste/fractal building blocks from all available terrain signals.

## Supersedes

This spec supersedes the current development direction of:

- `074-alpha-brush-library`: useful as a candidate/evidence extractor, but wrong when it treats tile-local connected alpha components as atomic brushes.
- `075-scar-mask-segmentation`: useful only as a coarse diagnostic baseline, not as the primary model path.
- The current interpretation of V18 paste outputs: useful prior art, but incomplete until full-map alpha fractal segmentation, height/normal provenance, MCLY context, and tileset-variant evidence are joined.

## Corrected Conceptual Model

WoW terrain is a structured 3D digital painting over a terrain mesh, not a set of independent minimap tiles. The closest working mental model is a stack of ZBrush-like documents/layers: artists paint source texture stamps, alpha masks, and terrain sculpting together until the mesh, alpha layers, and texture assignments form one cohesive terrain art primitive.

- The real canvas is the full map, not one ADT tile.
- ADT tiles are storage pages of a larger virtual artist canvas.
- MCAL alpha layers are artist brush/pressure masks over stacked texture layers.
- MCLY texture IDs and layer masks define which tileset texture participates in each chunk/cell.
- Height and normals are part of the same brush identity because terrain sculpting and alpha/texture painting were authored together.
- Tileset textures, decal-like textures, and effect textures may themselves contain source stamps or visual references that later appear as painted/sculpted terrain motifs.
- Candidate source BLP categories include FX/environment/weather/decal/particle textures and explicit brush-like paths such as `textures\BloodSplats`.
- A tile-local connected component can be a true atomic brush, a fragment of a larger fractal, a composite/chonker made of many brush placements, or a one-off hand-painted road/detail.

The goal is a trainable terrain-art primitive library built from real art signals with full provenance, not a JSONL dump of arbitrary connected components. A valid primitive may bundle source BLP/decal/effect evidence, alpha imprint, MCLY texture/layer assignment, height/normal sculpt response, and repeated placement lineage.

## User Scenarios & Testing

### User Story 1 — Assemble Full-Map Signal Canvases (Priority: P1)

As a terrain reconstruction researcher, I want full-map alpha/MCLY/height/normal canvases assembled from V18 Zarr stores, so segmentation operates on the actual coupled sculpt-and-paint document instead of tile-local fragments.

**Why this priority**: Any brush/paste/fractal that crosses tile boundaries is misrepresented by per-tile processing. Full-map canvases are the prerequisite for correct decomposition.

**Independent Test**: Run the canvas assembler for one bounded build/map and inspect output metadata proving each full-map pixel can be traced back to tile/chunk/local coordinates.

**Acceptance Scenarios**:

1. **Given** a V18 Zarr store with `alpha_256`, `height_257`, `normal_xyz`, `mcly_texture_ids`, and `mcly_layer_mask`, **When** the assembler runs for one map, **Then** it writes aligned map-canvas arrays plus a provenance index mapping canvas coordinates to build/map/tile/chunk/local-pixel coordinates.
2. **Given** a feature crossing an ADT boundary, **When** the canvas is inspected, **Then** it appears as one continuous map-canvas signal instead of two unrelated tile-local fragments.

---

### User Story 2 — Segment Full-Map Fractal Regions (Priority: P1)

As a researcher, I want full-map alpha layers segmented into fractal regions and virtual canvases, so fragments of the same authored structure can be grouped before building the brush library.

**Why this priority**: The existing 074 catalog over-counts tiny one-off strokes and includes massive glued-together chonkers. Fractal-aware segmentation is required before deciding what is a brush.

**Independent Test**: Run segmentation on a known visually complex map region and produce overlays showing fractal regions, rejected chonkers, one-off hand-painted roads/details, and linked 074 components.

**Acceptance Scenarios**:

1. **Given** full-map alpha layer canvases, **When** fractal segmentation runs, **Then** output regions preserve multi-tile bounding boxes and tile coverage.
2. **Given** a region made of repeated fractal motifs, **When** members are emitted, **Then** the members link to their source 074 component IDs where overlaps exist.
3. **Given** large random connected swaths or one-off road strokes, **When** curation labels are assigned, **Then** those entries are marked `composite_chonker` or `one_off_detail` and excluded from default training manifests.

---

### User Story 3 — Build A Provenance-Preserving Trainable Library (Priority: P1)

As a model trainer, I want a curated Zarr/Parquet training library of accepted brush/fractal/paste terrain-art primitives with height, normal, alpha, MCLY, tileset, and source-texture context, so models train on real reusable sculpt-and-paint building blocks instead of arbitrary masks.

**Why this priority**: The harvested library only becomes useful when it is transformed into a trainer-consumable dataset with preserved provenance and rejection reasons.

**Independent Test**: Generate a bounded trainable library and verify sample rows include tensor references, source provenance, curation label, spatial statistics, and split assignment.

**Acceptance Scenarios**:

1. **Given** segmented fractal regions and existing 074/024 candidate outputs, **When** library construction runs, **Then** accepted samples are written to a Zarr-backed tensor store and metadata is written to Parquet/JSONL with stable IDs.
2. **Given** rejected candidates, **When** metadata is inspected, **Then** rejection reason is explicit (`too_small_unique`, `one_off_detail`, `composite_chonker`, `low_repeatability`, `bad_provenance`, etc.).
3. **Given** a training sample, **When** it is loaded, **Then** the sample exposes alpha, height, normals, MCLY texture IDs/layer masks, minimap context, object masks where available, optional BLP/decal/effect source matches, and exact provenance.

---

### User Story 4 — Join Tileset Variant Evidence (Priority: P2)

As a researcher, I want tileset texture variants, transparent BLP effect textures, and overlay-like texture modifications linked to alpha/fractal candidates, so minimap color and texture contribution can be explained before terrain reconstruction.

**Why this priority**: Minimap appearance is not only alpha. Tileset variants and texture-level brush/overlay patterns affect the visible result and must be separated from terrain geometry.

**Independent Test**: Produce a report linking MCLY texture IDs, decoded tileset/variant fingerprints, and likely FX/environment/weather BLP brush-source fingerprints to accepted fractal/brush candidates on a bounded map.

**Acceptance Scenarios**:

1. **Given** MCLY texture ID usage and available decoded tileset texture assets, **When** variant joining runs, **Then** each candidate records texture-family/variant evidence where resolvable.
2. **Given** visually similar alpha candidates with different texture variants, **When** catalog rows are compared, **Then** they remain distinguishable by texture provenance.
3. **Given** client BLP assets whose paths or alpha channels suggest FX, environment, weather, decal, particle, brush-like source textures, or paths such as `textures\BloodSplats`, **When** source-texture matching runs, **Then** visually similar candidates are linked to candidate source BLP asset paths with similarity scores and review flags.

---

### User Story 5 — Define Correct Downstream Model Targets (Priority: P3)

As a trainer, I want model targets derived from the curated library, so every model predicts one useful signal rather than an underspecified whole-tile mask.

**Why this priority**: Training before the library is curated repeats the 075 mistake. Models must follow the corrected dataset contract.

**Independent Test**: Write model-target notes and sample loaders that can feed one-signal models from the curated library without starting a training run.

**Acceptance Scenarios**:

1. **Given** the curated library, **When** model target planning runs, **Then** it emits separate candidate targets for alpha-fractal segmentation, tileset-variant identification, and height/normal residual restoration without multi-head training.
2. **Given** the target list, **When** reviewed, **Then** no target trains on raw whole-tile binary scar masks as the primary product.

## Edge Cases

- Full-map canvases may be sparse or have missing tiles; provenance gaps must be explicit.
- Alpha regions can cross ADT tile boundaries and chunk boundaries.
- One ADT tile can contain multiple virtual artist canvases or unrelated pasted blocks.
- Roads and other one-off hand-painted details can be large and visually meaningful but not reusable brush families.
- Large chonkers can contain real motifs but are not themselves accepted atomic training units.
- Some useful fractals may have weak alpha repetition but strong height/normal repetition.
- Texture IDs can change between builds while visual texture variants remain related.
- Some alpha/fractal brush shapes may originate from small transparent BLP source assets, especially FX, environment, weather, decal, particle, `textures\BloodSplats`, or similar brush-like textures.
- Object occlusion and minimap bake artifacts must be tracked so model input is not polluted by non-terrain content.

## Requirements

### Functional Requirements

- **FR-001**: System MUST assemble full-map canvases per build/map for alpha layers, height, normals, MCLY layer masks, and MCLY texture IDs from existing V18 Zarr stores.
- **FR-002**: System MUST preserve pixel/sample provenance from map-canvas coordinates back to build, map, tile_id, tile_x, tile_y, chunk/cell, layer, and local pixel coordinates.
- **FR-003**: System MUST treat terrain mesh, height, normals, alpha masks, and MCLY texture/layer context as one coupled terrain-art primitive during segmentation and curation.
- **FR-004**: System MUST segment alpha/fractal regions in full-map coordinates before deciding brush identity.
- **FR-005**: System MUST link full-map regions back to 074 component IDs when overlaps exist.
- **FR-006**: System MUST classify candidates at minimum as `accepted_candidate`, `fractal_member`, `composite_chonker`, `one_off_detail`, `too_small_unique`, or `rejected_unknown`.
- **FR-007**: System MUST exclude `composite_chonker`, `one_off_detail`, and `too_small_unique` rows from default training manifests while preserving them for review.
- **FR-008**: System MUST compute spatial signatures from height and normals for each accepted or reviewable candidate.
- **FR-009**: System MUST record MCLY texture ID and layer-mask context for each candidate.
- **FR-010**: System MUST support joining available tileset texture/variant fingerprints to candidate metadata without rewriting BLP or format readers.
- **FR-011**: System MUST support a read-only BLP source-candidate scan using existing BLP decode/tooling surfaces, prioritizing asset paths or alpha channels that suggest FX, environment, weather, decal, particle, `textures\BloodSplats`, or brush-like textures.
- **FR-012**: System MUST compare likely BLP source assets to accepted alpha/fractal candidates with stable fingerprints, similarity scores, and asset-path provenance.
- **FR-013**: System MUST emit a trainable Zarr/Parquet dataset contract, not only JSONL analysis rows.
- **FR-014**: System MUST produce visual overlays/contact sheets separated by curation label and layer, with source provenance visible.
- **FR-015**: System MUST keep all implementation under `wow-viewer/` and read only existing staged/Zarr dataset paths.
- **FR-016**: System MUST not start new model training phases until the full-map segmentation and curated library validation gates pass.

### Key Entities

- **Map Canvas**: Full build/map coordinate surface assembled from ADT tile signals.
- **ZBrush-Like Document**: Coupled terrain mesh, alpha masks, MCLY texture/layer assignments, and source texture/decal/effect evidence treated as one sculpt-and-paint authoring surface.
- **Virtual Canvas**: Subregion of a map canvas representing one coherent authored painting/sculpting/paste area that may not align to ADT tile boundaries.
- **Fractal Region**: Full-map alpha/height/normal structure with self-similar or repeated motif evidence.
- **Terrain-Art Primitive**: Reusable sculpt-and-paint unit bundling alpha imprint, height/normal response, MCLY texture/layer context, optional BLP/decal/effect source evidence, and placement lineage.
- **Brush Candidate**: Candidate reusable unit derived after full-map segmentation, not from tile-local connected components alone.
- **Fractal Member**: Local part of a larger fractal region; can become a training sample but is not independently a complete brush family.
- **Composite Chonker**: Large connected or high-coverage region likely composed of multiple unrelated brush placements.
- **One-Off Detail**: Hand-painted road/detail/retouch that may be valid art but lacks repeatability as a reusable brush family.
- **Tileset Variant Evidence**: Texture-family/variant signals that explain minimap appearance beyond alpha placement.
- **BLP Source Candidate**: Decoded client BLP texture, often transparent or effect-like, that may be the original brush/decal/effect source later painted into alpha layers.
- **Training Library Sample**: Provenance-preserving Zarr/Parquet entry with tensors and metadata for a curated unit.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A bounded build/map run writes full-map alpha/MCLY/height/normal canvases and provenance metadata with no missing required provenance fields for valid tiles.
- **SC-002**: Visual overlays show at least one cross-tile alpha/fractal structure preserved as a single map-canvas region.
- **SC-003**: Review sheets separate accepted candidates from chonkers and one-off details; default training manifests include no `composite_chonker` or `one_off_detail` rows.
- **SC-004**: At least 90% of accepted samples include height and normal spatial statistics, and 100% include alpha + MCLY provenance when source arrays exist.
- **SC-005**: The trainable library can be loaded by a smoke dataset reader that returns coupled alpha, height, normal, MCLY, optional source-texture evidence, and provenance for at least 32 accepted samples.
- **SC-006**: No new training run is recommended until SC-001 through SC-005 pass.

## Assumptions

- Existing V18 Zarr stores contain the needed `alpha_256`, `height_257`, `normal_xyz`, `mcly_texture_ids`, and `mcly_layer_mask` arrays for primary builds.
- Existing 074 outputs are useful evidence, but not authoritative brush labels.
- Existing V18 paste scripts are useful references for candidate metadata, but must be corrected to full-map/fractal-aware semantics before they are treated as training truth.
- Tileset/effect BLP source extraction may require a follow-up slice if decoded texture fingerprints are not already available in a usable form.
- The brush library data already harvested is valuable, but it requires refinement against full-map fractal structure, mesh response, MCLY context, and possible original BLP source assets before it can be used as training truth.
- The first model after this spec will be planned separately and will obey the one-signal/one-output model rule.
