# Feature Specification: Alpha Brush Library

**Feature Branch**: `074-alpha-brush-library`

**Created**: 2026-06-22

**Status**: Deprecated as primary direction; retained as candidate/evidence extraction for `076-full-map-fractal-brush-library`

> Deprecation note (2026-06-23): This spec's tile-local connected-component definition is not sufficient for the intended brush library. Contact-sheet review showed tiny one-off details, roads, and large composite chonkers being treated as brush candidates. Future work must use `076-full-map-fractal-brush-library`, where 074 components are evidence rows only and not authoritative brush labels.

**Input**: User description: "Extract, cluster, and catalog unique fractal brush patterns from MCAL alpha masks across all builds, then train segmentation models to identify them from minimap input."

## Conceptual Model: ADT as Layered Photoshop Canvas

The ADT file format encodes a terrain tile as a layered digital painting:

- **MCAL alpha layers** are the transparency/selection masks (like Photoshop layer masks).
- **MCLY entries** define the layer stack order and which mask controls each layer.
- **MTEX entries** are the actual tileset textures applied through the masks.
- Artists painted with reusable **fractal brushes** on a grid of 16×16 cells, and the same brush stamps recur across tiles, maps, and even builds.

This means the alpha layers are not arbitrary signals — they are the **unadulterated brushwork** of the terrain artists. Our job is to reverse-engineer the artists' brush library by clustering the stamp patterns.

### Expected Layer Roles (heuristic, validated per map)

| Layer | Typical Role | Examples |
|-------|--------------|----------|
| L0 | Base fill / ground plane | dirt, grass, stone, diagnostic checkers.blp |
| L1 | Primary terrain detail | ridge lines, paths, banks |
| L2 | Secondary / transition detail | riverbeds, lava beds, thick blends |
| L3 | Highlight / accent | specular highlights, rim lighting, subtle detail |

Layer roles are consistent within a map/zone but may differ across continents or art-direction revisions. Phase 2 of this work validates these heuristics statistically.

## Existing Signals Used

This feature consumes signals already present in the V18 Zarr stores under `wow-viewer/output/datasets/v18/<build>.zarr/`:

| Signal | Array Key | Shape (per tile) | Purpose |
|--------|-----------|------------------|---------|
| MCAL alpha pack | `alpha_256` | 256×256×4 float32 | The layer masks from which brush components are extracted. |
| MCLY texture IDs | `mcly_texture_ids` | 16×16×4 int32 | Which tileset texture is bound to each layer per chunk cell. |
| MCLY layer mask | `mcly_layer_mask` | 16×16×4 float32 | Which layers are active per chunk cell. |
| Minimap | `minimap_rgb` | 256×256×3 uint8 | Future segmentation model input (not used in Phase 1). |
| Height / normals | `height_257`, `normal_xyz` | 257×257, 257×257×3 | Future terrain reconstruction context (not used in Phase 1). |

No new dataset is harvested. All analysis reads directly from the existing Zarr stores.

## User Scenarios & Testing

### User Story 1 — Extract and Catalog Unique Brush Patterns from Alpha Masks (Priority: P1)

As a researcher, I want to discover all unique fractal brush patterns used across the MCAL alpha layers of every map tile, so that I can build a library of terrain building blocks.

**Why this priority**: This is the foundation for everything downstream. Without knowing what the brushes look like, we cannot build segmentation or reconstruction models. All other phases depend on having a cataloged set of patterns.

**Independent Test**: Running the extraction script across the existing V18 zarr stores produces a JSON catalog and visualization grid of unique brush patterns, grouped by similarity, with tile coordinates and area metadata.

**Acceptance Scenarios**:

1. **Given** a set of V18 zarr stores for client builds 0.5.3.3368, 3.3.5.12340, and others, **When** the brush extraction script runs, **Then** for each tile with valid MCAL alpha data, each alpha layer (L0-L3) is decomposed into connected components where alpha > 0.05 threshold.

2. **Given** a set of extracted brush components from all tiles, **When** feature vectors are computed (size, shape moments, boundary fractal dimension, alpha histogram), **Then** components are clustered by cosine similarity with a configurable threshold, and the cluster centroids are saved as reference patterns.

3. **Given** the clustered brush patterns, **When** the catalog is written, **Then** each entry contains: component ID, source (build, map, tile_x, tile_y, layer_index), cluster ID, feature vector, bounding box within the 256x256 tile, area in pixels, boundary fractal dimension, and area name from AreaTable.dbc if available.

4. **Given** the clustered output, **When** a visualization script runs, **Then** a montage grid is produced showing up to 16 example components per cluster, arranged by similarity, with cluster size annotations.

---

### User Story 2 — Identify Semantic Layer Roles Across Maps (Priority: P2)

As a researcher, I want to verify that L1-L3 have consistent semantic roles across an entire continent map, so that I can use layer identity as a prior for segmentation models.

**Why this priority**: If layers have consistent roles per map, a segmentation model can use layer index as a strong prior. If not, the model must learn per-map or per-tile behavior, which is harder.

**Independent Test**: A report showing the distribution of brush pattern clusters across L0-L3 for each map, with statistical tests for layer-to-cluster consistency.

**Acceptance Scenarios**:

1. **Given** the clustered brush catalog, **When** a per-map layer analysis runs, **Then** for each map, a contingency table shows which clusters appear in which layers, and a chi-squared test reports whether layer assignments are non-random.

2. **Given** the layer analysis, **When** the report is generated, **Then** maps with consistent layer semantics (e.g. L1 always contains ridge clusters) are flagged, and maps with inconsistent semantics are flagged separately for manual review.

---

### User Story 3 — Train Alphaprint Segmentation Model from Minimap (Priority: P3)

As a researcher, I want to train a model that predicts which brush patterns are present in each 16x16 patch of a tile from its minimap rendering alone, so that terrain can be reconstructed from brush identifications.

**Why this priority**: This is the eventual goal, but it depends on having a mature brush catalog first. Until we know what the brushes are, we cannot train a model to find them.

**Independent Test**: A trained model that, given a held-out minimap tile not seen in training, outputs per-patch brush cluster probabilities, with per-patch accuracy above 70%.

**Acceptance Scenarios**:

1. **Given** the brush catalog and a training dataset of minimap tiles with per-pixel brush labels, **When** a U-Net segmentation model is trained, **Then** on a held-out test split, per-patch brush presence F1 score exceeds 0.7.

2. **Given** a trained segmentation model, **When** run on a tile whose alpha layers are known, **Then** the model's predicted brush regions match the ground-truth alpha components with at least 0.5 IoU on the dominant brush per layer.

---

### Edge Cases

- What happens when a tile has zero alpha (all 4 layers empty)?
- How does the system handle tiles with only L0 populated (single-texture tiles)?
- What about alpha layers where values are constant 1.0 (saturated)?
- How to handle overlapping brush components across layers on the same tile?
- What happens when a brush component extends beyond a single tile (edge-crossing patterns)?
- How are MCCV tinting layers treated when analyzing brush shapes?

## Requirements

### Functional Requirements

- **FR-001**: System MUST extract connected components from each alpha layer (L0-L3) where alpha > configurable threshold (default 0.05), using 4-connectivity or 8-connectivity.
- **FR-002**: System MUST compute per-component feature vectors: area (pixel count), bounding box (x, y, w, h), centroid (cx, cy), perimeter, compactness (perimeter / area), boundary fractal dimension (box-counting), Hu moments (7 values), alpha value histogram (mean, std, percentiles), and Sobel gradient magnitude histogram.
- **FR-003**: System MUST filter out components smaller than a minimum area (default 16 pixels) and components that touch the tile boundary (likely partial patterns).
- **FR-004**: System MUST cluster components using cosine similarity on normalized feature vectors, with HDBSCAN or DBSCAN for density-based clustering (no fixed K). Feature weights must be configurable.
- **FR-005**: System MUST export a catalog JSON file with all extracted components and their cluster assignments, plus a per-cluster summary.
- **FR-006**: System MUST produce a visualization montage showing example components for each cluster with metadata overlays.
- **FR-007**: System MUST include per-brush metadata: source build, map name, tile coordinates, layer index (0-3), and AreaTable-derived zone/subzone name.
- **FR-008**: System MUST support analyzing all available builds in the V18 zarr store, not just the two primary builds.
- **FR-009**: Analysis script MUST be runnable with a single command using uv run.

### Key Entities

- **Brush Component**: A connected region of alpha > threshold within a single MCAL layer on a single tile. The atomic unit of terrain painting.
- **Brush Cluster**: A group of similar brush components across tiles/layers/maps, representing a reused pattern.
- **Brush Catalog**: The complete set of unique brush clusters with metadata, stored as JSON and visualization assets.
- **Alphaprint**: The set of brush clusters present in a given tile's alpha layers, forming a "fingerprint" of that tile's terrain composition.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Catalog contains at least 1000 unique brush clusters across all available builds.
- **SC-002**: At least 80% of all alpha pixels across all tiles belong to a component that maps to a cluster with > 10 members (non-trivial cluster).
- **SC-003**: Visualization montage is visually interpretable — a human reviewer can identify which clusters look like similar terrain features (ridges, riverbeds, shorelines).
- **SC-004**: Per-map layer analysis produces a statistically significant contingency table for at least one map (p < 0.01).

## Assumptions

- The V18 zarr stores contain valid, complete MCAL alpha data for all tiles. Missing alpha data is skipped gracefully.
- Connected component extraction at alpha > 0.05 is sufficient to capture meaningful brush shapes without including noise. This threshold can be tuned.
- The existing AreaTable.dbc mapping for zone/subzone names is available and can be joined by map name and tile coordinates.
- Brush patterns are mostly tile-internal (not crossing tile boundaries). Edge-touching components are filtered out to avoid partial patterns.
- Density-based clustering (HDBSCAN) is appropriate for discovering irregularly-shaped clusters without a fixed K. If feature engineering yields well-separated clusters, K-means can be used instead.

## Out of Scope

- Terrain height reconstruction from brush identifications (future phase)
- Segmentation model training and inference (User Story 3 — future phase)
- MCCV tinting layer analysis
- Real-time brush identification in a viewer context
- Integration with any C# tooling — this is entirely Python/data-harvester
