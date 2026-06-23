# Research: Phase 4 Texture, Variant, And BLP Evidence

## Goal

Identify which existing `wow-viewer` surfaces can be reused for Spec 076 Phase 4 without rewriting file readers or inventing a second texture pipeline.

## Reusable Now

- `alpha_256`, `mcly_texture_ids`, and `mcly_layer_mask` already exist in V18 Zarr stores and are copied into Spec 076 Phase 1 canvases.
- `fractal_segments.py` records per-region `mcly_texture_ids` and `mcly_active_layers` from the full-map canvas.
- `fractal_library.py` now records per-sample `mcly_texture_id_counts`, `dominant_mcly_texture_id`, and `mcly_active_layer_coverage` in `samples.parquet`.
- `src/harvester/compositor.py` provides alpha/weight compositing helpers, but it uses placeholder colors and is not real tileset texture evidence.

## Useful Prior Art Only

- `scripts/build_v18_composition_graph.py` and `scripts/build_v18_paste_library_catalog.py` contain paste/family grouping metadata, but not canonical terrain tileset or BLP fingerprints.
- `src/harvester/object_roof.py` has `variant_fingerprint_from_rgb`, a deterministic perceptual hash used for object-roof RGB renders. The hash approach is reusable, but the object-roof output is not terrain texture evidence.
- `scripts/build_v18_object_roof_library.py` writes `variant_fingerprint` for WMO roof captures. This is object evidence, not terrain tileset or transparent/effect BLP source evidence.

## Not Canonical Yet

- No standalone decoded terrain tileset texture fingerprint dataset was found under `wow-viewer/data-harvester/`.
- No canonical `wow-viewer/output` artifact for terrain `texture`, `tileset`, `variant`, `fingerprint`, or `blp` evidence was found by the Phase 4 inventory.
- V16/V18 capture stubs include a `textures` list field, but the reviewed dataset-builder code initializes it as empty and does not make it a canonical texture fingerprint source.
- Viewer/export code contains BLP decode surfaces, including `AssetProbe`, `GlbExporter`, and `MapGlbExporter`, but those are app/export probes rather than a data-harvester terrain texture evidence dataset.

## BLP Source Candidate Plan

Use existing BLP decode/tooling surfaces only. Do not rewrite BLP readers.

1. Inventory client asset paths from staged client roots or existing listfile/catalog surfaces.
2. Prioritize paths that look like transparent or stamp-like sources: `textures\BloodSplats`, FX, environment, weather, decal, particle, alpha-bearing, and brush-like names.
3. Decode candidate BLPs through existing `wow-viewer` BLP decode/tooling paths into small RGBA thumbnails.
4. Compute stable fingerprints from alpha and luminance channels using a shared helper derived from the existing `variant_fingerprint_from_rgb` pattern.
5. Join candidate fingerprints to accepted Spec 076 samples as optional review evidence, with asset path, fingerprint ID, similarity score, and review state.

## Phase 4 Status

- T025 inventory is complete for current repo/output surfaces.
- T026 MCLY texture summary fields are implemented in the trainable library metadata.
- T027-T031 remain open until a canonical decoded terrain tileset/BLP fingerprint artifact exists or a bounded evidence extractor is added in `wow-viewer/data-harvester/`.

## Curation Correction: Composite Canvases And Minimum Footprint

- `composite_chonker` is a review/composite harvest label, not a synonym for invalid data. Many such regions may be correct when the intended target is a composite terrain canvas instead of an atomic brush.
- The smaller sub-segments inside a composite can be over-segmented artifacts and should not automatically become accepted atomic samples.
- Default atomic samples now require a minimum `8x8` alpha-pixel footprint, the smallest authoring block size for the data we care about.
- If later validation proves that the correct lower-level unit is an `8x8` set of cells within a chunk, update the footprint constant and rerun the Phase 2/3 smoke gates before training.

## Raw Analysis Mode

- `segment_full_map_fractals.py --curation-mode raw` emits every detected region as `raw_component`.
- Raw mode is for inspection and broad analysis only. It does not decide whether a region is atomic, composite, too small, or trainable.
- `analyze_fractal_raw_components.py` runs canvas assembly, raw segmentation, and exact binary-shape dedupe across selected builds/maps in one command.
- `--tile-limit 0` loads every tile for a map and processes the map in horizontal strips of configurable ADT-tile width/overlap.
- Exact dedupe hashes thresholded alpha crops by shape and bitmap. It is intentionally strict.
- Full-map Azeroth 0.5.3 produced 12,906 raw components, 12,163 exact patterns, and only 566 exact duplicates. Exact matching is therefore too brittle; most repeated motifs are near-duplicates (translated, mirrored, or slightly varied), not pixel-identical bitmaps.
- Near-duplicate clustering groups raw components by translation/mirror/rotation-invariant normalized binary thumbnails. With a 16x16 thumbnail and radius 0, full Azeroth 0.5.3 collapsed to 11,976 clusters (668 duplicate clusters, max size 40).
- Rectangle-page detection finds solid axis-aligned rectangular alpha pages (extent >= 0.85) likely to be authored paste/boundary regions, separately from fractal connected components. Full Azeroth 0.5.3 produced 72 rectangle_page regions.
