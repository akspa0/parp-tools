# ML Dataset Packaging Reference

This document describes the current ML dataset package contract used by `ml-export` and `ml-harvest`.

Existing `mk-*` and `vlm-*` commands remain as compatibility aliases, but new work should use `ml-*` naming. For downstream tooling, prefer the JSON and manifest paths recorded here instead of reconstructing filenames by convention.

Machine-readable schema files now live here:

- `docs/schemas/ml-dataset-tile.schema.json`
- `docs/schemas/ml-dataset-manifest.schema.json`

## Purpose

The ML dataset package is meant to be a durable interchange surface for:

- training pipelines that need stable paths to terrain supervision assets
- curation and dedupe tooling that needs image signatures and coverage summaries
- viewer or reconstruction tooling that needs the full per-tile metadata, not just baked PNGs

## Current Root Layout

An exported dataset root can contain the following directories and files:

```text
<dataset-root>/
|- dataset/
|  |- <tile>.json
|  |- <tile>.bin
|- images/
|  |- <tile>.png
|  |- <tile>_heightmap.png
|  |- <tile>_heightmap_global.png
|  |- <tile>_normal.png
|  |- <tile>_mccv.png
|- liquids/
|  |- <tile>_liq_mask.png
|  |- <tile>_liq_height.png
|- depths/
|  |- <tile>_depth.png
|- stitched/
|  |- <tile>_shadow.png
|  |- <tile>_alpha_l1.png
|  |- <tile>_alpha_l2.png
|  |- <tile>_alpha_l3.png
|  |- <tile>_alpha_l4.png
|  |- <tile>_alpha_atlas.png
|  |- <map>_full_minimap.png
|  |- <map>_full_shadows.png
|  |- <map>_full_alpha_l1.png
|  |- <map>_full_alpha_l2.png
|  |- <map>_full_alpha_l3.png
|  |- <map>_full_alpha_l4.png
|  |- <map>_full_alpha_atlas.png
|  |- <map>_full_heightmap.png
|  |- <map>_full_heightmap_global.png
|- tilesets/
|  |- <texture>.png
|- texture_database.json
|- ml_dataset_manifest.json
|- reference_minimaps/
|  |- <tile>_reference_minimap.png
```

Not every export contains every directory:

- `depths/` exists only when depth generation was requested.
- `ml_dataset_manifest.json` exists only after `ml-harvest` or the viewer finalize flow writes it.
- `reference_minimaps/` is optional and is not part of the default viewer finalize path anymore.

## Packaging Rules

- Tile JSON files in `dataset/` are the canonical per-tile source of truth.
- Paths inside JSON and manifest files are dataset-root-relative and use `/` separators.
- Tile identity is anchored by `terrain_data.adt_tile` and matches the file stem, for example `Azeroth_32_48`.
- Cross-build training should currently treat dataset-root provenance as the primary build selector. The documented tile schema guarantees stable tile/map identity, but it does not currently document a canonical per-tile client-build field.
- `dataset/<tile>.bin` is a compatibility binary payload emitted alongside JSON. New tooling should prefer JSON and manifest metadata unless it explicitly needs the binary layout.
- Stitched terrain supervision outputs now live under `stitched/`. Tooling must not assume old root-level `shadows/` or `masks/` directories exist.
- Viewer validation captures are not a core package directory. They may be written outside the dataset root, so training or audit tooling should not depend on them.

## Per-Tile JSON Contract

Each `dataset/<tile>.json` file is serialized from `VlmTrainingSample` and has this top-level structure:

```json
{
  "image": "images/Azeroth_32_48.png",
  "depth": "depths/Azeroth_32_48_depth.png",
  "terrain_data": {
    "adt_tile": "Azeroth_32_48"
  }
}
```

### Top-Level Fields

| Field | Meaning | Training relevance |
| --- | --- | --- |
| `image` | Source minimap tile path when a minimap was found and converted | Primary rendered raster input |
| `depth` | Optional depth image path when depth generation was requested | Optional auxiliary supervision |
| `terrain_data` | Full structured tile payload | Canonical source for terrain, liquids, objects, and stitched outputs |

### `terrain_data` Fields

| Field | Meaning | Recommended usage |
| --- | --- | --- |
| `adt_tile` | Stable tile identifier, usually `<map>_<x>_<y>` | Primary join key |
| `heights` | Raw 145-sample height array per chunk | Reconstruction or custom rasterization |
| `chunk_positions` | Flattened chunk world positions | Spatial indexing |
| `holes` | MCNK hole masks | Terrain masking |
| `heightmap` | Legacy generic heightmap path | Compatibility fallback |
| `heightmap_local` | Per-tile normalized heightmap PNG | Local terrain target |
| `heightmap_global` | Map-global normalized heightmap PNG | Cross-tile terrain target |
| `normalmap` | Stitched normal map PNG | Shading or geometry supervision |
| `mccv_map` | Vertex color map PNG | Ground tint or color supervision |
| `shadow_maps` | Tile-scope stitched shadow outputs | Shadow supervision; currently usually one stitched shadow image |
| `shadow_bits` | Raw per-chunk shadow bit payload | Fine-grained shadow analysis or reconstruction |
| `shadow_analysis` | Derived shadow region summaries, candidate associations, and first-pass explained-vs-residual scar labels | Audit, curation, and object-recovery features; still heuristic |
| `alpha_masks` | Tile-scope stitched grayscale alpha masks | Explicit per-layer terrain texture weights |
| `alpha_atlas` | Packed alpha atlas PNG | Compact terrain texture supervision |
| `liquid_mask` | Stitched liquid occupancy mask | Water or liquid supervision |
| `liquid_height` | Stitched liquid height image | Water surface regression |
| `liquid_min` / `liquid_max` | Min/max liquid heights for decoding | Required to interpret liquid height images |
| `textures` | Referenced terrain texture paths | Lookup surface for texture semantics |
| `chunk_layers` | Raw per-chunk layer metadata | Reconstruction and fine-grained terrain analysis |
| `liquids` | Raw per-chunk liquid metadata | Reconstruction or custom rasterization |
| `objects` | MDDF/MODF placement list with bounds when available | Object-aware supervision or masking |
| `wdl_heights` | Low-resolution WDL height payload | Far-terrain or low-frequency terrain context |
| `height_min` / `height_max` | Tile-local terrain height range | Decode local heightmap |
| `height_global_min` / `height_global_max` | Map-global height range | Decode global heightmap |
| `is_interleaved` | Format note for source height data | Decoder compatibility |

### Alpha And Shadow Packaging

Current behavior matters here because older docs were wrong.

- `terrain_data.alpha_atlas` is alpha-only compact packing.
- `stitched/<tile>_alpha_atlas.png` stores alpha layers 1-3 in RGB.
- Shadow data is not packed into the alpha atlas anymore.
- `terrain_data.shadow_maps` remains the separate lookup surface for stitched shadow images.
- `terrain_data.alpha_masks` remains the explicit per-layer lookup surface when tooling needs separate grayscale masks instead of the packed atlas.

### Chunk-Level Legacy Fields

Within `terrain_data.chunk_layers`:

- `shadow_path` is a legacy per-chunk path and may be null.
- `layers[*].alpha_path` is a legacy per-chunk alpha path and may be null.
- `layers[*].alpha_bits` carries decoded per-chunk 64x64 grayscale alpha bytes and is the more durable structured payload.

New tooling should not depend on per-chunk PNG files being present.

## Manifest Contract

When `ml-harvest` or the viewer finalize flow writes `ml_dataset_manifest.json`, it provides a dataset-wide index intended for training and curation tooling.

### Top-Level Manifest Fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Current manifest schema identifier |
| `harvested_at_utc` | Manifest creation timestamp |
| `dataset_root` | Absolute dataset root used during harvest |
| `dataset_name` | Map name derived from tile naming |
| `source_format` | Current input family label |
| `tile_json_directory` | Relative path to the canonical tile JSON directory |
| `reference_minimap_directory` | Relative path to optional baked reference minimaps |
| `coverage` | Dataset-wide counts by asset family |
| `tiles` | Per-tile lookup records |

### Per-Tile Manifest Fields

Each `tiles[]` entry includes:

- canonical identity: `tile_name`, `map_name`, `tile_json_path`
- existence tracking for `source_minimap`, local/global heightmaps, normal map, and MCCV map
- `shadow_map_paths`, `existing_shadow_map_count`, and `shadow_map_signature`
- `alpha_atlas_path`, `alpha_atlas_exists`, and `alpha_atlas_signature`
- `alpha_mask_paths`, declared count, and existing count
- object and chunk-layer counts
- completeness classification such as `core-terrain-ready`
- optional `reference_minimap_path` and generation/existence flags

### Example Manifest Snippet

```json
{
  "schema_version": "mk-dataset-manifest.v1",
  "dataset_name": "Azeroth",
  "tile_json_directory": "dataset",
  "coverage": {
    "tiles_processed": 2,
    "tiles_with_source_minimap": 2,
    "tiles_with_local_heightmap": 2,
    "tiles_with_global_heightmap": 2,
    "tiles_with_shadow_maps": 2,
    "tiles_with_any_alpha_mask": 2,
    "tiles_with_alpha_atlas": 2,
    "declared_alpha_mask_images": 8,
    "existing_alpha_mask_images": 8,
    "declared_shadow_map_images": 2,
    "existing_shadow_map_images": 2,
    "tiles_with_objects": 2,
    "tiles_with_chunk_layer_metadata": 2,
    "tiles_with_reference_minimap": 0,
    "reference_minimaps_generated": 0
  },
  "tiles": [
    {
      "tile_name": "Azeroth_32_48",
      "tile_json_path": "dataset/Azeroth_32_48.json",
      "source_minimap_path": "images/Azeroth_32_48.png",
      "source_minimap_exists": true,
      "alpha_atlas_path": "stitched/Azeroth_32_48_alpha_atlas.png",
      "alpha_atlas_exists": true,
      "shadow_map_paths": [
        "stitched/Azeroth_32_48_shadow.png"
      ],
      "completeness_class": "core-terrain-ready"
    }
  ]
}
```

### Image Signatures

The manifest records compact signatures for selected image families:

- `source_minimap_signature`
- `alpha_atlas_signature`
- `shadow_map_signature`

Each signature includes:

- `width`
- `height`
- `sha256`
- `average_hash64`

These are meant for dedupe, clustering, and coverage selection without forcing a separate post-process step.

## Recommended Consumer Strategy

For downstream ML or curation tooling, use this lookup order:

1. If `ml_dataset_manifest.json` exists, load it first for coverage, existence checks, and signatures.
2. Use each tile's `tile_json_path` to load the canonical `dataset/<tile>.json` payload.
3. Read asset paths from JSON or manifest rather than rebuilding names manually.
4. Treat `image`, `heightmap_local`, `heightmap_global`, `alpha_atlas`, `alpha_masks`, `shadow_maps`, `liquid_mask`, `liquid_height`, and `objects` as the main training-facing payloads.
5. Use raw chunk arrays such as `heights`, `chunk_layers`, `shadow_bits`, and `liquids` only when you need reconstruction-grade detail.

## Practical Notes For Training Tooling

- For rendered minimap supervision, use `image`.
- For terrain height targets, choose between `heightmap_local` and `heightmap_global` based on whether the model needs per-tile normalization or map-wide normalization.
- For terrain texture blending, use `alpha_atlas` for compact transport and `alpha_masks` when you need explicit per-layer grayscale maps.
- For shadow supervision, use `shadow_maps`, not the alpha atlas.
- For liquid supervision, use `liquid_mask` and `liquid_height` together with `liquid_min` and `liquid_max`.
- For object-aware training, use `objects` and their optional bounds.
- Do not assume `reference_minimaps/` or viewer validation outputs exist.

## Recommended Model Split

The exported dataset supports two separate model families and should not force
them back into one mixed contract:

1. Terrain reconstruction model:
  `image` + `normalmap` + `wdl_heights` + `liquid_mask` + `objects` + height bounds -> `heightmap_local` / `heightmap_global`
2. Texture decomposition model:
  `image` + palette context from `textures` / `chunk_layers` / `tilesets/` -> terrain texture assignments + `alpha_masks`
3. Shadow-scar object recovery model:
  `image` + `shadow_maps` / `shadow_bits` / `shadow_analysis` + surviving `objects` -> missing-object candidates and restored placement hypotheses

For the current development-map direction, the limited terrain-texture palette
available in the `4.0.0.11927` client is a practical first target for the
separate texture model.

The third model should not be framed as "predict every object from the
minimap." The more defensible target is narrower:

1. treat placed `objects` as the surviving positive object state
2. treat `MCSH`-derived shadows as evidence that an object used to exist or cast
   into that terrain region
3. learn the residual where a shadow imprint survives but no matching placement
   still exists

That residual is the useful recovery signal. A "shadow scar" is the footprint
or silhouette implied by `MCSH` that is not already explained by the current
MDDF/MODF placement set. In practice this gives a better problem statement for
missing-object reconstruction:

- input evidence: rendered minimap tile plus stitched/raw `MCSH` shadow
  payloads
- context: current surviving object placements and optional bounds/footprints
- target: one or more candidate missing placements, object-family labels, or
  footprint masks for objects that appear to have existed before being moved or
  deleted

Before training that model, we should derive a cleaner supervision layer from
the current exported payloads:

1. rasterize current object footprints from `terrain_data.objects`
2. compare them against `shadow_maps` and `shadow_bits`
3. mark unexplained persistent shadow regions as `shadow scar` candidates
4. cluster those candidates against nearby historical/similar object families
   so the model learns object-family recovery, not just generic dark blobs

The existing `shadow_analysis` field is the right staging surface for this. It
should become the durable handoff for shadow-to-object association features,
candidate masks, and later recovered-placement labels instead of leaving that
reasoning implicit in ad hoc notebooks.

## Quick Export Commands

```bash
cd src/WoWMapConverter/WoWMapConverter.Cli
dotnet run -- ml-export --client "H:\CLIENTS\World of Warcraft Cata beta 11927" --map LostIsles --out "I:\parp\parp-tools\datasets\4_0_0_11927\LostIsles"
dotnet run -- ml-harvest --dataset "I:\parp\parp-tools\datasets\4_0_0_11927\LostIsles" --output "I:\parp\parp-tools\datasets\4_0_0_11927\LostIsles\ml_dataset_manifest.json"
```

## Fixed-Client Corpus Wrapper

For the current fixed local client roots, use the checked-in wrapper instead of
retyping per-map commands:

```bash
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1
```

The wrapper reads [gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json](gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json), exports each configured map into `datasets/<label>/<map>/`, and runs `ml-harvest` to generate `ml_dataset_manifest.json` plus the root-level `metadata.jsonl` and `dataset_info.json` files for each dataset root. The current checked-in subset is intentionally narrow and now also includes the split-root `original_development` seam.

## Shadow-Scar Recovery Reference

For the object-recovery use of `MCSH`, see [gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md](gillijimproject_refactor/docs/SHADOW_SCAR_OBJECT_RECOVERY.md).

That note captures the actual third-model goal:

- `MCSH` is treated as historical object evidence, not just generic shadow supervision
- the target is the residual shadow not already explained by current placements
- retrieval from repeated object patterns elsewhere in the copied/pasted world is expected to be part of the solution
