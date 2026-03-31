# VLM Dataset Reconstruction Plan

## Mar 31, 2026 - Source-Of-Truth Reset

- the active goal is not "train on whatever the exporter currently dumps".
- the active goal is to build a provenance-rich real-map dataset contract that can support a v7-like terrain reconstruction model for missing layers on damaged targets such as `development`.
- `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM/VlmDatasetExporter.cs` is already more capable than the current docs imply.
- current public continuity is stale because the old docs still talk like a narrower v6/v30 pipeline, while the exporter already emits additional real-data channels such as:
  - per-chunk heights
  - per-tile local and global heightmaps
  - MCNR-derived normals
  - MCCV-derived color maps
  - raw MCSH shadow bits
  - derived shadow-region / candidate-object summaries
  - per-chunk layer metadata and alpha-mask paths
  - liquids, objects, and WDL low-res heights
  - root-level binary tile output (`VLM1`, version `1`)

## Goal

Build an auditable dataset workflow that:

- uses real maps and real client data as supervision
- preserves per-tile provenance instead of flattening everything into anonymous training blobs
- separates observed ground-truth layers from generated or fallback layers
- lets future training runs target specific missing-layer reconstruction tasks without pretending the whole map is already fully observed

## Current Exporter Boundary

The active exporter in `WoWMapConverter.Core.VLM` already owns these real seams:

- Alpha monolithic WDT/WDT.MPQ input handling
- LK/Cata split-ADT handling with `_tex0` and `_obj0`
- shared archive access through `WowViewer.Core.IO`
- WDL low-res height extraction
- M2/WMO placement extraction with cached bounds lookup
- stitched shadow/alpha/liquid outputs
- global-height post-pass that rewrites each tile JSON with a map-wide height range

This means the next problem is not "invent dataset channels". The next problem is to make the active channels trustworthy, versioned, and curated.

## Non-Negotiable Rules

- train on real exported maps from real clients; do not use synthetic supervision as the main teacher corpus
- treat `test_data/development/World/Maps/development` as the reconstruction target and evaluation corpus, not as the only training ground-truth source
- do not treat `test_data/WoWMuseum/335-dev/World/Maps/development` as canonical training truth for the damaged `development` map; it is reference-only
- do not mix Alpha, Wrath, and Cataclysm-like exports into one undifferentiated corpus without explicit profile tags and split-aware validation
- do not claim a "v7" dataset exists just because the exporter writes files; the schema must be explicit and the curation rules must be documented

## Proposed V7-Like Dataset Contract

Every exported map dataset should carry a map-level manifest plus per-tile provenance.

### Map-level manifest

At minimum:

- dataset schema version
- exporter build/version identifier
- source client version/profile
- source map name and resolved map directory
- export timestamp
- tile counts: discovered/exported/skipped
- channel coverage counts per modality
- map-wide height bounds
- texture database summary
- notes about any fallback path used during export

### Per-tile contract

Keep the current observed channels, but add provenance fields that make them usable for reconstruction work:

- source file presence:
  - root ADT present
  - `_tex0` present
  - `_obj0` present
  - WDL present
  - WL-present / liquid source present where relevant
- observed channels:
  - minimap image
  - local/global heightmap
  - normal map
  - MCCV map
  - alpha masks
  - shadow bits and shadow analysis
  - liquids
  - chunk layers
  - objects
  - WDL heights
- channel status flags per modality:
  - observed-from-source
  - stitched-from-source
  - generated-from-fallback
  - missing

## Reconstruction Strategy

Do not start with one monolithic "solve every missing layer" model.

Use a staged dataset strategy:

1. complete-map teacher corpora
   - export fully observed real maps from matching client profiles
   - use these as supervision for channels that are genuinely present on disk

2. damaged-map target corpora
   - keep `development` as the main reconstruction target
   - use it to measure how well the pipeline handles partial observation and fallback tagging

3. task-specific targets
   - terrain-layer reconstruction: alpha/liquids/shadows/MCCV
   - geometry-conditioned reconstruction: local/global height + normals + WDL
   - object-context enrichment: shadow/object association and placement-aware context

This is the practical path to a v7-like model. First stabilize the data contract, then train per-task baselines, then decide whether one larger multimodal model is justified.

## Immediate Missing Pieces

The current exporter/docs gap is concrete:

- `docs/VLM_Training_Guide.md` still frames the workflow around an older v6-style dataset/training story
- `docs/VLM_DATASET_EXPORTER.md` does not document newer channels like `shadow_analysis`, `mccv_map`, binary tiles, or the global-height rewrite pass
- there is no map-level manifest describing dataset schema, coverage, and provenance
- there is no explicit curation step that classifies tiles by completeness before training

## Minimal Next Implementation Slice

The next production-worthy slice should be:

1. add a map-level `manifest.json` to `VlmDatasetExporter`
2. add explicit per-tile provenance / completeness flags to `VlmTerrainData`
3. add a curation script that builds train/val/test splits by map/profile from real exported datasets
4. update the docs to match the actual exporter schema after the manifest lands

That slice is higher value than immediately changing model code because it creates auditable data quality gates.

## Real-Data Validation Standard

For VLM dataset work, validation must be stated at three levels:

- build validation
- real export validation on at least one complete real map and one damaged/partial target map
- curation/manifest validation that counts channels and flags missing/generated data correctly

Do not describe a dataset as "ready" based only on exporter compile success.

## First Candidate Corpora

- reconstruction target:
  - `test_data/development/World/Maps/development`
- in-repo reference/evidence:
  - `test_data/0.5.3/alphawdt/World/Maps/*`
- future teacher corpora:
  - real full-map exports from actual client installs run through `vlm-export`, tagged by profile/version and kept separate by dataset root

## What Not To Waste Time On Yet

- generic VLM finetuning tweaks before schema/provenance are frozen
- synthetic-only terrain targets as a substitute for real map exports
- mixing museum/reference outputs into the teacher corpus without explicit provenance tags
- claiming a single giant model should replace dataset audit and curation