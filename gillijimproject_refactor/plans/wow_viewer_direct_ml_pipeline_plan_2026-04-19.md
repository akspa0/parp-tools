# wow-viewer Direct ML Pipeline Plan

## Intent

- stop treating harvested dataset trees as the required starting point for ML training
- move the canonical terrain-training pipeline into `wow-viewer` so it can run from a real game root to a final curated training cache without a separate manual export phase
- keep training caches and curated manifests as valid outputs, but make them the result of the pipeline instead of a prerequisite for it
- bias the whole lane toward raw terrain semantics and shared format decode, with raster images only as auxiliary signals where they still add value

## Why This Is The Right Boundary

- the current `v9` lane already proved that native terrain targets are stronger than image-derived terrain truth
- the current bottleneck is architectural: training still depends on pre-harvested dataset roots and legacy bridge scripts even though `wow-viewer` already owns the shared file-access direction we want long-term
- game assets are not natively image datasets; the actual signal lives in ADT/WDT/WDL/liquid/object/texture/placement payloads
- if the loader and dataset contracts stay separate forever, every future model lane will keep paying a duplicated export-and-cleanup tax before training can even start

## Core Recommendation

- the canonical ML pipeline should be:
  - game root or staged client root
  - shared `wow-viewer` discovery and sampling
  - first-pass bucket or audit artifacts
  - second-pass curation
  - final curated cache manifest plus tensor shards or JSON rows
  - training or inference consumers

- do not require a permanent intermediate harvested dataset tree as the normal path
- keep a materialized curated cache as the durable training surface because reproducibility and resume semantics still matter
- treat exploratory exports under `datasets/` as compatibility or inspection artifacts, not as the permanent architectural center of the ML system

## What Should Become Canonical

### Source Surface

- fixed local client roots under `H:\CLIENTS\...`
- staged WoWArchive client copies under `output/tmp/wowarchive-clients/`
- optional loose overlay roots for development-map or patched-map workflows

### Shared Decode Layer

- `wow-viewer/src/core/WowViewer.Core`
- `wow-viewer/src/core/WowViewer.Core.IO`
- runtime-facing terrain or liquid or placement seams that can already read real game assets directly

### Dataset Tool Owner

- a dedicated `wow-viewer` dataset-builder tool under `wow-viewer/tools/`
- the desktop `Dataset Tooling` workspace should become a consumer of that tool, not a launcher for legacy export-first scripts

## Proposed Pipeline Shape

### Phase 1 - Direct Tile Discovery

- input:
  - client root or archive root plus build label
  - map filter or tile filter
  - optional loose overlay root
- output:
  - stream or manifest of candidate tiles with direct provenance back to shared readers

- this phase should answer:
  - which maps and ADTs exist
  - which companion files exist (`_obj0`, `_tex0`, `_lod`, WDL, minimaps)
  - which build or root or overlay actually supplied each tile

### Phase 2 - Raw Signal Sampling

- sample chunk and tile signals directly from shared readers:
  - native terrain heights
  - WDL priors
  - liquid masks and liquid heights
  - hole masks
  - chunk flags and area ids
  - object and WMO placement footprints
  - PM4 masks when available
  - texture-layer metadata and alpha masks when the consumer lane needs them
  - cleaned minimap variants only as auxiliary guidance, not as the canonical truth surface

- this phase should not require pre-written dataset JSON per tile just to move data between steps

### Phase 3 - Bucketed Audit Pass

- write lightweight per-tile or per-chunk audit rows first, not final tensors immediately
- bucket useful statistics for later curation:
  - height range
  - liquid coverage
  - object coverage
  - hole coverage
  - texture-layer count
  - minimap gradient or variance when minimap is present
  - WDL-vs-native divergence
  - build and map family

- this is the scientific pass the current workflow is missing: the pipeline should rank and stratify samples from raw signals before expensive cache writes

### Phase 4 - Curation Pass

- select the final training pool from audit buckets using explicit rules:
  - diversity by map and tile region
  - bounded liquid-heavy and object-heavy coverage
  - bounded flat or low-signal rejection
  - optional cohort-size targets

- write a durable curated manifest only here
- this curated manifest becomes the stable training input for resume and experiment reproducibility

### Phase 5 - Curated Cache Materialization

- materialize only the curated subset into final cache artifacts:
  - compact JSONL or manifest rows
  - tensor shards or NPZs
  - optional preview or debug assets

- this keeps training fast without forcing the whole repo to live on giant rasterized dataset exports forever

### Phase 6 - Training And Inference Consumers

- training scripts consume the curated cache manifest and tensor shards
- inference tools consume the same shared contracts for feature parity and easier ablation work
- the cache format can evolve per model family, but the upstream direct-sampling pipeline stays shared

## V9.2 Recommendation

- treat `v9.2` as the first lane designed around this direct pipeline target even if the first implementation still keeps a transitional cache writer
- keep the model grounded in raw terrain math first:
  - native height targets
  - WDL priors
  - liquid masks and heights
  - object footprints and recovery masks
  - holes and chunk semantics

- keep minimap or normalmap usage auxiliary and explicitly masked where liquids or objects contaminate the visible surface
- do not frame `v9.2` as another image-to-image experiment; frame it as a structured terrain-signal model with optional visual side channels

## Proposed wow-viewer Command Surface

- `dataset scan --client-root <path> --build <label> --map <name|id> [--loose-overlay-root <path>]`
  - discovers candidate tiles directly from shared readers
- `dataset audit --input <scan-manifest> --output <audit.jsonl>`
  - computes raw tile and chunk metrics for curation
- `dataset curate --input <audit.jsonl> --output <curated-manifest.json>`
  - selects the final training population
- `dataset build-cache --input <curated-manifest.json> --output <cache-root>`
  - writes model-family-specific tensor caches only for the curated subset
- later optional convenience command:
  - `dataset pipeline ...`
  - runs scan plus audit plus curate plus cache-build end to end for users who want one command

## Transitional Rule

- keep current `gillijimproject_refactor` scripts working as compatibility consumers while the shared wow-viewer path is built
- do not deepen the legacy export-first pipeline as the design owner for new ML work
- whenever a new shared sampling or artifact rule is needed for training, land it in `wow-viewer` shared code first and let the legacy scripts consume it temporarily only if needed

## Validation Standard

- prove direct sampling on fixed local roots before calling the pipeline change real:
  - `H:\CLIENTS\WoW335\3.X_Retail_Windows_enUS_3.3.5.12340\World of Warcraft`
  - `H:\CLIENTS\World of Warcraft Cata beta 11927`
- when wider build coverage is needed, stage archive-backed roots locally under `output/tmp/wowarchive-clients/`
- do not claim the direct pipeline is closed until it can:
  - discover tiles from a client root
  - audit and curate them
  - emit a final curated cache without requiring a pre-existing harvested dataset tree

## Immediate Implementation Order

1. define a shared `wow-viewer` terrain-training sample contract over direct client-root reads
2. add the first dataset `scan` and `audit` commands in `wow-viewer`
3. build the curation manifest writer over audit buckets
4. add a `build-cache` step that emits the current `v9`-style native tensor shards for the curated subset only
5. cut the current `v9` trainer over to that curated cache contract
6. only after that, collapse the shell scripts and desktop tooling onto the new direct pipeline entrypoint

## Recommendation

- yes, the pipeline should move to direct game-path-to-cache ownership in `wow-viewer`
- no, the long-range architecture should not require permanent harvested datasets as the normal starting point
- the right scientific shape is direct shared decoding, bucketed audit passes, explicit curation, and only then a materialized training cache for reproducibility and speed