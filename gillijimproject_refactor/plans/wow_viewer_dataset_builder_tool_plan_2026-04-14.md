# wow-viewer Dataset Builder Tool Plan

## Apr 20, 2026 - landed the first direct wow-viewer multimap or multibuild composition seam and wrapper

- status: partial execution update
- landed surface:
  - `WowViewer.Tool.Converter dataset-merge --input <manifest.json> ... --output <merged.json>` now exists as the first direct manifest-composition step for shared terrain-training manifests inside `wow-viewer`
  - the merge command rejects mixed schema or manifest-kind inputs and duplicate `SampleId` collisions instead of depending on ad hoc external JSON stitching
  - `wow-viewer/scripts/run_v9_direct_pipeline.ps1` now exists as the first `wow-viewer`-owned direct wrapper over `dataset-scan`, `dataset-merge`, `dataset-audit`, `dataset-curate`, `dataset-build-cache`, and downstream `train_v9.py`
  - the wrapper defaults `IncludeBuilds` to `0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_3_5_12340`, and `4_0_0_11927`, with early builds expected from staged roots under `output/tmp/wowarchive-clients`
- current proof:
  - `dataset-merge` merged two bounded direct `3.3.5.12340` scan manifests (`Azeroth` plus `EmeraldDream`) into one `terrain-training-scan.v2` manifest, and `dataset-audit` consumed that merged result without special handling
  - `run_v9_direct_pipeline.ps1` completed end to end on the fixed local `3.3.5.12340` root in bounded audit mode, scanning one tile per default map, merging `8` scans, auditing `4` samples, curating `1`, building `1` cache shard, and finishing `train_v9.py --audit-only` with `rejected = 0`
- current guardrail:
  - do not describe this as proof that early-build direct reads are closed; `0.5.x` and `0.7.0` are only wired into the wrapper defaults so far because the staged client roots were not present during validation
  - do not describe this as proof that default minimap or WDL gates are closed across the broad direct workflow; the bounded wrapper proof still ran with `--no-require-minimap --no-require-wdl`

## Apr 20, 2026 - first direct wow-viewer curate plus build-cache smoke is now landed

- status: partial execution update
- landed surface:
  - `WowViewer.Tool.Converter dataset-curate --input <audit.json> --output <curated.json> ...` now exists as the first direct curation step over audited manifests
  - `WowViewer.Tool.Converter dataset-build-cache --input <audit-or-curate.json> --output-dir <dir> ...` now exists as the first direct cache-materialization step for v9-style training shards and per-tile debug JSON sidecars
  - the direct cache now writes the core trainer-required arrays from shared terrain/liquid seams without routing through harvested dataset-folder ownership
- current proof:
  - bounded real-data smoke on the fixed local `3.3.5.12340` Azeroth sample (`2` tiles) completed end to end through `dataset-curate`, `dataset-build-cache`, and `train_v9.py --audit-only --no-require-minimap --no-require-wdl`
  - the resulting direct cache manifest reported `processed = 2`, `skipped = 0`, and the trainer audit reported `rejected = 0`
- current guardrail:
  - do not describe the current direct cache slice as proof that archive-backed minimap sourcing is closed; the smoke tiles still had `has_minimap_rgb_256 = false`
  - do not describe the current direct cache slice as proof that WDL alignment is closed; `wdl_17` is now intentionally withheld from the direct cache until the shared terrain/WDL absolute-height seam is validated
  - treat this as the first working direct cache baseline, not the end of auxiliary-signal recovery

## Apr 19, 2026 - direct game-root-to-curated-cache ML pipeline directive

- status: active ownership and workflow directive
- user direction:
  - the ML path should not require a separate harvested dataset tree as the normal starting point
  - dataset and training prep should run as one direct pipeline from a real game root through shared `wow-viewer` loaders, with curation and final cache materialization happening inside that pipeline instead of ahead of it
- immediate implication:
  - treat pre-harvested dataset roots under `datasets/` as compatibility or inspection artifacts, not as the canonical long-range ML starting surface
  - the canonical future shape is direct client-root or staged-client discovery, raw signal sampling, audit buckets, curation, and then final curated cache writing
  - new dataset-builder and training-orchestration work should bias toward a direct shared-reader command surface such as `scan`, `audit`, `curate`, and `build-cache` instead of another export-first bridge script
- continuity:
  - use `gillijimproject_refactor/plans/wow_viewer_direct_ml_pipeline_plan_2026-04-19.md` as the focused plan for this direct pipeline cutover

## Apr 20, 2026 - first direct wow-viewer dataset-audit slice is now landed, but build-cache is still open

- status: partial execution update
- landed surface:
  - `WowViewer.Tool.Converter dataset-scan` remains the direct discovery step for client-root or archive-root tiles
  - `WowViewer.Tool.Converter dataset-audit --input <scan.json> ...` now exists as the first direct post-scan audit step over shared terrain/liquid readers
  - the current audit slice recomputes real terrain height min/max/range, MH2O-derived liquid coverage, bounded MCLQ fallback coverage, and hole coverage from root ADT data instead of using the old `HasWater ? 1 : 0` placeholder
- current guardrail:
  - treat the new audit command as proof of direct raw-signal sampling only for terrain/liquid/hole availability and coverage
  - do not describe the current audit slice as final WDL-alignment proof; WDL tile availability is surfaced, but WDL delta metrics remain intentionally withheld until the shared terrain/WDL absolute-height seam is validated
  - do not describe this as `dataset build-cache` ownership; cache materialization still remains the next major implementation gap after audit

## Apr 14, 2026 - Dataset-builder convergence directive

- status: active ownership and workflow directive
- user directive:
  - all shared dataset or export or terrain-supervision logic should converge into `wow-viewer`
  - the long-range dataset-builder surface should be a new `wow-viewer` tool, not more architecture inside `WoWMapConverter`
- immediate implication:
  - stop treating `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM` as the design owner for new shared dataset-builder behavior
  - use `WoWMapConverter` and `MdxViewer` exporter code as extraction or compatibility references only, except for bounded hotfixes explicitly requested in the legacy path

## Canonical Target Shape

- shared contracts in `wow-viewer/src/core/WowViewer.Core`
- shared readers or writers or artifact builders in `wow-viewer/src/core/WowViewer.Core.IO`
- thin shared tool helpers in `wow-viewer/src/tools-shared/WowViewer.Tools.Shared` only where cross-tool orchestration really belongs there
- dedicated dataset-builder tool in `wow-viewer/tools/` as the long-range owner of corpus export and manifest packaging

## Target User-Facing Surfaces

1. Shared `wow-viewer` library seams for format decode, artifact contracts, manifests, and training metadata.
2. CLI command surface for corpus export or validation or curation or training-orchestration entrypoints.
3. Viewer or editor workflows for inspection, correction, supervised labeling, and artifact review.
4. Dataset explorer surface for browsing tiles, masks, metadata, labels, and provenance.
5. Local supervised training tooling that orchestrates training on the user’s own hardware without shipping first-party model outputs.

## Shared Seams That Must Converge

1. Tile discovery and archive or loose asset resolution for dataset jobs.
2. ADT or WDT or WDL terrain sampling and split-ADT family ownership.
3. Shared terrain artifact contracts:
   - heightmaps
   - normalmaps
   - MCCV maps
   - alpha or shadow masks and atlases
   - liquid masks and liquid heights
   - object and PM4 masks
   - cleaned minimap variants such as no-liquid, no-MCCV, no-object, and terrain-only outputs

Cross-build expectation for object and PM4 masks:
- for the current fixed local clients, treat `3.3.5.12340` and `4.0.0.11927` as the paired minimum target builds for object-mask and other footprint-derived artifact work
- do not assume `4.0.0` mask behavior is automatically identical just because the current `M2` footprint extraction is shared-format; require a bounded proof on each root before calling the seam closed
4. Dataset JSON or manifest or metadata ownership.
5. Batch corpus orchestration and resumable tile-map export semantics.
6. Dataset explorer indexing, filtering, preview, and provenance ownership.
7. Training manifests, experiment configs, labels, and local orchestration metadata.

## Bring Your Own Data And Distribution Policy

- The tooling must stay Bring Your Own Data.
- Do not ship copyrighted client data, harvested corpora, trained model weights, or model outputs derived from proprietary data.
- The distributable product is the toolchain, configs, schemas, manifests, labels, and validation flow, not precomputed outputs from local corpora.
- The goal is reproducibility on the user’s own hardware and data roots whenever practical, not redistribution of local results.

## Compute Backend Policy

- Do not make CUDA the only architectural assumption for long-range training or inference tooling.
- Keep backend-specific launch or tensor-runtime code behind a shared orchestration seam so alternate local runners can be added later.
- Current candidate backends worth preserving in the architecture discussion are CUDA, Vulkan, OpenCL, and MLX for macOS users.
- CPU-only fallback or validation paths are still useful for correctness and portability even when they are not the primary performance target.

## Non-Negotiable Guardrails

- No new shared dataset-builder capability should default to `WoWMapConverter.Core/VLM` when the ask is really about long-range ownership or new shared behavior.
- Training and inference scripts stay downstream consumers of exported artifacts; they do not own format decode, packing, or manifest semantics.
- `MdxViewer` and `WoWMapConverter` remain valid extraction-reference inputs for parity recovery, but they are not the final home for new shared artifact semantics.
- Do not maintain two permanent first-party dataset-builder pipelines after the new `wow-viewer` tool exists.
- Dataset explorer and supervised training tooling belong to the same shared-contract family as export and curation; they should not become a second disconnected ownership track.
- No workflow plan should assume the repo can distribute a ready-made model or proprietary output bundle.

## First Migration Wave

1. Define a shared `wow-viewer` terrain-dataset contract for one tile and one corpus row.
2. Move the current artifact-semantics hotspots into shared `Core` or `Core.IO` seams:
   - alpha or shadow or atlas packing rules
   - minimap cleanup and mask union rules
   - loose-vs-archive override rules
3. Build the first thin `wow-viewer` dataset command over those seams for bounded real-data export.
4. Add the first dataset-explorer metadata index surface over the same shared contracts instead of inventing a second viewer-only schema.
5. Define the first backend-agnostic training-run manifest shape before any new local launcher claims long-range ownership.

## Validation Standard

- `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
- real-data validation on the fixed client roots and dataset roots from `gillijimproject_refactor/memory-bank/data-paths.md`
- when the slice touches object masks, PM4 masks, no-object cleanup, or geometry-derived footprint extraction, include both the fixed `3.3.5.12340` and fixed `4.0.0.11927` roots in the bounded real-data proof unless the user explicitly narrows the build target
- do not describe a build or a narrow test pass as full dataset-builder cutover proof without regenerated real artifacts
- do not describe a launcher or explorer build as distribution proof for models or outputs; the policy target is local reproducibility, not shipped results

## What This Plan Does Not Claim Yet

- no full `wow-viewer` dataset-builder tool implementation is landed yet beyond the current direct `dataset-scan` plus `dataset-audit` CLI slices
- no shared dataset contract has been fully cut over yet
- no parity claim is made for the current legacy VLM exporter beyond existing bounded proofs
- no `wow-viewer` dataset explorer or training orchestration surface is landed yet
- no backend-portable runner abstraction is implemented yet