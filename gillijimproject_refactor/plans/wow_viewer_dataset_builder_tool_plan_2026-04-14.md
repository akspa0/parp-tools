# wow-viewer Dataset Builder Tool Plan

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

- no `wow-viewer` dataset-builder tool implementation is landed yet
- no shared dataset contract has been fully cut over yet
- no parity claim is made for the current legacy VLM exporter beyond existing bounded proofs
- no `wow-viewer` dataset explorer or training orchestration surface is landed yet
- no backend-portable runner abstraction is implemented yet