---
description: "Plan the new wow-viewer dataset-builder tool and the cutover of shared dataset/export and supervised-training logic out of legacy tooling. Use when the task is ML corpus export ownership, terrain-supervision artifact generation, dataset explorer design, supervised training-tooling ownership, dataset manifest or harvest cutover, minimap cleanup or mask-stitch semantics, BYOD distribution policy, or deciding what shared data contracts must move into wow-viewer before replacing WoWMapConverter VLM tooling."
name: "wow-viewer Dataset Builder Plan"
argument-hint: "Describe the dataset-builder seam, artifact family, or legacy exporter path to converge"
agent: "agent"
---

Design the first-class `wow-viewer` dataset and supervised-training tool stack and the shared-library cutover it depends on.

## Read First

1. `gillijimproject_refactor/memory-bank/activeContext.md`
2. `gillijimproject_refactor/memory-bank/progress.md`
3. `gillijimproject_refactor/plans/wow_viewer_shared_io_library_plan_2026-03-26.md`
4. `gillijimproject_refactor/plans/wow_viewer_dataset_builder_tool_plan_2026-04-14.md`
5. `wow-viewer/README.md`
6. `.github/copilot-instructions.md`
7. `.github/prompts/wow-viewer-tool-inventory-cutover-plan.prompt.md`
8. `.github/prompts/wow-viewer-shared-io-library-plan.prompt.md`

## Goal

Define the `wow-viewer`-owned dataset-builder stack so dataset export, terrain-supervision artifacts, dataset exploration, and supervised training orchestration stop depending on `WoWMapConverter` or `MdxViewer` as long-term owners.

## Non-Negotiable Constraints

- Do not keep adding new shared dataset/export logic to `gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Core/VLM` when the ask is architectural cleanup or new capability.
- Shared data contracts and artifact builders must live in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO` before a tool surface claims ownership.
- The dataset-builder should be a dedicated `wow-viewer` tool over shared services, not another legacy sidecar.
- `WoWMapConverter`, `MdxViewer`, `parpToolbox`, and older scripts are extraction or compatibility references only unless the user explicitly asks for a bounded legacy hotfix.
- ML training and inference scripts are downstream consumers; they do not own format decode, artifact packing, or manifest semantics.
- The long-range user-facing surfaces should be explicit: shared library, CLI, viewer/editor workflows, dataset explorer, and supervised training tooling.
- The workflow must be Bring Your Own Data. Do not assume the tool can ship copyrighted source data, harvested corpora, trained models, or model outputs.
- Prefer reproducible local-run configs and metadata over redistributed outputs.
- Do not assume CUDA is the only long-range backend. Call out where backend seams should stay open for Vulkan or OpenCL or MLX or other local runners.

## Shared Surfaces To Classify Explicitly

- ADT and WDT and WDL access and tile discovery
- split ADT companion ownership (`_tex0.adt`, `_obj0.adt`, `_lod.adt`)
- heightmaps, normalmaps, MCCV, alpha and shadow, liquid, and object or PM4 mask artifacts
- minimap cleanup variants such as no-liquid, no-MCCV, no-object, and terrain-only outputs
- archive-backed versus loose override resolution
- dataset JSON contracts, manifests, metadata rows, and harvest packaging
- corpus batch orchestration and resumable tile-map jobs
- dataset explorer indexing, preview, filtering, and provenance surfaces
- supervised labels, training manifests, experiment configs, and local launch metadata
- backend-runner abstraction for local training or inference orchestration

## What The Plan Must Produce

1. The target module tree in `wow-viewer`.
2. The boundary between shared libraries and the new CLI or viewer/editor or explorer or training-tool surfaces.
3. The artifact ownership matrix.
4. The absorb vs rewrite vs reference-only map for legacy exporter code.
5. The first vertical slice that proves the cutover is real.
6. The validation plan using real fixed client roots and dataset outputs.
7. What not to port or not to deepen in the legacy repo.
8. The BYOD and no-model-distribution policy surface.
9. The backend portability seams for local training or inference.

## Deliverables

Return all items:

1. target `wow-viewer` module or tool layout
2. shared-contract ownership matrix
3. legacy exporter cutover map
4. first migration wave
5. highest-risk artifact seams
6. real-data validation plan
7. explicit no-go list for legacy ownership
8. BYOD and redistribution policy notes
9. backend portability strategy

## First Output

Start with:

1. the exact dataset-builder problem you think the user is trying to solve
2. the shared data seams that must move first
3. the proposed `wow-viewer` tool boundary across CLI or viewer/editor or explorer or training surfaces
4. the legacy exporter code that should become reference-only first