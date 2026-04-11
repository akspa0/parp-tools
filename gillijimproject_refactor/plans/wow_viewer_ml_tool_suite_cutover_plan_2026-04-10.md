# wow-viewer ML Tool-Suite Cutover Plan

## Apr 10, 2026 - Direction Reset

The ML dataset and training workflow should stop being split across `WoWMapConverter`, ad hoc scripts, and `MdxViewer`-local tooling.

The new default direction is:

- `wow-viewer` becomes the canonical home for ML dataset gathering, corpus auditing, validation-capture orchestration, and training-workflow contracts.
- `MdxViewer` remains a transitional GUI host only where it already has working runtime/capture surfaces that are still needed before the `wow-viewer` app owns those workflows directly.
- Python training can remain Python for now, but it should move into the `wow-viewer` repo/tool surface and consume one first-party dataset contract instead of hanging off legacy repo-local exporter assumptions.

This is a repo-shape and ownership correction. It is not a claim that the current ML export path is already correct.

## Problem Statement

The current ML workflow is fragmented in ways that make it hard to reason about truth ownership:

- shared format reading is partly moving into `wow-viewer`, but real ML export still depends heavily on legacy `WoWMapConverter.Core.VLM`
- `MdxViewer` owns useful validation capture behavior, but not the actual dataset contract
- training scripts live in the legacy repo and silently depend on exporter quirks
- signal audits, corpus fixes, and training runs can drift because the data contract is not owned in one place

The immediate concrete evidence behind this direction is that recent V7 audits proved the active corpora were missing effective liquid/object supervision even while the model still expected those channels.

## Canonical Ownership

### `wow-viewer/src/core/WowViewer.Core`

Own shared ML-domain contracts only:

- dataset/tile manifest contracts
- per-tile signal descriptors
- signal-coverage and provenance contracts
- training-profile metadata contracts

### `wow-viewer/src/core/WowViewer.Core.IO`

Own format reading and signal extraction inputs:

- ADT/WDT/WDL family reading
- split ADT companion discovery (`_tex0.adt`, `_obj0.adt`, `_lod.adt`)
- terrain bake inputs and coherent tile reconstruction inputs
- liquid extraction (`MH2O`, `MCLQ`, related masks/heights)
- object placement extraction (`MDDF`, `MODF`, names, bounds)
- chunk-layer/alpha/shadow/MCCV/normal inputs

### `wow-viewer/src/core/WowViewer.Core.Runtime`

Own shared execution services that both CLI and GUI can call:

- dataset-build orchestration
- validation-capture job descriptions
- minimap/capture variant definitions (`default`, `noliquids`, later others)
- progress/cancellation/reporting contracts

### `wow-viewer/tools/converter/WowViewer.Tool.Converter`

Own the headless ML workflow surface:

- `ml-export` or successor command for real dataset emission
- `ml-corpus` for fixed-client batch orchestration
- `ml-harvest` for manifest generation
- `ml-audit-signals` for coverage/effective-activation audits
- `ml-validate-captures` or equivalent batch capture orchestration over shared services

### `MdxViewer`

Remain a transitional GUI host only:

- keep the existing `Build ML Dataset` panel and validation-capture UI while the new app surface is not ready
- stop owning parsing/export/business logic directly
- call shared wow-viewer services or command surfaces instead of duplicating ML logic locally

### Python training

Training should remain Python-based for now, but move under the `wow-viewer` tool/repo surface once the dataset contract is stable.

That means:

- do not try to port PyTorch training into the viewer app or .NET runtime first
- do move the trainers, profiles, audits, and inference helpers so they consume the same wow-viewer-owned dataset contract

The goal is one repo and one contract, not one language for everything.

## Non-Negotiable Boundaries

- Do not make `MdxViewer` the long-term owner of ML export logic again.
- Do not leave dataset gathering half in `WoWMapConverter.Core.VLM` and half in `wow-viewer`.
- Do not move training first while the dataset contract is still unstable.
- Do not merge terrain-height, texture-layer, and missing-object recovery into one giant tool or model family.
- Do not force GUI-only workflows for export/audit/harvest; headless batch execution still matters.

## Tool-Family Classification

### Rebuild as first-class wow-viewer tool family

- ML corpus export and harvest
- signal auditing / corpus validation
- validation minimap capture orchestration
- no-liquid minimap synthesis and related dataset transforms

### Dual-surface workflow

- dataset build / harvest / validate
- capture queue definition and output review
- per-map/per-build corpus selection

CLI owns headless batch execution.
GUI owns inspection, preview, and interactive job setup.
Both must call the same shared services.

### Keep transitional in `MdxViewer`

- orthographic validation capture flow
- existing ML dataset panel while the wow-viewer app surface is not ready
- viewer-side terrain/minimap preview used to inspect outputs

### Keep as reference-only during cutover

- legacy `WoWMapConverter.Core.VLM` exporter assumptions once equivalent wow-viewer services exist
- one-off dataset repair scripts that encode old file-layout assumptions

## What Actually Differs From The Old V7 Workflow

The useful old V7 idea was not "old code." It was a coherent supervision bundle:

- minimap appearance
- terrain normals
- WDL low-res prior
- known-loss masks for liquids and obscuring objects
- auxiliary supervision that helped shape/boundary learning converge fast

The current failure mode is mostly not the U-Net itself.
It is that the active dataset contract is fragmented enough that some of those signals silently died in real exported corpora.

The cutover plan should therefore prioritize restoring trustworthy signals, not inventing more exotic model inputs first.

## First Migration Wave

### Wave 1 - Stabilize the wow-viewer ML dataset contract

- create a first-class shared ML tile schema in wow-viewer
- port the current signal audit into wow-viewer tooling
- require explicit coverage reporting for all expected signal families before any training run is treated as valid

Exit condition:

- wow-viewer can say exactly which signals a corpus really has, with effective activation counts rather than only field presence

### Wave 2 - Port real signal extraction ownership into wow-viewer

- liquids from `MH2O` / `MCLQ`
- object placements and bounds from `MDDF` / `MODF`
- coherent terrain height/normal baking
- WDL prior extraction
- chunk-layer/MCCV/shadow signal ownership

Exit condition:

- a fresh wow-viewer-produced corpus contains live liquid/object signals where the source map data actually has them

### Wave 3 - Rehost the active MdxViewer ML workflow as a thin consumer

- `Build ML Dataset` in `MdxViewer` becomes a shared-service frontend
- validation capture requests are described in shared contracts instead of viewer-local exporter assumptions

Exit condition:

- `MdxViewer` no longer owns ML export logic; it only hosts interactive setup/preview on top of shared wow-viewer services

### Wave 4 - Move trainers under the wow-viewer repo surface

- relocate `train_v7.py`, `infer_v7.py`, texture trainers, audits, and related scripts/workflows
- keep Python runtime for now
- make them depend only on the wow-viewer dataset contract

Exit condition:

- one repo owns both dataset generation and training workflows, even if training stays Python-based

## Recommended Immediate Slice

Do this next before more model tuning:

1. Make wow-viewer own the ML signal audit command surface.
2. Port or repair liquid/object extraction in the wow-viewer-side dataset path.
3. Regenerate one narrow real corpus (`3.0.1.8303 Northrend` first).
4. Re-audit that corpus.
5. Only then rerun V7 training.

This keeps the next step bounded and directly addresses the proven current failure: dead known-loss channels in the real corpus.

## What Not To Waste Time On Yet

- porting PyTorch training into C#
- native-client memory hacks for minimap capture before the basic signal contract is stable
- inventing many new input channels before existing required channels are alive again
- broad repo-wide tool rewrites before the ML dataset contract and extraction services are canonical

## Longer-Range Option: Terrain-Only Render Teachers

Once the base signal stack is fixed, a useful follow-up is a derived teacher-image family generated from viewer/runtime terrain rendering without baked minimap textures.

Use that as auxiliary supervision first, not as a required inference-time input.

Good candidates:

- terrain-only orthographic render
- no-liquid terrain-only orthographic render
- hillshade/slope/curvature derived from ground-truth heightmaps

These can help teach shape without forcing the deployment pipeline to supply another fragile input surface.

## Success Criteria

The migration is working when all of the following are true:

- wow-viewer owns the ML dataset contract and extraction services
- MdxViewer is only a thin interactive host for the same services
- headless corpus export/audit/harvest still works
- training scripts consume the wow-viewer-owned contract instead of legacy exporter quirks
- signal audits show non-dead liquid/object supervision where source data supports it
- model-quality discussions can focus on real encoding/model tradeoffs instead of hidden corpus breakage