# ML Dataset Grounding And Harvest

This document explains what grounds the active terrain model in real data, how the dataset is harvested, and which channels are trusted today.

The short version is simple:

- the training corpus is harvested from real WoW client data and checked-in real map roots
- the GAN is a training-time refinement objective, not a data generator for the corpus
- deterministic cleanup and mask-generation steps are allowed only when their inputs are themselves harvested from real data
- the brush channel is active and trusted enough to document as part of the real dataset contract
- the prefab channel is not trusted enough yet to present as active supervision and is intentionally deferred from the public grounding story

## Why This Document Exists

One of the easiest ways to misread this project is to assume the terrain model is being trained on synthetic or GAN-invented pairs.

That is not the intended workflow.

The active dataset policy is:

1. harvest supervision from real client or map data first
2. preserve that evidence in traceable dataset roots
3. derive only deterministic helper channels from those harvested assets
4. train on that corpus
5. treat GAN output as a training objective or refinement aid, not as a source of new truth labels

If a channel cannot be traced back to a real exported tile or a deterministic transform of real exported tile assets, it should not be described as grounding the model in reality.

## Real Data Sources

The active corpus is built from real client roots and one checked-in real split-root development seam.

Current fixed roots are defined in `gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json` and currently include:

- `original_development` via checked-in terrain and explicit minimap root
- `0.5.3.3368`
- `0.5.5.3494`
- `0.6.0.3592`
- `0.7.0.3694`
- `3.0.1.8303`
- `3.3.5.12340`
- `4.0.0.11927`

The output corpus lives under `datasets/`.

For archive-backed clients, the mounted WoWArchive surface is treated as a source surface only. Heavy export work should stage a local working copy first instead of streaming directly from the mount.

Current archive workflow:

- mount source: `G:\WoW\WoWArchive-0.X-3.X\Mount`
- mount entrypoint: `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`
- staging root: `i:/parp/parp-tools/output/tmp/wowarchive-clients`

This matters because the provenance claim is stronger when the export path is explicit:

- fixed local root
- or mounted archive source plus staged local copy

## Harvest Flow

The active dataset build is a staged harvest pipeline, not a one-shot opaque export.

### 1. Resolve The Client Root

Use the fixed-config wrapper when possible:

```powershell
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1 -DryRun
pwsh ./gillijimproject_refactor/scripts/export_ml_corpus.ps1
```

That wrapper reads `gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json` and does the following:

- prefers `local_client_path` when it already exists
- otherwise stages `archive_client_path` from the mounted archive into the configured staging root
- uses `all_maps: true` roots for full map discovery
- writes outputs under `datasets/<label>/<map>/`

There is also a direct CLI path for the same workflow shape:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-corpus --config i:/parp/parp-tools/gillijimproject_refactor/scripts/ml_corpus_fixed_clients.json
```

### 2. Discover Maps

When a client entry uses `all_maps: true`, the workflow discovers maps from the real client root:

```powershell
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- ml-list-maps --client H:\053-client
```

This is still grounded in the real client layout. No synthetic map list is invented by the trainer.

### 3. Export Per-Tile Dataset Assets

The exporter writes the canonical per-tile JSON and image payloads:

```powershell
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- ml-export --client H:\053-client --map Azeroth --out i:/parp/parp-tools/datasets/0_5_3_3368/Azeroth
```

The canonical source of truth is the tile JSON under `dataset/`.

The exporter also writes the derived image families that later training and audit steps use, including heightmaps, normal maps, liquid maps, stitched alpha/shadow surfaces, cleaned minimaps, and object masks.

### 4. Harvest Dataset Metadata

After export, run `ml-harvest` so the dataset root gets traceable metadata surfaces:

```powershell
dotnet run --project i:/parp/parp-tools/gillijimproject_refactor/src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- ml-harvest --dataset i:/parp/parp-tools/datasets/0_5_3_3368/Azeroth
```

That writes:

- `ml_dataset_manifest.json`
- `metadata.jsonl`
- `dataset_info.json`

Those files are how readers and downstream tooling can inspect what was actually harvested without reverse-engineering filenames.

### 5. Harvest Brush Evidence

Brush harvesting is a second pass over already-exported real tiles, not a separate synthetic generator.

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-harvest-brushes --dataset-root i:/parp/parp-tools/datasets/original_development/development --output-dir i:/parp/parp-tools/output/build-validation/brush-imprints/original_development --write-previews
```

`ml-harvest-brushes` reads the exported tile JSON and heightmaps, identifies patch-scale terrain edits, groups them, and writes:

- `brush_imprints/brush_imprint_manifest.json`
- `brush_imprints/groups/*.json`
- `brush_imprints/tile_masks/*_brush_mask.png`

This is why the brush channel is still part of the real-data story: it is a deterministic analysis pass over harvested terrain, not a guessed semantic object library.

### 6. Audit Before Training

The audit stage is there to check coverage and spot broken or missing channels before the model sees the corpus:

```powershell
dotnet run --project i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- ml-audit-signals --dataset-root i:/parp/parp-tools/datasets/3_0_1_8303/Northrend --output i:/parp/parp-tools/output/build-validation/ml-audit/northrend_signal_audit.json --limit 32
```

## Current Channel Provenance

The active V7.5.1 model consumes `13` input channels. Not every exported channel is a direct model input, so it helps to separate active inputs from supporting derived assets.

### Active V7.5.1 Input Channels

| Channel family | How it is harvested or derived | Current role |
| --- | --- | --- |
| `terrain_only_minimap` / `no_object_minimap` / `no_mccv_minimap` / raw `image` RGB | Starts from a real exported minimap tile and may be deterministically cleaned with real exported masks | Primary RGB evidence |
| `normalmap` RGB | Deterministically rendered from real ADT terrain heights | Local surface-shape cue |
| `wdl_heights` prior | Exported from real WDL data when present | Low-resolution global terrain prior |
| `height_min` / `height_max` hint masks | Scalar bounds from the real tile height range | Height decoding context |
| `liquid_mask` | Exported from real liquid payloads | Water or liquid occupancy cue |
| `liquid_height` | Exported from real liquid heights | Water surface prior |
| `object_visibility_mask` | Built from real object placements plus real model geometry or bounds when available | Object-occlusion context |
| `brush_mask_path` | Built by `ml-harvest-brushes` from real exported heightmaps | Terrain-edit context |

### Real Exported Supporting Channels

These are not all direct inputs to the current terrain model, but they are still grounded in the export pipeline and matter for cleanup, auditing, or future supervision work.

| Channel family | How it is harvested or derived | Current role |
| --- | --- | --- |
| `heightmap_local` / `heightmap_global` | Deterministically rendered from real ADT heights | Current targets for V7.5.1 |
| `alpha_masks` / `alpha_atlas` | Exported from real terrain texture-layer data | Cleanup, texture supervision, audit |
| `mccv_map` | Exported from real MCCV bytes | Cleanup and tint analysis |
| `shadow_maps` | Exported from real shadow payloads | Diagnostic and audit only for terrain-only cleanup; not currently removed from `terrain_only_minimap` |
| `pm4_mask` | Exported from real PM4 overlays when available | Cleanup and later collision-context work |
| `objects` | Real MDDF/MODF object placements with bounds when available | Object analysis and mask construction |

## What The GAN Does And Does Not Do

The GAN belongs to training, not to corpus creation.

What it does:

- adds adversarial pressure during selected training epochs
- helps sharpen local detail when the base geometry path is already learning something real
- participates in model optimization only after the exporter has already produced the dataset

What it does not do:

- invent dataset tiles
- create source minimaps
- create ground-truth heightmaps
- replace exported object or liquid evidence
- justify calling a synthetic output a harvested channel

The current project should therefore be described as training on harvested real data with optional GAN-assisted refinement, not as using a GAN to manufacture the corpus.

## Deterministic Derived Channels Are Allowed

Some active channels are not raw client captures. They are deterministic transforms over harvested evidence.

Examples:

- `terrain_only_minimap` starts from the exported minimap and removes known contaminants using exported masks
- `normalmap` is rendered from exported terrain heights
- `object_visibility_mask` is projected from real placements plus geometry or bounds
- `brush_mask_path` is a deterministic terrain-analysis result over exported heightmaps

Those are still acceptable grounding channels because they can be traced back to real tile data and re-derived reproducibly.

## Brush Versus Prefab Policy

The current documentation and public progress framing should treat these two channels differently.

### Brush Channel

The brush channel is active and trusted enough to document as part of the current grounding story because:

- it is harvested from real tile geometry already present in the exported dataset
- it writes reproducible manifests and tile masks
- it already participates in the current terrain model as a conditioning channel

### Prefab Channel

The prefab channel is currently experimental and should be treated as deferred for public grounding claims.

That means:

- do not present prefab outputs as an active trusted training channel
- do not present prefab exports as proof that the model is grounded in real data
- do not rely on prefab outputs to explain the current terrain model inputs

The prefab tooling can remain in the repo as research, review, and future dataset work, but until it is validated to the same standard as the brush path it should stay outside the main grounding narrative.

## How To Verify A Dataset Claim

If someone wants to check whether a training claim is grounded in real data, they should be able to follow this chain:

1. identify the dataset root under `datasets/<label>/<map>/`
2. inspect `ml_dataset_manifest.json`, `metadata.jsonl`, and `dataset_info.json`
3. open the tile JSON in `dataset/<tile>.json`
4. inspect the dataset-relative asset paths named in that tile JSON
5. if brush evidence is referenced, inspect `brush_imprints/brush_imprint_manifest.json` and the tile-level brush mask

One current limitation remains important: the tile schema does not yet carry a canonical per-tile client-build field, so dataset-root provenance is still the primary build selector.

## Current Public Framing

The safest accurate public description today is:

- the terrain model is trained on datasets harvested from real WoW client and development-map data
- the exporter writes traceable tile JSON plus image artifacts under `datasets/`
- brush evidence is a real harvested auxiliary channel
- prefab work exists, but it is not currently part of the trusted active supervision story
- GAN is a refinement objective inside training, not the source of the dataset