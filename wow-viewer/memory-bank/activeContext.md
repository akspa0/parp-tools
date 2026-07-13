# Active Context — wow-viewer

Last updated: 2026-07-13

## Current target — Spec 103: revive the v7 terrain regressor on clean signals

- **Governing law (image-only):** the only deployment input is one image tile. Every other signal is generated from it; no model reads a ground-truth signal at inference; downstream trains on generated (not ground-truth) upstream; a target the image cannot support is invalid. Validation is **label-free** (self-consistency), never label-comparison. See `specs/103-image-only-reconstruction/spec.md`.
- **Model:** the real **v7** = `MultiChannelUNetV7` (single U-Net, no stages), read-only reference at
  `gillijimproject_refactor/src/WoWMapConverter/scripts/v7_model.py`. Ported into
  `wow-viewer/data-harvester/src/harvester/spec103/` (reference repo never modified). Input **13 channels**
  (0-2 minimap RGB, 3-5 normal RGB, 6 WDL prior, 7-12 aux) → terrain height; residual over the WDL "trestle" prior.
- **WDL prior = verified transform:** `outer = height257[::16,::16]`, `inner = height257[8::16,8::16]`
  (`WdlWriter.ExtractTileHeightsFromAlpha`). Derive from existing `height_257` — no reharvest.
  **Never** `wdl_height_33` (the `::8` mistake).
- **Approach — synthetic-first (MVP):** author synthetic ADT tiles with known height patterns (existing ADT
  tooling, AlphaWdtWriter frozen) → capture minimaps in WoWViewer → derive the prior → train v7 → verify
  known-pattern reconstruction → catalog caveats. Then apply the proven recipe to real clean data.
- **Quick and dirty:** plain height regression, **no object-mask loss gating** by default; WDL-prior channel
  **dropout** so one model handles prior-present and prior-absent tiles.
- **Exploratory lanes (recorded, not built):** measure terrain-shadow↔height on synthetic known-height tiles
  (Spec 102 N011-N013 shadow-capture contract) → teacher (rich signals) distilled to an image-only student;
  image-only `minimap → WDL-prior` front-end; output-space object segmentation+inpaint cleanup.
- **The USER runs all training/capture/heavy jobs.** The agent prepares scripts + commands only (AGENTS RULE 0).

## Dropped / paused

- **V24 / Spec 094 is NOT functional — dropped.** Do not revive it.
- **Spec 102 M0 object-mask lane is paused/superseded** by Spec 103. Its work is preserved: a full-stack simple
  M0 trainer (`scripts/train_spec102_m0_simple.py`: complete-map holdout, blank-tile filter, AMP, EMA,
  warmup+cosine, early-stop, resumable, threshold sweep) + image-only inference (`infer_spec102_m0_simple.py`);
  the strict fragment-trace target and 42/42-green tests remain but are not the active path.

## Boundaries

- New work in `wow-viewer/`; `gillijimproject_refactor` is read-only reference (port from, never edit).
- Staged clients only: `output/tmp/wowarchive-clients/`. Never `H:\CLIENTS`.
- Spec 080 owns the UI lane.
