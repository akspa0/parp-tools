# Active Context — wow-viewer

**Last updated**: 2026-06-24 | **Focus**: 076 macro paste/scar segmentation correction

## Current State

- **Spec 071 `071-left-right-sidebar-split`**: complete and committed. User is now testing/validating the viewer build.
- **Spec 074 `074-alpha-brush-library`**: deprecated as primary direction. Outputs remain useful candidate/evidence rows only; tile-local connected components are not authoritative brush labels.
- **Spec 075 `075-scar-mask-segmentation`**: deprecated as primary direction. The trainer is a coarse diagnostic baseline only; do not continue it as the brush-family route unless explicitly reopened.
- **Spec 076 `076-full-map-fractal-brush-library`**: active plan. Phases 1-3 are implemented for bounded compact and full-map strip proofs. Raw connected-component/near-dedupe outputs are diagnostic only; active review target is macro paste/scar/digital-painting regions via `--macro-pastes`.
- **V21/V21c height training**: paused. Multiple runs (with and without scheduler/normal/fractal changes, restored to d0929e2 baseline) failed to reproduce the earlier 0.3126 convergence; model stalls at ~0.83 height L1. Pivoting to a deconstruction-first approach.

## Why the Pivot

End-to-end height regression from minimap was not converging despite identical code/data to the earlier baseline. The working hypothesis is that the model needs to understand terrain as a **layered Photoshop-style composition** — MCAL alpha masks + tileset layers + fractal brush stamps — before it can predict height. The alpha masks contain the unadulterated brushwork; identifying and cataloging those brushes is the prerequisite.

## Active Work: 076 Full-Map Fractal Brush Library

- Location: `wow-viewer/specs/076-full-map-fractal-brush-library/`
- Architecture doc: `wow-viewer/docs/architecture/full-map-fractal-brush-library-2026-06-23.md`
- Purpose: recover real reusable brush/fractal/paste terrain-art primitives from the full map canvas, not per-ADT-tile connected components.
- Conceptual model: each map is a stack of ZBrush-like sculpt-and-paint documents/layers; terrain mesh, alpha masks, MCLY texture/layer assignments, and possible source BLP/decal/effect stamps are one coupled unit.
- Required signals: `alpha_256`, `height_257`, `normal_xyz`, `mcly_texture_ids`, `mcly_layer_mask`, minimap/object/liquid/shadow context where available, plus likely transparent/effect BLP source candidates when decoded/fingerprinted.
- Phase 1 canvas/provenance proof landed: `fractal_canvas.py`, `build_full_map_fractal_canvas.py`, and `tests/test_fractal_canvas.py`.
- Smoke output: `wow-viewer/output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/` with alpha shape `(256,1024,4)`, height shape `(257,1025)`, MCLY shape `(16,64,4)`.
- Phase 2 bounded segmentation landed: `fractal_segments.py`, `segment_full_map_fractals.py`, and `tests/test_fractal_segments.py`.
- Strict curation correction: `composite_chonker` can be a valid composite-canvas harvest target, while default atomic brush samples require a minimum `8x8` alpha-pixel footprint.
- Segment smoke output: `.../smoke_0_5_3_3368_Azeroth_tile16_compact/segments/`, 961 regions: 11 accepted candidates, 24 fractal members, 1 composite chonker, 2 one-off details, 923 too-small rows.
- Phase 3 trainable library landed: `fractal_library.py`, `build_fractal_brush_library.py`, and `tests/test_fractal_library.py`.
- Library smoke output: `wow-viewer/output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/`, 35 default atomic samples, 926 review/rejected rows, loader read 32 samples with no rejected labels.
- Phase 4 T025/T026 landed: `research.md` inventories texture/fingerprint surfaces; `fractal_library.py` now records MCLY texture counts, dominant texture ID, and active-layer coverage per sample.
- One-shot raw analysis landed: `analyze_fractal_raw_components.py` runs selected builds/maps sequentially (supports `--maps all`) and writes cross-target exact alpha-shape dedupe under `dedupe/`.
- Validated two-build Azeroth run: 7,317 raw components, 3,957 exact patterns, 233 duplicate patterns under `two_build_test1`.
- Footprint correction landed: default minimum atomic footprint is `8x8` alpha pixels; corrected two-build Azeroth run produced 2,025 raw components, 2,002 exact patterns, 17 duplicates under `two_build_test2`.
- Full-map strip processing landed: `--tile-limit 0` loads every tile for each selected map, writes tile-chunked canvases, segments horizontal strips, offsets bboxes to global canvas coordinates, and dedupes strip overlaps. Full Azeroth 0.5.3 (622 tiles) produced 12,906 raw components, 12,163 exact patterns, 566 exact duplicates under `full_map_Azeroth_0_5_3_3368`. Canonical runs use `--maps all`.
- Near-duplicate clustering landed: `fractal_near_dedupe.py` groups raw components by translation/mirror/rotation-invariant normalized binary thumbnails. Full Azeroth 0.5.3 collapsed to 11,976 clusters (688 duplicate clusters after rectangle pages added, max size 76) with a 16x16 thumbnail and radius 0. These raw-stroke clusters are diagnostic evidence, not the current training target.
- Rectangle-page detection landed: `detect_rectangle_pages()` finds solid axis-aligned rectangular alpha pages (extent >= 0.85). Full Azeroth 0.5.3 produced 72 rectangle_page regions.
- Macro paste/scar grouping landed: `segment_macro_pastes()` plus analyzer `--macro-pastes` groups nearby alpha strokes into larger authored paste/scar regions using streamed max-pooled alpha, coarse dilation, full-res bbox reprojection, and original-alpha area filtering. Full-map macro mode skips raw strip segmentation because macro regions replace those diagnostic rows.
- Macro visual review landed: analyzer `--visualize-macro` writes `macro_review/index.html`, `macro_paste_overview.png`, and macro crop contact sheets; `--visualize-composite-signal` adds `composite_signal_overview.png` using V18-style height/normal/alpha/MCLY/object/liquid hard-region signals.
- Macro validation: bounded `0_5_3_3368` Azeroth tile-limit 16 produced 7 `macro_paste` regions under `macro_visual_smoke_tile16`; full-map `0_7_0_3694` `PVPZone02` with close-radius 8/min-area 1024 produced 4 regions under `macro_visual_composite_pvpzone02_close8_area1024`.
- Macro sweep: `macro_sweep_pvpzone02_r8_16_32_area1024_4096/index.html` compares close radius 8/16/32 and min-area 1024/4096; all settings produced only 3-4 regions, showing alpha-only grouping remains broad/layer-wide on this map.
- Blocky child segmentation landed: analyzer `--blocky-pastes` emits dense middle-scale child regions from a 16x16 alpha coverage grid. `PVPZone02` proof with `--block-min-coverage 0.45 --block-close-radius 0 --block-max-footprint 160` produced 10 `blocky_paste` regions under `blocky_visual_pvpzone02_b16_cov045_close0_max160` and removed the giant parent-zone boxes.
- Single-map runs are smoke/proof shortcuts; real validation must run `--maps all` across every map present in each build index.
- Contact-sheet visualizer `visualize_fractal_raw_patterns.py` renders repeated exact-pattern pages (200 patterns / 5 pages proven).
- Near-duplicate cluster contact-sheet visualizer added: `scripts/visualize_fractal_near_patterns.py`; rendered 100 repeated clusters across 10 pages for full Azeroth 0.5.3.
- Analyzer now writes a top-level `index.html` summarizing every processed target and linking per-map overlays plus cross-map dedupe/near catalogs.
- No canonical decoded terrain tileset/BLP fingerprint dataset exists yet under `data-harvester` or `wow-viewer/output`; BLP source matching remains a bounded follow-up.
- Known gap: blocky alpha-coverage segmentation now has the right middle scale on `PVPZone02`, but it still needs composite-signal cross-check/tuning before all-map validation.
- Training remains blocked until macro paste/scar outputs are visually validated at all-map scale and Phase 5 model targets are approved.
- Immediate next slice: tune blocky child segmentation against composite hard-region overview on more maps, then run canonical `--maps all`; no model training until Phase 5 targets are approved.

## Historical Evidence: 074 Alpha Brush Library

- Location: `wow-viewer/specs/074-alpha-brush-library/`
- Status: deprecated as primary route. It extracted connected components from MCAL alpha layers, embedded them with DINOv2 (`transformers`), clustered, and built a candidate catalog.
- Output: `components.jsonl`, `clusters.jsonl`, `catalog.jsonl`, contact-sheet montages. Layer-role reports still pending.
- Phase 0 done: `_research_alpha_components.py` ran on `0_5_3_3368` V18 Zarr, 12 Azeroth tiles, threshold sweep, DINOv2 embeddings, PCA projections.
- Decision: default alpha threshold remains `0.05`; default embedding token strategy is mean-pooled patch tokens, with `[CLS]` retained for comparison.
- Phase 1 done: `alpha_brush.py` plus `tests/test_alpha_brush.py`; lint and tests pass.
- Phase 2 full output exists at `wow-viewer/output/analysis/alpha-brush-library/two-build-full/`: 320,368 components, 1000 clusters, 320,368 catalog rows.
- Phase 3 partial: `visualize_alpha_brush_catalog.py` renders contact sheets with an explicit layer-color legend. Full `montages_all` rendered for all 1000 clusters.
- Docs added: `data-model.md` and `quickstart.md` explain setup, smoke/full runs, output schemas, result inspection, and current limitations.
- Exact scar dedupe added: `dedupe_alpha_brush_patterns.py` found 263,188 exact binary scar patterns from 320,368 components; largest exact scar repeats 715 times. `pattern_neighbors.jsonl` ranks non-exact variants by similarity.
- Neighbor sheets added: `visualize_alpha_brush_pattern_neighbors.py` rendered top-200 exact scars with nearest non-exact neighbors under `two-build-full/dedupe/neighbor_montages/`.
- User interpretation: clusters are atomic scars; useful building blocks are likely multi-component/multi-tile sprites/prefabs/pastes. C35 appears to be very low-resolution legacy heightmap-like shapes, plausibly inherited from Warcraft 3 editor-era content.
- Correction: contact sheets showed many candidates are tiny unique one-off details/roads or large composite chonkers. 074 rows should feed 076 linkage/review, not direct training.

## Diagnostic Only: 075 V21 Scar Mask Model

- Files: `v21_scar_dataset.py`, `v21_scar_model.py`, `train_v21_scar_mask.py`, `test_v21_scar_mask.py`.
- Contract: `minimap_rgb_256 -> alpha_scar_mask_256`, one logits tensor `(B,1,256,256)`, BCE-with-logits + Dice. This does not solve brush/fractal/paste decomposition.
- Dataset: patched V18 Zarr stores are still the on-disk substrate; model lane is V21-era.
- Smoke command passed: `uv run python scripts/train_v21_scar_mask.py --builds 0_5_3_3368 3_3_5_12340 --max-steps 2 --val-max-steps 1 --batch-size 2 --max-tiles 64 --base-channels 8 --run-name smoke`.
- Smoke outputs: `models/v21/scar-mask/runs/smoke/{best.pt,latest.pt,metrics.json,preview.png}`.
- Full 20-epoch user run reached about val loss `0.6982`, IoU `0.7647`, F1 `0.8519`, but this remains the wrong primary target. Resume support was added to the trainer for general hygiene.

## In Validation

- `071-left-right-sidebar-split` branch: user testing the viewer UI with left/right sidebars, Model Viewer sub-tabs, and toolbar layout.

## Open Questions

1. Which bounded build/map should be the first 076 Phase 1 validation target? Teldrassil/root-heavy regions are preferred if present.
2. Which existing tileset/texture/BLP effect fingerprint output is canonical enough to join in Phase 4, especially for paths like `textures\BloodSplats`?
3. What macro-close radius/downsample/min-area settings best match human-visible paste/scar objects across all maps?
4. Which V18 composite signals should gate or split `blocky_paste` child regions before all-map validation?

## Files Touched Recently

- `wow-viewer/specs/074-alpha-brush-library/{spec,plan,tasks}.md`
- `wow-viewer/specs/074-alpha-brush-library/research.md`
- `wow-viewer/specs/074-alpha-brush-library/{data-model,quickstart,visualization_notes}.md`
- `wow-viewer/data-harvester/scripts/_research_alpha_components.py`
- `wow-viewer/data-harvester/src/harvester/alpha_brush.py`
- `wow-viewer/data-harvester/scripts/extract_alpha_brush_catalog.py`
- `wow-viewer/data-harvester/scripts/visualize_alpha_brush_catalog.py`
- `wow-viewer/data-harvester/scripts/dedupe_alpha_brush_patterns.py`
- `wow-viewer/data-harvester/scripts/visualize_alpha_brush_pattern_neighbors.py`
- `wow-viewer/specs/075-scar-mask-segmentation/{spec,plan,tasks}.md`
- `wow-viewer/specs/076-full-map-fractal-brush-library/{spec,plan,tasks,quickstart}.md`
- `wow-viewer/specs/076-full-map-fractal-brush-library/research.md`
- `wow-viewer/docs/architecture/full-map-fractal-brush-library-2026-06-23.md`
- `wow-viewer/data-harvester/scripts/analyze_fractal_raw_components.py`
- `wow-viewer/data-harvester/scripts/visualize_fractal_raw_patterns.py`
- `wow-viewer/data-harvester/scripts/visualize_fractal_near_patterns.py`
- `wow-viewer/data-harvester/src/harvester/fractal_raw_analysis.py`
- `wow-viewer/data-harvester/src/harvester/fractal_near_dedupe.py`
- `wow-viewer/data-harvester/src/harvester/fractal_canvas.py`
- `wow-viewer/data-harvester/src/harvester/fractal_segments.py`
- `wow-viewer/data-harvester/tests/test_analyze_fractal_raw_components.py`
- `wow-viewer/data-harvester/tests/test_fractal_near_dedupe.py`
- `wow-viewer/data-harvester/tests/test_fractal_segments_rectangle.py`
- `wow-viewer/docs/architecture/full-map-fractal-brush-library-2026-06-23.md`
- `wow-viewer/data-harvester/src/harvester/fractal_canvas.py`
- `wow-viewer/data-harvester/src/harvester/fractal_segments.py`
- `wow-viewer/data-harvester/src/harvester/fractal_library.py`
- `wow-viewer/data-harvester/scripts/build_full_map_fractal_canvas.py`
- `wow-viewer/data-harvester/scripts/segment_full_map_fractals.py`
- `wow-viewer/data-harvester/scripts/build_fractal_brush_library.py`
- `wow-viewer/data-harvester/tests/test_fractal_canvas.py`
- `wow-viewer/data-harvester/tests/test_fractal_segments.py`
- `wow-viewer/data-harvester/tests/test_fractal_library.py`
- `wow-viewer/data-harvester/src/harvester/{v21_scar_dataset,v21_scar_model,test_v21_scar_mask}.py`
- `wow-viewer/data-harvester/scripts/train_v21_scar_mask.py`
- `wow-viewer/data-harvester/tests/test_alpha_brush.py`
- `wow-viewer/memory-bank/{activeContext,progress}.md`
- `wow-viewer/data-harvester/scripts/train_v16_1_common.py` (reverted to d0929e2 baseline during debugging)
