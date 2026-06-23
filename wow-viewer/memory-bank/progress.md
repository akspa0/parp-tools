# Progress — wow-viewer

## 2026-06-23 — Spec 076 replaces 074/075 brush-model direction

### Phase 1 implementation

- Added `src/harvester/fractal_canvas.py` for tile-local to map-canvas transforms, compact tile-window selection, dense bounded canvas assembly, Zarr/Parquet output, and seam overlay rendering.
- Added `scripts/build_full_map_fractal_canvas.py` CLI.
- Added `tests/test_fractal_canvas.py`; `uv run pytest tests/test_fractal_canvas.py` -> `4 passed`.
- Lint passed: `uv run ruff check src/harvester/fractal_canvas.py tests/test_fractal_canvas.py scripts/build_full_map_fractal_canvas.py`.
- Real-data smoke passed on V18 Zarr `0_5_3_3368`/`Azeroth`, tile-limit 4.
- Smoke output: `wow-viewer/output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/`.
- Output shapes: alpha `(256,1024,4)`, height `(257,1025)`, MCLY `(16,64,4)`.

### Phase 1 status

- Phase 1 is implemented and validated for bounded compact tile windows and full-map strip processing.
- Full-continent chunk streaming is now implemented via tile-chunked Zarr writes and horizontal strip segmentation.
- Next route: review full-map strip artifacts, then improve dedupe/clustering before Phase 5 model target selection.

### Phase 2 implementation

- Added `src/harvester/fractal_segments.py` for full-map alpha region extraction, region stats, curation labels, optional 074 catalog linkage, Parquet/JSONL output, and overlay rendering.
- Added `scripts/segment_full_map_fractals.py` CLI.
- Added `tests/test_fractal_segments.py`; `uv run pytest tests/test_fractal_segments.py` -> `3 passed`.
- Lint passed: `uv run ruff check src/harvester/fractal_segments.py tests/test_fractal_segments.py scripts/segment_full_map_fractals.py`.
- Real-data strict-footprint smoke passed on the Phase 1 tile16 compact canvas: 961 regions; 11 accepted candidates, 24 fractal members, 1 composite chonker, 2 one-off details, 923 too-small rows.
- Segment output: `wow-viewer/output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments/`.
- Curation correction: `composite_chonker` is preserved as a composite-canvas harvest target, while default atomic samples require an `8x8` alpha-pixel minimum footprint.

### Phase 2 status

- Phase 2 is implemented and validated for bounded compact canvases and full-map strip views.
- Next route: inspect `fractal_regions_overlay.png` and metadata; add near-duplicate clustering because exact dedupe is too brittle.

### Phase 3 implementation

- Added `src/harvester/fractal_library.py` for terrain-art sample schema, stable IDs, deterministic splits, accepted/rejected filtering, Zarr tensor output, Parquet metadata output, and a smoke loader.
- Added `scripts/build_fractal_brush_library.py` CLI.
- Added `tests/test_fractal_library.py`; `uv run pytest tests/test_fractal_library.py` -> `3 passed`.
- Lint passed: `uv run ruff check src/harvester/fractal_library.py tests/test_fractal_library.py scripts/build_fractal_brush_library.py`.
- Real-data smoke passed on the Phase 2 tile16 strict-footprint segments: 35 default trainable atomic samples, 926 review/rejected rows, split counts `train=26`, `val=8`, `test=1`.
- Smoke loader read 32 samples and returned only `accepted_candidate`/`fractal_member` labels.
- Library output: `wow-viewer/output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/`.

### Phase 3 status

- Phase 3 is implemented and validated for the bounded compact canvas.
- Next route: Phase 4 texture/variant/BLP source-evidence inventory and joins. Do not start model training yet.

### Phase 4 initial inventory

- Added `specs/076-full-map-fractal-brush-library/research.md` documenting reusable texture evidence and gaps.
- Confirmed reusable-now evidence is MCLY texture IDs/layer masks already present in V18 Zarr, Phase 1 canvas, Phase 2 region metadata, and Phase 3 sample tensors.
- Added per-sample Phase 4 fields in `fractal_library.py`: `mcly_texture_id_counts`, `dominant_mcly_texture_id`, and `mcly_active_layer_coverage`.
- No canonical decoded terrain tileset/BLP fingerprint artifact was found in `data-harvester` or `wow-viewer/output`; object-roof fingerprints are useful prior art only.
- Next route: add a bounded decoded terrain texture/BLP evidence extractor or join a canonical fingerprint artifact when one exists.

### Raw two-build analysis runner

- Added `src/harvester/fractal_raw_analysis.py`, `src/harvester/fractal_near_dedupe.py`, `scripts/analyze_fractal_raw_components.py`, and `scripts/visualize_fractal_raw_patterns.py`.
- The runner processes each build/map sequentially, writes per-target `canvas/` and `segments_raw/`, then writes cross-target exact dedupe under `dedupe/`. It supports `--maps all`.
- Validated two-build Azeroth run: 7,317 raw components, 3,957 exact patterns, 233 duplicate patterns under `two_build_test1`.
- Footprint correction to 8x8 alpha pixels: 2,025 raw components, 2,002 exact patterns, 17 duplicates under `two_build_test2`.
- Full-map strip processing: `--tile-limit 0` loads every tile for each selected map, writes tile-chunked Zarr canvases, segments horizontal strips, offsets bboxes to global coords, dedupes strip overlaps. Full Azeroth 0.5.3 (622 tiles) produced 12,906 raw components, 12,163 exact patterns, 566 exact duplicates under `full_map_Azeroth_0_5_3_3368`. Canonical validation runs use `--maps all`.
- Near-duplicate clustering: groups raw components by translation/mirror/rotation-invariant normalized binary thumbnails. Full Azeroth 0.5.3 collapsed to 11,976 clusters (668 duplicate clusters, max size 40) with a 16x16 thumbnail and radius 0.
- Rectangle-page detection: `detect_rectangle_pages()` finds solid axis-aligned rectangular alpha pages (extent >= 0.85). Full Azeroth 0.5.3 produced 72 rectangle_page regions; with rectangles included near-duplicates became 688 clusters, max size 76.
- Added `tests/test_analyze_fractal_raw_components.py`, `tests/test_fractal_near_dedupe.py`, and `tests/test_fractal_segments_rectangle.py`; pytest passes.
- Contact-sheet visualizer renders repeated exact-pattern pages from the dedupe catalog (200 patterns / 5 pages proven).
- Near-duplicate cluster contact-sheet visualizer added: `scripts/visualize_fractal_near_patterns.py`; rendered 100 repeated clusters across 10 pages for full Azeroth 0.5.3.
- Use these for broad inspection; they still detect connected alpha/fractal components, not obvious rectangular paste/canvas-page boundaries.

### What landed

- Added `specs/076-full-map-fractal-brush-library/{spec,plan,tasks}.md`.
- Added `docs/architecture/full-map-fractal-brush-library-2026-06-23.md`.
- Marked 074 as deprecated for primary training labels; its connected components are evidence rows only.
- Marked 075 as diagnostic only; whole-tile scar-mask segmentation is not the brush/fractal/paste target.
- Marked V18 paste and fractal-height-loss docs as historical/paused for current route.

### Corrected direction

- Assemble full-map alpha/MCLY/height/normal canvases before segmentation.
- Segment fractal/virtual-canvas structures in map coordinates, not ADT-tile-local coordinates.
- Treat mesh, alpha masks, MCLY texture/layer assignments, and possible source BLP/decal/effect stamps as one coupled ZBrush-like terrain-art primitive.
- Preserve chonkers as composite-canvas harvest targets; exclude one-off roads/details, tiny unique strokes, and low-repeatability noise from default atomic training manifests.
- Preserve provenance and build a Zarr/Parquet trainable library before any new model training.
- Phase 4 should also investigate likely transparent/effect BLP source assets (`textures\BloodSplats`, FX/environment/weather/decal/particle-style textures) as possible original brush sources for alpha/fractal motifs.

### Next

- Run canonical `--maps all` validation across both builds and review the resulting cross-map contact sheets/overlays.
- Tune near-dedupe thumbnail size and Hamming radius against cross-map contact-sheet review.
- Tune rectangle-page thresholds against cross-map overlays; some detected rectangles may be roads/rivers rather than authored paste pages.

## 2026-06-23 — Spec 075 V21 scar-mask segmentation Phase 1 complete

### What landed

- Created spec/plan/tasks under `specs/075-scar-mask-segmentation/`.
- Corrected naming: spec number is 075, model lane is V21 (`v21_scar_*`) because V18 is only the patched Zarr substrate.
- Added `src/harvester/v21_scar_dataset.py`: minimap input + binary scar target from `alpha_256` layers L1-L3 at threshold `0.05`.
- Added `src/harvester/v21_scar_model.py`: single-output scar-mask logits model `(B,1,256,256)`.
- Added `scripts/train_v21_scar_mask.py`: standalone trainer with BCE+Dice loss, checkpoints, metrics, and preview.
- Added `src/harvester/test_v21_scar_mask.py` and architecture doc `docs/architecture/v21-scar-mask-segmentation-2026-06-23.md`.

### Validation

- `uv run ruff check src/harvester/v21_scar_dataset.py src/harvester/v21_scar_model.py src/harvester/test_v21_scar_mask.py scripts/train_v21_scar_mask.py`
- `uv run pytest src/harvester/test_v21_scar_mask.py` -> 3 passed.
- `uv run python -m py_compile src/harvester/v21_scar_dataset.py src/harvester/v21_scar_model.py scripts/train_v21_scar_mask.py`
- Smoke: `uv run python scripts/train_v21_scar_mask.py --builds 0_5_3_3368 3_3_5_12340 --max-steps 2 --val-max-steps 1 --batch-size 2 --max-tiles 64 --base-channels 8 --run-name smoke`.

### Status

Phase 1 is mechanically complete. Smoke outputs exist at `models/v21/scar-mask/runs/smoke/`. The smoke proves the model lane runs, not that the model is useful yet. Next: choose a real training schedule/subset, then add inference + connected-component extraction for predicted scar masks.

## 2026-06-23 — Spec 074 contact-sheet visualizer added

### What landed

- Added `scripts/visualize_alpha_brush_catalog.py` to render contact-sheet PNGs from `components.jsonl`/`clusters.jsonl`/`catalog.jsonl` by reopening V18 `alpha_256` crops.
- Rendered top-100 sheets under `wow-viewer/output/analysis/alpha-brush-library/two-build-full/montages/`.
- Rendered full 1000-cluster library under `wow-viewer/output/analysis/alpha-brush-library/two-build-full/montages_all/` with 20 paginated PNG sheets and `index.html`.
- Added explicit legend: gray=L0 base/fill, blue=L1 primary brush, green=L2 transition/detail, orange=L3 highlight/detail.
- Captured human review notes in `specs/074-alpha-brush-library/visualization_notes.md`: current clusters are atomic strokes, while useful authored units are likely multi-component/multi-tile sprites/prefabs/pastes; C35 looks like low-resolution legacy heightmap-like stamps, plausibly Warcraft 3 editor-era reuse.

### Status

Next useful 074 slice is not more component clustering. It is a grouping pass that reconstructs larger multi-tile sprite/paste candidates from co-occurring component clusters, then renders those as the actual prefab library.

### Exact scar dedupe follow-up

- Added `scripts/dedupe_alpha_brush_patterns.py`: hashes exact binary alpha crops, writes `exact_patterns.jsonl`, and ranks non-exact variants in `pattern_neighbors.jsonl` by embedding similarity.
- Full run on `two-build-full`: 320,368 components -> 263,188 exact binary scars; largest exact pattern has 715 members; 2,105,504 near-neighbor rows.
- Added `scripts/visualize_alpha_brush_pattern_neighbors.py`: renders exact canonical scar + nearest non-exact neighbors per row.
- Rendered top-200 exact scars to `two-build-full/dedupe/neighbor_montages/`.

## 2026-06-23 — Spec 074 Phase 1 complete, Phase 2 implemented

### What landed

- Added `wow-viewer/data-harvester/src/harvester/alpha_brush.py` with component/cluster/catalog dataclasses, extraction, patch rendering, DINOv2 embedding, clustering, catalog builders, and JSONL serializers.
- Added `wow-viewer/data-harvester/tests/test_alpha_brush.py`; targeted pytest passes.
- Added `wow-viewer/data-harvester/scripts/extract_alpha_brush_catalog.py` for V18 Zarr bulk extraction and catalog output.
- Phase 2 two-build smoke passed: `0_5_3_3368` + `3_3_5_12340`, `--tile-limit 2`, 179 components, 16 clusters, 16 non-singleton clusters.

### Validation

- `uv run ruff check src/harvester/alpha_brush.py scripts/extract_alpha_brush_catalog.py tests/test_alpha_brush.py`
- `uv run pytest tests/test_alpha_brush.py`
- `uv run python -m py_compile src/harvester/alpha_brush.py scripts/extract_alpha_brush_catalog.py tests/test_alpha_brush.py`

### Status

074 cannot move to Phase 3 yet. T022 full two-build validation is still open because it requires DINOv2 over 1,629 + 5,134 V18 alpha tiles and should not be marked complete from a smoke run.

### Documentation follow-up

- Added `specs/074-alpha-brush-library/data-model.md` with the exact current schemas and output files.
- Added `specs/074-alpha-brush-library/quickstart.md` as the operator/user guide for setup, smoke runs, full T022 extraction, result inspection, and troubleshooting.
- Linked the 074 quickstart from `data-harvester/README.md`.
- T030 remains open for final visualization command coverage because Phase 3 visualization is not implemented yet.

## 2026-06-23 — Spec 074 Phase 0 research complete

### What landed

- Added `wow-viewer/data-harvester/scripts/_research_alpha_components.py` for one-off alpha component research.
- Ran against `wow-viewer/output/datasets/v18/0_5_3_3368.zarr` on 12 alpha-bearing Azeroth tiles.
- Threshold counts: `0.03` -> 215 components, `0.05` -> 247, `0.10` -> 333.
- DINOv2 `facebook/dinov2-small` loaded through `transformers`; 96 component patches embedded.
- Outputs written under `wow-viewer/output/analysis/alpha-brush-library/research/`, including `projection.png`, `[CLS]`/mean projections, embeddings, patch examples, and `summary.json`.
- `research.md` records the Phase 0 decision: threshold `0.05`; mean-pooled patch-token embeddings by default.

### Status

074 Phase 0 complete. Next is Phase 1 shared library `alpha_brush.py` plus synthetic-shape smoke tests.

## 2026-06-22 — Pivot from V21 height regression to 074 Alpha Brush Library

### What happened

- V21/V21c height training could not reproduce the earlier 0.3126 baseline. Runs restored to commit `d0929e2` still stalled at ~0.83 height L1 after 35 epochs.
- Decided the end-to-end minimap→height approach skips the actual terrain construction process.
- New direction: treat the ADT as a layered Photoshop canvas and reverse-engineer the artists' fractal brush library from MCAL alpha masks.

### What landed

- Spec `074-alpha-brush-library` created at `wow-viewer/specs/074-alpha-brush-library/`.
- Plan written: 5 phases (research → shared library → bulk extraction → visualization → docs/handoff).
- Tasks broken down in `tasks.md`.
- DINOv2 (`transformers`) confirmed available and loadable in the data-harvester environment.
- Memory bank updated.

### Status

074 ready to start with Phase 0 research. 071 in user testing.

---

## 2026-06-22 — 073a: Toolbar / left sidebar dedup and alignment (complete)

### What landed

- Removed `DrawWorkspaceToolbarControls`, "Open Game Folder", and "Open File" from `DrawToolbar`.
- Toolbar now shows only scene status + centered terrain controls.
- Source/workspace controls remain in the left sidebar (`DrawWorkspaceBarsPanelContent`).
- Legacy mode preserved.
- Build: 0 errors, 284 pre-existing warnings.
- Commit `b11dd518` pushed to `071-left-right-sidebar-split`.

### Status

073a complete. 073b (Tools tab converter integration) spec'd and ready for implementation in fresh chat.

## 2026-06-22 — 072: Sidebar resize + toolbar layout hotfix (complete)

### What landed

- Removed `DrawFixedSidebarWidthControl` sliders from inside tab-mode left/right sidebars.
- `DrawFixedSidebarSplitters` now draws left/right edge splitters in tab mode.
- `DrawToolbar` spans only the scene viewport width (`viewportX`..`viewportWidth`).
- `DrawToolbar` is called after sidebars in `DrawUI` so it stays on top if edges overlap.
- Build: 0 errors, 284 pre-existing warnings.
- Commit `bcdcb752` pushed to `071-left-right-sidebar-split`.

### Status

072 hotfix complete.

## 2026-06-22 — Spec 071 Phase H: Memory bank + spec sync + final build (complete)

### What landed

- Updated `specs/071-left-right-sidebar-split/spec.md` to match final implementation.
- Updated memory bank with full 8-phase history.
- Final build: 0 errors, 286 pre-existing warnings.
- Commit `8190fb65` pushed to `071-left-right-sidebar-split`.

### Status

Spec 071 complete.

## 2026-06-22 — Spec 071 Phases A-G (complete)

Summary of earlier 071 phases:
- **A**: Viewport subtracts left/right sidebars.
- **B**: Left sidebar with workspace bars, file browser, world maps.
- **C**: Right sidebar = workbench anchored to right edge.
- **D**: 3 top tabs (Model/World/Tools) with `WorkbenchNavigator` and typed `OpenWorkbenchTab` helpers.
- **E**: Model > Info sub-tab with path line.
- **F**: Model > Animations sub-tab with Play/Pause/Stop, loop, speed buttons, timeline slider; added `PlaybackSpeed`/`Loop` to `IAnimationController`.
- **G**: Model > Actions + LOD sub-tabs; selected world object auto-switches to Model > Info.

All phases built clean and pushed to `071-left-right-sidebar-split`.

## 2026-06-21 — Spec 071 drafted

- Two-side layout + Model Viewer mode, 8 phases, branch cut from `069-viewer-ui-overhaul`.

## 2026-06-21 — Spec 069: Viewer UI overhaul (tab system → workbench)

- Cells overlay, tab data model, archeology playback, sticky settings, headless content variants.
- Learned: top/bottom tab bars failed (debug overlay look), per-sub-tab popouts failed (window sprawl), single Workbench panel succeeded.
- 14 phases committed to `069-viewer-ui-overhaul`.

## Previous work
- Spec 068: fractal-aware height loss + curation hardening (V21c)
- Spec 067: V20 multi-modal terrain intent
- Spec 066: V19 minimal-signal height regressor
- PM4 surface correlation, PM4 simplification reverse-engineering

## Branch summary

- `071-left-right-sidebar-split` — 071 + 072, active, user testing.
- `069-viewer-ui-overhaul` — legacy tab UI work, salvageable concepts extracted into 071.
- `074-alpha-brush-library` — implemented candidate/evidence extraction, deprecated as primary brush truth.
- `075-scar-mask-segmentation` — diagnostic baseline only, deprecated as primary model route.
- `076-full-map-fractal-brush-library` — active dataset-truth plan; Phase 1-3 and full-map strip processing are implemented.

## Out-of-Phase Work

- 070: Per-map workbench windows (deferred, large rewrite).
- 073b: Tools tab converter integration (spec'd, implementation deferred to fresh chat).
- V21/V21c height regression and fractal-aware height loss: paused pending 076 curated library validation.
