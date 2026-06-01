# PROGRESS — wow-viewer

## Position
- **Renderer Improvements Convergence (036) — CREATED 2026-06-01**
  - New owner plan at `wow-viewer/specs/036-renderer-improvements/`
  - Artifacts written:
    - `spec.md`
    - `plan.md`
    - `research.md`
    - `data-model.md`
    - `quickstart.md`
    - `tasks.md`
    - `contracts/renderer-capability-slice.schema.json`
    - `contracts/renderer-validation-scenario.schema.json`
  - Specs 030, 031, and 032 now carry convergence notes in their `plan.md` files pointing readers to 036
  - Intent:
    - keep 030-032 as source slices
    - make 036 the active owner plan for renderer modernization sequencing in `wow-viewer`
    - keep spec 035 M2 recovery as a separate adjacent feature lane
- **M2 build-profile note (2026-06-01)**
  - staged `3.0.1.8303` Northrend currently exposes a separate renderer-risk boundary:
    - some `.mdx` assets fail `.skin` lookup and converter fallback
    - logs suggest a possible prototype `MD20` / `Model2` family seam
  - this is now recorded as deferred research in:
    - `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md`
    - `wow-viewer/specs/035-m2-render-parity-recovery/research.md`
  - future action is Ghidra on staged `3.0.1.8303` `wow.exe` plus repo archaeology for older `MD20` handling
- **M2 3.3.5 wrong-axis animation hotfix (2026-06-01)**
  - `WowViewer.Core.Runtime.M2.M2TrackSampler` now reads `M2CompQuaternion` payloads in the same direct little-endian component order as `M2ToMdxConverter`
  - shared helper added at `WowViewer.Core.M2.M2CompQuaternion.FromRawLittleEndian(...)` so runtime and converter no longer drift on the same on-disk payload
  - synthetic runtime fixture encoding in `WowViewer.Core.Tests` was updated to match the real payload order
  - focused proof: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter M2RuntimeTests` passed `25/25`
  - remaining proof gap: active viewer runtime signoff still requires a viewer restart because the live `ParpToolsWoWViewer` process holds the app output binaries open
- **M2 3.3.5 skin bone-remap hotfix (2026-06-01)**
  - `WowViewer.Core.Runtime.M2.M2StaticRenderModelBuilder` now uses `.skin` `BoneEntries` for runtime render-vertex bone indices when present
  - new runtime regression test proves skin-owned bone remap overrides raw M2 vertex bone indices
  - focused proof: `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter M2RuntimeTests` passed `26/26`
  - remaining proof gap: live viewer/world signoff still requires restarting the running viewer process and rechecking affected 3.3.5 animated doodads
- V16 dataset generation + training is the primary active workflow.
- Harvest-first is canonical:
  - `WowViewer.Tool.Harvest`
  - staged clients under `output/tmp/wowarchive-clients/`
- **MdxViewer Migration (033) + WoWViewer Rename (034) — COMPLETED 2026-05-30/31**
  - MdxViewer moved from `gillijimproject_refactor/src/MdxViewer/` → `wow-viewer/src/viewer/WoWViewer/`
  - Renamed to WoWViewer, version bumped to v0.5.0 in csproj
  - MDX-L_Tool vendored into `wow-viewer/libs/WoW-Tools/MDX-L_Tool/`
  - gillijimProject-csharp.csproj vendored into `wow-viewer/libs/WoW-Tools/GillijimProject/`
  - WowViewer.App moved to `wow-viewer/src/viewer/WowViewer.App.Defunct/`
  - WoWViewer added to `wow-viewer/WowViewer.slnx`
  - **Remaining gap**: Many source files still use `MdxLTool.Formats.Mdx` namespace (not yet ported to WowViewer.Core.IO)
  - **MDX rendering is BROKEN** — recent M2 rendering fixes broke MDX rendering; namespace/type mismatches after migration
- **Ghidra RE lane (2026-05-30):** WMO render pass architecture + terrain cell system fully decompiled from build 3368.
  - Spec 029 (WMO minimap BLP harvest): spec+plan+tasks written, ready for Phase 1 implementation
  - Spec 030 (WMO render pass architecture): spec written, architecture doc written at `docs/architecture/wmo-render-pass-architecture-2026-05-30.md`, **plan+tasks now complete** (created 2026-05-31, 27 tasks total)
  - Spec 031 (terrain cell awareness): spec written, **plan+tasks now complete** (created 2026-05-31, 34 tasks total)
  - Key RE findings: 11 WMO render passes, per-batch MOMT flags, interior fog, liquid type dispatch, 145-vertex terrain layout, 8x8 cell grid, hole masks, 13-bit cell addressing
  - **MDX rendering is BROKEN** — many source files still use `MdxLTool.Formats.Mdx` namespace (not yet ported to WowViewer.Core.IO); recent M2 rendering fixes broke MDX rendering path
  - Key RE findings: 11 WMO render passes, per-batch MOMT flags, interior fog, liquid type dispatch, 145-vertex terrain layout, 8x8 cell grid, hole masks, 13-bit cell addressing
- New V18 planning lane is now documented with Spec Kit:
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/spec.md`
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/plan.md`
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/tasks.md`
  - target outcome: canvas-scale paste mining + cross-build dedupe + refined manifests for smaller/smarter V18 models
- Old converter-side `dataset-scan` / `dataset-audit` / `dataset-build-cache` flows are not the primary terrain-AI path.
- V16-facing docs were rewritten into shorter source-of-truth surfaces:
  - `wow-viewer/README.md`
  - `wow-viewer/data-harvester/README.md`
  - `wow-viewer/docs/architecture/v16-terrain-model-spec-2026-05-16.md`
- **Spec 009 landed (2026-05-22):** Master design specification at
  `wow-viewer/specs/009-full-project-reimplementation-spec/spec.md` (2,650 lines).
  28 sections covering every subsystem with byte-level detail. The doc audit
  (`speckit-doc-audit-2026-05-18.md`) updated to reflect this.

## Validated Now

### wow-viewer validation capture extraction
- `wow-viewer/specs/012-real-validation-batch-extraction/` now has bounded wow-viewer GPU proof on staged `0_5_3_3368 / Azeroth_30_48` and staged `3_3_5_12340 / Azeroth_30_48`.
- Command proof: `dotnet run --project wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture/WowViewer.Tool.ValidationCapture.csproj -- capture ... --gpu-viewer-style`
- Result: `4/4` variants succeeded on both anchors, with PNG output under `wow-viewer/output/tmp/validation-capture-gpu-viewer-style/` and `wow-viewer/output/tmp/validation-capture-gpu-viewer-style-335/`.
- Phase 5.1 is now landed on the same bounded proof surface: the tool writes compatible `images/<tile>_object_visibility_mask.png` and `images/<tile>_no_objects.png` outputs under `datasetRoot/images/`, with focused real-data proof roots at `wow-viewer/output/tmp/validation-capture-phase5-053/` and `wow-viewer/output/tmp/validation-capture-phase5-335/`.
- Current proof level is precise: `ValidationWorldSceneAdapter` now owns render/readback behind `IValidationWorldSceneAdapter`, the host uses explicit `ValidationCaptureCameraFrame` matrices and bypasses `WowViewerWorldScenePlanner`, but it still reuses `WorldGpuPreviewRenderer` as a temporary backend and does not yet mean the long-range dedicated viewer-style renderer extraction is done.

### V16 Corpus
- Finalized stores built for:
  - `0_5_3_3368`
  - `0_5_5_3494`
  - `0_7_0_3694`
  - `3_0_1_8303`
  - `3_3_5_12340`
  - `4_0_0_11927`
- All six current `signal_validation.json` files pass.
- Human-eye QA artifacts exist for all six under `wow-viewer/output/datasets/v16/inspection/`.
- `0_7_0_3694` carries the expected allowed warning for zero `has_holes_16` coverage.

### V16 Recovery / Build Surfaces
- In-memory archive harvest path is landed.
- Lean `ARRY` stream profile is landed.
- Map-level resume / `_resume_state.json` is landed.
- Completed-store skip guards and `--rebuild-existing` behavior are landed.
- `repair-index` is landed for coordinate-only fixes.
- `patch-liquids` is landed for in-place liquid rewrites.
- Signal validation gate is landed and passing on the current finalized corpus.
- Dataset inspection / summary / visual QA tooling is landed.
- Default Zarr compression is now `lz4` / `1` / `shuffle`.

### Critical V16 Fixes
- Mixed Cata `_tex0` fallback fixed:
  - `ReadTextureDataFromBytes(...)` now falls back to inline root `MCLY` / `MCAL`.
  - Focused repro on staged `4_0_0_11927 / AhnQiraj / (27,46)` restored alpha/MCLY truth.
- Alpha placeholder `map=memory` metadata fix is landed.
- Liquid presence-mask fix for valid type-`0` water is landed.
- Object instance mask + `placements.parquet` are landed.
- Bounds-based MDDF clutter filtering is now landed in
  `AdtTensorPackBuilder` for archive-backed harvest:
  - normalized asset-path clutter regex expanded beyond the old basename-only
    tree check
  - resolved doodad model bounds now classify tiny clutter and tall clutter for
    `object_filtered_mask_257`
  - focused proof: harvest project build succeeded and raw preview rerun for
    `3_3_5_12340 / Azeroth` completed cleanly
- Raw harvest preview now renders `raw mddf`, `raw modf`, and
  `filtered loss mask` panels, so preview-first QA can confirm whether clutter
  is leaking into terrain-loss gating before any Zarr mutation
- MdxViewer validation capture object artifacts are now a real renderer-truth
  path instead of WMO-only truth:
  - doodads now follow world-object visibility during the capture batch, so
    MDX/M2 silhouettes are included in `object_visibility_mask`
  - object artifact generation now prefers direct `objectsonly` silhouettes for
    `0.x` builds and prefers `primary` vs `noobjects` visibility diffs for
    later builds so terrain occlusion wins where underground geometry should
    stay hidden
  - bounded hotfix (2026-05-28): `ViewerApp_CaptureAutomation`
    `ShouldPreferDirectObjectsOnlyMask(...)` was still stubbed to `false`, so
    the documented `0.x` direct-`objectsonly` path was not actually enforced at
    artifact generation time; this is now fixed in the real `MdxViewer` proof
    lane
  - startup automation can now run a bounded validation batch directly from a
    dataset root and exit when complete, which makes real-data capture proof
    scriptable instead of UI-only
  - focused proof: `dotnet build gillijimproject_refactor/src/MdxViewer/MdxViewer.csproj -c Debug` succeeded after the capture-mask hotfix
  - focused real-data proof roots:
    - `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48`
    - `output/tmp/mdxviewer_validation_smoke/3_3_5_12340_Azeroth_30_48`
    - `output/tmp/mdxviewer_validation_smoke_fix_wmo/3_3_5_12340_Azeroth_30_48`
    - `output/tmp/mdxviewer_validation_smoke_heightfilter/3_3_5_12340_Azeroth_30_48`
  - runtime result on `Azeroth_30_48`:
    - `0.5.3.3368` final mask exactly matched direct `objectsonly`
      - refreshed bounded rerun after the policy fix succeeded at:
        - `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48/viewer_validation_minimaps/Azeroth_30_48_viewer_validation.png`
        - `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48/images/Azeroth_30_48_object_visibility_mask.png`
    - `3.3.5.12340` final mask differed from direct `objectsonly`, confirming
      the later-build occluded-diff policy executed at runtime
  - bounded follow-up fixes now landed in the MdxViewer proof path:
    - WMO near-camera culling hotfix in `WmoRenderer`
    - longer validation settle delay before capture
    - MDX bounds-height filtering during capture to suppress very tall clutter
  - throughput warning remains open:
    - one bounded single-tile rerun in the real `MdxViewer` lane still took a
      little over 3 minutes, which is too slow for broader proof sweeps
    - batching tiles within one loaded world session is now the clearest next
      bounded follow-up if this proof lane stays active
  - current proof boundary is still only `0_5_3_3368` + `3_3_5_12340`; the
    remaining four builds still need real renderer-truth capture proof before
    this lane can claim broad coverage
  - active `V16.2` direction is sidecar-first instead of mutating finalized
    base V16 stores immediately
  - renderer-truth capture pipeline is now integrated into the V16 build:
    - `build_v16_dataset.py generate-viewer-stubs` generates per-tile JSON stubs
    - `generate_all_renderer_truth_captures.bat` runs MdxViewer capture batches
    - `build_v16_dataset.py patch-renderer-truth` patches PNGs into Zarr stores
    - all 20,627 stubs generated across 6 builds (2026-05-22)
    - MdxViewer CLI requires: --game-path, --build, --listfile, --world,
      --validation-dataset-root, --validation-output, --validation-resolution,
      --force-validation-regeneration, --exit-after-validation
  - harvest stream produces all core terrain signals; viewer produces only
    renderer-truth overlay (object_visibility_mask, no_object_minimap)

### V16 Training Surfaces
- `V16Dataset` is the live loader.
- `V15Model` is the current V16 terrain model host.
- `validate_v16_training_ready.py` passed on staged `3_3_5_12340`.
- `validate_v16_training_ready.py` also passed across all six finalized stores and wrote:
  - `wow-viewer/output/datasets/v16/validation/all-builds.training_readiness.json`
- Current terrain lane supervises:
  - height
  - normals
  - alpha
  - holes
  - liquid mask
  - MCLY
- `liquid_height` remains in the dataset but is deferred from the current terrain model.
- Short-lived terrain-lane `liquid_height` supervision was superseded.
- Multi-build training smoke is now proven:
  - run: `smoke_v16_full_corpus_post_fix`
  - output: `wow-viewer/models/v16/runs/smoke_v16_full_corpus_post_fix/`
  - result: 1 CPU epoch completed cleanly on curated tiles from the six-build corpus
- Validation alpha QA fix is now landed:
  - the trainer no longer renders raw `alpha[...,0]` as the only alpha GT view
  - validation snapshots now write `alpha_gt_painted_max.png` / `alpha_pred_painted_max.png`
  - focused proof run: `wow-viewer/models/v16/runs/smoke_alpha_validation_fix/`
- Epoch-rotating train subsets are now landed:
  - `train-max-tiles` defines the persistent train pool for the run
  - `train-epoch-tiles` can draw a fresh no-replacement subset from that pool every epoch
  - `train_epoch_orders.jsonl` now records both selected positions and final order per epoch
  - focused proof run: `wow-viewer/models/v16/runs/smoke_epoch_rotation/`
- Loader defaults are less throttled for CUDA runs:
  - `--num-workers=-1` auto-resolves a worker count
  - `persistent_workers` defaults on when workers are active
  - `prefetch-factor` default is now `4`
- Basic trainer-side quality curation is now landed:
  - `--curation-quality-profile basic` is the default
  - low-signal flat tiles are dropped before subset selection
  - weighted curation now favors richer tiles when `train-max-tiles` / `val-max-tiles` cap the pool
  - focused proof run: `wow-viewer/models/v16/runs/smoke_quality_curation/`
- Alpha/minimap discrepancy audit is now landed:
  - script: `data-harvester/scripts/audit_v16_alpha_minimap_alignment.py`
  - sampled corpus result: `edge_f1_mean≈0.54`, `median≈0.64`, `p10=0.0`
  - this confirms a real mismatch bad tail rather than purely subjective screenshot reading
- Best-epoch qualitative snapshots are now landed:
  - every new best `val_h` writes a fresh random validation sample set under `validation/best_epoch_XXXX/`
  - this is separate from the normal interval snapshots so review is not pinned to one repeating tile set
- The main doc routing is now cleaner:
  - root README for repo orientation
  - data-harvester README for operational commands
  - V16 spec for dataset / trainer / inference contract
- The next architecture lane is now named V16.1:
- The first V16.1 implementation slice is now landed:
  - `src/harvester/v16_1_dataset.py`
  - `src/harvester/v16_1_models.py`
  - `scripts/train_v16_1_common.py`
  - `scripts/train_v16_1_height.py`
  - `scripts/train_v16_1_normal.py`
  - `scripts/train_v16_1_holes.py`
  - `scripts/train_v16_1_liquid.py`
  - `scripts/train_v16_1_texcomp.py`
  - `scripts/infer_v16_1.py`
  - trainer CLI now supports bounded smoke caps through
    `--max-train-samples` / `--max-val-samples`
  - `V161Dataset` now exposes shared object-loss weights plus the first coarse
    liquid-type labels from `mcnk_flags_16`
  - the texture-decomposition family now includes recomposition validation
    output in the initial implementation
- Focused V16.1 proof now exists:
  - 1-epoch CPU normal smoke run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_cpu/`
  - 1-epoch CPU height smoke run:
    - `wow-viewer/models/v16_1/height/runs/smoke_height_cpu/`
  - stitched inference smoke using the normal checkpoint:
    - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_normal/3_3_5_12340.pred.zarr`
  - stitched inference smoke using the height checkpoint:
    - `wow-viewer/output/datasets/v16_1_inference/smoke_infer_height/3_3_5_12340.pred.zarr`
- The V16.1 normal trainer now uses terrain-aware loss gating:
  - `normal_mask`
  - object-filter-derived terrain weights
  - `mddf_mask` / `modf_mask`
  - `liquid_mask`
  - blended objective: angular alignment + vector agreement + `z` stabilization
- The V16.1 normal trainer now also has deformation-aware steering:
  - target height gradients and local normal variation increase loss weight on
    terrain-shape transitions
  - CLI knob: `--normal-detail-boost`
  - focused proof run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_detail_steering_cpu/`
    - logged `train_normal_detail_mean≈1.88`, `val_normal_detail_mean≈1.75`
- The V16.1 normal lane now also carries raw supervision guidance channels:
  - `terrain_valid_mask_257`
  - `object_presence_257`
  - `alpha_painted_256`
  - `mcly_any_16`
  - `what_plate_flag`
  - focused proof run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_supervision_channels_cpu/`
    - logged `what_plate_rate=0.0`, `alpha_painted_cov≈0.66`, `mcly_cov=1.0`
- New quilt tooling now exports validation-style hard-region maps as full stitched PNGs:
  - script: `wow-viewer/data-harvester/scripts/quilt_v16_1_debug_signals.py`
  - reproduces the same `hard_region` + `transition` signal math used in `train_v16_1_common.py`
  - writes per-`build/map` local-normalized and global-normalized quilts plus `train_mask` for context
  - bounded smoke proof completed with `--builds 3_3_5_12340 --maps Azeroth --max-tiles 2 --quilt-tile-size 32`
    - output: `wow-viewer/output/tmp/quilt_debug_smoke/`
- Archive-backed ADT harvest now has a first bounded WMO-mask repair:
  - `AdtTensorPackBuilder.BuildObjectMasks` now tries to raster transformed WMO
    mesh triangles into `modf_mask` / `object_filtered_mask` instead of always
    painting projected bounds rectangles
  - the harvest path passes archive asset reads into the builder so WMO render
    documents can be opened during pack generation
  - raw harvest QA no longer requires a finalized store:
    - `inspect_v16_harvest_samples.py` can inspect NPZ samples without loading a
      `.zarr` store unless `--compare-zarr` is explicitly requested
  - Zarr-mutating dataset commands now require `--allow-zarr-write`:
    - `build`
    - `patch-liquids`
    - `patch-objects`
    - `merge-builds`
  - current proof is targeted compile-only:
    - `dotnet build wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -c Debug`

### V16.1/V17.1 Trainer Updates (Session)

- `v17_1_normals` no longer uses refiner by default; training path is now direct minimap->normals + supervisor-only height guidance.
- `v17_1_normals` default `height_supervision_weight` is now `1.0` and startup guard rejects disabled height supervision for this variant.
- Autotune now measures warmup+steady probe steps (not one startup step) and records measured reserved peaks for candidate decisions.
- Added autotune controls:
  - `--autotune-safety-factor`
  - `--autotune-probe-warmup-steps`
  - `--autotune-probe-measure-steps`
- Added conservative `v17_1_normals` loader defaults to reduce host RAM spikes (workers/prefetch/persistent worker behavior).
- Added invalid-region neutral normal loss to reduce object-mask leakage in predictions:
  - CLI: `--invalid-neutral-weight`
  - metrics: `train_normal_invalid_neutral`, `val_normal_invalid_neutral`

### Paste Mining / Dataset Refinement Surface

- New script landed: `wow-viewer/data-harvester/scripts/mine_v17_pastes.py`
  - extracts candidate paste regions from normal guidance maps
  - emits candidate manifests, brush library seed manifests, and optional cell-library outputs
  - supports cross-build/map dedupe via perceptual hash (`--dedupe`)
- New script landed: `wow-viewer/data-harvester/scripts/mine_v18_pastes_canvas.py`
  - stitches tiles into build/map canvases before extraction
  - emits `canvas_bbox` + `tile_coverage` per candidate for multi-tile paste preservation
  - `tile_coverage` now carries `tile_id` for downstream trainer-manifest projection
  - writes machine-readable evidence:
    - `summary.json`
    - `candidates.jsonl`
    - `canvas_summary.json`
    - `config.snapshot.json`
  - writes debug overlays/signals:
    - `overlays/*_canvas_overlay.png`
    - `canvas_debug/*_signals.png`
  - now includes Phase 2 deterministic dedupe + library seams:
    - per-candidate `rgb_fingerprint`
    - alpha-layer descriptors:
      - `layer_means`
      - `layer_coverage`
      - `dominant_layers`
      - `alpha_layer_signature`
    - cluster lineage fields:
      - `cluster_id`
      - `canonical_id`
      - `variant_rank`
      - `cluster_size`
      - `is_canonical`
    - dedupe artifacts:
      - `candidates_deduped.jsonl`
      - `cluster_summary.jsonl`
      - `dedupe_stats.json`
    - cluster QA atlases:
      - `cluster_atlas/*.png`
      - `clusters_canonical_top_atlas.png`
- New script landed: `wow-viewer/data-harvester/scripts/quilt_v16_1_debug_signals.py`
  - emits per-build/per-map stitched `hard_region`, `transition`, and `train_mask` quilts
- Bounded proof for Spec 024 Phase 1 (canvas mining) now exists:
  - command:
    - `uv run python -u scripts/mine_v18_pastes_canvas.py --builds 3_3_5_12340 --maps Azeroth --max-tiles 1024 --seed 42 --component-threshold 0.28 --out-dir ../output/tmp/v18_canvas_smoke_dense`
  - result:
    - `tiles_considered=1024`
    - `candidates=24`
    - `multi_tile_candidates=6`
    - `multi_tile_ratio=0.25`
- Bounded proof for Spec 024 Phase 2 (cross-build deterministic dedupe) now exists:
  - command:
    - `uv run python -u scripts/mine_v18_pastes_canvas.py --builds 0_5_3_3368 3_3_5_12340 --maps Azeroth --max-tiles 2048 --seed 42 --component-threshold 0.28 --dedupe --out-dir ../output/tmp/v18_canvas_phase2_run1`
  - result:
    - `candidates=48`
    - `clusters=48`
    - `duplicates_dropped_if_canonical_only=0` on this bounded corpus slice
    - deterministic hashes:
      - `selection_hash=999c2e6880225c24fd979b70538f1353d60f8187b51ba2abfe5c43b40cefabe0`
      - `cluster_hash=4dc8ffa09cac92cd3c07ede9f8ad88aec91167c99837ffc418ec438622db14a8`
  - rerun stability proof:
    - identical command with `--out-dir ../output/tmp/v18_canvas_phase2_run2`
    - `FC /B` reported no differences for `cluster_summary.jsonl`
- New script landed: `wow-viewer/data-harvester/scripts/build_v18_refined_manifest.py`
  - consumes `candidates_deduped.jsonl` and emits trainer-compatible refined manifests:
    - `kept_tiles.parquet`
    - `tiles.parquet`
    - `tiles.jsonl`
  - applies normal-aware quality gates and cluster-balanced selection controls
  - writes evidence package:
    - `summary.json` (selection hash, cluster distribution, duplicate-ratio metrics, bucket/build counts)
    - `selected_candidates.jsonl`
    - `config.snapshot.json`
- Bounded proof for Spec 024 Phase 3 (refined manifest generation + trainer load) now exists:
  - manifest build command:
    - `uv run python -u scripts/build_v18_refined_manifest.py --deduped-candidates ../output/tmp/v18_canvas_phase3_source --run-name v18_refined_manifest_phase3_smoke --output-dir ../output/tmp/v18_refined_manifest_phase3_smoke --max-clusters 32 --max-variants-per-cluster 2 --max-tiles 128 --min-score-mean 0.10 --min-transition-mean 0.70 --min-hard-mean 0.30 --min-train-mask-mean 1.0`
  - manifest result:
    - `candidate_rows_in=48`
    - `candidate_rows_selected=32`
    - `kept_tiles=42`
    - `selection_hash=83cc25fa09b855297103feba95ae267de1444d772885da950337a916f3b4b71c`
  - trainer compatibility smoke:
    - `uv run python -u scripts/train_v16_1_normal.py --builds 0_5_3_3368 3_3_5_12340 --curation-manifest ../output/tmp/v18_refined_manifest_phase3_smoke --device cpu --epochs 1 --batch-size 2 --train-max-tiles 24 --train-epoch-tiles 8 --val-max-tiles 8 --rotate-val-tiles --val-epoch-tiles 4 --num-workers 0 --no-compile --run-name v18_refined_manifest_load_smoke`
    - completed with `train=24`, `val=3`, `new best val_loss=0.7973`
- New script landed: `wow-viewer/data-harvester/scripts/build_v18_composition_graph.py`
  - builds composition graph from deduped canvas candidates:
    - cluster-node summaries
    - adjacency/co-occurrence edges
    - stable composition-family IDs
  - emits AreaID-aware candidate/group metadata:
    - `area_id_coverage`
    - `dominant_area_ids`
  - AreaID source is optional (`--area-id-map`); missing labels fall back to `unknown` as soft labels
  - output artifacts:
    - `composition_candidates.jsonl`
    - `composition_nodes.jsonl`
    - `composition_edges.jsonl`
    - `composition_families.jsonl`
    - `summary.json`
- Bounded proof for Spec 024 Phase 4 (composition graph + deterministic rerun) now exists:
  - run1 command:
    - `uv run python -u scripts/build_v18_composition_graph.py --deduped-candidates ../output/tmp/v18_canvas_phase3_source --output-dir ../output/tmp/v18_composition_phase4_run1 --adjacency-margin-px 64 --cooccur-edge-min 2`
  - run2 command:
    - same command with `--output-dir ../output/tmp/v18_composition_phase4_run2`
  - deterministic result:
    - `graph_hash=eccd70abda63f7e5dcbabe2528f2809b62aed7455780c2cef125291a7333c09a`
    - `family_hash=800a2029757e7988f736d70dc787087d8b4f72e692c3007e32fa4bdbdd8c2771`
    - `FC /B` reported no differences for `composition_edges.jsonl`
- Phase 4 integration into refined manifests is now landed:
  - `build_v18_refined_manifest.py --composition-graph ../output/tmp/v18_composition_phase4_run1`
  - refined output now includes composition-family balancing metadata:
    - `source_composition_family_ids`
    - `source_composition_family_count`
    - `composition_balance_weight_mean`
  - trainer smoke with composition-augmented manifest completed:
    - run: `v17_1_v18_refined_manifest_phase4_load_smoke`
    - `train=24`, `val=3`, `new best val_loss=0.7973`
- New script landed: `wow-viewer/data-harvester/scripts/build_v18_paste_library_catalog.py`
  - deterministic paste-family naming and catalog emission
  - emits stable IDs and naming metadata:
    - `paste_id`
    - `canonical_name`
    - `aliases`
    - `name_confidence`
    - `review_state` / `review_required`
  - catalog outputs:
    - `paste_library_catalog.json`
    - `paste_library_catalog.jsonl`
    - `summary.json`
- Bounded proof for Spec 024 Phase 5 (auto-naming + catalog stability) now exists:
  - run1 output: `../output/tmp/v18_paste_library_phase5`
  - run2 output: `../output/tmp/v18_paste_library_phase5_run2`
  - deterministic result:
    - `stable_name_hash=1ae9a2d2900a24aba4f7b34c260f747bd683527317278abdb2a22a783f372a2f`
    - `FC /B` reported no differences for `paste_library_catalog.jsonl`
- New script landed: `wow-viewer/data-harvester/scripts/run_v18_baseline_contract.py`
  - defines baseline profile contract and writes `baseline_profiles.json`
  - runs bounded refined baseline + non-ref baseline and emits comparison report
- Bounded proof for Spec 024 Phase 6 (baseline launch + comparison) now exists:
  - command:
    - `uv run python -u scripts/run_v18_baseline_contract.py --refined-manifest ../output/tmp/v18_refined_manifest_phase4_smoke --builds 0_5_3_3368 3_3_5_12340 --profile small --output-dir ../output/tmp/v18_baseline_contract_phase6`
  - refined run:
    - `v17_1_v18_baseline_small_refined`
    - epoch1 `val_loss=0.6008`, `elapsed_s=7.02`
  - non-ref run:
    - `v17_1_v18_baseline_small_nonref`
    - epoch1 `val_loss=0.6505`, `elapsed_s=7.22`
  - report outputs:
    - `comparison_report.json`
    - `comparison_report.md`

### Spec 025 — Object Roof Mask Library (Session)

- Phase 1 bounded roof-library proof remains valid on staged `3_3_5_12340`:
  - run: `wow-viewer/output/datasets/object_roof_library/smoke_spec025_phase1_335/`
  - summary: `placements_selected=744`, `exemplars_kept=52`, `families_total=36`
  - validator: `scripts/validate_v18_object_roof_library.py`
  - current re-check status: `pass`

- Phase 2 object-roof mask generation lane is now landed:
  - new scripts:
    - `scripts/infer_v18_object_roof_masks.py` (learned fallback inference host)
    - `scripts/validate_v18_object_roof_masks.py` (bounded mask quality validator)
    - `scripts/patch_v18_object_roof_masks.py` now writes label-contract/report artifacts under `output/tmp/object_roof_patch_reports/` (not inside `.zarr`)
  - label contract proof:
    - `output/tmp/object_roof_patch_reports/smoke_spec025_patch_335/3_3_5_12340/object_roof_label_contract.json`
  - bounded anchor inference proof:
    - `output/tmp/v18_object_roof_infer_smoke_335_30_53/summary.json`
    - `tiles_non_empty=1`, `mean_mask_coverage≈0.0945`
  - bounded mask validation proof:
    - `output/tmp/v18_object_roof_infer_smoke_335_30_53/mask_validation_report.json`
    - status: `pass`

- Phase 3 training integration lane is now landed:
  - dataset/model/trainer consume object-roof auxiliary signals:
    - `v16_1_dataset.py` object-roof channels/weights
    - `v16_1_models.py` `V161NormalObjectRoofModel`
    - `train_v16_1_common.py` variant `v18_object_roof_aux`, object-roof sieve in normal loss, and evidence fields (`object_roof_mask_source_counts`, coverage metrics)
  - bounded CUDA smoke run (aux enabled):
    - `models/v18/normal/runs/v18_oroof_smoke_spec025_v18_oroof_aux_cuda/`
    - confirms `resolved_input_contract=minimap_rgb+object_roof_mask`
  - bounded same-pool baseline run:
    - `models/v18/normal/runs/smoke_spec025_v18_baseline_samepool_cuda/`

- Phase 4 operational proof is now captured for the bounded lane:
  - roof library run + validation pass recorded
  - object-mask generation run + validation pass recorded
  - bounded comparison artifacts recorded between aux and same-pool baseline normal runs

- Spec task checklist updated at:
  - `wow-viewer/specs/025-object-roof-mask-library-and-minimap-sieve/tasks.md`
  - completed: `T001`, `T003`-`T022`
  - remaining deliberate open seam: `T002` (MdxViewer one-at-a-time asset capture with explicit pose metadata)

- Spec 025 T002 object-capture audit + first wow-viewer capture-policy slice landed:
  - architecture note:
    - `wow-viewer/docs/architecture/spec025-t002-object-capture-audit-2026-05-26.md`
  - shared-runtime capture-policy propagation is now wired end-to-end:
    - `ValidationCaptureScenePolicy` includes explicit culling-override knobs
    - `ValidationWorldScenePolicyApplier` maps these into policy state
    - `ValidationWorldSceneAdapter.BuildFrameRequest(...)` forwards fog/object-streaming/MDX-height/culling knobs into runtime request
    - `WowViewerWorldRuntimeFrameRequest`/`WowViewerWorldRuntimeBridge` carry those knobs into `WorldObjectVisibilityContext`
    - `WorldObjectVisibilityCollector` now honors capture override flags and MDX max-bounds-height suppression
  - focused test proof (filtered suite):
    - `ValidationCaptureScenePolicyTests`
    - `ValidationWorldScenePolicyApplierTests`
    - `ValidationWorldSceneAdapterTests`
    - `WorldObjectVisibilityCollectorTests`
    - result: pass (`18/18`)
  - bounded staged proof:
    - `WowViewer.Tool.ValidationCapture capture --real-scene-dry-run`
    - staged client root: `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`
    - tile: `Azeroth_30_48`
    - variants: Primary / NoLiquids / NoObjects / ObjectsOnly all reported `sceneContent=True`, `tileLoaded=True`, `pendingObjects=0`
  - open follow-up remains:
    - dedicated object-render backend parity and one-at-a-time per-asset pose-capture orchestration for full T002 closure

- Spec 025 T002 second bounded slice (automation cutover) landed:
  - `WowViewer.Tool.ValidationCapture` now includes `capture-batch` command
    - required: `--client-root --map-input --dataset-root --output-root --ledger-path`
    - reads `manifest_capture_ledger.json`, skips `captured_complete`, expands pending entries into all 4 variant requests per tile
    - reuses default scene + variant policy composition from single-tile `capture`
  - `build_v16_dataset.py generate-viewer-stubs` messaging/help now routes to wow-viewer `capture-batch` as primary next step (legacy MdxViewer scripts remain for compatibility comparison only)
  - focused tests added and passing under `ValidationCaptureCommandTests`:
    - `Execute_CaptureBatchMissingLedger_ReturnsOne`
    - `Execute_CaptureBatchDryRun_ReturnsZeroAndPrintsSummary`
  - focused proof command:
    - `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ValidationCaptureCommandTests"`
    - result: `5/5` passed

- Spec 025 T002 third bounded slice (Python automation bridge) landed:
  - `wow-viewer/data-harvester/scripts/build_v16_dataset.py` now has `capture-renderer-truth`
  - command behavior:
    - discovers `WowViewer.Tool.ValidationCapture.exe`
    - loads per-build `manifest_capture_ledger.json`
    - skips `captured_complete` rows
    - groups pending rows by `map`
    - emits temporary per-map ledgers
    - invokes wow-viewer `capture-batch` per map group using staged client roots and forwarded mode/resolution/build flags
  - focused proof:
    - `uv run python scripts/build_v16_dataset.py --help` lists `capture-renderer-truth`
    - `uv run python scripts/build_v16_dataset.py capture-renderer-truth --build 3_3_5_12340 --dry-run` completed cleanly in this environment (tool/root resolved; no ledger present so group count remained 0)

- Spec 025 T002 fourth bounded slice (pose metadata carry-through) landed:
  - `WowViewer.Tool.ValidationCapture capture-batch` now accepts optional pose metadata fields in ledger rows:
    - `asset_path`, `instance_type`, `unique_id`, `rot_x`, `rot_y`, `rot_z`, `scale`
  - render-mode batch runs now emit per-tile pose artifacts:
    - `<dataset-root>/pose-metadata/<tile_name>_pose.json`
  - `build_v16_dataset.py generate-viewer-stubs` now enriches ledger rows from real `<build>.zarr/placements.parquet` (prefers `modf`, then `mddf`) so pose metadata is sourced from dataset placement truth, not synthetic tile stubs
  - focused test proof:
    - `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ValidationCaptureCommandTests"`
    - result: `6/6` passed
  - bounded functional proof:
    - regenerated ledger for `3_3_5_12340`
    - ran `capture-renderer-truth --stub-scene` on single-tile trimmed ledger
    - verified emitted pose artifact at `output/tmp/mdxviewer_validation_smoke/3_3_5_12340/pose-metadata/AhnQiraj_46_27_pose.json` contains non-null placement-derived metadata (`asset_path`, `instance_type`, `unique_id`, `rot*`, `scale`)

- Spec 025 T002 fifth bounded slice (full per-tile placement resolution) landed:
  - `build_v16_dataset.py generate-viewer-stubs` now hydrates each ledger tile row with all placement rows from `<build>.zarr/placements.parquet`:
    - `object_instance_count`
    - `object_instances[]`
  - representative top-level pose fields remain for compatibility, but no longer represent the full tile placement set.
  - `capture-batch` pose artifact outputs now preserve both full-instance fields in emitted JSON.
  - focused parity proof (`3_3_5_12340`):
    - regenerated ledger tile rows: `5134`
    - `placements.parquet` rows: `1,015,470`
    - mismatches between ledger per-tile counts and placement-table per-tile counts: `0`
    - multi-instance sample tiles confirmed (`Northrend_21_23=3580`, `Northrend_22_22=3437`, `Azeroth_32_39=3015`).

- V18 undecoded blob-preservation sketch documented:
  - `wow-viewer/docs/architecture/v18-undecoded-blob-datastore-sketch-2026-05-27.md`
  - defines sidecar `raw_blobs` manifest + content-addressed payload layout, phased migration path, and validation contract without reopening existing reader implementations.

- Dataset contract hardening for "single-script full indexing" is now landed in the main build flow:
  - `build_v16_dataset.py` now writes `decoded_metadata.parquet` during build with one row per harvested tile (`tile_id`) including decoded metadata payload + key provenance fields.
  - new `decoded_metadata_validation.json` parity validation checks:
    - row-count equality vs `index.parquet`
    - 1:1 `tile_id` coverage (no missing/extra/duplicate tile rows)
    - JSON object validity of `decoded_metadata_json`
  - build command now exposes decoded metadata validation toggles and runs that validation by default.
  - merge command now preserves/remaps decoded metadata into merged stores and validates parity post-merge.
  - validate-signals command now validates both signal coverage and decoded metadata integrity.

### Spec 024 Expansion (Session Close)

- Spec/plan/tasks for `024-v18-canvas-paste-refinement-layer` were expanded to reflect the macro-artwork thesis:
  - map-as-canvas detection first, tile metadata second
  - alpha-layer-aware dedupe required (layer signatures)
  - MCNK AreaID overlap required for macro-zone grouping/lineage
  - paste-library metadata contract added (stable IDs, canonical names, aliases, role/shape tags)
  - deterministic auto-naming + confidence + review/lock workflow added as a dedicated phase
  - refined manifests must support family-balanced sampling (not raw-frequency sampling)
- Next implementation anchor for fresh chat:
  - `wow-viewer/specs/024-v18-canvas-paste-refinement-layer/tasks.md`
  - Spec 024 task checklist is now fully implemented (`T001` through `T028`)
  - next likely lane is operational tuning on broader corpus slices (dedupe compression, AreaID map feeds, medium/large baseline runs)
  - keep `mine_v17_pastes.py` as transitional tile-local tooling only
- The shared V16.1 trainer now has real gradient accumulation:
  - CLI flag: `--grad-accum-steps`
  - trainer prints now flush immediately instead of hiding early startup
  - focused proof run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_accum_cpu/`
    - micro-batch `1`, accumulation `4`, effective batch `4`, `opt_steps=1`
- The shared V16.1 trainer now preserves the useful V16 runtime seam instead of
  stripped-down defaults:
  - `torch.compile`
  - auto CUDA-friendly `--num-workers -1`
  - `--persistent-workers`
  - `--prefetch-factor`
  - focused proof run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_compile_gpu/`
    - completed on GPU with `torch.compile: enabled`
- The shared V16.1 trainer now also preserves the useful V16 small-epoch seam:
  - run-level curated pool caps:
    - `--train-max-tiles`
    - `--val-max-tiles`
  - rotating per-epoch train subsets:
    - `--train-epoch-tiles`
  - focused proof run:
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_curated_epoch_rotation_cpu/`
    - `32/63` train-pool cap, `8` sampled train tiles for epoch `1`
    - evidence written:
      - `train_pool_summary.json`
      - `val_pool_summary.json`
      - `train_epoch_orders.jsonl`
- V16.1 now has a separate reusable curation layer between Zarr and trainers:
  - module: `src/harvester/v16_curation.py`
  - builder: `scripts/build_v16_curation_manifest.py`
  - trainer/dataset consumption: `--curation-manifest`
  - first profile: `normal_terrain_v1`
  - builder now supports multi-process tile auditing via `--workers` and
    `--chunk-size`
  - focused proof outputs:
    - `wow-viewer/output/datasets/v16/curation/smoke_normal_curation_335/`
    - `wow-viewer/models/v16_1/normal/runs/smoke_normal_curated_cpu/`
    - `wow-viewer/output/datasets/v16/curation/smoke_normal_curation_335_mt/`
    - `wow-viewer/output/datasets/v16/curation/smoke_normal_curation_335_whatplate/`
  - sampled `3_3_5_12340` proof rejected `59/128` low-signal or blank-normal
    cases before training (`keep_ratio≈0.54`)
  - the current curation proof also rejected explicit blank genesis tiles:
    - `blank_what_plate_tile=4`
- Operator docs now expose that workflow directly:
  - `wow-viewer/data-harvester/README.md` documents manifest build, curated
    V16.1 normal training, resume, and VRAM tuning ladder
  - `wow-viewer/README.md` now points root-level readers at the curation-first
    V16.1 path and explicitly calls out that V16.1.x resume uses
    `--resume-checkpoint` rather than the older V16 `--resume-from auto`
  - current recommended command contract is now documented:
    - curation: `--workers -1 --chunk-size 128`
    - train: `--batch-size 16 --grad-accum-steps 1`
    - small scouting pool: `--train-max-tiles 400 --train-epoch-tiles 128 --val-max-tiles 48`
    - VRAM-first ladder:
      - `16 x 1`
      - `20 x 1`
      - `24 x 1`
      - fallback `12 x 1`, `8 x 1`, then accumulation rungs
- V16.1 direction has shifted from height-first to normal-first for terrain
  signal learning, with height follow-on meant to absorb what the normal lane
  teaches about minimap-to-terrain supervision.
- The next architecture lane is now named V16.1:
  - separate `minimap -> target-family` trainers
  - no shared trainable weights across height / normals / holes / liquids / texture decomposition
  - liquids become footprint + type, not only a soft mask
  - alpha moves into a dedicated MCLY/MCAL decomposition + recomposition family
  - existing D1 minimap-to-tileset work is now explicitly the migration baseline for that family
  - shared object-mask loss gating remains part of the trainer contract
  - V16 remains the baseline/reference path while V16.1 is brought up
  - the split-up families are linked back together to build final output signals
- The next bounded fresh-chat implementation pack is now V16.1.1:
  - spec pack:
    - `wow-viewer/specs/007-v16-1-1-curated-normal-acceleration/`
  - purpose:
    - keep V16.1 as the landed base
    - improve sample efficiency in the normal lane before more long training
  - planned upgrades:
    - difficulty-aware curated manifests
    - bucket-aware epoch sampling
    - stronger hard-region normal weighting
    - optional uncertainty-guided normal loss
    - mixed `400`-tile scouting workflow before larger reruns
- The first V16.1.1 implementation slice is now landed:
  - curation builder:
    - new profile `normal_terrain_v16_1_1`
    - per-tile usefulness scoring
    - difficulty buckets:
      - `easy`
      - `medium`
      - `hard`
      - `pathological`
    - `summary.json` now writes bucket counts, examples, and scouting guidance
  - trainer:
    - manifest ingestion carries bucket metadata into the normal lane
    - new CLI flag:
      - `--bucket-sampling-profile v16_1_1_normal`
    - per-epoch sampler evidence:
      - `train_epoch_orders.jsonl`
      - `train_epoch_bucket_usage.jsonl`
    - hard-region weighting now includes painted alpha / MCLY transition signal
      while terrain-valid masking stays authoritative
    - best-model tracking is explicit again:
      - `best_val`
      - `best_epoch`
    - validation previews are best-gated again:
      - `validation/best_epoch_XXXX.png`
    - validation previews now render multiple samples again:
      - up to `8` rows from the selected validation batch per preview image
    - validation previews now also carry visible labels again:
      - panel labels
      - per-sample row headers with tile metadata
    - startup VRAM autotune is now landed:
      - dormant `--target-vram-gb` seam is active
      - batch-size ladder probe runs before loader creation
      - `train_epoch_tiles` can rescale automatically to preserve steps/epoch
      - evidence writes at `evidence/batch_autotune.json`
      - focused proof run:
        - `wow-viewer/models/v16_1/normal/runs/smoke_v16_1_autotune_gpu/`
  - first longer real-run proof now exists at:
    - `wow-viewer/models/v16_1/normal/runs/v16_1_1_normal_pool800_epoch256_autotune12_compile/`
  - current continuation rule for that lane:
    - resume the existing run with `--resume-checkpoint .../checkpoints/v16_1_normal_last.pt`
    - increase `--epochs` to the new total target instead of treating it as extra epochs to add
    - the shared V16.1 trainer now also extends the resumed cosine schedule to that higher total instead of restoring the old run ceiling unchanged
- Focused proof now exists for that slice:
  - curation smoke:
    - `wow-viewer/output/datasets/v16/curation/smoke_v16_1_1_curation_335/`
    - result: `21/32` kept, bucket mix `hard=14`, `pathological=7`
  - manifest-ingestion normal smoke:
    - `wow-viewer/models/v16_1/normal/runs/smoke_v16_1_1_bucket_cpu/`
    - result: 1 CPU epoch completed cleanly with bucket mix printed at startup
  - bucket-rotation smoke:
    - `wow-viewer/models/v16_1/normal/runs/smoke_v16_1_1_bucket_rotation_cpu/`
    - result: epoch `1` sampled `hard=3`, `pathological=1` from a `16`-tile train pool and wrote the new bucket-evidence logs

### Alpha / LK Conversion Lane
- `AlphaToLk` and `LkToAlpha` are both landed in shared `wow-viewer` surfaces.
- Real-data `LkToAlpha` proof exists:
  - `4_0_0_11927 / Azeroth`
  - `839/839` tiles
  - terrain + WMOs rendered in MdxViewer
- Current shared alphaWDT rules that still matter:
  - `MAIN` is row-major
  - always emit all `256` MCNKs
  - `MCRF` stays FourCC-wrapped
  - top-level chunks are contiguous
  - doodads use single-owner chunk routing
  - shared placement rotation stays in raw-file convention

## In Progress
- V16.1.3 height-channel normal model 1000-epoch run:
  - run: `v16_1_3_height_normal_pool4000`
  - `--height-channel` adds `height_norm` as 4th input channel
  - autotune selected batch-size=48 for 12GB VRAM target
  - torch.compile enabled, ~172s/epoch
  - resumed from smoke checkpoint (epoch 10), currently at epoch 12+
  - first two resumed epochs: val_loss=0.3160, 0.3199
  - run dir: `wow-viewer/models/v16_1/normal/runs/v16_1_3_height_normal_pool4000/`
- First real V16 training run:
  - `v16_full_corpus_epoch_rotation`
  - `train-max-tiles 4000`
  - `train-epoch-tiles 1350`
  - `val-max-tiles 150`
  - `batch-size 72`
  - `gpu-duty-cycle 100`
- WL* partial chunk-fill semantics in the loader / trainer.
- V16.1 spec pack and continuity routing for the dense-correlation model family.
- V16.1 liquid, texcomp, and holes smoke proof.
- Additional target-aware curation profiles beyond `normal_terrain_v1`.
- Full stitched multi-family output proof into one `.pred.zarr` bundle.
- Object segmentation Model A.
- Global asset vocabulary for instance/asset follow-up work.
- PM4 cross-reference / object mapping follow-up.
- PM4 `MSHD.Field04` region-id promotion is now landed in `wow-viewer` and consumed by `MdxViewer` for overlay grouping/coloring/debug export, selected-region peer inspection, and LLM-oriented visible-overlay evidence bundles; the viewer compile blocker from the `M2ToMdxConverter` ambiguity was also cleared.

## High-Value Open Gaps
- Forward `AlphaToLk` AreaID wiring.
- Exact doodad-border ownership for large cross-chunk placements.
- Full chunk-preservation closure is still open:
  - `MFBO`
  - `MCCV`
  - `MCLV`
  - `MTXF`
  - higher-fidelity `MH2O`

## Spec 013 / 014 — Terrain MCAL & Object Mask Rendering Fixes (2026-05-23)

### Spec 013 (Object Mask Rendering Fix) — Done
- Root cause: `ComputeTilePlanarMin/Max` had swapped `tileX`/`tileY` axes.
- Fix: `rendererX = MapOrigin - tileX * TileSize`, `rendererY = MapOrigin - tileY * TileSize`.
- `BuildChunkPositions` IndexX/IndexY swap fixed in `WorldGpuPreviewRenderer.cs`.
- UniqueID deduplication added in bridge instance builder.
- Validation: `WMO visible=1, MDX visible=53` on `3_3_5_12340 / Azeroth_30_48`.

### Spec 014 (Terrain MCAL Rendering Parity) — In Progress
- `FillAlphaShadowSlice` now writes implicit 255 alpha for layers without MCLY `0x100` flag.
- `WorldTerrainChunkData` now has `ShadowMap` property, threaded from `AdtTextureChunk.ShadowMap` via `WorldTerrainTileBuilder` (LK path).
- Shadow map written into channel 3 of alpha-shadow texture array in `FillAlphaShadowSlice`, with edge-clamped indexing matching MdxViewer reference.
- Terrain shader now applies shadow darkening from `alphaShadow.a`: `result *= mix(1.0, 0.4, shadow)`.
- Terrain shader UV (`vec2(-vWorldPosition.y, -vWorldPosition.x) * texScale`) verified identical to MdxViewer.
- Build passes, 477/492 tests pass (15 pre-existing failures unrelated to these changes).
- Remaining: GPU validation capture with MCAL fixes on both anchors.

## Not Yet
- Completed / running proof for a production-oriented V16 training run with epoch rotation is still pending final training outcomes.
- V16.1 liquid footprint/type trainer smoke proof.
- V16.1 texture decomposition/recomposition trainer smoke proof.
- V16.1 holes trainer smoke proof.
- Asset-attribute model / PM4 cross-ref workflow.
- Broader chunk-for-chunk terrain conversion closure.

## Spec Status Update (2026-05-31)
- **Spec 029** (WMO minimap BLP harvest): spec+plan+tasks complete, not yet implemented.
- **Spec 030** (WMO render pass architecture): spec+plan+tasks now complete (plan+tasks created 2026-05-31, 27 tasks total).
- **Spec 031** (terrain cell awareness): spec+plan+tasks now complete (plan+tasks created 2026-05-31, 34 tasks total).
- **Spec 032** (native renderer parity): spec+plan+tasks complete, not yet implemented (depends on 033 completion).
- **Spec 033** (MdxViewer migration): spec+plan+tasks complete, **PHASES 1-2 COMPLETE** (moved, vendored, renamed to WoWViewer v0.5.0), **Phase 3 INCOMPLETE** (WoWMapConverter.Core decoupling + MDX namespace porting remaining).
- **Spec 034** (WowViewer rename): spec+plan+tasks complete, **COMPLETE** (WowViewer.App → WowViewer.App.Defunct, MdxViewer → WoWViewer, v0.5.0).

## MDX/M2 Rendering Breakage (2026-05-31)
- **MDX rendering is BROKEN** after recent M2 rendering fixes and migration.
- Many source files in `wow-viewer/src/viewer/WoWViewer/` still use `MdxLTool.Formats.Mdx` namespace.
- `MdxLTool` was vendored to `wow-viewer/libs/WoW-Tools/MDX-L_Tool/` but types not fully ported to `WowViewer.Core.IO`.
- `Rendering/WmoRenderer.cs` line 32: still references `WoWMapConverter.Core`'s `WmoV14Data` model for geometry.
- Remaining external reference in `WoWViewer.csproj` line 40: `gillijimProject-csharp.csproj` — but this is INSIDE `wow-viewer/libs/` (vendored), so repo-independence rule satisfied.
- Next chat should: port `MdxLTool.Formats.Mdx` types to `WowViewer.Core.IO/Mdx/`, then fix all `using MdxLTool...` references in WoWViewer source files.
