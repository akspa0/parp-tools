# Active Context — wow-viewer

Last updated: 2026-07-30

## Spec 122 — Canonical Dataset Curation and Signal-Mismatch Bucketing — US1-US4 IMPLEMENTED (this session)

- **What it is**: the long-requested C# curation layer. Consolidates dataset quality classification
  (bucketing + mismatch detection), previously reimplemented ad hoc every model generation in
  Python, into one canonical library `WowViewer.Core.Curation` + a `curate` subcommand on
  `WowViewer.Tool.Harvest`, so it can never again get silently dropped between model versions.
- **Full speckit docs written and then implemented same session**: `specs/122-dataset-curation/`
  (spec, plan, research — includes two background-agent external research reports on comparable
  aerial/satellite image-to-terrain ML projects, folded into research.md Part B — data-model,
  contracts, quickstart, tasks).
- **Governing principle (explicit user correction mid-session)**: curation MUST partition, never
  filter. Every tile gets a bucket + finding record, including bad/mismatched ones; every bucket is
  equally queryable. This directly reverses the failure mode of the pre-existing Python tooling
  (see below).
- **US1 (canonical classification)**: `WowViewer.Core.Curation` — `DifficultyBucketClassifier` (4
  buckets, ports `build_v16_curation_manifest._score_row_v16_1_1`'s exact weighted formula, adapted
  for v50: roof coverage always 0 since v50 dropped roof masks, object coverage prefers the strict
  `ObjectGeometryVisibleMask257` else the alpha-builder footprint masks), `CoverageBucketClassifier`
  plus `BlankTileDetector` (ports `is_blank_what_plate`), `HeightNormalMismatchDetector` (exact port of
  `mismatch_detector.py`'s thresholds/severity bands/reason strings), `NonFiniteSignalDetector`,
  `HasFlagTruthfulnessDetector`. Output: two Parquet tables (`curation_manifest.parquet` one row per
  tile, `curation_findings.parquet` one row per finding) + `curation_run.json`
  (`v50-curation-run-v1`) under `<store>/curation/<curation_run_id>/`, plus a `latest` pointer. This
  is the repo's first C#-side Parquet **writer** (Parquet.Net was previously read-only, via
  `V18StorePlacementsReader.cs`).
- **`curate` orchestration**: since this codebase has no C# Zarr reader, `curate` re-derives each
  tile's `TerrainTileTensorPack` fresh from the client (reusing the exact same
  `AlphaTensorPackBuilder`/`BuildPackFromArchiveAdt` path `harvest-stream` already uses — extracted
  a shared `BuildEnrichedTensorPackForTile` helper, a behavior-preserving refactor, not a rewrite)
  and only reads the store's `index.parquet` for row identity. Dry-run-first; `--write` required to
  persist; refuses to write a partial manifest if any tile fails re-extraction (FR-008 full coverage
  is a hard gate, not best-effort).
- **US2 (equal access)**: `harvester/curation_store.py` — `load_curation_manifest`/
  `load_curation_findings`, zero default filtering, resolves `curation/latest` by default. Proven:
  querying `coverage_bucket == "blank"` is the identical column-filter call as `"well_covered"`.
- **US3 (synthetic fidelity)**: `SyntheticFidelityDetector` reuses `MinimapShadingMatch`'s existing
  best-candidate correlation (no new comparison invented) as `synthetic_fidelity_score`; below 0.30
  (same threshold `MinimapLightingProvenance.Infer` already uses for its own mcsh-correlation
  judgement) produces a `synthetic_fidelity_gap` finding. This is the durable, queryable answer to
  the user's mid-session observation that synthetic minimap shading doesn't match authored — it
  measures and exposes the gap; it does not implement a loss function that consumes it (future work).
- **US4 (legacy consolidation) — corrected mid-implementation by a real-caller search**: the
  original plan assumed 4 scattered scripts could become thin shims. A real search found 14+3 real
  callers across V16/V18/V23-era scripts that depend on `v16_curation.py`/`mismatch_detector.py`/
  `spec111/lighting_buckets.py`/`build_v16_curation_manifest.py`'s current behavior for store shapes
  `curate` was never built to read — converting them to shims would have broken real, still-needed
  functionality. **Corrected disposition**: documentation-only pointers added to all four (plus a
  **newly-discovered fifth scattered script**, `spec103_curate_dataset.py` — the one that actually
  produced the real curation output already on disk under
  `output/datasets/v50/v50.1/curation-0_5_3_3368-<Map>*/`, schema `spec103-curation-v1`, a drop
  filter that discarded 779/951 Kalimdor tiles with only aggregate reasons — exactly the anti-pattern
  this feature exists to fix). No logic changed in any of the five; `docs/architecture/
  v50-clean-room-dataset-repo-audit-2026-07-15.md` gained a canonical-entrypoint pointer section.
  One real bug caught and fixed during this pass: a docstring edit that used a truncated
  `old_string` match accidentally closed a multi-paragraph docstring early, turning the rest of it
  into a syntax error — caught by `py_compile`, fixed, verified clean.
- **Real-data validation (not just fixtures)**: `curate --write` run twice against the real
  `H:\CLIENTS\0_5_3_3368` client + the real on-disk `0_5_3_3368-PVPZone02.zarr` store (64/64 tiles,
  ~45-53s). Real, non-degenerate output: all four difficulty buckets populated, both coverage
  buckets populated, three of four lighting statuses populated, 25/64 tiles fidelity-evaluated
  (score range 0.0–0.69, 22 flagged high-severity gap) — directly, quantitatively corroborating the
  user's own synthetic-minimap-shading concern on real data, not just in principle.
  `StoreIndexReader` needed one real-data fix mid-session: pyarrow can write an int64 index column
  as nullable even with no actual nulls, and Parquet.Net returns `long?[]` for it — fixed with a
  defensive reader that fails loudly on a genuine null rather than silently coercing.
- **Proof**: 39/39 `WowViewer.Core.Curation.Tests` pass; full solution Debug build 0 errors; 9/9
  `data-harvester/tests/test_curation_store.py` pass (includes a real C#→pyarrow cross-language
  fixture round-trip, not just same-language self-consistency); full `data-harvester` suite 1154
  passed / 45 skipped / 3 pre-existing unrelated failures (v24 export-map fixture, 2× v25 h1_coarse
  — unchanged from every prior session).
- **Not done this session (explicitly, not silently)**: the SC-003 numeric comparison between
  `mismatch_detector.py`'s legacy output and the new C# `HeightNormalMismatchDetector` on identical
  real tiles was not run (no real height-normal mismatches occurred on the one map validated,
  PVPZone02 — a larger map with more terrain would be needed for a meaningful comparison). Full
  manual visual sign-off on synthetic-fidelity scores (SC-004) is a user step per this project's
  established convention; only the quantitative distribution was assistant-verified. Kalimdor/
  Azeroth/Kalidar (the larger, more terrain-diverse maps) have not yet been run through `curate`.

## Spec 121 — V7-Style WDL-Prior Height Reconstruction (Small Model Lane) — CLOSED (architecture failure)

- **Status**: CLOSED 2026-07-25. The RGB→WDL approach is fundamentally wrong for this project's
  needs. Three architectures (LatticeNet v2/v5, MitB0LatticeNet) hit the same wall: zone-local
  mapping that does not transfer cross-region. The within-map reframe produced a model that
  looked good visually but the detailer (5% improvement over the prior) confirmed the prior was
  already at the noise floor — the detailer had nothing to refine because the RGB→WDL task is
  harder than what v7 actually did.
- **What was learned**: (1) RGB→WDL prediction is a fundamentally harder problem than WDL
  consumption. v7 consumed WDL as input, never predicted it. (2) The detailer works (5%
  improvement, sc002=True) — that piece is salvageable. (3) The correct architecture is: merged
  WDL prior (real + synthetic, no ML) → detailer refines. No RGB→WDL model at all.
- **Salvageable artifacts**: `DetailerMitB0Net` (SegFormer-B0 trunk, ~3.8M params), the
  within-map split machinery, the object-mask tile loss, the store check helpers, the diagnostic
  tooling. All in `harvester/spec121/` and `harvester/v50/geometry_detailer_model.py`.
- **Specs 119/120/121 all closed**: the minimap object identity line (119/120) and the
  RGB→WDL prediction line (121) are both recorded dead ends. The detailer is the only survivor.
- **Next**: user break. When ready, the path forward is: merged WDL prior (Spec 094 Stage 0) →
  detailer refines. No RGB→WDL model.

- **Status (2026-07-24)**: T001–T012 DONE. `harvester/spec121/` (store_check, lattice_backbone_model, object_mask_tile_loss, lattice_backbone_train) + `scripts/spec121_train_lattice_prior.py` + `tests/spec121/` (30 tests, all pass). Full suite 1136 passed / 3 pre-existing unrelated failures; ruff+compileall clean. Real dry-run smoke on `curriculum-0_5_3_3368-dual_v3.zarr` + spec116 split-dual_v2: violation_count=0, lattice arrays present, object-mask absent → graceful `object_mask_signal_present=false` (documented degradation), exits without training.
- **Key numbers**: `MitB0LatticeNet` = **3,469,922 params** at default B0 config (inside 3–30M band; plan refuses `--confirm-run` outside band). Heads are native-direct (LatticeNet v5 rule): outer 17×17 off the encoder's 32×32 stage via learned k2/s2/p1, inner 16×16 off the 16×16 stage; no interpolation in output path.
- **v50.2 naming (user decision)**: this lane's substrate = **v50.2 release** (v50.1 signals + Spec 117 lattice + Spec 118 object-mask arrays). Trainer `--release` defaults to `v50.2`; existing dual_v3 is v50.1 (works for unweighted runs; object-mask weighted runs need the v50.2 rebuild — user-run).
- **G1 evidence (2026-07-24, in-flight run diagnosed at epoch 54)**: train tiles +18.9% vs tile-mean (signal IS learnable locally), held-out −73% vs tile-mean (does NOT transfer across the region-isolated split; both maps fail; third architecture, identical wall). Failure class = zone-local mapping, not capacity, not no-signal. Decision fork recorded in research.md R-1: (A) record negative+stop, (B) reframe as within-map WDL completion (v7's real constraint), (C) synthetic-source discriminating run first. **Awaiting user decision before Phase 4 work.**
- **Next**: T013 USER RUN — Stage A real training (G1 gate: SC-001 ≥15% below tile-mean). **Full runbook: `specs/121-v7-wdl-height/USERGUIDE.md`** (Phase 0 verify → Phase 2 train-now on v50.1 → Phase 1 v50.2 rebuild for mask-weighted runs → Phase 4 paired comparison). Then Phase 4 (T014–T020): prior→coarse bridge + detailer.
- **User decisions (2026-07-24)**: substrate named **v50.2** (research.md D-08); nothing in this lane may carry v24/v25 naming; backbone must be off-the-shelf (SegFormer-B0 HF, proven in-repo by Spec 114), not homebrew. Stage A reframed to **within-map WDL completion** (D-09): after cross-region diagnostic proved zone-local mapping (−73% val), user redirected to within-map held-out split. Split builder at `scripts/spec121_build_within_map_split.py`; trainer auto-detects schema; `diagnose.py` artifact exists for future use.
- **B-reframe code**: `harvester/spec121/within_map_split.py` + CLI + tests (39/39 spec121 tests pass; ruff + compileall clean).
- **What it is**: back to the v7 shape on v50 signals — Stage A `MitB0LatticeNet` (SegFormer-B0 encoder + 545-pt WDL lattice heads, ~3.4M params, pretrained `nvidia/mit-b0` optional, Spec 117 masked lattice contract) → bridge to the detailer's existing `--coarse-store` schema → Stage B residual detailer (existing `GeometryDetailerNet`; `detailer_mit_b0_v1` trunk option behind `--architecture`). Band: 3–30M params per model.
- **Object masks**: loss-side ONLY. Stage A gets tile-coverage `--object-mask-weight` (new); Stage B reuses Spec 118 pixel-level `--object-mask-weight`. No minimap segmentation/classification/retrieval anywhere — that line is dead (see below).
- **Key reuse**: `harvester/v50/direct_geometry_model.py` (MiT-B0 + pretrained loader), `harvester/spec117/lattice_model.py` (masked 545 contract), `harvester/v50/geometry_detailer_{model,train}.py` (residual detailer + `validate_coarse_store`), `harvester/spec118/object_loss.py`, `harvester/v50/spectral_guidance.py` (V7 spectral terms), Spec 116 split. `transformers>=4.52`/`timm>=1.0` already pinned. No new C#.
- **Gates (user-run)**: G1 = Stage A ≥15% below tile-mean (the bar Spec 117 failed; fail = lane stops with recorded negative). G2 = Stage B ≥9% below prior-only. G3 = paired mask-weight verdict (null = valid close) + visual sheets.
- **CLIs**: exact flags in `specs/121-v7-wdl-height/contracts/cli-contract.md`; run sequence (PowerShell-ready) in `specs/121-v7-wdl-height/quickstart.md`.

## Specs 119 + 120 — ARCHIVED 2026-07-24 (minimap object identity is a measured dead end)

- Both moved to `specs/archived/` with `CLOSED.md` notes; rows added to `specs/archived/ARCHIVED.md`.
- **Why dead**: Spec 119's own retrieval PoC measured real minimap object instances at **p50=10px, max=29px** — the 128px library embedding matched every crop to unrelated round blobs at ~0.99 cosine. Object identity does not survive minimap scale; DINOv2 (120) inherits the same physics. Do NOT retry with another backbone — blocker is input resolution, not embedding quality.
- **What survives**: precise masks (Spec 118 `object_geometry_visible_*` arrays) repurposed as loss-side weight in Spec 121; `curated_embeddings.parquet` + trained 119 checkpoints kept read-only as reference.
- Historical detail (classifier 0.9137 / segmenter IoU 0.9921 / curation audit numbers) preserved in `specs/archived/119-object-library-classifier/` and the 2026-07-24 section of `progress.md` below.

## Previous: Spec 119 object-library classifier/segmenter — TRAINED + gated on the full library (ARCHIVED)


- **Full pipeline executed on `objlib_0_5_3_3368.zarr` (5,841 captured assets; classes present:
  mdx 85.6% majority, wmo minority, empty from blank relabel; NO m2 class — alpha client)**.
  Split: 4,486 train / 1,355 held-out, family-isolated, `verified_violation_count=0`.
- **US1 classifier trained (user-run)**: best_epoch=21, held-out 0.9137 vs majority baseline
  0.8562 (+5.75pp). Literal SC-001 (+15pp) is mathematically unreachable at 85.6% majority —
  substantive verdict via per-class recall (baseline scores 0% on minorities): empty 92%/mdx 94%/
  wmo 43% (wmo = genuine wmo↔mdx top-down confusability, 74 support). `sc001=False` flag is the
  literal gate; learning is real. Base-64 capacity comparison (1.55M params) remains an optional
  user-run (`--run-name classifier_v1_b64 --base 64`).
- **US2 segmenter trained (user-run)**: best_epoch=39, held-out IoU **0.9921** vs better trivial
  baseline 0.3529 → **SC-002 PASSES** (+0.64).
- **US3 quality lens written**: 130 mislabels, 200 near-dup pairs (top-k cap), 460 low-coverage
  flags; `embeddings.parquet` + `quality_report.json` + `mislabel_review.png` (24-capture visual
  sheet). **SC-004 PASSES on manual review**: all 24 top flags are either genuine wmo↔mdx
  confusables (~17) or junk/near-blank/test captures the lens correctly caught (missingwmo,
  plaguelandsgra01, arhflt01/03/04, sivsap01 — thin-line captures near the 0.01 threshold).
- Run artifacts under `output/object-library/runs/{classifier_v1,segmenter_v1}/`; checkpoints
  `classifier.pt` (98,067 params) + `segmenter.pt` (482,737 params), `model_stage_run.json`
  records with `promotion_verdict=pending` (user gate).
- **Minimap-retrieval PoC (negative result, decisive)**: hand-cropped object instances from real
  minimap tiles (`curriculum-0_5_3_3368-obj_v1.zarr`, `object_instance_mask`) and embedded them
  with the frozen classifier against `embeddings.parquet`. Measured instance sizes: **p50=10px,
  max=29px** at 257px tile scale (~2 yd/px — even buildings are ~5px blobs). Result: every crop
  classified `mdx(1.00)`, top-3 cosine matches all ~0.99 against unrelated round blobs
  (boulders/glow effects) — the 128px-capture embedding cannot discriminate 5–29px blobs;
  silhouette/texture detail does not survive minimap scale, color dominates what little signal
  remains. **Direct minimap-crop→library retrieval with the Spec 119 embedding does NOT work
  as-is.** A future retrieval feature needs scale-matched training (re-render/downscale library
  captures through the minimap compositor to 8–32px and train an embedding at that scale), not
  just a resize. Sheet: `output/object-library/runs/minimap_retrieval_poc.png`. Also noted:
  ~460/5,841 library captures (7.9%) are blank/near-blank (UI textures, nodxt details,
  thin-line renders) — the `empty` class flags them correctly; a harvest-quality improvement
  candidate.

## Previous: Spec 119 code-complete through Phase 5 dry-runs (earlier this session)

- Two small from-scratch, independently checkpointed specialists trained on the object-library
  zarr itself (Spec 118 capture output), plus a quality lens. Pure Python under
  `data-harvester/src/harvester/spec119/` + `scripts/spec119_*.py` + `tests/spec119/`; no C#.
- **Phase 0 (split, FR-004)**: `object_library_contract.py` (CoarseClassIndex
  `{"empty":0,"m2":1,"mdx":2,"wmo":3}`, `derive_asset_family` = parent dir, blank-threshold
  relabel 0.01, variant-stem helper) + `split.py` family-isolated held-out split with mandatory
  leakage check (`verified_violation_count` must be 0; a leaky split is a refusal, not a
  warning) + `spec119_build_split.py` dry-run-first CLI. `model_stage_contract.STAGES` widened
  with `object_library_classifier` + `object_library_segmenter` (schema unchanged).
- **Phase 1 (US1 classifier)**: `ObjectClassifier` (conv encoder 128→8 + global pool + linear,
  97,938 params @ base 16), inverse-freq class weights, majority-class baseline, per-class
  P/R, blank→`empty`, `--fine-labels` heuristic (run record marks it heuristic), onecycle LR
  with warmup-aware stale counter (reuses `v50.lr_schedule`), `v50-model-stage-run-v1` record
  with `promotion_verdict=pending`. Dry-run on real smoke store verified.
- **Phase 2 (US2 segmenter)**: `ObjectSegmenter` U-Net-lite (128→16 + skip decoder, 482,737
  params @ base 16), BCE, blank captures EXCLUDED from training (D-04), all-foreground/
  all-background trivial IoU baselines, per-coverage-bucket held-out IoU. Smoke dry-run verified.
- **Phase 3 (US3 infer + quality lens)**: `spec119_infer.py` loose-PNG inference for both
  checkpoints (FR-013; reconstructs architecture from `base`, refuses missing `base`);
  `spec119_quality_lens.py` frozen-classifier → penultimate embeddings (deterministic,
  FR-009) + mislabel report (sorted by wrong-class confidence) + cosine near-duplicate pairs +
  low-coverage flags; dry-run-first, `--write` emits `embeddings.parquet` + `quality_report.json`.
  Smoke dry-run verified end-to-end with a random-init checkpoint (mislabels=5 expected).
- Proof: 41/41 `tests/spec119/` pass; full data-harvester suite 1179 passed / 3 pre-existing
  failures (v24 export-map fixture, 2× v25 h1_coarse — unchanged, unrelated; run via
  `uv run python -m pytest` — plain `uv run pytest` hits a spec116 `tests.` import collection
  quirk). Ruff + compileall clean on all touched files.
- **Remaining, explicitly user-run (training gates, FR-010)**: (1) full-library
  `spec119_build_split.py --write`; (2) `spec119_train_classifier.py --confirm-run` → SC-001
  gate (≥15pp above majority baseline); (3) only if SC-001 passes:
  `spec119_train_segmenter.py --confirm-run` → SC-002 gate (≥0.20 IoU above better trivial
  baseline); (4) `spec119_quality_lens.py --write` + manual mislabel review (SC-004). Exact
  CLIs in `specs/119-object-library-classifier/contracts/cli-contract.md`. Requires the
  full-scale object-library build (user-run harvest, still pending from Spec 118 session).

## Active work: Object-library capture pipeline — WMO exclusion root-caused + fixed + smoke-validated (this session)

- The Spec 118 `capture-objects` harvest command + `--from-harvest-stream` Python zarr builder
  (`build_object_library.py`) + headless `ObjectCaptureRenderer` (`WmoObjectRenderer`/
  `MdxObjectRenderer`) + `WmoFullLoader` were built in a prior session that terminated before
  memory-bank sync. The initial test captured MDX/M2 but **zero WMOs**.
- **Root cause**: in the 0.5.3.3368 alpha client, WMOs are stored as listfile-less single-file
  `.wmo.MPQ` wrapper archives (one payload + MD5 sidecar), NOT inside normal listfile-bearing
  MPQs. `NativeMpqService.ScanNestedWrapperArchives` (added last session) registered these into
  `_scannedArchives` so `ReadFile`/`FileExists` could load them — but `ListFiles("*")` only
  returned `ExtractInternalListfiles()` (the `(listfile)` blocks of normal archives) and never
  merged `_scannedArchives` keys. So `capture-objects`' `catalog.ListFiles("*")` enumeration
  never discovered WMOs even though they were readable → WMOs silently absent while MDX/M2
  (inside listfile-bearing archives) were captured.
- **Fix 1** (`NativeMpqService.ListFiles`): merge `_scannedArchives.Keys` into the enumerated
  set so wrapper-archived objects become discoverable. Minimal, dispatch-unchanged.
- **Fix 2** (`ScanNestedWrapperArchives`): canonicalize each wrapper's virtual path by stripping
  a leading `Data\` segment. Without this, the same `.wmo.mpq` was registered twice (once as
  `Data\World\wmo\X.wmo` from the game-root scan, once as `World\wmo\X.wmo` from the Data-subdir
  scan) → `ListFiles` would emit every WMO twice and the capture loop would render each twice.
  Stripping makes both roots produce the same key, so `TryAdd` collapses to one entry, and the
  virtual path matches the MPQ-internal-path convention (no `Data\` prefix) used by listfiles.
- **Smoke validation (run this session, H:\CLIENTS\0_5_3_3368)**: curated 5-WMO asset list →
  `build_object_library.py --from-harvest-stream --run` → `Captured 5 objects, 0 skipped, 0
  errors` → `smoke_wmo.zarr` with `capture_rgb (5,128,128,3) uint8` + `capture_mask (5,128,128)
  uint8`. Per-WMO mask coverage 0.377–0.704, image means 26.5–64.6 (all non-blank, textured).
  Validation contact sheet at `output/object-library-smoke/smoke_wmo_validation.png`
  (5 panels, wmo=5, captured=5). Both `image_rgb` and `mask` signals proven to flow through
  enumerate→read→parse(v14 root+embedded MOGP)→render→stream→zarr for WMOs.
- Build: `dotnet build WowViewer.slnx -c Debug` → 0 errors. No trainer/training touched.
- **Remaining, explicitly user-run**: the full-scale object-library build (whole client, no
  `--asset-list`, no `--capture-limit`) is a data harvest — hand off, do not self-launch. The
  exact CLI is in the completion notes. M2/MDX were already proven in the prior session; this
  session's smoke was WMO-focused to prove the exclusion fix.

## Active work: Spec 118 per-object occlusion-aware masks (US1–US3 implemented and code-verified; user-run gates remain)

- Reintroduces the object signal dropped from v50 — correctly: per-object, occlusion-aware
  (visible-portion-only) mask + class, used loss-side first (US2) then as supervision for a small
  from-scratch segmenter whose prediction feeds the geometry chain (US3, the Spec 115 pattern).
  Spec docs (spec/plan/research/data-model/contracts/quickstart/tasks, 30 tasks) written this
  session under `specs/118-object-occlusion-masks/`.
- **Key discovery (mirrors Spec 117 US1): almost no C# was needed.** The strict object-geometry
  target (`TerrainVisibleObjectMaskRasterizer` + `AdtTensorPackBuilder.BuildStrictTerrainVisibleObjectMask`)
  already rasterized transformed M2/WMO triangles retained only above the raw MCVT surface (+0.25
  clearance, liquid-aware, front-most=highest-elevation overlap rule) and already streamed
  `object_geometry_visible_mask_257`/`object_geometry_visible_source_257` in the Full/V16 profiles
  (NOT V22). The ONE new C# addition: dense `object_geometry_visible_instance_257` (int32,
  per-tile compact ids 1..K painted in the same raster pass under the same front-most rule, FR-002)
  + `object_geometry_visible_instances` per-tile metadata table + serializer/NPZ writes + 5 new C#
  tests (23/23 rasterizer+serializer pass). Legacy footprint masks stay deferred.
- **US1**: 3 rows added to the frozen v50 signal catalog (mask float32, source uint8, instance
  int32, all (257,257) copy-if-verified) + regenerated configs via the existing generator
  (26 signal entries; drift guard passes unmodified). `object_mask_audit.py` + CLI emit
  `v118-object-mask-audit-v1` (marked-fraction percentiles, per-instance consistency,
  visible-vs-footprint reduction factor). One real test-caught bug: map-filtered audits read
  arrays positionally against the filtered index — fixed to filter (position, row) pairs.
- **US2**: `--object-mask-weight` (default 0.0 = bit-parity) on BOTH geometry trainers, mirroring
  `--liquid-mask-weight` exactly (plan echo, missing-array warn+disable, per-point
  `1 - w * mask`, run-record echo) + FR-008 object-touched vs untouched region MAE in the run
  record. Shared helpers in `harvester/spec118/object_loss.py`. Ground truth is loss-side only
  (FR-014); detailer's coarse-only baseline stays unmasked so the relative gate stays honest.
- **US3**: `harvester/spec118/` package — `object_segment_model.py` (`ObjectSegmentNet` U-Net-lite
  skip-decoder, RGB→3-class logits at 256×256, from scratch, base-only reconstructable),
  `object_segment_train.py` + CLI (dry-run-first, `--held-out-split` REQUIRED, class-weighted CE
  with background cap, per-class IoU/recall + median visible-object IoU selection, D-07 gate
  thresholds 0.40/0.50 recorded, `object_config.base` in checkpoint), `object_segment_infer.py` +
  CLI (two mutually exclusive modes: loose `--inputs` OOD with no store, `--store` batch with
  ground-truth scoring; `v118-object-infer-v1` audit), `object_feature_bridge.py` + CLI (frozen
  checkpoint → `v115-feature-map-v1` store, class_count=2 doodad/building softmax channels, none
  dropped as redundant). STAGES widened with `"object_segmentation"` (one schema change).
- **Real integration proof (T027 analogue)**: fixture store + Spec 116 split + random-init
  checkpoint → real bridge `--write` → dry-ran BOTH `v50_train_direct_geometry.py` and
  `v50_train_geometry_detailer.py` with `--feature-store` — both accepted with zero trainer code
  changes (`input_channels: 5`, `generated_terrain_feature_map` in deployment inputs). FR-011
  proven for real, not just unit tests.
- Proof: 48/48 `tests/spec118/` pass (incl. `--help` argparse verification of both trainer CLIs);
  full data-harvester suite 1080 passed / 3 pre-existing failures (v24 export-map fixture, 2× v25
  h1_coarse — unchanged, unrelated); ruff clean on all touched files; compileall clean; full
  solution Debug build 0 errors.
- **US3 augmentation — multi-feature-store (2026-07-22, this session's in-flight work completed).**
  The bridge produced a `v115-feature-map-v1` object store, but the geometry trainers accepted only
  ONE `--feature-store`, and the promoted deconfounded chain (Spec 115 v3) already occupies that
  slot with the terrain-feature map — so the object prior could only REPLACE the roads-as-slopes
  deconfounding, never sit alongside it, even though objects occlude ground height (a different
  confound). "Object segmenter → geometry as a real deconfounding input" was therefore NOT actually
  wired; the T027 dry-run "proof" passed only because it exercised the single-store replacement
  path. Fixed: new shared `harvester/v50/feature_stores.py` (`FeatureBinding`, `load_feature_stores`,
  `total_class_count`, `feature_channels_for_row` concat-in-CLI-order, `plan_entries`,
  `road_feature_binding` — the FR-008 road diagnostic argmaxes the terrain-feature prior's OWN
  channels, found by its `taxonomy_revision` attr, not the concatenation); `--feature-store` is now
  `action="append"` on `direct_geometry_train.py`, `geometry_detailer_train.py`, AND
  `direct_geometry_materialize.py` (order-sensitive — the materializer must be fed the priors in the
  same order the coarse checkpoint trained with); `in_channels = 3 + Σ class_counts`; plan/record
  emit `feature_stores` (list) + `feature_input_channels`. `height_relative_evaluate.py` preview/
  evaluate helpers gained a `feature_bindings` param, backward-compatible with the legacy
  `feature_group`/`feature_row_to_position` via `as_bindings`, so `height_relative_train.py` is
  untouched. 10 new `tests/v50/test_feature_stores.py` (two-store concat order, per-store coverage
  refusal, schema refusal, road-binding selection, `plan_entries`, all 3 CLIs advertise REPEATABLE);
  full suite 1138 passed / 3 pre-existing unrelated failures / 46 skipped; ruff clean. Tasks
  T031–T033 (new Phase 5b), quickstart §3b. The paired terrain-only vs terrain+objects geometry
  comparison (SC-003, relief-stratified object-touched MAE) is user-run.
- **Remaining, explicitly user-run**: the `H:\CLIENTS` store rebuild to pick up the 3 new arrays
  (same Spec 109 build command; Full profile required — V22 omits the strict arrays), the US1 audit
  + eyeball proof on a city tile and an underground-heavy tile, the US2 paired with/without
  `--object-mask-weight 1.0` training comparison (relief-stratified object-touched MAE, SC-003 —
  a null result is a valid reportable outcome that stops the line before US3 training), US3 real
  segmenter training + hand-painted OOD eyeball + paired geometry comparison. Exact CLIs in
  `specs/118-object-occlusion-masks/quickstart.md`.

## Active work: Spec 117 WDL-lattice coarse prior for terrain geometry (US1–US3(i) implemented and code-verified)

- Adds a third generated input to the v50 coarse+detailer chain: a per-tile 545-point WDL-scale
  height lattice (17×17 outer + 16×16 inner, Spec 108 FR-001), predicted from minimap RGB alone.
  Three user stories in priority order: US1 export the lattice as a real v50 signal (data
  plumbing) → US2 prove a standalone predictor can learn it, scored only on the honest held-out
  split → US3 bridge the frozen predictor's output into the existing `--feature-store` contract so
  the already-validated coarse/detailer trainers consume it with zero trainer changes. No GAN,
  adversarial loss, or generative-image technique anywhere (explicit spec boundary).
- **Key mid-implementation discovery: no C# work was needed for US1.** `TerrainWdlLattice` was
  already computed in `AdtTensorPackBuilder` and already streamed by
  `RawArraySerializer.WriteTerrainVertexArrays` as `wdl_outer_17`/`wdl_inner_16`/`wdl_outer_present`/
  `wdl_inner_present` in every stream profile (Full/V16/V22), predating this spec entirely. The spec
  docs were drafted before this was known and named the arrays `wdl_lattice_outer17` etc.; corrected
  to the real names during implementation. The only real gap was the frozen v50 signal catalog
  (`docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`) not yet declaring these four
  arrays, so the existing 1:1-name-matched store builder never selected them. Fixed by adding four
  catalog rows and regenerating `v50_configs/v50-manifest-template-0_5_3_3368.json` +
  `v50-signals-0_5_3_3368.json` via the existing `v50_generate_manifest_template.py` generator —
  zero hand-editing, zero new ingestion code, drift-guard test (`test_committed_053_template_
  matches_the_frozen_catalog`) still passes unmodified.
- **US2**: `harvester/spec117/{lattice_model.py, lattice_train.py}` + `scripts/spec117_train_
  lattice.py`. `LatticeNet` (~178K params at base=8) predicts minimap RGB → 545 values. The target
  contract (`encode_lattice_target`/`decode_lattice_target`) is a masked analogue of
  `height_relative_model`'s v112.1 per-tile min-max floor contract — absent lattice samples never
  contribute to the normalization range or the loss. Reuses (imports, does not reimplement)
  `height_relative_train`'s curriculum/source validation and `direct_geometry_train.
  apply_held_out_split`. Unlike the existing trainers, `--held-out-split` is REQUIRED with no
  `--val-key` fallback (FR-004: refuse an unspecified split, don't default away from one).
  `STAGES` in `harvester/v50/model_stage_contract.py` was widened to add `"lattice_prior"` — the one
  actual schema change needed to make research.md D-01's "reuse v50-model-stage-run-v1 verbatim"
  claim true.
- **US3(i)**: `harvester/spec117/lattice_bridge.py` + `scripts/spec117_lattice_to_feature_map.py`
  bridges the frozen checkpoint's output into a `v115-feature-map-v1`-shaped, `class_count=1` store
  (independently bilinear-upsamples the two regular outer/inner grids to 256×256 and averages them
  — a documented approximation, not a precision quincunx reconstruction).
- **Real end-to-end proof, not just unit tests**: built a real fixture v50 store + Spec 116
  held-out split + a real (untrained, random-init, no CUDA training) checkpoint, then actually ran
  `spec117_train_lattice.py` (dry-run and the missing-array refusal), `spec117_lattice_to_feature_
  map.py --write`, and dry-ran BOTH `v50_train_direct_geometry.py --feature-store` and
  `v50_train_geometry_detailer.py --feature-store` against the bridged output — both accepted it
  with zero code changes (`input_channels: 4`, `deployment_inputs` gained `generated_terrain_
  feature_map`). Caught one real integration bug this way: the checkpoint's `architecture` block
  only carries a config hash, not the raw `base` width, so `lattice_bridge.py` needed a separate
  `lattice_config: {"base": ...}` field to reconstruct an architecturally-identical `LatticeNet`
  before `load_state_dict` — a bug no isolated unit test would have caught.
- Proof: 26/26 new `tests/spec117/` pass; full `data-harvester` suite green (no regressions beyond
  pre-existing unrelated failures); ruff clean; `py_compile` clean.
- **Remaining, explicitly user-run**: a real store rebuild against `H:\CLIENTS` to pick up the new
  catalog signals for real tiles, real `--confirm-run` training of the standalone predictor (the
  learnable/not-learnable verdict is not yet known against real data), and the real US3(ii) paired
  coarse/detailer comparison against the existing structure-augmented baseline.
- **Scheduling bug found + fixed (2026-07-22) from the first real US2/US3 runs** (real store
  `curriculum-0_5_3_3368-dual_v3.zarr`, train=679/val=446, 43 steps/epoch, Spec 116 split). All
  three geometry trainers paired `OneCycleLR` (default `pct_start=0.3` → 30-epoch warmup) with a
  patience-15 early-stopper that counted *every* non-improving epoch as stale. When `patience <
  warmup_epochs` the run died mid-warmup, before the LR ever reached its peak. The detailer was the
  worst case: its zero-init residual head starts AT the coarse baseline and cannot improve val until
  the LR rises, so `detailer-with-lattice-run1` froze at val 0.2301 (coarse 0.2333), best epoch 2,
  early-stop epoch 17 — "goes stale very early, does not progress." `lattice-run1` survived warmup
  (no zero-init head) but still plateaued at 0.2427 vs tile-mean 0.1277 (did not beat baseline).
  Fix: new shared `harvester/v50/lr_schedule.py` (`make_onecycle_scheduler` +
  `warmup_complete`/`warmup_epochs_for`); the stale counter is now **suppressed until the warmup
  phase completes**, and `--pct-start` is exposed on all three trainers (default 0.3 = torch parity;
  quickstart recommends 0.1 for this 43-steps/epoch dataset → 10-epoch warmup). `pct_start`/
  `warmup_epochs` now appear in the dry-run plan. 7 new `tests/v50/test_lr_schedule.py` pass
  (incl. a reproduction of the patience<warmup kill); 63 affected tests pass; ruff/py_compile clean.
  This is a scheduling fix, NOT a learnability verdict — whether the lattice predictor beats
  tile-mean, and whether the lattice-augmented detailer beats coarse-only, remain user-run
  questions. The exact rerun CLIs are in `specs/117-wdl-lattice-prior/quickstart.md` §2b/§4.
- **Architecture fix (2026-07-22) after the first post-fix run still plateaued.**
  `lattice-authored-v2` (warmup-aware, `--pct-start 0.1`) survived warmup — best epoch 52, ran to
  67, early-stopped cleanly *after* warmup — but still val 0.2307 vs tile-mean 0.1277. Diagnosis:
  train MAE was *also* above tile-mean → **underfit, not overfit**. v1 `LatticeNet` was a plain
  4-conv encoder whose two heads read only the 16×16 bottleneck (no skip connections), so it could
  not localize the 17×17 height field. Redesigned `LatticeNet` to a **U-Net-lite (v2)**: bottleneck
  decoded back up with skip connections (e3, e2) and each head fuses all four feature levels
  (16/32/64/128) at the lattice resolution. Capacity 178K → 675K params at `--base 24`; still
  constructable from `base` alone so `lattice_bridge.py` is unchanged; `architecture_identity`
  config now carries `"arch": "lattice_net_v2"` so run records distinguish it. 3 new v2 tests
  (skip-decoder structure, skip-path differentiability, base-only reconstruction round-trip +
  cross-base load refusal); 36 spec117+lr_schedule tests pass; ruff/py_compile clean; dry-run
  confirms 675170 params. Verdict on whether v2 beats tile-mean is user-run. If it overfits 679
  tiles, lower `--base` (e.g. 16) before adding regularization.
- **Visibility + V7 insight (2026-07-22).** The lattice trainer previously emitted ONLY checkpoints
  + a val_mae number — no per-epoch visuals (unlike the detailer), which is why progress was
  disconcerting to judge. Added: per-epoch `validation/best_previews/epoch_XXXX.png` (8 fixed
  held-out tiles: minimap RGB / truth lattice / predicted lattice / tile-mean baseline / signed +
  abs error, where the lattice is the dense 256×256 bilinear-average the bridge actually emits) +
  final `validation/final_best/{fixed_rows,worst_cases}.png`. Reuses
  `height_relative_evaluate.render_validation_sheet`/`compute_row_metrics`; `visual_evidence` now
  recorded in the stage run. Also added a loss-only `--gradient-weight` (V7-ported 2D
  finite-difference gradient term; 0 = parity) targeting the "right values, wrong arrangement"
  failure. **Key reframe from reading the V7 doc**: the V7 height regressor that "worked" used WDL
  as an INPUT prior (channel 6) + normals + masks + a residual-around-WDL head — it did NOT predict
  WDL from RGB alone; the V7 doc states outright "the minimap does not directly encode enough
  elevation signal to reconstruct valid terrain on its own." Spec 117's RGB→WDL-alone experiment is
  strictly harder, so a "does not beat tile-mean from RGB alone" verdict is a valid, reportable spec
  outcome — the previews now let us SEE which failure mode it is instead of guessing. 3 new tests
  (gradient zero-on-match/masked, dense-field averaging); 39 spec117+lr_schedule tests pass; ruff/
  py_compile clean; dry-run confirms `gradient_weight` in the plan.

## Active work: Spec 116 relational terrain layer reconstruction (FULLY IMPLEMENTED — all 5 user stories)

- Spec 116 reframes terrain reconstruction as a **relational schema**: layer entries are ordered
  rows, texture references are foreign keys into each tile's own MTEX table, and the corpus is a
  discrete alphabet of reused pieces. Five user stories: US1 family→slot consistency (decides
  output vocabulary, no model), US2 shape→coverage coupling (decides derivability, no model), US4
  spatially-isolated held-out set + relief-stratified eval, US3 structure prediction from minimap
  alone, US5 feed predicted structure into geometry.
- **All 35 tasks (T001–T035) implemented and validated.** 121 spec116 tests pass; ruff clean;
  compileall clean; full data-harvester suite 1017 passed / 46 skipped / 3 pre-existing failures
  (unrelated to Spec 116). No regressions.
- **US1 (Phase 3)**: `family_slot_consistency.py` + CLI — per-family P(s|f), summary consistency
  score, `slot_keyed`/`family_keyed` recommendation. `v116-analysis-report-v1` artifact.
- **US2 (Phase 4)**: `shape_coverage_coupling.py` + CLI — GradientBoosting explained variance,
  SAS bimodality coefficient, GMM BIC, `coverage_derivable`/`coverage_independent` decision.
- **US4 (Phase 5)**: `held_out_split.py` (8-neighbour-isolated split, `verified_violation_count`
  must be 0) + `relief_stratification.py` (chunk strata, stratified MAE, tile-mean baseline,
  dihedral NCC reused-piece overlap) + CLIs. `rescore_geometry_checkpoint` (T019, in
  `structure_train.py`) re-scores an EXISTING Spec 114/115 geometry checkpoint against the split,
  stratified by relief — read-only, no training — via `spec116_train_structure.py
  --rescore-checkpoint`.
- **US3 (Phase 6)**: `structure_model.py` (`StructureSlotNet` — one independent U-Net-lite per
  detail slot 1–3, 16×16 chunk-resolution head, base slot 0 never predicted FR-008);
  `structure_train.py` (dry-run-first, class-weighted CE, per-class IoU/recall gate D-08,
  `promotion_verdict=pending`, `v50-structure-run-v1` record, refuses leaky split);
  `structure_infer.py` (legality resolver picks same-family local id from tile MTEX table SC-004,
  OOD never fabricates reference D-05, `v50-structure-infer-v1` audit record) + CLI with two
  mutually exclusive modes: `--inputs`/`--tile-table` (loose PNG files/dirs, no store required —
  runs unchanged on a hand-painted OOD image) and `--store`/`--dumps` (batch over a v50 store).
- **US5 (Phase 7)**: `structure_materialize.py` (frozen checkpoint → derived structure store with
  `structure_family`/`structure_confidence`/`structure_legal`, row-aligned `index.parquet`,
  source stores immutable, checkpoint sha256 bound) + CLI. `structure_feature_bridge.py` (new) +
  `spec116_structure_to_feature_map.py` CLI upsample that derived store into the
  `v115-feature-map-v1` shape `harvester.v50.direct_geometry_train`'s `--feature-store` already
  validates (predicted class keeps its confidence as probability mass; remainder spread uniformly
  — a valid per-pixel distribution, not a finer-grained claim). `direct_geometry_train.py` gained
  `apply_held_out_split`/`--held-out-split` so the trainer consumes a Spec 116 split directly
  (read-only; curriculum store never mutated), overriding `--val-key`/`--val-value`; the dry-run
  plan's `split_counts`/`train_steps_per_epoch` are overridden to match so they're never wrong
  about what will actually train. Geometry comparison documented in quickstart.md section 5b
  (`v50-structure-geometry-comparison-v1` template, SC-007 bar).
- Reuses v50 Zarr store (no new harvest), Spec 115 `v115.1` taxonomy, Spec 114 sha256 helpers.
  User runs all training (FR-018). All CLIs are dry-run-first (FR-015).
- **Verification correction (2026-07-21, same session as "fully implemented" above)**: a
  from-scratch verification pass (reading every CLI's real argparse against
  quickstart.md/cli-contract.md/tasks.md, not trusting the prior "all 35 tasks done" claim) found
  three tasks that were marked done but were not actually runnable as documented: T019's
  `--rescore-checkpoint`/`--print-only` flags did not exist on `spec116_train_structure.py`; T027's
  `--inputs`/`--tile-table` loose-image interface did not exist on `spec116_infer_structure.py`
  (only store-batch mode did); and the US5 5b payoff referenced a script that does not exist
  (`spec114_train_geometry.py`) whose real counterpart (`v50_train_direct_geometry.py`) had no
  `--split` mechanism and would hard-refuse `--feature-store` pointed at a
  `v116-structure-store-v1` store (it validates for `v115-feature-map-v1`). All three gaps were
  closed this session (code above); do not repeat the "all 35 tasks implemented" claim without
  actually exercising each documented CLI invocation — passing focused tests only proves the
  library functions work, not that the documented commands parse.
- **Three more gaps found running the actual `--write`/rescore paths against the real corpus
  (2026-07-21, same day)**: (1) `spec116_build_held_out_split.py --write` crashed —
  `v50-held-out-split-v1` requires non-empty `build_id`, CLI defaulted it to `""`; fixed by
  auto-deriving `build_id` from the store's own `index.parquet` `build` column. (2)
  `rescore_geometry_checkpoint` only ever built a 3-channel RGB tensor, so any Spec 115
  deconfounded checkpoint (8 channels) crashed; fixed by adding `feature_store`/`--feature-store`
  (concatenates the same generated `feature_map` array the trainer does, RGB first) with
  `in_channels` auto-derived as `3 + class_count`. (3) the real feature-map store only covers the
  1,629 authored rows, not the dual curriculum's 1,361 synthetic rows; fixed by adding
  `source`/`--rescore-source` (mirrors `direct_geometry_train.py`'s `--source`) to filter held-out
  rows to the matching domain. Also fixed `quickstart.md`'s 5b `--source authored_only` (not a
  real choice — `SOURCE_CHOICES = {"all","authored","synthetic"}`) to `--source authored`. Real
  corpus location (not the quickstart placeholder): `../output/datasets/v50/v50.1/curriculum-
  0_5_3_3368-dual_v1.zarr` (2,990 rows), one directory above `data-harvester/`.
- **Major finding from rescoring all six direct-geometry checkpoints on the new split (2026-07-21,
  same 444-row authored-only held-out subset, same `trivial_baseline_mae=44.984`):**

  | checkpoint | channels | relief MAE | vs trivial | SC-007 |
  |---|---|---|---|---|
  | v1 (RGB baseline) | 3 | 40.518 | -9.9% | true |
  | v2-spectral (RGB+spectral loss) | 3 | 29.887 | -33.6% | true |
  | **v3-deconfounded** | 8 | **26.689** | **-40.7%** | true |
  | v4-brush | 8 | 33.766 | -24.9% | true |
  | v5-normals | 8 | 34.987 | -22.2% | true |
  | v6-mcly-brush | 8 | 36.785 | -18.2% | true |

  **This overturns two standing findings, both artifacts of the OLD leaky split (99.6% train/val
  spatial adjacency), not properties of the models**: (1) "no model beats the tile-mean baseline"
  (recorded across Specs 112-115) — all six checkpoints clear it here; (2) v6-mcly-brush is not
  the best generalizer despite being recorded as "best road MAE run to date" — relief MAE gets
  monotonically WORSE from v3→v4→v5→v6 even as each was introduced as an improvement. The
  deconfounding itself (v3, RGB→8ch) does the real work; brush loss (v4), normal guidance (v5),
  and mcly-brush weighting (v6) each improved the metric they were tuned against (road-region MAE
  on the leaky split) while eroding general relief-region generalization — a real
  overfitting-to-leaky-evaluation signature. Follow-up: re-examine whether v4-v6's loss terms are
  worth keeping now that a trustworthy eval exists to check them against. MAE here is raw
  world-height units (decoded via each tile's own min/max), not the `[0,1]` normalized units
  elsewhere in memory-bank, and the split differs from what those figures used — not directly
  comparable to prior recordings, only to each other within this rescore setup. 125/125 spec116
  tests pass after all six total fixes this session.

## Active work: Spec 114 direct minimap-to-terrain (original zarr-based spec restored)

- **Viewer-only Spec 110 lighting repair is implemented; real 3.x visual proof remains.** The
  interactive renderer now establishes an always-present global directional/ambient light at noon
  before evaluating profiles. Raw exact-build DBC/LightData local profiles blend by spatial weight;
  absent/out-of-range/failed locals leave the base unchanged and departed local fog is reset. The
  Lighting panel reports global and local layers separately. This does not alter fixed-noon-white
  synthetic minimaps. Focused composition/DBC tests pass 15/15; viewer Debug build has 0 errors.

- **The deployment contract is the authored WoW minimap; the dataset is the project-owned v50
  Zarr store.** An unauthorized "universal arbitrary-raster" reset of Spec 114 (DINOv2 student,
  DPT-Hybrid/MiDaS pseudo-label teacher, third-party image folders) was reverted on 2026-07-19:
  spec docs restored to the pre-reset state, all universal/teacher code and docs deleted (commit
  `06151357`). That lane never ran — no weights, corpus, training, or inference existed. Pretrained
  encoders remain only an optional FR-013 pinned ablation against the from-scratch baseline.
- The 1,561,537-parameter WoW-only CNN completed 100 epochs and is rejected: evaluator MAE
  0.1493349 versus tile-mean 0.1387470, gradient MAE 0.0058671, border MAE 0.1607286. Its fixed-scale
  quantile/worst/per-row artifacts are immutable negative evidence. **Do not optimize/rerun it.**
  Recorded in research.md as the T003/T017/T018 evidence block.
- Phase 1-2 (T002-T009) implemented: `model_stage_contract.py` (dependency-free validator for all
  three schema variants + sha256 identity binding), `reconstruction_curriculum.py` +
  `v50_build_reconstruction_curriculum.py` (dual-view admission: grouped-split leak refusal,
  stale-lighting exclusion without zero-fill, mixed-provenance refusal, dry-run-first CLI). Against
  today's store the builder keeps 1,629 authored rows and excludes 1,361 stale synthetics.
- Phase 3 code (T014-T015) implemented: `direct_geometry_model.py` (`direct_cnn_v112` baseline +
  `mit_b0_regression` SegFormer-B0-scale candidate, one bounded 257×257 output, DepthAnything
  refusal, FR-013 pinned-pretrained path) and `direct_geometry_train.py` +
  `v50_train_direct_geometry.py` (flat+tile-mean baselines, SC-001 vs frozen Spec 112 run,
  SC-002 border-vs-interior-p95, quantile/worst sheets, schema-validated `model_stage_run.json`
  with `promotion_verdict=pending`; optional AMP/OneCycle/clip, defaults at bootstrap parity).
- **`mit_b0-authored-v1` completed (user-run, 2026-07-19)**: best epoch 93, val MAE 0.187802,
  SC-001 false (tile-mean 0.138747), SC-002 true. User verdict: strongest VISUAL geometry to date
  (correct layout/mountains on unseen tiles) but smooth and under-amplituded — diagnosed as
  spectral bias against fractal terrain structure (research.md T018 follow-on record).
- T056 deployment inference implemented: `direct_geometry_infer.py` +
  `v50_infer_direct_geometry.py` run any loose 256x256 minimap tile(s) through a checkpoint →
  16-bit relief PNGs + fixed-scale review sheet + per-tile hash-bound manifest (FR-015). Dry-run
  default; relative-relief-only caveat recorded in every manifest.
- Spectral guidance (Spec 068 US1 revived, loss-only): `spectral_guidance.py` adds radial
  log-power L1 (DC-removed; spectral slope = fractal-dimension proxy) and multi-octave gradient L1,
  wired into `direct_geometry_train.py` as `--spectral-weight`/`--multiscale-weight` (default 0 =
  bootstrap parity). No aux head (one-output constitution), no deployment change.
- **`mit_b0-authored-v2-spectral` (user-run) done**: best epoch 130, val MAE 0.193435, SC-001
  false, SC-002 true. Visually sharper structure (islands/ridges) at a small MAE cost; both runs
  plateau at ≈0.19 with train loss ≈0.016 — the single stage is capacity-saturated at 1,384
  tiles. More epochs/loss tuning is not the lever.
- **Residual detailer stage implemented (T058-T060)**: `direct_geometry_materialize.py` +
  `v50_materialize_coarse_relief.py` (frozen checkpoint → derived `coarse_relief` Zarr bound to
  checkpoint hash, source stores immutable, 1:1 row alignment validated);
  `geometry_detailer_model.py` (U-Net-lite residual refiner, RGB + generated coarse → one
  residual field, zero-init head so epoch 1 starts AT the coarse baseline);
  `geometry_detailer_train.py` + `v50_train_geometry_detailer.py` (coarse-only strong baseline,
  ≥5% relative gate, SC-002, fixed/quantile/worst sheets, `upstream_models` provenance in
  `model_stage_run.json`). 16 focused tests; full v50 suite 273 passed / 4 skipped; Ruff clean.
- **T061 DONE (user-run, 2026-07-20)**: `detailer-mit_b0-authored-v1` best epoch 91, val MAE
  0.170665 vs coarse-only 0.187800 — **9.1% relative, gate=True, sc002=True**. First Spec 114
  geometry checkpoint to clear its numeric gate. User visual verdict positive ("a lot more
  detailed, getting REALLY close"). Two-stage chain proven: coarse owns layout, detailer owns
  high-frequency residual, each independently replaceable.
- **T062 DONE**: V7 (April) vs current detailer (July) comparison. V7 had four structural loss
  terms the current detailer lacks: full 2D frequency (not radial average), Laplacian curvature,
  Sobel edge, transition-focus 3× weighting. V25 had explicit LF/HF band split. V7 "still worked"
  on dirty data because the structural prior compensated for input noise — the model learned
  "produce correct edge/curvature statistics" regardless of object/liquid channel cleanliness.
  Decision: port all five as loss-only flags into the detailer trainer (T063), not the 117M V7
  architecture (wrong constitution/scale).
- **T063 DONE + USER-RUN COMPLETE + CONTINUED**: multi-frequency band-split loss stack
  implemented in `spectral_guidance.py` (5 new loss-only terms from V7/V25).
  - Run 1 (100 epochs): `detailer-mit_b0-authored-v2-bandsplit` best epoch 89, val MAE 0.170494
    vs coarse-only 0.187800 (9.2% relative), gate=True, sc002=True. User: "visually stunning."
  - Run 2 (continued, 200 epochs total): resumed from epoch 89 checkpoint via `--init-weights`,
    best epoch 164, val MAE 0.166769 vs coarse-only 0.187800 (11.2% relative), gate=True,
    sc002=True. Plateaued at epoch 164 after 36 epochs of no improvement; 100 additional epochs
    yielded 0.003725 absolute MAE improvement over the 100-epoch run.
  - Run used bf16 AMP, val-tolerance 0.01, liquid-mask-weight 0.5, per-epoch validation previews
    (9-panel comprehensive sheets). Also added: `--amp-dtype bf16` (both trainers),
    `--val-tolerance` (noise-robust early stopping), `--liquid-mask-weight` (loss masking in
    liquid regions), per-epoch validation previews (9-panel comprehensive sheets),
    `--init-weights` for checkpoint resumption.
- Next: user visual promotion gate for bandsplit-v2 (epoch 164 checkpoint). Dual-view
  `--source all` stays gated on the Spec 113 NoonWhiteGlobal rerender; object-mask phase starts
  only after geometry promotion. If more training is needed, the plateau suggests the single-stage
  detailer has hit its capacity limit — the next lever would be a multi-stage frequency-band
  chain (T064) or architecture change.
- Source-image UV projection provides immediate mesh texture. Object cleanup, terrain semantics,
  editable texture families, and alpha remain later independent models with separate checkpoints.

- **Spec 112**: real corpus correction and dual-source curriculum are proven. Kalimdor/Azeroth
  coverage reports pass parity; `curriculum-0_5_3_3368-dual_v1.zarr` has 2,990 rows (1,629 authored
  + 1,361 synthetic; 2,545 train / 445 val). The `v112.1` relative-height model and CUDA-only
  trainer are implemented with schema/map/leak gates, target round-trip/offset-invariance tests,
  baseline reporting, and epoch-1 structural-failure detection. Its failed run is historical input
  to Spec 114 only; it is not the universal-model route.
- **Spec 113**: T001-T010c, T011/T012, and T014 are implemented. The detail compositor uses production 8×/chunk UVs and
  footprint-selected mips (not unfiltered base texels); the v50 builder applies `--detail` only to
  1024, preserves synthetic-minimap authority, and records store provenance. The cross-map analyzer
  searches all 8 dihedral transforms plus one fixed LR-pixel offset without wrap and emits the hard
  gate. The completed staged builds exposed a correctable manifest-policy mismatch: 220
  synthesized-minimap rows are honestly unavailable, so `finalize --policy-template ...` now
  derives 731/951 coverage without a rebuild. The real 120-tile cross-map report then failed
  `fail_inconsistent` (NCC p50 0.211 / p05 0.000; SC-001 detail gain 16.10): authored minimaps and
  synthetic detail renders are different domains, not one transform apart. The user confirmed this
  is intentional terrain-only cross-domain supervision: authored objects/icons are not expected in
  synthetic targets, so raw full-frame NCC is diagnostic, not the promotion owner.
- Visual review exposed an actual renderer bug: raw ADT MCNR axes were dotted directly with a
  renderer-space solar vector, reversing relief lighting. The shared compositor now applies
  `TransformAdtNormalToRenderer` first. Existing `minimap_rgb`/`minimap_rgb_1024` store arrays are
  stale; numeric height/normal/liquid/material/mask/flag signals and authored minimaps remain sane.
- The first 2.4.3 comparison proved a policy error: runtime local Light DBC color was purple-tinting
  and crushing minimap terrain. `synthetic-minimap` is now fixed to one 12:00 achromatic global
  light for every era; v6 records `NoonWhiteGlobal`, rejects non-noon/`--dbd-dir` inputs, and never
  evaluates LIT/DBC. The interactive viewer alone retains exact-build DBC lighting/status. Next:
  user reruns T010b, then refreshes only synthetic RGB before pair-set/training work.
- The original Expansion01 32,32 handoff was wrong: it is WDT-occupied but produced an all-black
  one-material synthetic PNG. `synthetic-minimap` now supports bounded `--tile-list` and
  `--authored-reference`, emitting authored/synthetic/liquid/side-by-side files per tile and rejecting
  missing/all-black references or results. The replacement 2.4.3 set is `24,24;21,28;28,30;26,26;
  27,27;23,30`, proven occupied, authored-nonblack, and backed by 5-10 nonblack decoded terrain BLPs.
- Architecture ruling for later Spec 113 US3: ComfyUI-native RealPLKSR via spandrel; DAT-2 ceiling,
  RRDBNet floor. The visual-review surface, guarded `sr_pairset.py`, its contract tests, and
  `sr_model.py` wrapper are implemented pre-gate; no real pair set is promoted yet.
  `v50_train_minimap_superres.py` is intentionally absent until T010b/T013 complete.
- Proof: authored training focus 23/23, full v50 Python focus 178 passed / 4 skipped, Ruff/pycompile,
  and real dry-run pass; stale synthetic input is refused. Prior combined
  compositor/detail/DBC/lookup C# focus remains 41/41; Harvest build 0 errors.

## Prior active work: Spec 109 v50 clean-room dataset — Setup and Build Pipeline Fully Operational

- **Phase 1 Setup (T002-T005) is complete**: Signal table frozen in the docs, approved/protected roots recorded in `research.md`, package directories validated, and CLI/contracts verified.
- **Wired-Up Build Pipeline (T037) is complete and tested**: The `build` subcommand in `scripts/v50_build_dataset.py` has been upgraded from a printing stub to a fully operational stream-consuming compiler. It compiles the C# harvester once and launches both `synthetic-minimap` synthesis processes (256x256 and 1024x1024) in parallel using the compiled DLL directly (avoiding stdout compilation pollution). It streams real ADT signals (including `mccv_rgb` vertex colors) and compiles all inputs into the Zarr store.
- **Client-Root Auto-Resolution**: Resolves the library `--clients-root H:\CLIENTS` to the build-specific directory `H:\CLIENTS\0_5_3_3368` by reading the template build ID, ensuring all main archives (like `texture.MPQ`) are loaded and textures successfully decoded.
- **Multithreading/Parallel Synthesis**: Parallelized the C# tile composition loop using `Parallel.ForEach` over concurrent collections, transforming the slow sequential loop into a high-performance multithreaded process.
- **Config Files Generated**:
  - `v50_configs/v50-signals-0_5_3_3368.json` (defines signals, blacklists `holes_16`, and binds has-flags).
  - `v50_configs/v50-manifest-template-0_5_3_3368.json` (defines release `v50.1` and Zarr structures).
- **Audit/Lineage Decisions (T002/T003)**:
  - Real client-decoded ground truth is kept (heights, MCNR normals, MCAL overlays, MCSH shadows, MCNK flags, and MCCV vertex colors).
  - High-resolution 1024x1024 synthesized minimap tiles (`minimap_rgb_1024`) are added as a first-class signal for Real-ESRGAN upscaler training.
  - All interpreted, interpolated, or buggy signals are dropped (object masks, roof masks, and inpainted `ground_intent_height_257`).
  - Approved write-eligible output directories and protected read-only source directories recorded as Decision 7 in `research.md`.
- **Proof**: `uv run python -m pytest tests/v50/ tests/test_v50_build_command.py -q` -> 112 passed, 2 skipped, 0 failed.
- **Handoff**: The build pipeline is fully optimized, parallelized, and ready. The next step is for the user to run the fresh-build extraction command against `H:\CLIENTS` (documented in `quickstart.md` section 5).

## Prior active work: Spec 111 minimap lighting calibration (implemented through the T019 gate)

- Spec/plan/research/data-model/contract/quickstart/tasks written and all code phases implemented
  (`specs/111-minimap-lighting-calibration/`). Remaining: the user-run real-client bucketing pass
  (T009's bounded `harvest-stream --stream-profile v22` + eyeball proof) and the explicitly gated
  T019 training run.
- Implementation shape: `MinimapLightingProvenance` gained six additive shading-match fields;
  `MinimapShadingMatch` (Core.IO, because it must call `TerrainMinimapCompositor`) sweeps 24 hourly
  candidates through the production compositor and scores mean/variance-normalized luma Pearson
  correlation — gradient-direction cosine was tried first and could not discriminate hours at all
  once the azimuth became fixed; value correlation is tint-invariant for a single material (luma =
  materialLuma × lightingValue) and genuinely elevation-discriminative. Empirical facts baked into
  its tests: `TerrainSolarDirection`'s 0.05 elevation floor makes hours 0–6/18–23 render
  byte-identically (a genuine tie, not a metric bug — test at noon, which is unique), and material
  channels near 255 clip at high-lambert hours, breaking multiplicative tint-invariance (test
  colors stay ≤150).
- The Harvest wiring is NOT a new command: `AnalyzeAuthoredMinimapLighting` chains
  `MinimapShadingMatch.Evaluate` after the tint `Infer()` on the harvester's existing
  full-texture-decode streaming pathway; the internal 0.5.3.3368 fingerprint gate renders zero
  candidates for other builds. `minimap_shading_match_v1` joins AvailableSignals only when a tile
  was actually evaluated.
- **Lane correction (user feedback, 2026-07-17): the active dataset/model lane is v50, not the
  legacy V22/spec103/spec108 naming.** Phase 3 now delegates to the canonical
  `v50_train_wdl_prior.py` entry (`--release v50.1` passthrough; wrapped trainer enforces
  `require_store_release`, so non-v50 stores fail closed). The stream profile is transport only;
  the dataset-wide bucketing pass depends on Spec 109's clean-room V50 store builder, which must
  (1) extract with a full-texture-decode profile so the analysis runs and (2) carry
  `minimap_lighting` incl. shading-match fields as a DatasetSignal. Recorded as a research.md
  decision in spec 111.
- Python side: `harvester/spec111/lighting_buckets.py` (reconciled distribution report; pre-111
  tiles missing the field are surfaced as `tiles_without_shading_match_field`, never folded into
  not-evaluated), `rebalance_lighting_variants.py` (largest-remainder allocation into bare-float
  `lighting_times` for the existing spec103 store builder — structurally cannot leak bucket labels
  into model input), `checkpoint_comparison.py` (regressed AND inconclusive both keep the deployed
  checkpoint). `scripts/train_spec111_reconstruction.py` validates and prints the delegated
  `train_spec103_wdl_prior.py` command but refuses to run without `--confirm-run`.
- The drifted `terrain_lighting.py` direction formula is now a documented value-for-value port of
  the corrected C# `TerrainSolarDirection` with a regression test; a streamed (not ported)
  architecture stays a labeled follow-up.
- Proof: focused C# sweep 42/42; Harvest Debug build 0 errors; `tests/spec111/` 16/16;
  `tests/spec103/test_terrain_lighting.py` 10/10; gate smoke-run refused to train and wrote no
  checkpoint. Full data-harvester suite: 3 pre-existing failures (v24 export-map fixture, v25
  h1_coarse neighbor-context API) reproduce without these changes and are unrelated.

## Durable project-wide constraints (carried forward across specs, not relitigated)

- No DepthAnything-family/multi-head/shared-weight model paths (Spec 102 Constitution Check).
- Ground-truth lighting/time is never a deployed-model input (Spec 103/106).
- Canonical storage stays per-build Zarr, no NPZ (constitution principle V).

## Prior active work: Spec 110 viewer stabilization (current state; full session history archived)

Full chronological detail (every individual correction with its own test-count proof) moved to
`memory-bank/archive/2026-07-18-spec110-viewer-stabilization-detail.md`. See `progress.md`'s Spec 110
section for the condensed current-state summary — kept in one place rather than duplicated here.

## Next implementation slice after the visual gate

- Native M2 recovery only: world-object and WMO-doodad code still has adapter-backed MdxRenderer and
  M2→MDX runtime fallback branches, which are forbidden. Route 1.0.0 through
  `BuildEra100StaticRenderModel`/native `M2Renderer`; then remove all MDX conversion runtime paths.
- Do not use converted MDX as renderer proof. M2→MDX is explicit Alpha export only.

## Audited follow-ons

- WMO v14→v17 and v17→v14 Core.IO converters and fixture tests exist. Their real-client fidelity is
  not yet signed off.
- Core M2→MDX has synthetic conversion tests only; do not call it reliable for a client profile yet.
- Main Tools menu audit, removal of MK Dataset/VLM Dataset, and Inspect/Converter launch repair are
  planned in Spec 110 Phase 3 after the fog and native-M2 phases.

## Separate active lane

- Spec 109 v50 clean-room dataset work remains separate. `H:\CLIENTS` is the approved configured
  client library; legacy workspace output was cleared. Do not recreate pre-v50 outputs. V50 now has
  a real per-build store writer (`harvester/v50/store.py` + `scripts/v50_build_dataset.py`,
  fixture-proven and smoke-tested; see Phase 5 above) — the former Spec 108 mixed-copy wrapper is
  fully replaced, not just failing closed. Its frozen liquid policy preserves `liquid_mask`/
  `liquid_height` as useful targets but makes them fresh-only; a WL source requires all three
  contiguous/above-terrain/typed markers, while non-WL sources retain reader identity in row
  lineage. A real user-run build against `H:\CLIENTS` Kalimdor now exists on disk (491 MB, 951
  tiles, real content hashes) — a real dataset has now actually been built, not just user-run-only.
- **Phase 8 (2026-07-17)**: that real build appeared to "randomly" wipe itself and restart. Root
  cause confirmed against the actual store on disk (it was genuinely complete and valid): the new
  `v50_pipeline_runner.py` fed `finalize` the blank manifest template (`row_count: 0`, placeholder
  hashes) instead of the manifest `build` actually produced, so `finalize` always reported
  `finalization_state=incomplete`; a retry against the same `--write-store` path then destroyed the
  good store because `write_v50_store` opened it with unconditional `zarr.open_group(mode="w")`.
  Fixed: `build` now takes `--write-manifest` to persist its real manifest to disk, the pipeline
  runner now feeds that (not the template) to `finalize`, and `write_v50_store` now writes to a
  staging directory and only replaces the target once the new write fully succeeds (with
  retry-with-backoff for transient Windows rename/rmtree denials). Full incident write-up in
  `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`; tasks T053-T057. Full
  data-harvester suite: 580 passed, 43 skipped, 3 pre-existing unrelated failures (same as every
  prior phase). Not yet fixed (documented follow-up, not implicated in this incident): `_cmd_build`
  still accumulates a whole map in memory before one final write, and the harvest-stream/minimap
  pass runs inside one `tempfile.TemporaryDirectory()` that would delete synthesized PNGs on a
  genuine mid-run crash.
- **Docs (2026-07-17)**: user reported not knowing how to run the full corpus. Documented
  `scripts/v50_pipeline_runner.py --confirm` (build → finalize → curate for all four
  terrain-bearing world maps of `0_5_3_3368` — Kalimdor, Azeroth, PVPZone02, Kalidar; the client's
  other WDTs are dungeon/instance interiors with no outdoor MCNK terrain and are correctly excluded)
  as the one-command full-corpus path in `docs/dataset-preparation-userguide.md` (new §8) and
  `data-harvester/README.md` (new "Current active lanes" entry + "V50 quickstart"), with the manual
  per-map fallback and the current one-client-build (`0_5_3_3368`-only; other `H:\CLIENTS` builds
  have no V50 config files yet) scope limit both called out explicitly.
- **Phase 9 (2026-07-18)**: the user's actual first full-corpus run immediately hit two more gaps —
  a legitimate (not false-negative) `finalize=incomplete` on Azeroth aborted the whole multi-map run
  via unconditional `check=True`, and the pipeline's only curation pass silently dropped every
  object-touched tile (correct for minimap→height reconstruction specifically, wrong as the only
  curated view given v50 keeps object masks as real signals). Fixed and completed by hand: all four
  maps now built, finalized, and have both a strict object-free and an object-inclusive curation
  manifest. **The v50.1 `0_5_3_3368` full corpus now actually exists on disk**, not just
  documented — see `progress.md`'s Phase 9 entry for exact kept-tile counts per map/manifest.

## Spec 115 terrain-feature deconfounding + liquid cells + normal supervision (2026-07-20)

**Motivating failure:** the promoted geometry chain read roads as slopes — colour used as a depth
proxy. Confirmed on out-of-distribution input (hand-painted `ek.jpg` tiles).

**Delivered, measured against frozen baselines (not projections):**

- **Terrain-feature classifier** (`terrain_feature_{labels,model,train,infer}.py`): RGB → per-pixel
  family (unknown/terrain/road/water/structure). Feeding its *generated* map to geometry as extra
  input channels cut **road-region height MAE 0.2075 → 0.1632 (−21.35%)**, non-road −8.18%. The
  baseline was worse *inside* road regions than outside (0.2075 vs 0.1877) — direct evidence it was
  reading roads as geometry. v3 overall val MAE 0.1723 vs baseline 0.1878 (−8.3%).
- **Liquid cell classifier** (`liquid_cell_{labels,model,train}.py`): RGB → per-cell none/river/ocean.
  **river IoU 0.7345 at 16×16 chunk grid, 0.8244 at 128×128 quad grid** (recall 0.955), baseline 0.0.
- **Normal gradient supervision** (`normal_guidance.py`, `--normal-weight`): constrains predicted
  height's slope to authored MCNR normals at real vertices. Loss-only; never enters inference input.
- **Depth-aware liquid loss** (`--liquid-depth-aware`): penalty scales with water depth, raising
  effective liquid loss weight 0.50 → 0.83. Terrain is visible through shallow water (depth p50 ≈21)
  and not through ocean (p90 ≈514); one flat constant cannot serve both.
- `direct_geometry_model.py` gained `in_channels` (default 3, hashed into the config identity, so
  existing RGB-only checkpoints stay bit-identical and cannot be confused with deconfounded ones).
- New C# `dump-texture-names` recovers per-tile MTEX tables (verified against Kalimdor 24,40).

**Two structural lessons worth keeping:**

1. **Classify at the authoring unit; don't segment pixels.** Road *segmentation* topped out at IoU
   0.17 (road is 0.26% of pixels). Liquid *cell classification* hit 0.82 with the same architecture
   family. Predict at the mesh's real resolution — a tile is 128×128 quads, not 16×16 chunks.
2. **A target must be visible in the RGB before class balance matters.** `impass` is a collision
   marker with no rendered footprint; `has_mcsh` measured r=-0.006 against minimap luminance (MCSH
   is not baked into minimaps). Both were dropped despite attractive balance.

**Verified facts that corrected working assumptions:**

- Normals are NOT higher-resolution than heights: `mcnr_mask_257` is bit-identical to the height
  quincunx mask (145 samples/chunk).
- Half of `height_257` is format-level interpolation (gap cells reconstruct from orthogonal
  neighbours to within 0.6–5% of height std). Real, but it caps detail; it is not the blur's cause.
- The global `mcly_tileset_ids` → name list is **not persisted** anywhere; the plausible
  `asset_inventory.parquet` substitute was tested and falsified. Use local `mcly_texture_ids` + the
  texture-name dump.
- `uniqueId` is a real object-placement chronology but did NOT predict alpha↔height brush coupling
  (r=-0.007) — it times doodad placement, not terrain authoring.
- Alpha-brush structure and height curvature ARE coupled (mean r +0.158, top tiles +0.64), clustered
  by zone (Wetlands high; Silithus/Desolace/placeholder-heavy negative).

**Known gap, deliberately accepted:** ~9.6% of tiles have MCNR normals that disagree with their own
heights, where normal supervision would push toward flatness. `spec103_curate_dataset.py` has a
`height_normal_mismatch` drop reason but it only catches "flat height + relieved normals", and the
v50.1 curation ran with no-op thresholds (`min_rgb_std: 0.0`, `max_object_coverage: 1.0`,
951/951 kept). User's call: leave it — deployment inputs are poorly upscaled 2002 imagery, so
tolerating imperfect rows is closer to the real inference distribution than over-curating.

**Proof:** full v50 suite 331 passed / 4 skipped; Ruff clean on all new modules; every trainer
dry-runs and refuses to train without `--confirm-run`.

## Known open bug (unrelated lane, reported 2026-07-18, not yet fixed)

- Built-in map GLB export: textures are mirrored along the Y axis. User confirmed via testing GLB
  output on a phone; otherwise export is correct. Not investigated or fixed this session — flagged
  for follow-up.

## Durable boundaries

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- User runs client-backed visual proof, training, capture, and heavy work. Report client root,
  build identity, and fingerprint with any real-data conclusion.
- `AlphaWdtWriter.cs` is frozen. Renderer reader ownership is native M2; export conversion is separate.

## Corpus structure supersedes the raster-regression framing (2026-07-21)

- An ADT is a **relational schema**, not an image: MTEX/MMID/MWID are lookup tables,
  `MCLY.textureId` is a foreign key into the tile's own MTEX, MDDF/MODF are placement joins.
  `terrain_feature_labels.py` already performs that join. Treat terrain reconstruction as
  structured prediction under referential constraints, not continuous raster regression.
- **Terrain is assembled from a reused fractal brush library** (9.5% of L1 alpha blocks are >=0.99
  cross-tile copies under rotation). With the relational framing this makes blur a symptom of
  averaging a *discrete* target space, not just spectral bias.
- **L0 never has an alpha map** (opaque base zone texture). Never include it in an alpha stack, and
  never collapse MCLY layers with `max(axis=2)` — that defect in `brush_mask_from_alpha` made the
  brush loss boost road edges, the exact opposite of the intent.
- **Two validation problems block honest evaluation**: 99.6% train/val spatial adjacency (adjacent
  ADTs share edge vertices, so val is interpolation), and no model has ever beaten the tile-mean
  baseline (0.1387 vs 0.1723+), because 39% of terrain is near-flat. Stratify metrics by relief
  before trusting any aggregate MAE.
- Before reporting that a signal is absent, verify the detector could have found it. Two null
  results on brush reuse were both wrong and both structurally underpowered.
