# Progress — wow-viewer

Last updated: 2026-07-24

## Spec 121 — V7-Style WDL-Prior Height Reconstruction (Small Model Lane) — Stage A code complete (T001–T012)

- **2026-07-24**: Stage A implemented + verified without training. `harvester/spec121/` (4 modules) + CLI + 30 tests (30/30 pass); full suite 1136 passed / 3 pre-existing unrelated failures (v24 export-map, 2× v25 h1_coarse); ruff+compileall clean. Real dry-run on dual_v3 + spec116 split: violation_count=0, lattice arrays present, object-mask absent → warn+disable path proven for real. `MitB0LatticeNet` 3,469,922 params (band-enforced). User decision: substrate named **v50.2** (v50.1 + lattice + object-mask arrays); trainer `--release` defaults v50.2.
- Remaining: T013 USER RUN G1 (Stage A training, SC-001 ≥15% below tile-mean — fail = recorded negative, lane stops). T014–T020 bridge+detailer, T021–T023 paired mask-weight runs, T024–T027 chain materializer, T028–T031 bookkeeping (119/120 already archived).
- Full Spec Kit set written: `specs/121-v7-wdl-height/{spec,plan,research,data-model,quickstart,tasks}.md` + `contracts/cli-contract.md` + `checklists/requirements.md` (validation passes). 31 tasks (T001–T031); T013/T020/T021/T022/T027 are USER RUN training gates.
- **Design**: Stage A `MitB0LatticeNet` (SegFormer-B0 + 545-pt lattice heads, Spec 117 masked contract, ~3.4M params, pretrained optional) → `prior_coarse_bridge` into the detailer's existing `--coarse-store` schema → Stage B existing residual detailer (+ `detailer_mit_b0_v1` trunk option). 3–30M band per model. Object masks loss-side only (tile-coverage weight for A; Spec 118 pixel weight for B).
- **Gates**: G1 Stage A ≥15% below tile-mean (Spec 117's failed bar; fail = recorded negative, lane stops). G2 Stage B ≥9% below prior-only. G3 paired mask-weight verdict + user visual gate.
- Next: T001–T012 (Stage A code + tests + dry-run smoke), then user-run G1.

## Specs 119 + 120 — ARCHIVED 2026-07-24

- Moved to `specs/archived/` with `CLOSED.md` + ARCHIVED.md rows. Minimap object identity is a **measured** dead end (p50=10px instances; ~0.99 cosine matches to unrelated blobs). Do not retry with another backbone — input-resolution physics, not embedding quality. Precise masks survive as loss-side signal in Spec 121. Historical outcomes (classifier 0.9137, segmenter IoU 0.9921, 4,368 curated vectors) preserved in the archived dirs.

## Spec 120 — Minimap OBB Object Detector & Metadata Sidecar Generator (ARCHIVED — see above)

- **Outcome**: curation pipeline worked (1,473 junk pruned → 4,368 verified vectors), but the retrieval goal it served is dead per the 119 PoC scale measurement. `scripts/spec120_dinov2_retrieval.py` kept read-only.

## Spec 119 — object-library classifier/segmenter + quality lens (ARCHIVED — see above)


- **Outcome**: full user-run pipeline completed on `objlib_0_5_3_3368.zarr` (5,841 assets;
  mdx-majority 85.6%, no m2 class in alpha client). Classifier: held-out 0.9137 vs 0.8562
  baseline (best_epoch=21; literal SC-001 +15pp unreachable at this majority share — per-class
  recall is the real verdict: empty 92 / mdx 94 / wmo 43). Segmenter: held-out IoU 0.9921 vs
  0.3529 trivial → SC-002 PASS. Quality lens: 130 mislabels / 200 near-dup pairs / 460
  low-coverage; manual review of top-24 sheet (`mislabel_review.png`) → SC-004 PASS (all flags
  genuine confusables or genuine junk/near-blank captures).
- Optional follow-ups (user-run): base-64 classifier comparison for wmo recall; the
  `empty→mdx` flags show thin-line captures near the 0.01 blank threshold may want a threshold
  or coverage-bucket revisit. Promotion verdicts remain `pending` (user gate).

## Previous: Spec 119 code-complete; user-run training gates remained

- All 30 tasks (T001–T030) implemented: `harvester/spec119/` package (contract, split,
  library_data, classifier_model/train, segmenter_model/train, infer, quality_lens) + 5 thin
  `scripts/spec119_*.py` CLIs + `tests/spec119/` (41 tests, all pass). Full suite 1179 passed /
  3 pre-existing failures (unchanged, unrelated). Ruff + compileall clean. No C#.
- Classifier: 97,938 params @ base 16 (SC-005); segmenter: 482,737 @ base 16. Both from
  scratch, independently checkpointed, constructable from `base` alone (D-02).
- Family-isolated split with mandatory leakage check (`verified_violation_count=0` on the real
  smoke store); blank captures → `empty` class (classifier) / excluded (segmenter) per D-04;
  majority-class + trivial IoU baselines always recorded (FR-005/SC-002).
- `model_stage_contract.STAGES` widened with `object_library_classifier` +
  `object_library_segmenter`; run records reuse `v50-model-stage-run-v1` verbatim,
  `promotion_verdict=pending`.
- All CLIs dry-run-first (FR-010); smoke-store dry-runs for split/train-classifier/
  train-segmenter/quality-lens + loose-PNG infer all verified end-to-end on the real
  `smoke_wmo.zarr` with zero heavy launches.
- Remaining (user-run): full-library split `--write`, classifier `--confirm-run` (SC-001 gate),
  segmenter `--confirm-run` (SC-002 gate, only if SC-001 passes), quality lens `--write` +
  manual review (SC-004). CLIs in `specs/119-object-library-classifier/contracts/cli-contract.md`.

## Object-library capture pipeline — WMO exclusion root-caused + fixed + smoke-validated

- Prior session built `capture-objects` (harvest C#) + `build_object_library.py
  --from-harvest-stream` (Python zarr) + headless `ObjectCaptureRenderer`
  (`WmoObjectRenderer`/`MdxObjectRenderer`) + `WmoFullLoader`, but terminated before
  memory-bank sync. Initial test captured MDX/M2 but **zero WMOs**.
- **Root cause**: 0.5.3.3368 alpha WMOs are listfile-less `.wmo.MPQ` wrapper archives
  registered into `NativeMpqService._scannedArchives` (readable via `ReadFile`) but
  `ListFiles("*")` only returned `ExtractInternalListfiles()` (normal-archive
  `(listfile)` blocks) — never merging `_scannedArchives` — so `capture-objects`
  enumeration never discovered WMOs. MDX/M2 live in listfile-bearing archives, so they
  were found.
- **Fix 1**: `ListFiles` now merges `_scannedArchives.Keys` into the enumerated set.
- **Fix 2**: `ScanNestedWrapperArchives` canonicalizes virtual paths by stripping a
  leading `Data\` — otherwise each `.wmo.mpq` registered twice (game-root scan gives
  `Data\World\wmo\X.wmo`, Data-subdir scan gives `World\wmo\X.wmo`) → every WMO emitted
  twice by `ListFiles` and rendered twice. Stripping collapses both to one canonical
  key matching the MPQ-internal-path convention.
- **Smoke (this session, H:\CLIENTS\0_5_3_3368)**: 5 curated WMOs →
  `build_object_library.py --from-harvest-stream --run` → `Captured 5, 0 skipped, 0
  errors` → `smoke_wmo.zarr` (`capture_rgb (5,128,128,3)`, `capture_mask (5,128,128)`,
  both uint8). Mask coverage 0.377–0.704, image means 26.5–64.6 (non-blank). Validation
  sheet `output/object-library-smoke/smoke_wmo_validation.png` (wmo=5, captured=5).
  Both signals proven end-to-end for WMOs. Build 0 errors. No training touched.
- Remaining (user-run): full-scale whole-client object-library build (data harvest —
  hand off). CLI in activeContext.md.

## Spec 118 — per-object occlusion-aware masks (US1–US3 implemented and code-verified; user-run gates remain)

- Spec Kit docs + all 30 tasks (T001–T030) implemented this session. 48/48 `tests/spec118/` pass;
  full data-harvester suite 1080 passed / 46 skipped / 3 pre-existing failures (unchanged,
  unrelated); ruff/compileall clean on touched files; solution Debug build 0 errors; 23/23 C#
  rasterizer+serializer tests pass.
- **Discovery (Spec 117 US1 shape)**: the strict object-geometry target already computed the
  visibility-correct mask (transformed M2/WMO triangles above raw MCVT +0.25 clearance,
  liquid-aware, front-most rule) and streamed `object_geometry_visible_mask_257`/
  `object_geometry_visible_source_257` (Full/V16, not V22). Only new C#: dense
  `object_geometry_visible_instance_257` (compact per-tile ids, same front-most rule) + instance
  metadata table. US1 = 3 catalog rows + regenerated configs + `object_mask_audit.py`
  (`v118-object-mask-audit-v1`).
- **US2**: `--object-mask-weight` on both geometry trainers (parity-defaulted, mirrors
  `--liquid-mask-weight`) + object-touched/untouched region MAE in run records (FR-008). Ground
  truth loss-side only (FR-014). Shared `harvester/spec118/object_loss.py`.
- **US3**: `ObjectSegmentNet` (from-scratch U-Net-lite, RGB→3-class 256×256), trainer (held-out
  split required, class-weighted CE, D-07 gate 0.40 IoU / 0.50 recall, `object_config.base`
  checkpoint), two-mode infer (`v118-object-infer-v1`), and a feature bridge emitting
  `v115-feature-map-v1` class_count=2 (doodad/building softmax). STAGES +=
  `"object_segmentation"`. One test-caught design fix: absent classes now get the rarest-present
  weight, not 0.
- **Real integration proof**: fixture store + split + random-init checkpoint → real bridge
  `--write` → both geometry trainers' dry-runs accepted the bridged store with ZERO trainer code
  changes (input_channels 5). One test-caught audit bug fixed (map-filter positional misalignment).
- **2026-07-22 — US3 multi-feature-store augmentation (the in-flight work that was actually
  incomplete).** The bridge produced an object `v115-feature-map-v1` store, but the geometry
  trainers accepted only ONE `--feature-store`, already occupied by the Spec 115 terrain-feature map
  in the promoted deconfounded chain — so the object prior could only REPLACE the roads-as-slopes
  deconfounding, never augment it. Objects occlude ground height (a different confound), so it must
  sit alongside. Made `--feature-store` repeatable (`action="append"`) on `direct_geometry_train.py`,
  `geometry_detailer_train.py`, and `direct_geometry_materialize.py` via a new shared
  `harvester/v50/feature_stores.py` (validate/concat-in-CLI-order/road-binding-by-taxonomy);
  `in_channels = 3 + Σ class_counts`; eval helpers gained a backward-compatible `feature_bindings`
  param (legacy single-store callers untouched). 10 new `tests/v50/test_feature_stores.py`; full
  suite 1138 passed / 46 skipped / 3 pre-existing unrelated failures; ruff clean. Tasks T031–T033
  (new Phase 5b), quickstart §3b. Paired terrain-only vs terrain+objects comparison is user-run.
- Remaining (user-run): `H:\CLIENTS` store rebuild (Full profile), US1 audit + eyeball, US2 paired
  with/without-mask training comparison (SC-003; null result stops the line), US3 real training +
  OOD eyeball + geometry comparison (now incl. the terrain+objects augmentation). CLIs in
  `specs/118-object-occlusion-masks/quickstart.md`.

## Spec 117 — WDL-lattice coarse prior for terrain geometry (US1–US3(i) implemented and code-verified)

- Third generated input to the v50 coarse+detailer chain: a per-tile 545-point WDL-scale height
  lattice (Spec 108 FR-001), predicted from minimap RGB alone. US1 export → US2 standalone
  learnability → US3 bridge into the existing `--feature-store` contract, no trainer changes.
- **Discovery**: the C# harvester already streamed the lattice (`wdl_outer_17`/`wdl_inner_16`/
  `wdl_outer_present`/`wdl_inner_present`, `TerrainWdlLattice` in `AdtTensorPackBuilder`) before this
  spec existed. US1 was pure catalog/config wiring (4 rows added to the frozen signal catalog +
  regenerated manifest template/signals config) — zero new C#, zero new ingestion code.
- **US2**: `harvester/spec117/{lattice_model.py, lattice_train.py}` — `LatticeNet` (lean conv
  encoder + two pooled heads, ~178K params at base=8) predicts RGB → 545 values; masked
  encode/decode target contract (absent samples never affect normalization range or loss); reuses
  `height_relative_train`/`direct_geometry_train`'s validated curriculum/split machinery;
  `--held-out-split` REQUIRED (no fallback, FR-004). Widened `model_stage_contract.STAGES` to add
  `"lattice_prior"` so the reused `v50-model-stage-run-v1` schema actually validates the new stage.
- **US3(i)**: `harvester/spec117/lattice_bridge.py` bridges the frozen checkpoint into a
  `v115-feature-map-v1`, `class_count=1` store (bilinear-upsample outer/inner grids, average).
- **Verified for real, not just unit tests**: built a fixture v50 store + held-out split + a
  random-init checkpoint (no CUDA training), then actually ran the trainer dry-run, the
  missing-array refusal, the bridge `--write`, and dry-ran BOTH `v50_train_direct_geometry.py
  --feature-store` and `v50_train_geometry_detailer.py --feature-store` against the bridged output
  — both existing trainers accepted it with zero code changes. Caught one real bug this way: the
  checkpoint needed an explicit `lattice_config.base` field since `architecture.config_sha256` only
  hashes the config, doesn't carry it — `lattice_bridge.py` couldn't otherwise reconstruct the exact
  `LatticeNet` shape before `load_state_dict`.
- 26/26 new tests pass; full data-harvester suite green; ruff/py_compile clean. Remaining
  (explicitly user-run): real store rebuild against `H:\CLIENTS`, real `--confirm-run` training
  (learnable/not-learnable verdict unknown against real data), real US3(ii) paired comparison.
- **2026-07-22 — scheduling bug found + fixed from the first real US2/US3 runs.** All three
  geometry trainers paired `OneCycleLR` (default `pct_start=0.3` → 30-epoch warmup) with a
  patience-15 early-stopper that counted every non-improving epoch as stale; when `patience <
  warmup_epochs` the run died mid-warmup before the LR reached its peak. Detailer worst case
  (zero-init residual head starts at coarse baseline): `detailer-with-lattice-run1` best epoch 2,
  val 0.2301 (coarse 0.2333), early-stop epoch 17. `lattice-run1` survived but plateaued at 0.2427
  vs tile-mean 0.1277 (did not beat baseline). Fix: shared `harvester/v50/lr_schedule.py`
  (`make_onecycle_scheduler` + `warmup_complete`/`warmup_epochs_for`); stale counter suppressed
  until warmup completes; `--pct-start` exposed on all three trainers (default 0.3 = torch parity;
  quickstart recommends 0.1 → 10-epoch warmup for this 43-steps/epoch dataset); `pct_start`/
  `warmup_epochs` now in the dry-run plan. 7 new `tests/v50/test_lr_schedule.py` (incl. the
  patience<warmup kill reproduction); 63 affected tests pass; ruff/py_compile clean. Scheduling fix
  only — learnability verdicts remain user-run. Rerun CLIs in quickstart §2b/§4.
- **2026-07-22 — LatticeNet v2 (U-Net-lite) after the post-fix run still plateaued.**
  `lattice-authored-v2` survived warmup (best epoch 52, early-stop 67) but val 0.2307 vs tile-mean
  0.1277, and train MAE was also above tile-mean → underfit. v1's plain 4-conv encoder pooled to a
  16×16 bottleneck with no skips and couldn't localize the 17×17 field. v2: bottleneck decoded back
  up with skip connections (e3, e2); each head fuses all four levels (16/32/64/128) at the lattice
  resolution. 178K → 675K params at `--base 24`; still constructable from `base` alone (bridge
  unchanged); `architecture_identity` config carries `"arch": "lattice_net_v2"`. 3 new v2 tests;
  36 spec117+lr_schedule tests pass; ruff/py_compile clean; dry-run confirms 675170 params.
  Whether v2 beats tile-mean is user-run; if it overfits 679 tiles, lower `--base` first.
- **2026-07-22 — lattice trainer visibility + V7 insight.** The trainer previously emitted only
  checkpoints + a val_mae (no visuals). Added per-epoch `validation/best_previews/epoch_XXXX.png`
  + final `fixed_rows`/`worst_cases` sheets (reuses `render_validation_sheet`; shows minimap /
  truth lattice / prediction / tile-mean baseline / errors as the dense 256×256 bridge field) and a
  loss-only `--gradient-weight` (V7-ported 2D gradient term; 0 = parity). Reframe from the V7 doc:
  V7 "worked" with WDL as an INPUT prior + normals + masks + residual-around-WDL head, NOT RGB→WDL
  alone; the V7 doc says minimap alone lacks enough elevation signal. Spec 117's RGB→WDL-alone test
  is strictly harder, so "doesn't beat tile-mean from RGB alone" is a valid reportable outcome —
  previews now let us see which failure mode it is. 3 new tests; 39 spec117+lr_schedule pass; ruff/
  py_compile clean; dry-run confirms `gradient_weight` in plan.

## Spec 116 — relational terrain layer reconstruction (FULLY IMPLEMENTED — all 35 tasks done)

- Spec Kit plan + all 35 tasks (T001–T035) implemented and validated. 121 spec116 tests pass;
  ruff clean; compileall clean; full data-harvester suite 1017 passed / 46 skipped / 3
  pre-existing failures (unrelated). No regressions.
- **US1**: `family_slot_consistency.py` + CLI — vocabulary decision (`slot_keyed`/`family_keyed`).
- **US2**: `shape_coverage_coupling.py` + CLI — derivability decision.
- **US4**: `held_out_split.py` (8-neighbour isolation) + `relief_stratification.py` (chunk strata,
  stratified MAE, tile-mean baseline, dihedral NCC overlap) + CLIs. `rescore_geometry_checkpoint`
  (T019) re-scores an existing geometry checkpoint against the split, relief-stratified, read-only
  (`spec116_train_structure.py --rescore-checkpoint --print-only`).
- **US3**: `structure_model.py` (`StructureSlotNet` per detail slot, 16×16 chunk head, FR-008);
  `structure_train.py` (dry-run-first, class-weighted CE, per-class IoU/recall gate D-08,
  `v50-structure-run-v1`); `structure_infer.py` (legality resolver SC-004, OOD audit D-05,
  `v50-structure-infer-v1`) + CLI with two modes: `--inputs`/`--tile-table` (loose images, no
  store, runs on a hand-painted OOD tile) and `--store`/`--dumps` (batch).
- **US5**: `structure_materialize.py` (frozen checkpoint → derived structure store, source
  immutable, checkpoint sha256 bound) + CLI. New `structure_feature_bridge.py` +
  `spec116_structure_to_feature_map.py` adapt that store into the `v115-feature-map-v1` shape the
  existing geometry trainer's `--feature-store` requires. `direct_geometry_train.py` gained
  `apply_held_out_split`/`--held-out-split` to consume the Spec 116 split directly (read-only,
  overrides `--val-key`/`--val-value`, dry-run plan counts kept in sync). Geometry comparison
  documented in quickstart.md 5b.
- Reuses v50 store (no new harvest), Spec 115 `v115.1` taxonomy, Spec 114 sha256 helpers.
  User runs all training (FR-018). All CLIs dry-run-first (FR-015).
- **2026-07-21 verification pass found and fixed 3 gaps between "done" and actually runnable**:
  T019's rescore CLI flags, T027's loose-image infer interface, and the US5 5b geometry-comparison
  handoff (wrong script name `spec114_train_geometry.py`; real trainer had no split mechanism and
  would reject the Spec 116 structure store's schema) were all documented/task-complete but not
  wired into working code. Root cause: tests validated library functions in isolation, never the
  documented CLI invocations end-to-end. Lesson: "tests pass" is not "the quickstart commands
  work" — verify by reading the actual argparse against the docs, not just running the suite.
- **Same day, running the real `--write`/rescore paths surfaced 3 more gaps**: `--build-id`
  defaulted to `""` but the split schema requires non-empty (fixed: auto-derive from the store);
  `rescore_geometry_checkpoint` only built a 3-channel RGB tensor so any 8-channel Spec 115
  checkpoint crashed (fixed: `--feature-store` reconstructs the same generated channels, with
  `--rescore-source` to match the feature store's row-domain coverage). `quickstart.md`'s
  `--source authored_only` was also invalid (real choices: `all`/`authored`/`synthetic`).
- **Rescoring all six direct-geometry checkpoints on the same 444-row authored held-out subset
  overturned two standing findings**: (1) every checkpoint (v1 through v6) beats the trivial
  tile-mean baseline on relief regions (v3-deconfounded best at -40.7%, v1 RGB baseline still
  -9.9%) — the old "no model beats tile-mean" result was an artifact of the leaky split, not a
  model property; (2) relief MAE gets monotonically WORSE from v3→v4→v5→v6 (26.7→33.8→35.0→36.8)
  even though v6 was recorded as the best road-MAE run — brush loss/normal guidance/mcly-brush
  weighting each optimized a leaky-split metric (road MAE) at the cost of general relief-region
  generalization. Deconfounding itself (v3) is the real driver, not the later refinements. Full
  per-checkpoint table in `activeContext.md`'s Spec 116 section. 125/125 spec116 tests pass after
  all six fixes.

## Spec 110 — viewer global light is unconditional

- The interactive viewer now starts every world with its global time-of-day directional/ambient
  light (default noon). Exact-build DBC/LightData bands are retained as raw local profiles and blend
  over that base only while in range; missing or failed locals are identity, and local fog cannot
  leak after leaving a zone. Lighting reports the global base separately from local overlay status.
- Synthetic minimap lighting is untouched. Focused composer/DBC tests: 15 passed. Active viewer
  Debug build: 0 errors. T019 still owns the user-run 3.x visual confirmation.

## Spec 114 — direct minimap-to-terrain (original spec restored; Phase 1-3 code complete)

- **Reverted the unauthorized universal-raster reset (2026-07-19, commit `06151357`).** The
  deployment contract is the authored WoW minimap over the project-owned v50 Zarr store — not
  arbitrary third-party rasters with a DINOv2 student and DPT-Hybrid/MiDaS teacher. All
  universal/teacher code, tests, scripts, and the plan doc were deleted; that lane never executed
  anything (no weights, corpus, training, or inference existed to lose).
- The authored `direct_cnn_v112-authored-v1` run completed all 100 epochs and failed SC-001: best
  epoch 92, validation MAE 0.149267 vs tile-mean 0.138747; evaluator MAE 0.1493349, gradient MAE
  0.0058671, border MAE 0.1607286 over 245 held-out rows. Immutable negative evidence, frozen as the
  mandatory comparison baseline (research.md T003/T017/T018 records). Do not rerun that recipe.
- Phase 1-2 (T002-T009) complete: contract fixtures + dependency-free three-variant validator
  (`model_stage_contract.py`, sha256 identity binding, generated-input provenance), dual-view
  admission-policy builder (`reconstruction_curriculum.py`) and dry-run-first CLI. It refuses
  grouped-split leaks and mixed lighting provenance, excludes stale synthetics honestly (1,629
  authored kept / 1,361 synthetic excluded on today's store), and never zero-fills missing rows.
- Phase 3 code (T014-T015) complete: `direct_geometry_model.py` architecture registry
  (`direct_cnn_v112` + `mit_b0_regression`, one bounded 257×257 output, DepthAnything refusal,
  FR-013 pinned-optional pretrained path) and `direct_geometry_train.py` — flat+tile-mean in-run
  baselines, SC-001 vs the frozen Spec 112 run, SC-002 border/interior-p95, per-row quantile/worst
  sheets, schema-validated `model_stage_run.json` (`promotion_verdict=pending`), optional
  AMP/OneCycle/clip at bootstrap-parity defaults.
- Proof: full v50 suite 242 passed / 4 skipped; Ruff clean; both CLIs dry-run without writing. No
  corpus build or training was launched.
- **`mit_b0-authored-v1` (user-run) done**: best epoch 93, val MAE 0.187802, SC-001 false, SC-002
  true; visually the strongest geometry so far. Diagnosis: spectral bias vs fractal terrain
  structure (smooth, under-amplituded relief) — recorded in research.md.
- T056 deployment inference complete: loose 256x256 tile(s) → 16-bit relief PNG + review sheet +
  hash-bound manifest via `v50_infer_direct_geometry.py` (FR-015; dry-run default).
- Spectral guidance (Spec 068 US1 revived as loss-only): radial log-power + multi-octave gradient
  terms behind `--spectral-weight`/`--multiscale-weight` (default 0). Full v50 suite 257 passed /
  4 skipped; Ruff clean.
- **`mit_b0-authored-v2-spectral` (user-run) done**: best epoch 130, val MAE 0.193435, SC-001
  false, SC-002 true. Both coarse runs plateau at ≈0.19 (train loss ≈0.016): single-stage
  capacity saturated at 1,384 tiles. Spectral term sharpened structure visually at small MAE
  cost — MAE and perceived detail now measure different things.
- **Residual detailer stage (T058-T060) implemented**: coarse-output materializer (frozen
  checkpoint → derived `coarse_relief` Zarr, checkpoint-hash-bound, source stores immutable,
  1:1 row alignment validated), residual U-Net-lite detailer (RGB + generated coarse → one
  residual field; zero-init head starts AT coarse baseline), and detailer trainer (coarse-only
  strong baseline, ≥5% relative gate, SC-002, fixed/quantile/worst sheets, `upstream_models`
  provenance). 16 focused tests; full v50 suite 273 passed / 4 skipped; Ruff clean.
- **T061 DONE (user-run)**: `detailer-mit_b0-authored-v1` best epoch 91, val MAE 0.170665 vs
  coarse-only 0.187800 (9.1% relative), gate=True, sc002=True. First Spec 114 geometry checkpoint
  to clear its numeric gate. Two-stage residual chain proven.
- **T062 DONE**: V7 vs current detailer comparison. V7's four structural loss terms (full 2D
  frequency, Laplacian, Sobel edge, transition-focus 3×) + V25's LF/HF band split are the missing
  multi-band prior. V7 compensated for dirty data via structural statistics, not clean inputs.
  Decision: port all five as loss-only flags into the detailer trainer (T063), not the V7 arch.
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
  `--source all` stays gated on Spec 113
  NoonWhiteGlobal rerender; object-mask phase starts only after geometry promotion.
- Phase order is fail-closed: corrected dual-view curriculum → direct geometry bakeoff → trusted
  object visibility → feature library → texture families → alpha. The user owns all heavy
  builds/training. Spec 113 still owns RealPLKSR/detail.

## Specs 112/113 — implementation continuation

- Corrected the shared terrain lighting seam: CPU minimap composition now transforms raw ADT MCNR
  normals into renderer coordinates before the Lambert dot. This invalidates only previously
  synthesized RGB outputs, not the numeric reconstruction arrays in the same v50 stores.
- Corrected lighting ownership after the 2.4.3 authored-reference proof: `synthetic-minimap` now
  uses one fixed-noon achromatic global light for every era, emits v6 `NoonWhiteGlobal` evidence,
  and rejects non-noon/DBC options. Exact-build Light DBC loading, anomaly recovery, and status are
  retained only in the interactive viewer. Focused compositor/detail/DBC tests pass 38/38; Harvest
  builds with zero errors.
- Corrected the visual-proof handoff after Expansion01 32,32 produced black despite WDT occupancy.
  Added bounded multi-tile authored-reference output and all-black rejection. The replacement six
  Expansion01 coordinates have readable nonblack authored minimaps and 5-10 nonblack decoded terrain
  textures each; the five Kalimdor coordinates come from the persisted same-row visual review.

- Spec 112 T012/T013/T016 real-data gates recorded; T017-T020 landed. Relative-height contract now
  follows the published floor formula exactly (flat→0, near-flat relief retained), and the trainer
  fails closed on wrong schema, out-of-scope maps, row misalignment, invalid source/split values,
  or split-group leakage. T021 remains the user-run CUDA training gate.
- Spec 113 T001-T008 landed. Corrected the initial false 1-repeat assumption to the production
  shader's 8 repeats/chunk and added deterministic mip selection to avoid base-texture minification
  aliasing. The canonical v50 build makes 256 material-average / 1024 detail explicit and records
  detail provenance on the store. Alignment now gates both big maps on one dihedral transform and
  fixed LR offset; wrapped-edge correlation is forbidden.
- Specs/runbooks/README/user guide synchronized to the ComfyUI-native RealPLKSR decision and the
  exact staged T009 commands. `H:\CLIENTS` policy in active data-path docs now matches AGENTS Rule 9.
- Follow-up: synthesized minimaps are now cataloged as optional partial-coverage signals; the
  finalizer's `--policy-template` reconciles stale manifests using only real row lineage, so the
  already-built Kalimdor detail staging store can finalize without re-harvest (731/951 synthesis
  coverage, 220 honest gaps).
- Real Spec 113 US1 proof: the 120-tile Kalimdor+Azeroth report returned `fail_inconsistent` (NCC
  p50 0.211 / p05 0.000; no fixed transform or offset) although SC-001 detail gain passed at 16.10.
  Visual comparison confirms an intentional authored-with-objects versus terrain-only target domain,
  plus the now-fixed light-direction bug. Continue under the explicit identity, terrain-only
  cross-domain contract after the user signs off the fresh comparison sets and synthetic RGB is rebuilt.
- Added the same-row datastore contact-sheet tool and fail-closed terrain-only pair builder. The
  builder requires explicit cross-domain mode plus persisted visual-review evidence when raw NCC
  fails, forbids per-tile transforms, counts every missing source, and preserves leak-safe splits.
  The ComfyUI-native RealPLKSR model wrapper is also present; real pair-set promotion and trainer
  implementation remain behind T010b/T013.
- Verification: `tests/v50/ + test_v50_contract.py + test_v50_build_command.py` = 175 passed,
  4 skipped; current combined DBC/lookup/compositor/detail C# focus = 41 passed; Harvest build =
  0 errors; full Debug solution build = 0 errors (existing warnings remain).

## Spec 109 — v50 clean-room dataset (Setup & Build Pipeline complete and tested; ready for user build run)

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

## Spec 111 — minimap lighting calibration

- Implemented all code phases (T001–T018, T020–T021) up to the explicitly gated T019 training run.
  C# side: six additive shading-match fields on `MinimapLightingProvenance`;
  `Core.IO/Maps/MinimapShadingMatch.cs` sweeps 24 hourly `TerrainMinimapCompositor` candidates and
  scores tint-invariant luma Pearson correlation (gradient-direction cosine was tried and discarded:
  with the fixed azimuth it cannot distinguish hours); chained onto the existing Full/V22
  `AnalyzeAuthoredMinimapLighting` streaming pathway with an internal 0.5.3.3368 fingerprint gate —
  no new command, zero cost for other builds. Python side: `harvester/spec111/`
  (`lighting_buckets.py` reconciled report, `rebalance_lighting_variants.py` bare-float
  largest-remainder `lighting_times`, `checkpoint_comparison.py` where regressed and inconclusive
  both keep the deployed checkpoint) plus three thin CLIs; `train_spec111_reconstruction.py`
  validates and refuses to train without `--confirm-run` (smoke-proven). The drifted
  `terrain_lighting.py` direction formula became a documented port of the corrected C#
  `TerrainSolarDirection` with regression coverage.
- Lane correction after user feedback: the target lane is **v50**, not the legacy
  V22/spec103/spec108 naming the first pass anchored on. Phase 3 delegation now goes through the
  canonical `v50_train_wdl_prior.py` entry with `--release` passthrough (its `require_store_release`
  makes non-v50 stores fail closed); docs/readers now state the harvester stream profile is
  transport only and the dataset-wide store pass depends on Spec 109's clean-room builder carrying
  `minimap_lighting` as a DatasetSignal from a full-texture-decode extraction profile.
- Proof: focused C# sweep 42/42; Debug Harvest build 0 errors; `tests/spec111/` 16/16;
  `tests/spec103/test_terrain_lighting.py` 10/10. Full data-harvester suite 548 passed with 3
  pre-existing unrelated failures (v24 export-map fixture, v25 h1_coarse neighbor-context API).
- Remaining user-run proof: bounded real-0.5.3.3368 `harvest-stream --stream-profile v22` bucketing
  pass, the quickstart side-by-side eyeball check on `matched` tiles, whole-build report, then the
  separately authorized T019 retrain/evaluate.

### Planning record

- Created the Spec Kit spec, plan, research, data-model, contract, quickstart, and dependency-ordered
  tasks (`specs/111-minimap-lighting-calibration/`).
- Three user stories: US1 shading-based lighting-bucket inference for the real 0.5.3.3368 dataset
  (MVP), US2 rebalance synthetic-lighting-variant training sampling to match the real distribution,
  US3 retrain-and-evaluate the existing reconstruction model with an explicit go/no-go gate.
- Confirmed with the user before writing the spec: build scope is 0.5.3.3368 only; training scope
  includes the full retrain-and-evaluate loop (not just data prep), with its GPU/cloud execution step
  explicitly gated on separate authorization at run time.
- Research surfaced that `data-harvester/src/harvester/spec103/terrain_lighting.py` independently
  reimplements the solar-direction model and has drifted to `v1` while the corrected C# path is now
  `v3` — the plan retires that duplication rather than syncing constants by hand.

## Spec 110 — viewer stabilization (current state; full session history archived)

Full chronological detail (every individual correction, each with its own test-count proof) moved to
`memory-bank/archive/2026-07-18-spec110-viewer-stabilization-detail.md` per the archive convention.
What's true now:

- Current phase: **Phase 1d terrain-minimap fidelity correction**, gated on a bounded real-client
  visual re-export proof (see "Required user proof" below) before M2/tool/conversion work starts.
- Fog: user Fog Start/Fog End apply after LIT/DBC recommendation with Core-side normalization for
  invalid/collapsed ranges; controls live in Lighting, Settings is load-default-only.
- `TerrainMinimapCompositor` synthesizes terrain-only tiles from MCLY/MCAL/MCNR/decoded BLPs using
  per-BLP phase-independent material averages (not renderer UV/mip sampling, which moiréd at minimap
  scale); MCSH is excluded from normal RGB. `TerrainMinimapLiquidCompositor` produces an aligned
  `_liquid` companion from real world-geometry quad rasterization, never sparse stamps.
  `TerrainSolarDirection` locks horizontal bearing to a fixed north-west azimuth (matching the traced
  native ray, theta=225°) with only elevation varying — this codebase's MCNR/MCVT convention is
  +X=North, +Y=West, +Z=Up, confirmed by cross-referencing the traced 1.0.0 native `SetDirection` ray.
- Alpha-format tensor/liquid decoding is corrected for: MCLY/MCAL `[chunkX,chunkY,layer]`→row-major
  transposition, WDT MAIN row-major tile enumeration (`tileY*64+tileX`, matching the reader not
  transposed), cross-tile WMO roof-footprint painters bounds-checked against their own 256² buffer
  (not the 257² terrain grid), and MCLQ per-cell (not per-chunk) liquid type/coverage with the
  corrected `0x01=Ocean/0x03=Slime/0x04=River/0x06=Magma` mapping.
  WL* liquid provenance requires all three of `wl_liquid_surface_quads_v1`/`wl_liquid_above_terrain_v1`/
  `wl_liquid_basic_type_header_v1`; V16/V18/V50 reject any incomplete WL fallback, so all earlier
  WL-liquid datasets are invalid and need re-harvest.
- Missing terrain diffuse BLPs fall back through: same-stem `_s.blp` companion →
  up to 16 catalog-scanned `.blp` candidates (exact/similar basename first) → a
  `catalog_rgb_last_resort_proxy` (same folder, then terrain family, then prior verified decode).
  Only named materials get a proxy; a tile with no non-empty MTEX name is an unlit solid-white
  baseline. Original MTEX identity is always preserved in metadata alongside the substitute.
- `Tools > Export > Synthesized Terrain Minimap...` resolves the in-repo Harvest executable/DLL/
  project itself (no external binary dependency); accepts minute-precise `HHmm`/`HH:mm` time input.
  UniqueId range/layer/playback controls live only under `Tools > Archeology`, with pause/stop safe
  against a vanishing world/range.

### Required user proof before next code phase

- Re-export one bounded terrain tile with the corrected compositor; confirm stable materials, MCAL
  blends, unshadowed RGB, smooth terrain lighting, and an aligned `_liquid` companion — no moire,
  MCNR checkerboard, unexpected MCSH bake, or blur — before a whole-map job. Run the remaining
  failing `Kalimdor` coordinates `(36,44)`/`(37,44)` first; verify one former texture-only skip now
  records `catalog_rgb_last_resort_proxy`.
- Inspect one bounded V22/full stream's `minimap_lighting` metadata for explicit tint/shadow evidence
  or an explicit not-evaluated reason — never a silently inferred exact historical time.
- Run Spec 110 quickstart against a configured LIT map and no-LIT map at dawn/noon/dusk/night;
  confirm active source/range changes, no terrain disappearance, LIT markers/camera framing, a
  per-tile then whole-map export with captured client root/build/fingerprint, and that Archeology
  playback is reachable/pausable/stoppable through every nested tab in both UI modes.

### Next phase (blocked on above proof)

- Make M2 runtime native-only and remove all M2→MDX / adapter-backed MdxRenderer fallback branches.
- Then clean Tools entries and publish conversion capability levels. WMO v14↔v17 is fixture-covered
  in both directions; M2→MDX remains synthetic-test-only until real-client export proof.

## Separate continuity

- Spec 109 v50 clean-room dataset work is separate from Spec 110. V50 now has a canonical
  per-build writer (`harvester/v50/store.py` + `scripts/v50_build_dataset.py migrate-v18/build/
  verify/finalize/curriculum`; see Phase 5 above) replacing the legacy mixed-copy builder. Its
  liquid contract keeps `liquid_mask`/`liquid_height` as fresh-only targets: historic payloads are
  rejected; fresh WL sources require contiguous, above-terrain, and typed provenance; non-WL
  sources must retain their reader identity in row lineage. Focused V50/WL contract coverage: 5
  Python tests passed. A real user-run build against `H:\CLIENTS` Kalimdor now exists on disk
  (491 MB, 951 tiles, real content hashes).
- **Phase 8 (2026-07-17) — fixed a real data-loss bug found in that build.** The build had actually
  succeeded, but `v50_pipeline_runner.py`'s `finalize` step fed it the blank manifest template
  (`row_count: 0`) instead of the real manifest `build` produced, so `finalize` always reported
  `finalization_state=incomplete`; retrying then destroyed the good store because
  `write_v50_store` used unconditional `zarr.open_group(mode="w")`. Root-caused against the actual
  store already on disk, not reproduced from a guess. Fixed: `build` gained `--write-manifest` to
  persist its real manifest; the pipeline runner now feeds that to `finalize`; `write_v50_store`
  now stages its write and only replaces the target once fully successful, with retry-with-backoff
  for transient Windows rename/rmtree denials (confirmed necessary — hit a real
  `PermissionError`/WinError 5 during the fix's own test run, stable across 3 reruns after the
  retry was added). Tasks T053-T057; full write-up in
  `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`. Full data-harvester suite:
  580 passed, 43 skipped, 3 pre-existing unrelated failures (unchanged from every prior phase).
- **Docs pass (2026-07-17)**: added `docs/dataset-preparation-userguide.md` §8 (V50 is now the
  documented current/canonical lane there, superseding the V16/V22/V23 sections above it for new
  work) and a `data-harvester/README.md` "V50 quickstart" entry, both centered on the one-command
  full-corpus run `uv run python scripts/v50_pipeline_runner.py --confirm` (builds, finalizes, and
  pre-curates all four terrain-bearing world maps of `0_5_3_3368`: Kalimdor, Azeroth, PVPZone02,
  Kalidar). Both docs call out that only `0_5_3_3368` has V50 config files today — the many other
  `H:\CLIENTS` builds need their own `v50-signals-*.json`/`v50-manifest-template-*.json` before V50
  can target them, which has not been done for any build beyond this one.
- **Phase 9 (2026-07-18) — the first real full-corpus run hit two more gaps immediately.** (1)
  Azeroth's `finalize` legitimately reported `incomplete` (2 real tiles lack `minimap_rgb` — no
  texture data to synthesize from) but `finalize` only ever printed the bare state, and
  `v50_pipeline_runner.py`'s unconditional `check=True` treated that as fatal and killed the whole
  run before PVPZone02/Kalidar ever started. (2) The pipeline's one curation pass
  (`--max-object-coverage 0.0`) is correct for minimap-to-height reconstruction specifically (object
  footprints occlude true ground height) but was silently discarding every object-touched tile
  (51.8% of Azeroth, 54.5% of Kalimdor) from the only curated view, even though v50 keeps
  `object_precise_mask`/`object_instance_mask` as real signals. Fixed: `finalize_store_report()`
  names every concrete mismatch reason; the pipeline runner is now per-map resilient (a `build`
  failure skips only that map, a non-complete `finalize` no longer aborts anything, a final summary
  table prints every run); a second `-object-inclusive` curation manifest (`--max-object-coverage
  1.0`) now ships alongside the strict one for every map — neither ever duplicates array data.
  Completed the interrupted run by hand: all four maps built and finalized; strict/object-inclusive
  kept counts are Kalimdor 421→939, Azeroth 328→683, PVPZone02 60→63, Kalidar 24→36 (out of
  951/685/64/56). Tasks T058-T064; full write-up in
  `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`. `tests/v50/
  tests/test_v50_contract.py tests/test_v50_build_command.py` → 120 passed, 2 skipped, 0 failed.
  **The v50.1 `0_5_3_3368` full corpus is now actually built and curated (both manifest flavors),
  not just documented.**

- **2026-07-20 — Spec 115: texture-confound deconfounding, liquid cell classification, normal
  supervision.** The promoted geometry chain was reading roads as slopes (colour as depth proxy),
  visible on out-of-distribution `ek.jpg` input. Built a separate terrain-feature classifier (RGB →
  per-pixel family), materialized its *generated* output as extra geometry input channels, and
  retrained: **road-region height MAE 0.2075 → 0.1632 (−21.35%)**, non-road −8.18%, overall val MAE
  0.1878 → 0.1723 (−8.3%). The baseline had been worse *inside* road regions than outside — direct
  evidence of the confound; v3 inverts that. Then built a per-cell liquid classifier (none/river/
  ocean): **river IoU 0.7345 at chunk resolution, 0.8244 at 128×128 quad resolution** (recall 0.955)
  against a 0.0 majority baseline. Added normal-derived gradient supervision (`normal_guidance.py`,
  `--normal-weight`) constraining predicted slope to authored MCNR normals — the non-adversarial
  answer to "PatchGAN for detail", using exact ground truth instead of a discriminator. Added
  depth-aware liquid loss weighting (effective weight 0.50 → 0.83; terrain is visible through
  shallow water, not ocean). Key lessons: **classify at the authoring unit rather than segmenting
  pixels** (road segmentation IoU 0.17 vs liquid cells 0.82, same architecture family) and **predict
  at the mesh's real resolution** (128×128 quads, not 16×16 chunks). Corrected three working
  assumptions with measurement: normals are NOT higher-resolution than heights (identical quincunx
  mask); half of `height_257` is format-level interpolation (caps detail, does not cause blur); the
  global tileset→name list is not persisted and the `asset_inventory` substitute is falsified. New:
  `terrain_feature_*`, `liquid_cell_*`, `normal_guidance.py`, `feature_map_materialize.py`, C#
  `dump-texture-names`, `relief-to-map`, `split-minimap-image`. Specs `115-terrain-feature-
  classifier/{spec,plan,research,data-model,quickstart}.md`. Full v50 suite **331 passed, 4
  skipped**; Ruff clean; all trainers dry-run and refuse to train without `--confirm-run`.

## 2026-07-21 — Corpus structure findings + MCLY layer guidance (road MAE best-yet)

**Trainer optimisation.** `--cache-samples` on `direct_geometry_train.py`: memoises built samples in
RAM. `--workers` is pinned to `choices=[0]` on Windows, so all Zarr decompression ran on the main
thread with the GPU at 50% util (measured 31.5 ms/sample of raw reads; ~44 s/epoch). Sample
construction is deterministic, so caching is **bit-identical** and keeps runs comparable. Measured
effect: full training run **~1 hour -> ~10 minutes** (~6x, larger than the ~2x predicted because the
cache also removes per-sample derived work — `encode_relative_height`, `interpolate`,
`brush_mask_from_alpha`, depth computation — which the I/O-only measurement missed).

**MCLY per-layer loss guidance (user's idea, delivered).** `brush_mask_from_alpha` collapses the 4
MCLY layers via `max(axis=2)`, making a road edge indistinguishable from a cliff-detail edge — so
`--brush-loss-weight` was actively teaching the model to put *more* height detail at road borders.
Added `--label-store` + `--flat-paint-weight` (up-weights point loss on road-family pixels, whose
target is already flat) and `--brush-exclude-roads` (withholds the brush boost there). Ground truth
is admissible because it is **loss-side only**; the feature map had to be *predicted* solely because
it is an input — and its road IoU is only 0.17, so exact labels are strictly better here.

| run | road MAE | non-road | best val | last-25 sd |
|---|---|---|---|---|
| v3-deconfounded | 0.16322 | 0.17234 | 0.17230 | 0.00156 |
| v5-normals | 0.18164 | 0.17297 | 0.17300 | 0.00715 |
| **v6-mcly-brush** | **0.152246** | 0.175126 | 0.175012 | 0.00158 |

v6 is the best road result to date: **-6.7% vs v3, -26.6% vs the v1 baseline**. v5's normal-gradient
term REGRESSED roads (confound signature returned: road worse than non-road) and made val 4.6x
noisier; removing it restored v3-level stability, confirming normals as the destabiliser.
Also fixed: `liquid_mask_weight` was applied but never recorded in `training_plan.json`, so v3/v4
liquid settings are permanently unrecoverable. Now recorded. Note `--liquid-mask-weight` is a
PENALTY (`point_weight = 1 - penalty*liq`), so 0.25 gives the 0.75 weight originally intended.

**Corpus structure (measured, changes the modelling problem).**
- **Copy-paste reuse confirmed**: 9.5% of L1 alpha 32x32 blocks match a block in a *different* tile
  at >=0.99 correlation under 8 dihedral transforms (median best-match 0.662). Terrain is assembled
  from a reused fractal brush library. Two earlier tests wrongly reported no reuse — both were
  structurally incapable of detecting it (see the memory note on detector power).
- **L0 has no alpha map** — always-opaque base zone texture. Three detail layers over a base.
- L1->L3 monotonically finer (centroid 18.2->20.9 cyc/tile) but a **gradient, not a partition**.
- Dominant layer: L0 76.35%, L1 15.30%, L2 6.21%, L3 2.13% — far better posed than road (0.53%).
- Alpha<->height coupling weak under a *linear* test (median |r| 0.05-0.16, no bimodality), but
  Pearson cannot see threshold rules; a per-tile (height, slope)->alpha fit is the deciding test.
- **99.6% of val tiles have a train edge-neighbour; 42.4% fully surrounded.** Adjacent ADTs share
  edge vertices exactly, so val measures interpolation, not generalisation. Needs a blocked split.
- **`sc001=False` in every run: no model has beaten the tile-mean baseline** (0.1387 vs 0.1723+).
  39% of height patches are near-flat and 51/120 tiles are >90% base-only, so a constant-per-tile
  predictor is strong. Stratify the metric by relief content.

**Reframe:** an ADT is a serialized relational schema (MTEX/MMID/MWID lookup tables,
`MCLY.textureId` a foreign key, MDDF/MODF placement joins, MCIN an index). We are doing continuous
raster regression on structured prediction with referential constraints — which, with the discrete
brush alphabet above, explains blur as averaging over a categorical space.

Tests **339 passed, 4 skipped**; Ruff clean.
