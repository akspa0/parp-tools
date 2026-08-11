# Progress — wow-viewer

Last updated: 2026-08-10

**This file is a dated ledger of what shipped, newest first.** One entry per session, a few lines
each. Findings and how-it-works go in the workstream file; this only records *that* it happened and
what the evidence was. See "Memory bank layout" in `coding_standards.md`.

Current state and open work: [activeContext.md](activeContext.md).

## 2026-08-10 — Spec 136 safe doodad submission batching

- Static legacy-backed M2s now expose the existing batch path unless particle/ribbon state requires
  unbatched rendering; native-runtime M2s remain isolated until a backend-specific batch key exists.
- Opaque WMO doodads sharing an `IModelRenderer` now share one batch setup. Transparent doodads keep
  distance order, and particle/ribbon renderers keep the unbatched fallback.
- Focused Spec 142 proof remains 34 passing tests. This is CPU/state submission batching, not GPU
  instancing or a measured FPS result; the runtime capture remains user-run.

## 2026-08-10 — Spec 142 residency-safe graph rebuild

- Removed synchronous WMO summary reads from residency-triggered scene-graph rebuilds. The graph
  now consumes cached summaries only; missing optional WMO group metadata remains fail-open.
- Focused graph/workload/traversal/adapter/portal proof passes all 34 tests. Isolated viewer build
  passes with 0 errors; the normal output was locked by the user's running viewer process.

## 2026-08-10 — Spec 142 runtime activation and residency diagnostics

- Promoted `WorldScene.UseHierarchicalSceneTraversal` to default-on while retaining a state-invalidating
  runtime toggle back to the legacy path for A/B diagnosis.
- Added viewer diagnostics for graph roots/traversal rejection, AOI camera and detailed/retained tile
  counts, and the last ADT unload with its WMO placement count. This distinguishes tile residency loss
  from WMO visibility culling before changing policy.
- Viewer build passes with 0 errors and existing warnings. No real-client capture, GPU measurement, or
  AOI/cull-policy change has been run.

## 2026-08-10 — Spec 142 independent ADT scene-graph roots

- Added `WorldSceneGraphBuildSet`: each resident ADT tile now owns an independent `Tile`-rooted
  graph, while external content remains in the existing separate graph path.
- Opt-in `WorldScene` traversal, portal lookup, visibility aggregation, and snapshot reporting now
  operate across those per-ADT graphs without changing the legacy path. Focused proof is 34 passing
  tests and the viewer build has 0 errors. No real-scene/GPU performance claim exists yet.

## 2026-08-10 — Spec 142 deferred ADT M2 leaf visibility

- The opt-in graph keeps terrain-chunk subtree rejection but defers ordinary ADT M2 leaf visibility
  to the existing M2 collector, removing the duplicate graph-level leaf predicate.
- Deferred leaves are attributed by node kind; focused proof is 33 passing tests and the viewer
 build has 0 errors. No renderer submission or real-client/GPU performance claim exists.

## 2026-08-10 — Spec 142 traversal rejection attribution

- Added per-node-kind diagnostics for individually tested nodes, rejected subtree roots, and
  skipped descendants. The existing rejected terrain-chunk test now proves one Chunk rejection
  skipped two M2 descendants.
- This is diagnostic evidence only; M2 submission, WMO submission, real-client parity, and GPU
  timing remain unproven. The focused suite remains 32 passing tests.

## 2026-08-10 — Spec 142 ADT M2 doodad chunk partition

- Added deterministic spatial-bucket metadata to the opt-in scene-graph object adapter and
  mounted resident, non-skybox ADT M2 placements as map → tile → chunk → M2 nodes.
- Chunk bounds union resolved placement bounds; unresolved M2 bounds keep the chunk and ancestors
  fail open. A rejected chunk skips its ordinary doodad descendants in focused traversal proof.
- Seven adapter tests pass and the runtime/viewer builds have 0 errors with existing warnings.
  Legacy traversal, WMO group submission, WMO doodad-set submission, real-client capture, and
  GPU performance evidence remain unchanged and unproven.

## 2026-08-10 — Spec 142 graph-side runtime portal traversal

- Added `WorldScenePortalVisibilityEvaluator`: it selects the containing WMO group, traverses
  adjacency through bounded portal volumes, and fails open to all graph groups on uncertainty.
- Opt-in `WorldScene` traversal applies the result to nested `WmoGroup` graph nodes only; whole-WMO
  collection and `WmoRenderer` submission remain unchanged. Focused proof is 29 passing tests and
  the viewer build has 0 errors with existing warnings.
- Runtime WMO submission/doorway parity, pass/query integration, and performance evidence remain
  open; no real-client or GPU capture was launched.

## 2026-08-10 — Spec 142 opt-in runtime portal bridge

- Exposed the existing loaded `WmoRenderer` portal data through the graph read-model contract and
  cached placement-keyed `WorldScenePortalAdapter` results during opt-in `WorldScene` graph rebuilds.
- Nested group IDs align with the graph children; unloaded WMOs fail open and the current WMO
  visibility path is unchanged. Viewer build passed with 0 errors; no runtime capture was launched.
- Runtime nested-volume traversal, doorway parity, pass/query integration, and performance proof
  remain open.

## 2026-08-10 — Spec 142 bounded portal view-volume contract

- Added `WorldScenePortalViewVolumeBuilder`: parent planes are preserved, portal-edge planes form
  a bounded doorway cone, and the decoded destination-side portal plane is retained.
- Depth limits, unknown sides, invalid geometry, degenerate edges, and camera-on-plane cases fail
  open with explicit reasons. Focused Spec 142 proof is now 26 passing tests; viewer build remains
  0 errors with existing warnings.
- `WorldScene` and `WmoRenderer` do not consume this contract yet; runtime doorway parity and all
  heavy capture/performance evidence remain open.

## 2026-08-10 — Spec 142 WMO portal read-model adapter

- Added `WorldScenePortalAdapter` to consume existing `WmoRenderDocument` portal vertices,
  geometry, and group references without changing format readers or duplicating `WmoRenderer`.
- Valid portal geometry is preserved with stable group IDs; malformed geometry and unknown groups
  become explicit graph fallback diagnostics. Focused Spec 142 proof is now 23 passing tests and
  the viewer build has 0 errors with existing warnings.
- Nested view-volume clipping, doorway parity, runtime `WorldScene` integration, and heavy capture
  remain open; no GPU or real-client capture was launched.

## 2026-08-10 — Spec 142 bounded portal adjacency contract

- Added `WorldScenePortalGraph`, a graph-only adjacency contract for stable WMO-group-like node
  IDs and portal links. Malformed/unknown links are rejected; traversal reports cycles, missing
  entries, absent portal data, and maximum-depth fallback deterministically.
- Four portal graph tests pass, bringing focused Spec 142 graph/workload/traversal/adapter/portal
  proof to 19 passing tests. The viewer build remains at 0 errors with existing warnings.
- This does not implement portal geometry, doorway clipping, renderer parity, or `WorldScene`
  integration. Existing `WmoRenderer` behavior remains the owner; no heavy capture was launched.

## 2026-08-10 — Spec 142 nested WMO group mounting

- Extended the scene-graph object adapter with nested child nodes and mounted client-backed
  `WmoMeshSummary.GroupSummaries` beneath WMO placements in the opt-in `WorldScene` graph.
- Group IDs, local bounds, asset keys, and portal-group metadata are preserved; malformed bounds
  are fail-open. Existing `WmoRenderer` portal traversal remains the behavior owner for now.
- Focused graph/workload/traversal/adapter proof is 15 passing tests and the viewer build has 0
  errors. Terrain chunks, graph-side portals, pass/query reuse, and performance evidence remain
  open; no heavy capture was launched.

## 2026-08-10 — Spec 142 opt-in WorldScene adapter and traversal selector

- Added `WorldSceneGraphObjectAdapter`, which maps existing resolved object placements into stable
  map/tile/external bucket nodes and preserves unknown bounds as fail-open.
- Wired `WorldScene.UseHierarchicalSceneTraversal` behind the existing WMO/MDX visibility
  collectors. The graph is rebuilt on residency/bounds changes and traversed once per frame when
  enabled; the legacy path remains default.
- Focused graph/workload/traversal/adapter proof is 14 passing tests and the `WoWViewer` project
  builds with existing warnings. This is not an FPS or GPU result; terrain chunks, WMO portals,
  pass/query reuse, and real-client parity remain open.

## 2026-08-10 — Spec 142 Phase 1 graph foundation

- Implemented `WorldSceneGraph` and `WorldSceneNode` in `WowViewer.Core.Runtime` with stable IDs,
  nested attachment, complete subtree detach, transform propagation, conservative bounds, and
  explicit non-rejectable handling.
- Implemented deterministic hashed synthetic-world manifests covering sparse regions, chunks,
  nested WMO groups, repeated M2 assets, PM4 overlays, render-pass mix, and portal metadata.
- Focused Spec 142 proof is 8 passing graph/workload tests; runtime build passes with existing
  repository warnings. The active renderer traversal is unchanged; next slice is shared
  conservative traversal behind a selector.

## 2026-08-10 — Spec 142 Phase 2 conservative traversal

- Added `WorldSceneTraversal` with injected visibility testing, renderable-node selection, subtree
  rejection attribution, skipped-descendant counts, and fail-open handling for unknown bounds.
- Focused graph/workload/traversal proof is 11 passing tests. The active `WorldScene` renderer is
  still unchanged; the next bounded slice is the runtime selector/adapter, not portal math yet.

## 2026-08-10 — Spec 142 renderer performance grounding

- Expanded `specs/142-world-scene-graph/spec.md` with a deterministic synthetic world-scene
  workload, explicit separation from synthetic minimap/2-D preview data, stage-level timing and
  provenance requirements, four-scale replay gates, and real-client parity promotion gates.
- Added the Spec Kit quality checklist at
  `specs/142-world-scene-graph/checklists/requirements.md`; no unresolved clarifications remain.
- The next implementation slice is the versioned fixture/benchmark manifest and instrumentation;
  no heavy capture or GPU run was launched.

## 2026-08-10 — Spec 139 reject MCSH and add observable RGB baseline

- Rejected the raw-MCSH route as a model-input path: minimaps do not carry `shadow_mask`, so it
  cannot answer the real inference question. No raw-MCSH corpus or training should be run.
- Added `real_minimap_rgb` preparation from actual Zarr `minimap_rgb`: deterministic raw luma and
  gradients, explicit absent confidence, `albedo_gate_status=not_run`, source filtering, and
  map-held-out splits. The real store has 1,325 authored rows (688 Kalimdor, 637 Azeroth).
- Verified the new focused tests and both actual dry runs. The next heavy action is user-owned
  authored raw-RGB training/evaluation; this is a learnability baseline, not albedo-normalized
  promotion evidence.

## 2026-08-10 — Spec 139 source-signal correction

- The first full-store command exposed a source mismatch: `curriculum-0_5_3_3368-obj_v1.zarr`
  has `shadow_mask` and `height_257`, but no `terrain_shadow_256`. The builder now fails with an
  actionable message instead of silently relabeling raw MCSH.
- Added explicit `--input-signal shadow_mask` support, labeled as
  `geometry_only_diagnostic_raw_mcsh`; the verified dry run still reports 1,330 rows with 688
  Kalimdor train and 642 Azeroth validation. Deployment-clean evidence still requires a
  post-Spec-133 store containing `terrain_shadow_256`.

## 2026-08-10 — Spec 139 complete v50.1 bridge correction

- Corrected the scope error: the 16-row `real-shadow-npz-v1` directory is an old Alpha/Azeroth
  diagnostic subset, not the full real-data bridge. The v50.1 mixed curriculum Zarr store contains
  1,330 synthetic rows: 688 Kalimdor and 642 Azeroth.
- Added the dry-run-first Zarr bridge builder with original `index.parquet` row-index provenance,
  source-index hashing, read-only source access, and a complete-family map-held-out split.
- Verified the actual dry run and focused contract tests: 2 new tests pass; the plan reports 688
  train rows and 642 validation rows. No corpus materialization or training was launched by Codex.

## 2026-08-10 — Spec 139 real-terrain bridge training result

- The user-run full `pyramid_cnn`/`v7_structural_v1` probe trained on 15 Alpha/Azeroth bridge rows
  with one validation row. Best epoch 4 reached MAE `0.313952` versus `0.109902` baseline
  (`-185.66%`); the epoch-24 snapshot was `0.380639`.
- CPU all-row evaluation of the best checkpoint reached MAE `0.293371` versus `0.157124`
  baseline (`-86.71%`). Coarse error dominates detail error, confirming a real-domain/bridge
  mismatch rather than a late-epoch fluctuation.
- The next action is to preserve source-integrity bands and expand to additional approved
  maps/builds. No repeat training on the same 16 rows; authored RGB remains behind albedo gating.

## 2026-08-10 — Spec 139 real-terrain synthetic bridge diagnostic

- Added the source-preserving `real_terrain_synthetic` bridge builder and dry-run CLI. It converts
  harvested `terrain_shadow_256` plus independent `height_257` NPZ rows into the v60 four-channel
  corpus without mutating source files; the authored RGB route remains separate.
- Added image-only checkpoint evaluation with source-kind filtering, per-row predictions, atlas,
  baseline-relative metrics, and an explicit empty forbidden-read audit. Full v60 verification is
  84 passing tests; changed-file Ruff and py_compile pass.
- Materialized the existing 16-row Alpha/Azeroth bridge and evaluated the reflect-padding
  checkpoint: MAE `0.323879` versus `0.157124` tile-mean baseline (`-106.13%`). This is a real
  domain failure diagnostic, not a promotion result. More map/build rows and the user-owned real
  bridge training probe are next; authored minimap RGB still requires albedo normalization.

## 2026-08-10 — Spec 139 reflect-padding confirmation result

- The user-run `pyramid-full-structural-complete-v2-reflect-padding` checkpoint completed on CUDA
  at best epoch 80: MAE `0.137891` versus `0.191047` baseline, improving the aggregate gate from
  `8.97%` to `27.82%`.
- CPU checkpoint diagnosis confirmed the invented flat-input ramp is gone. Cross-tile lightning is
  still `61.17%` worse than its baseline and cross-tile burn `30.15%` worse, so promotion and real
  transfer remain blocked.
- The next user-owned experiment is full-profile reflect-padding within-family training with all
  81 training rows and 27 held-out variants. It separates family-coverage failure from missing
  four-channel information; no further complete-family rerun is needed now.

## 2026-08-10 — Spec 139 complete-family confirmation result

- The user-run full-profile `pyramid_cnn` + `v7_structural_v1` CUDA gate completed at
  `output/datasets/v60/v7-clean-signal-runs/pyramid-full-structural-complete-v1` with 76 train
  rows, 32 held-out rows, and best epoch 37.
- Best final-height MAE was `0.173904` versus the `0.191047` tile-mean baseline, an `8.97%`
  aggregate improvement. The family report still rejects promotion: `cross_tile_burn` regressed
  `15.52%` and `cross_tile_lightning` regressed `229.79%`; the pathological bucket regressed
  `2.81%`.
- This satisfies the user-run execution evidence but not the generalized-winner acceptance scenario.
  The checkpoint is diagnostic; real transfer remains blocked. Codex launched no training.

## 2026-08-10 — Spec 139 checkpoint diagnosis consumer

- Added `clean_signal_diagnostics.py` and `v60_diagnose_clean_signal_checkpoint.py`. The command
  reconstructs the saved model identity, uses the checkpoint's exact held-out row IDs, performs
  image-only prediction, and writes compact per-row prediction/error NPZs plus full and cross-tile
  PNG atlases and `diagnostic_report.json`.
- Focused proof: diagnostics plus trainer tests pass (4 tests); changed-file Ruff and `py_compile`
  pass; CLI help passes. No training or GPU work was launched.
- Next user action is the CPU diagnostic command in the Spec 139 quickstart. Review the cross-tile
  atlas before changing architecture, data, loss, or transfer scope.

## 2026-08-10 — Spec 139 constant-field stability correction

- Atlas review found `flat-v00` and `cross_tile_lightning-v01` have almost identical four-channel
  inputs and the legacy checkpoint emits the same non-flat ramp for both near-zero targets. This is
  a model padding artifact, not evidence that 2×2 context is immediately required.
- New clean-signal models use versioned `reflect-3x3-v1` padding for spatial 3×3 convolutions.
  Legacy zero-padding checkpoint identities remain reconstructable for before/after comparison.
  Constant-field stability tests pass for `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2`.
- The user-owned next step is one fresh full-profile `pyramid_cnn` structural run under
  `pyramid-full-structural-complete-v2-reflect-padding`, followed by the checkpoint diagnostic.
  No training was launched by Codex.

## 2026-08-10 — Spec 139 clean-signal foundational contracts

- Implemented the v60 four-channel clean observation package: luma, deterministic x/y gradients,
  and measured or explicitly absent confidence, with stale/rejected/quarantined and forbidden
  target-signal gates.
- Implemented versioned per-tile relative-height targets with `box9-edge-replicate-v1` coarse
  relief and signed detail residual, plus corpus NPZ/hash/split/recomposition validation.
- Added the fail-closed `scripts/v60_validate_clean_signal_corpus.py` entrypoint; it writes
  `validation.json` only when requested and returns nonzero for invalid corpus evidence.
- Added CPU contract fixtures for malformed, stale, textured-rejected, missing-confidence,
  forbidden-signal, altitude-invariance, and source-group leakage cases.
- Evidence: 15 focused tests passed; full `tests/v60` passed 55 tests with a fresh writable
  basetemp. Changed-file Ruff and py_compile passed. Full-folder Ruff still reports unrelated
  pre-existing findings in `v60/store.py`. No corpus generation or GPU work was launched.

## 2026-08-10 — Spec 139 clean-signal model contract

- Added `clean_signal_model.py` with one shared four-channel feature adapter and independent
  coarse/detail heads for local `pyramid_cnn`, `segformer_b0`, and `unet_lite_v2` candidates.
- Outputs are finite coarse relief, signed detail residual, and clamped recomposed `height_257`;
  the detail head starts at zero for a coarse-only initial prediction.
- Added JSON identity/config hashing and reconstruction tests; no external or pretrained weights
  are loaded. Evidence: 8 model tests passed, with full v60 verification pending this slice.

## 2026-08-10 — Spec 139 synthetic clean-signal corpus packaging

- Added the dry-run-first `v60_build_clean_signal_corpus.py` path. It validates the existing C#
  control corpus, derives the four-channel synthetic observation and coarse/detail targets, writes
  seven named arrays plus SHA-256 hashes, and publishes only through a fresh output root.
- Added builder tests for no-write planning, valid hashed publication, overwrite refusal, and bad
  confidence. No control corpus was generated by Codex; the user still owns `--confirm-build`,
  visual review, and all later training/GPU work.

## 2026-08-10 — Spec 139 clean-signal visual review

- Added family and variant contact sheets plus stitched complete cross-tile sheets with a persisted
  JSON review report. The builder now preserves pattern and alignment metadata required by the
  cross-tile atlas.
- Added a visual-review fixture proving family, variant, and cross-tile outputs. Evidence: 1 new
  visual-review test passed; full v60 verification is 67 passing tests. No generated atlas has been
visually accepted by the user.

## 2026-08-10 — Spec 139 clean-signal loss contract

- Added `clean_signal_losses.py` with versioned `parity` and `v7_structural_v1` configurations.
  Point, first-derivative, full log-spectrum, Laplacian, Sobel edge, transition-focus, tile-border,
  and low/high-frequency band terms are independently weighted and reported; adversarial and
  object/recovery terms remain excluded from the clean lane.
- Added identity-zero, smoothing penalty, differentiability, and component-isolation tests. Evidence:
  4 focused loss tests passed; full `tests/v60` passes 71 tests, with changed-file Ruff and
  `py_compile` clean. No corpus generation or training was launched.

## 2026-08-10 — Spec 139 clean-signal trainer/evaluator contract

- Added `clean_signal_train.py` with deterministic within-family and complete-family split
  identities, lazy four-channel NPZ loading, independent final/coarse/detail metrics, per-family
  and per-complexity reports, and best/last checkpoints bound to model/loss/split provenance.
- Added report, split, checkpoint, and fresh-output refusal tests. Evidence: 3 focused trainer tests
  passed; full `tests/v60` passes 74 tests, with changed-file Ruff and `py_compile` clean. The tests
  inject a tiny CPU model only for contract proof; no real training run was launched.

## 2026-08-10 — Spec 139 clean-signal training CLI

- Added `v60_train_clean_signal.py`, a PowerShell-ready dry-run matrix planner for the shared
  architecture/loss split. It reports model identities, parameter counts, loss weights, split hash,
  and empty forbidden-signal audit; `--confirm-run` is required and nonempty output roots fail
  closed before any trainer call.
- Added dry-run/no-write and fresh-output refusal tests. Evidence: 2 focused CLI tests passed; full
  `tests/v60` passes 76 tests, with changed-file Ruff and `py_compile` clean. No user corpus or
  training cell was launched.

## 2026-08-10 — Spec 139 within-family matrix result and promotion gate

- User-run CUDA matrix completed six cells (`pyramid_cnn`, `segformer_b0`, `unet_lite_v2` x parity/
  structural) on the within-family split with 32 training rows and 27 validation rows. Best:
  `pyramid_cnn/v7_structural_v1`, validation MAE `0.145868` at epoch 48; parity control was
  `0.150999`, and the tile-mean baseline was `0.181995`.
- Structural guidance improved every architecture but missed the 10% same-architecture lift gate
  (3.40% pyramid, 4.23% SegFormer, 7.72% U-Net). This is not a promotion failure: it is a reason to
  run the prepared full-profile `pyramid_cnn` complete-family confirmation before transfer.
- Codex verified the dry-run plan only: full profile, 1,579,586 parameters, all 76 train rows, 32
  held-out rows, fresh output root. No additional GPU run was launched.

## 2026-08-10 — reset v60 to terrain-only learning

- Parked object-sieve and object-marker work after the user-run marker experiment failed identity
  retrieval; its checkpoint is diagnostic only.
- Added `v60/control_experiment.py` and `scripts/v60_run_experiment.py` for the validated
  `control-v1` NPZ corpus: fixed family holdouts, deterministic 8/16/32-row learning-curve plans,
  tile-mean baseline, and per-family/per-variant report metrics.
- Offline proof: focused control-experiment tests passed; Ruff and py_compile passed. No training
  was launched.

## 2026-08-09 — v60 footprint-guided object marker pivot

- Amended Spec 134 so object identity is a separate specialist from the optional terrain sieve.
- Added `v60-object-marker-v1`: deterministic candidate corpus derivation from the corrected
  real-library sieve, a small image-plus-footprint knownness/embedding model, frozen-gallery
  retrieval, `known_object_marker_256` export, and an identity sidecar table.
- Added PowerShell-ready build/validate/train/mark CLIs. Training and corpus generation remain
  user-run; no marker corpus or GPU work was launched.
- Focused offline proof: `tests/v60/test_object_marker.py` passed 6 tests; Ruff and py_compile
  passed for the new marker modules/scripts.
- Corrected the marker builder's overlap handling: fully occluded/overwritten source instances are
  recorded in `skipped_instances` instead of aborting the corpus, and builds publish atomically
  from a `.partial` directory. The focused v60 suite now passes 31 tests after this regression fix.

## 2026-08-09 — v60 object lane corrected to use the v50 object library

- Rejected the `real-object-masks-v1` result for the precision-object lane: it trained on v50
  curriculum tile-level placement projections, which are dot-like labels, and never read the actual
  object library.
- Added `v60-object-library-sieve-v1`: a read-only compositor over the 5,349-entry
  `object_mask_library_0_5_3_3368.zarr`, using real `capture_rgb`/`capture_mask` silhouettes over
  clean v60 terrain controls. Rows carry exact union masks, per-instance IDs, library identity,
  deterministic transforms, and family-isolated splits.
- Added PowerShell-ready builder/validator/visual-review/trainer CLIs and a deterministic focused
  test. Offline proof: v60 suite 20 passed; Ruff and py_compile passed. No real corpus generation or
  CUDA training was run by Codex.
- The user's first two corpus builds stopped after 251 NPZs before writing their manifests when the
  thin `world/nodxt/detail/elwgra06.mdx` silhouette was erased by nearest-neighbour downsampling;
  both partial directories are invalid and were left untouched. Replaced mask resizing with
  coverage-preserving BOX/BILINEAR rasterization and kept contextual failure text; the next user
  run should use a fresh output directory.

## 2026-08-09 — v60 real-library sieve corpus passed

- User-run `object-library-sieve-v3` completed and passed validation: 540 rows from 108 terrain
  controls, five complete placement regimes, 304 train / 236 validation rows, 1,033 sampled library
  objects, and 115 isolated library families. Visual atlas was generated successfully.
- The next gate is the user-run three-variant CUDA object-sieve ablation. No training has started.

## 2026-08-09 — first object-sieve training readout and residual correction

- User-run `library-guided-v1` reached non-empty mask IoU 0.4183 at epoch 35, with final precision
  0.6622 and recall 0.5038. This is useful mask evidence.
- Its clean output failed the identity test: best clean MAE 0.0372 versus 0.0066 for simply passing
  the contaminated input through. The model was rewriting clean terrain outside the small object
  regions, so the run is not a functional sieve despite the mask head learning.
- Changed `ObjectSieveNet` to a zero-initialized residual clean head (`input + learned correction`),
  added the identity-baseline gate to the report, and added focused tests. User must rerun with a
  fresh output directory; no residual model training has run yet.

## 2026-08-09 — v50 stale synthesis correction

- The active failure was stale synthesized minimap data in the old v50 datastore; the 0.5.3
  renderer remains the control.
- The old builder did not provide a synthesis-only refresh, so old RGB arrays survived later
  compositor lighting fixes.
- Added `scripts/refresh_v50_synthetic_minimaps.py` to regenerate only `minimap_rgb` and
  `minimap_rgb_1024` from existing tile indices, with fatal written/non-black validation and a new
  output store.
- Fixed its Windows/Zarr patch step to recreate the two refreshed arrays with one tile per chunk;
  the historical multi-row chunks caused `PermissionError` during repeated row replacement.
- Existing `0_5_3_3368-Azeroth.zarr` has 43 all-zero rows in both synthetic resolutions, proving
  the old synthesis needed regeneration.
- No refresh, harvest, training, or heavy run was started by Codex.

## 2026-08-09 — Spec 138 0.5.3 dataset audit from WoWClient.exe

- Queried the loaded 0.5.3.3368 binary in Ghidra and confirmed that authored minimap BLP tiles,
  terrain MCSH/LIT rendering, and object/icon overlays are separate native paths.
- Found blocking harvest defects: MCAL `MCLY.offsAlpha` is ignored; zero-valued absolute MCVT
  heights can be overwritten by gap filling; raw MCSH is overclaimed as a 256 terrain-shadow
  target; and Alpha shadow composition forces synthetic cast shadows without native LIT data.
- Found that Alpha MDDF/MODF masks are heuristic placement labels and omit MCRF/visibility
  semantics. The shared Alpha adapter also transposes WDT `MAIN` indexing, although the direct
  harvester reader is row-major and matches the client.
- Confirmed current MCVT absolute heights, MCNR transform, and minimap BLP path. No accepted
  0.5.3 real corpus, harvest, training run, or heavy build resulted from this audit.

## 2026-08-08 — Spec 138 archive-source research and plan

- Researched CascLib, WoW-Tools/CascLib, TACTSharp, TACT.Net, WoWTools.Minimaps, wow.export,
  wow-listfile, and pywowlib through their GitHub repositories and local reference checkout.
- Recorded the source matrix: existing MPQ for 0.x–5.x, CascLib as the early-CASC baseline,
  TACTSharp as a later-CASC candidate, and wow.export/WoWTools.Minimaps as comparative authorities.
- Corrected the plan to treat DBCD, WoWDBDefs, and wow-listfile as existing integrated authorities,
  not future work. TACT.Net remains isolated pending its GPL-3.0 boundary review.
- Added Spec 138 `research.md`, `plan.md`, `data-model.md`, `quickstart.md`, and the source-profile
  JSON schema. No archive adapter or heavy extraction was run.

## 2026-08-08 — Spec 138 scope widened to the cross-era terrain foundation

- Reframed the epic from a primarily 4.x renderer roadmap into a profile-gated basic-terrain
  foundation intended to span 0.5.3 through 11.x with minimal follow-up per era.
- 4.0.0.11792 remains the first modern evidence anchor because it exposes the vertex, lighting,
  shadow, and shader-permutation boundaries. 0.5.3 parity is downstream, not the first branch.
- Later client tools such as wow.export may supply comparative evidence, but remain outside the
  repo-independent runtime contract.

## 2026-08-08 — Spec 138 Build 11792 Ghidra evidence pass

- Queried the live `WOW-11792patch4.0.0_Alpha-INTERNAL.exe` Ghidra project and traced
  MCNK parsing, terrain vertex construction, shader permutation setup, `mapShadows`,
  MCSH composition, and the `TerrainBlend` shadow render-target path.
- Confirmed that this build uses separate `MCLV`, `MCCV`, and `MCSH` terrain signals;
  `MCTV`/`MCMT` were not observed. The CPU vertex builder preserves MCNR byte order,
  leaving the normal-axis transform unproven until the Terrain shader path is traced.
- No renderer code, harvest, training, or long-running work was run. The next proof owner
  is the Terrain shader input/constant path.

## 2026-08-08 — Spec 138 renderer baseline clarified

- Recorded that existing 4.0.0 support is partial but usable: terrain/world content and basic
  WMO/M2 paths work, while shaders, visual effects, lava-effect models, fog, lighting/point lights,
  batching, and CPU-bound submission are the main gaps.
- Spec 138 now requires separate visual-parity and frame-time proof; basic loading is not signoff.

## 2026-08-08 — Spec 138 Cataclysm 4.x renderer evolution note

- Added `specs/138-cataclysm-renderer-evolution/spec.md` and its requirements checklist.
- Captured the 4.0.0.11792 19-module audit as reference input for a future evidence-led epic:
  terrain layer/chunk evolution, 4.x M2/WMO paths, dense-scene performance, and provenance-
  preserving synthesis across client eras.
- First gate is source/profile inventory. No renderer rewrite, harvest, training, or long-running
  benchmark was run.

## 2026-08-08 — Spec 134 paired real/synthetic validation and mask inputs

- Added the read-only v50.1 authored/legacy-flat pair selector and absolute-difference report. It
  requires matching `source_group_id` plus map/tile identity, preserves the split, and records
  incomplete groups. The legacy synthetic image is explicitly not a terrain-shadow target.
- Added `v60_validate_real_synthetic_pairs.py`, which writes a small validation JSON report and
  visual atlas. The first 16-tile Azeroth slice: 1,325 complete groups out of 1,330, mean RGB MAE
  0.1812, mean RMSE 0.2120, and 69.4% of pixels differing by >0.10 normalized RGB.
- Removed the invalid synthetic-input guidance route from the real object-mask trainer. A fresh
  post-fix C# NPZ containing `terrain_shadow_256` is now required for shadow comparison. Real masks
  are labels only; no GPU training has run.
- Evidence: focused v60 suite 18 passed, ruff clean, JSON contracts valid, pair plan-only checks pass.

## 2026-08-08 — Spec 134 control-corpus and object-sieve implementation

Branch `134-v60-unified-dataset-model`.

- Reframed the initial v60 experiment around a small synthetic control corpus; the abandoned
  multi-client v50 harvest is not evidence of a working v60 dataset.
- Added deterministic fBm/ridged-fractal, dendritic lightning-burn, global 2×2 cross-tile
  lightning/burn, mountainous, arbitrary-angle sheer-dropoff, and zone-style-blend families to the
  C# control generator. The default taxonomy is 27 families × 4 variants = 108 rows across four
  complexity buckets. Non-grid fields carry deterministic sub-cell offsets; `chunk_grid` is the
  explicit aligned diagnostic.
- Added fail-closed cross-tile metadata validation, shared-pattern-ID checks, duplicate-position
  checks, alignment metadata validation, stitched height/shadow visual atlas output, and visual
  coverage contract fields.
- Added a sibling `object-sieve-v1` writer with 540 deterministic rows: four object families across
  none/sparse/dense/overlap/boundary-crossing regimes, plus clean terrain and contamination-mask
  targets. Added object manifest/hash validation, input/mask atlases, and clean-only,
  auxiliary-mask-loss, and predicted-mask-guided model variants.
- Evidence: harvest tool build 0 errors; focused Python checks remain the final local gate. No corpus
  generation, client harvest, or training was run by Codex.

## 2026-08-08 — Spec 134 object-sieve design extension

- Added the next bounded signal lane to the Spec 134 plan: synthetic objects over canonical terrain,
  exact clean terrain-shadow targets, and a separate screen-space contamination mask.
- Defined three ablations: clean-output-only, auxiliary mask loss, and predicted-mask guidance. The
  ground-truth mask remains loss-side supervision only; it is never an inference input.
- Added object-control data-model and experiment contracts, placement regimes (none/sparse/dense/
  overlap/boundary-crossing), task decomposition, and continuity notes. Implementation waits for the
  user-run terrain control corpus gate.

## 2026-08-08 — PM4/PD4 versioning & Specs 135, 136, 137 landed

- **PM4/PD4 Versioning**: Created `Pm4VersionFormatter.cs` for correct MVER version string parsing (`0x10` Cataclysm = v16, `0x30` WoD = v48). Wired to viewer status bar and CLI inspection output.
- **Spec 135 (Phased Terrain Dual-Map Overlay)**: `ITerrainAdapter`, `StandardTerrainAdapter`, `TerrainManager`, `WorldScene` support `SecondaryOverlayMap` / `OverlayMapName`. Real-time ADT tile replacement from overlay directories with live tile eviction & streaming. Searchable map dropdown picker added to UI.
- **Spec 136 (M2 Doodad Performance Optimization)**: Enabled batched instancing (`BeginBatch` + `RenderInstance`) for M2 adapter models without particles/ribbons by updating `ModelRenderer.RequiresUnbatchedWorldRender`. Deduplicated `UpdateAnimation` calls in `WorldScene.cs` so shared models advance at most once per frame. Restored smooth framerates on dense doodad maps (>60 FPS).
- **Spec 137 (Phased Minimap Overlay & Consistent Teleport)**: Updated `MinimapRenderer` & `MinimapHelpers` to render active secondary overlay tile BLPs on the minimap surface. Unified fullscreen minimap to use 3-click armed teleport (`MinimapTeleportMode.Armed`) matching small minimap panel.

## 2026-08-07 — Spec 134 route reset (not a working v60 dataset)

Branch `134-v60-unified-dataset-model`.

- The earlier datastore/harvest consolidation was not accepted as a working v60 dataset and is no
  longer the first experiment.
- Spec 134 now gates on the synthetic control corpus, followed only later by a tiny explicit 0.x/1.x
  albedo-normalized transfer sample. Later client builds remain out of the initial scope.

## 2026-08-05 — Spec 132 Phase 1: three-tier brush-signature classification

Branch `132-terrain-brush-signature-classification`, commit `f19fc774`.

- **`classify.py`** — new library module with `compute_signal_tier()`: strong/normal/weak/na with
  published criteria (weak < 5 range, normal 5-50 range or 8-64 levels or low correlation).
- **`v50_tile_classify.py`** — CLI: reads V50 Zarr store or NPZ dir -> per-tier CSV/JSON + summary.
- **`tile_inventory.py`** — gains `signal_class` / `signal_class_evidence` per row + `by_signal_class`
  summary. `tile_composite.py` gains green normal-tier outline. Both archaeology orchestrators run
  the classifier.
- **13 new tests pass**; 22 existing inventory/composite tests still pass (no regressions).
- **tasks.md** covers all 6 phases; Phase 2 (nested weak signal detection) is next.

## 2026-08-04 — Archaeology pipeline built, spec 132 drafted

- **Single-command archaeology pipeline** (`run-archaeology.ps1`): harvest MPQ → V50 Zarr store → tile inventory → synthesis → composites. Proven on TBC 2.0.0.5610 (Expansion01, 741 tiles).
- **Batch archaeology** (`run-batch-archaeology.ps1`): discovers all 15 1.x Windows clients in H:\CLIENTS, runs pipeline on each.
- **V50 store builder from NPZ** (`build_v50_store_from_npz.py`): reads NPZ shards, builds proper V50-format Zarr store with index.parquet, then runs full archaeology suite.
- **Spec 132** (`terrain-brush-signature-classification`): 6 user stories for three-tier classification, nested signals, brush-texture correlation, cross-map alignment, rescale boundary detection, and predictive model. Plan written.
- **Harvested data on disk**: `output/archaeology/2_0_0_5610/` with NPZ, Zarr store, and archaeology results for Expansion01.

## 2026-08-04 — PM4 scene graph: tree view restored, MSLK linking summary added

Spec 131, branch `131-pm4-scene-graph-doodads`. Detail:
[workstream-pm4-decode.md](workstream-pm4-decode.md).

- **PM4 Scene Graph panel restored as a full scene outliner** (Blender-style). Two modes:
  "Full Scene" (tile → CK24 → Part hierarchy with MSLK group/MPRL refs at each level, search
  filter, right-click select/frame-all context menu) and "Selected Object" (existing detailed
  decomposition, now with MSLK linking info). Clicking any row selects the object and frames the
  camera.
- **MSLK Linking Summary** section added to the outliner: anchor-only vs path-window link counts,
  component link coverage, RefIndex mismatch counts, and research leads — computed live from all
  loaded PM4 research contexts via `WorldScene.GetPm4MslkLinkingStats`.
- New public API on `WorldScene`: `GetPm4TileObjectSummaries()`, `SelectPm4ObjectByKey()`,
  `GetPm4MslkLinkingStats()` and the `Pm4MslkLinkingStats` struct. Build passes.

## 2026-08-04 — PM4 scene graph: CK24 decoded, doodad identity found

Spec 131, branch `131-pm4-scene-graph-doodads`. Detail:
[workstream-pm4-decode.md](workstream-pm4-decode.md).

- **PM4 placement confirmed correct in the viewer.** Tiles aligned, tents identified, previously
  rotated walls and buildings correct. Committed by the user as "finally right".
- **`pm4 doodad-split`** — a keyed (non-zero) CK24 is one placed WMO. 47 WMO-free tiles carry zero
  keyed objects (47 chances to falsify, none taken); keyed count matches WMO count exactly on
  136/179 tiles, within ±1 on 163. CK24 0 is the per-tile remainder — exactly one per tile.
- **`pm4 component-identity`** — the CK24 0 remainder splits into per-doodad components by mesh
  connectivity: 19,124 of 20,113 (95.1%) land on an MDDF placement, closest at distance 0.00.
  **`MSLK.GroupObjectId` is the per-doodad identity** — 3,343 of 3,345 pure components unique on
  their tile. `MSUR.GroupKey` scores 100% purity but 0% distinctness and would have been a false
  positive without the distinctness column.
- **New ground truth**: Blizzard WoW Editor 1.9.0 screenshots of this data, with Karazhan Crypts
  loaded for comparison.
- Memory bank restructured into per-workstream files (this change).

## 2026-08-03 — PM4 placement solved

Spec 130. Detail: [workstream-pm4-decode.md](workstream-pm4-decode.md).

- **The coordinate transform**, verified against ADT ground truth at 92.4% vs 0.7% for the
  alternative. MSVT is ADT placement space; the map tile index is `31 - band`.
- **Region-scoped frames refuted** via `pm4 bounds-audit --by-region` (1,877 of 1,895 objects on one
  frame).
- **The MPRL-scored placement fitter disproven and removed from the render path.**
  `pm4 yaw-evidence`: containment 93.3% canonical vs 88.2% with the fitted yaw vs 79.0% for a
  known-wrong 45° control; hurt 96, helped 3. `Pm4PlacementMath` left untouched — all 16
  `PlacementMath_*` tests still pass.
- **Both PM4 disk caches bumped to version 9.** They store post-transform geometry, so a stale cache
  replayed the old placement and made a correct fix look broken.
- `Pm4CoordinateService.TryGetObj0PathForPm4` replaces a padded-name lookup that matched none of the
  616 corpus files.
- Earlier the same day: MSPV/MSPI decoded as a vertical planar quad mesh (the walls), walls rendered
  in the viewer, `pm4 mprr` structural hypothesis eliminated.

## 2026-08-03 — terrain ML: stacked-height trainer unblocked

Specs 114/125/126. Detail: [workstream-terrain-ml.md](workstream-terrain-ml.md).

- Residual→height feed-forward **proven dead** (r = 0.0024; three approaches agree).
- Forward-model-as-referee built; it establishes the information limit of single-view shading.
- Stacked height model (4-channel `direct_cnn_v112`) implemented; the channel-count crash is fixed
  via one shared `build_model_input_channels`. **Not yet trained** — user-run gate.
- Full data-harvester suite: ~1150 passed / ~45 skipped / 3 pre-existing unrelated failures.

## 2026-08-10 — v60 terrain architecture bakeoff prepared

Spec 134. The old U-Net-only v60 control run was rejected against its tile-mean baseline
(`0.228693` best MAE vs `0.191047`). Implemented a shared random-init bakeoff registry and trainer
for `unet_lite_v2`, ConvNeXtV2/FPN `pyramid_cnn`, local HF DPT `dpt_small`, and from-scratch HF
SegFormer-B0. The trainer uses one seeded nested training schedule and writes per-architecture,
per-family metrics. Tiny CPU contract tests pass (`9 passed`); the real-corpus dry run reports the
four contracts and parameter counts without training. User must run the CUDA command in Spec 134
quickstart and record whether any candidate beats the baseline.

## 2026-08-10 — Spec 139 v7 clean-signal pivot planned

Created branch `139-v7-clean-signal-reconstruction` and the complete Speckit design package under
`specs/139-v7-clean-signal-reconstruction/`. The plan preserves v7's coarse/detail outputs and
structural loss guidance but removes its WDL trestle, height hints, normals, liquid/object channels,
and all target-derived inference inputs. The new deployment contract is an albedo-normalized
observation plus deterministic luma gradients and albedo confidence. No implementation or training
has started; next is the small contract/model/corpus slice, followed by a user-owned loss/architecture
matrix and only then tiny 0.x/1.x transfer.

## 2026-08-10 — Spec 140 paste/fractal/tileset evidence pipeline planned

Created branch `140-terrain-paste-motif-archaeology` and the Speckit design package under
`specs/140-terrain-paste-motif-archaeology/`. The new lane decomposes minimap-to-terrain work into
observation normalization, tileset profiling, alpha/fractal descriptors, cross-tile paste
retrieval, Spec 139 terrain guidance, and a deferred object-slot/refinement lane. The 10.2 workflow
map is recorded as corroborating workflow evidence only; recurrence must be proven against actual
client-backed data and deterministic synthetic controls. No implementation, harvest, or training
started. First gate: visual atlas plus transformed-motif retrieval and leakage report.

## 2026-08-10 — Spec 140 paint-order hypothesis added

Updated Spec 140 to test a stronger authoring model: opaque layer-0 base/“brain” texture, layer-1
recurring rocky paste, later alpha-painted terrain intent, then sculpted relief and refinement.
Ordered alpha evidence is now an intermediate paint/sculpt scaffold, with MCLY order, MCAL offsets,
layer-0/layer-1 distinction, cumulative/incremental occupancy, paste references, and explicit
intact/retextured/resculpted/unknown status. Source-side alpha remains supervision/evidence; the
minimap-only deployment contract must predict the scaffold rather than consume client alpha.
No implementation or training started; the next gate is synthetic known-order validation.

## 2026-08-10 — Spec 140 complementary brush scales recorded

Refined Spec 140 to retain both historical approaches: the early Python connected-alpha extractor
is the atomic brush scale, while the later C# full-map implementation supplies middle-scale
paste-block children and broad macro-prefab context. The C# implementation is a different spatial
ontology, not a bug. Added parent/child hierarchy records, per-scale metrics, and fail-closed
review rules. Also specified frozen-reference validation scoring as curriculum difficulty guidance
only (`easy`, `learnable_hard`, `pathological`), never as staleness, pseudo-target, or provenance.
No implementation, harvest, or training started; next gate remains synthetic known-order and
hierarchy validation.

## 2026-08-10 — Spec 140 alpha-first evidence boundary recorded

Promoted alpha preservation to the fan-out boundary. Spec 140 now requires lossless source-layer
references and provenance before deriving raw occupancy, transition/stroke, atomic, paste-block,
macro-context, ordered-layer, or cross-tile views. The interpretations remain independently
available so a weak view cannot erase useful evidence elsewhere; unavailable/opaque alpha cannot be
replaced with an empty mask. No implementation, harvest, or training started.

## Before 2026-08-01

Condensed into [archive/2026-08-01-progress-detail.md](archive/2026-08-01-progress-detail.md) at the
feature-complete declaration and spec audit. Older session history is in
[archive/](archive/README.md).

## 2026-08-10 — v60 dataset catalog and viewer selector

Spec 134 was extended with the viewer-consumption boundary. Verified on-disk v50.1 stores include
liquid masks/heights, variable liquid-type coverage, object placement/mask evidence, MCLY
layer/tileset IDs, and texture/path inventories. The v60 output tree currently has no unified map
Zarr; control NPZs and model runs are not renderable datasets.

Added `WowViewer.Core.Maps.DatasetVersionCatalog`, focused catalog tests, current liquid aliases and
object/tileset/texture/placement summary fields to `ZarrTileDatasetLoader`, and a Settings dataset
version selector with persistent catalog/selection paths. VLM projects can be activated in-session
and preserve camera state. Zarr stores remain summary-only until compressed chunk decoding and
`TerrainTileTensorPack` rehydration are proven. Viewer build passed; the two new catalog tests
passed. Existing repository warnings remain, including the known Snappier advisory.

## 2026-08-10 — real tile observations added

The viewer/data contract now has an explicit `RealTileObservation` class for client tiles, authored
minimaps, and low-resolution media/reference imagery. Catalog discovery recognizes explicitly named
real-observation folders and marks them reference-only. RGB-only observations are eligible as model
inputs but never targets or terrain renderers; missing liquid/alpha/tileset/object/height signals
remain unknown. Source-preserving manifest/materialization and image inspection are the next tasks.

## 2026-08-10 — Spec 141 terrain-method translation Phase 0/1 complete

Created branch `141-terrain-method-translation` and its Speckit design package. The plan records
DSM2DTM and ResDepth as height-prior architecture references, SMRF/CSF as point-cloud diagnostics,
and aerial segmentation models as predicted-mask auxiliaries. It explicitly separates those lanes
from the current RGB-only minimap contract and adds a provenance-bound research-lead workflow for
new signal discoveries. Implemented `terrain_method_translation.py` and
`v60_audit_terrain_methods.py` with six method records, four input branches, canonical aliases, and
forbidden-read audits. Focused proof is `17 passed`; the full v60 regression is `106 passed`; Ruff,
`py_compile`, and the dry-run CLI pass. The manifest-only RGB planner now handles authored and
object-library sources, preserves separate split hashes, and fails closed on target reads. The real
`object-library-sieve-v3` plan is valid at 540 rows (304 train / 236 validation) but has zero runtime-
eligible RGB rows because its input is `objectified_terrain_shadow_256`. The planner tests add `5`
passing cases; full v60 is now `111` passing. No external-weight use, harvest, or training occurred.
Next: build the authored RGB corpus and review the combined plan.
