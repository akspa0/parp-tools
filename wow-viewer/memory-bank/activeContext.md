# Active Context — wow-viewer

Last updated: 2026-07-20

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

## Known open bug (unrelated lane, reported 2026-07-18, not yet fixed)

- Built-in map GLB export: textures are mirrored along the Y axis. User confirmed via testing GLB
  output on a phone; otherwise export is correct. Not investigated or fixed this session — flagged
  for follow-up.

## Durable boundaries

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- User runs client-backed visual proof, training, capture, and heavy work. Report client root,
  build identity, and fingerprint with any real-data conclusion.
- `AlphaWdtWriter.cs` is frozen. Renderer reader ownership is native M2; export conversion is separate.
