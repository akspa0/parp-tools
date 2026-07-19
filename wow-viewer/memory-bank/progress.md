# Progress — wow-viewer

Last updated: 2026-07-19

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
- Next (user-owned): T010 curriculum build dry run + `--write`, then the `mit_b0-authored-v1`
  training command (quickstart). Dual-view `--source all` runs stay gated on the Spec 113
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
