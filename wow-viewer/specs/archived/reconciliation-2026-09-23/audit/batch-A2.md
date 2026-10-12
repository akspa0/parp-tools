# Batch A2 audit — ML terrain reconstruction / minimap decomposition / terrain archaeology

Specs: 125, 126, 127, 132, 133, 134, 139, 140, 141, 194, 196. None of these specs carry an
`evidence/` directory (all predate or sit outside the Spec 224 §9.2 receipts regime except where
noted). Findings below are from direct code/grep verification against spec.md/tasks.md claims.

---

### 125 Minimap DXT1 Artifact Inversion
- Stated status: Draft | Tasks: 0/42 checked (all `[ ]`), but code exists for most P1/P2 stories.
- Scope: Detect/reproduce authored-tile DXT1 degradation for fair comparison (US1/US2), restore
  toward pre-compression appearance (US3), decode terrain shadow -> heightmap -> mesh (US4),
  super-resolution (US5), export textureless residuals (US6), extract residual from arbitrary RGB
  and strip shading (US7).
- Verified implemented:
  - US1/US2 (parity, encoding survey, lighting baseline) -> `src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs`
    (`EncodeDecode`, `DecodeAuthored`, `RoundTripAgreement`), `MinimapEncodingSurvey.cs`,
    `MinimapLightingBaseline.cs`; CLI flags `--encoding-survey`, `--lighting-baseline`,
    `--no-dxt1` in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`. Tests:
    `tests/WowViewer.Core.Tests/Dxt1TileCodecTests.cs`, `Dxt1PythonParityTests.cs`,
    `MinimapLightingBaselineTests.cs`. Notably the mechanism evolved past the spec: DXT1 is now the
    **primary** synthesized output by default (`--no-dxt1` opts out; `--dxt1-parity` is a retained
    no-op alias) — a superset of FR-015, not a gap.
  - FR-004 (degenerate-tile exclusion) -> `Program.cs:4035-4057` (`degenerate = uniqueColors <= 2`).
  - US6 (textureless residual export, FR-022/023) -> `--textureless-residuals` flag, stitched output,
    Zarr signal write, all in `Program.cs` (explicitly commented "Spec 133" — this story was actually
    delivered under spec 133, see below).
  - US4 (shadow decode -> height, FR-017/018/019) -> `data-harvester/src/harvester/v50/residual_height_model.py`
    (docstring literally cites "Spec 125 US4"), `data-harvester/scripts/v50_measure_residual_shading_law.py`
    (the E1 shading-law experiment from spec 126's research.md).
  - US7 (residual extractor, FR-024/025) -> `data-harvester/src/harvester/v50/residual_extractor_model.py`,
    `residual_extractor_train.py` (542 lines), `residual_extractor_infer.py` (includes
    `_stripped_albedo_rgb` = FR-025 subtraction), `scripts/v50_train_residual_extractor.py`,
    `scripts/v50_build_residual_extractor_curriculum.py`.
- Partial: US7 confidence reporting (FR-026) — no `confidence` field found in
  `residual_extractor_infer.py`/`residual_extractor_model.py`; per-tile confidence flagging not
  verified present.
- Not implemented: US5 super-resolution (FR-021) — no `super_resolve`/`SuperResolution` module found
  anywhere in `data-harvester/` outside the unrelated archived spec 113. `dxt1_restore.py` (US3,
  T021-T026) not found under that name; only a `v50/dxt1_approx.py` + test exists, which is a
  different, smaller artifact than the spec's residual-restoration-network design — restoration
  proper (FR-007..FR-012, hallucination gate) is unverified.
- Checkbox accuracy: N/A — all boxes unchecked, but a substantial majority of US1/US2/US4/US6/US7
  work exists in code (checkboxes understate progress rather than overstate it).
- Operator gates owed: SC-004 (25% colour-error reduction), SC-009 (95% block-agreement round-trip),
  SC-011 (lighting baseline improves agreement), SC-013 (shadow decode consistency), SC-015
  (super-res) — all require real training/eval runs not evidenced here.
- Open residue (spec-stated only): FR-021 (super-resolution, entirely unbuilt), FR-010/FR-012
  (restoration hallucination gate + resolution separation, unverified), FR-026 (residual-extractor
  confidence reporting).
- Superseded by / overlaps: 126 (US4 height-from-shadow), 133 (US6 textureless residual — code
  comments literally attribute it to Spec 133), 139/141 (v60 clean-signal lane absorbed the
  shadow->height modeling direction).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: minimap-decomposition-and-codec-parity
- Confidence: medium (broad grep coverage; did not execute tests or training runs)

---

### 126 Minimap-to-Terrain Reconstruction Stack
- Stated status: Draft | Tasks: no tasks.md (plan.md only, heavily gated Phase-0-research design).
- Scope: Full 8-user-story stack — albedo/shading decomposition, single-tile relief recovery,
  occlusion-aware loss masking, heightmap+mesh export via WDL composition, multi-tile/seam
  continuity, graded texture-tier decode, iterative refinement, per-layer paint-by-numbers.
- Verified implemented:
  - Shading-law experiment (E1) -> `data-harvester/scripts/v50_measure_residual_shading_law.py`.
  - Height-from-residual model (US2 core claim) -> `residual_height_model.py` (target contract
    `TARGET_CONTRACT_VERSION = "v125.1"`, altitude-offset-invariant per-tile normalization matching
    FR-007's relative-height contract).
  - Occlusion masking (US3, FR-012/013) -> `ObjectInstanceMask257` referenced across
    `AdtTensorPackBuilder.cs`, `TerrainTileTensorPack.cs`, `RawArraySerializer.cs`,
    `NpzTileSerializer.cs`, `AlphaTensorPackBuilder.cs` (matches the project's established
    occlusion-aware-mask work per memory notes).
  - WDL composition (US4, FR-008) -> `data-harvester/src/harvester/v50/wdl_prior_evaluate.py`,
    `wdl_prior_visualize.py`; refinement (US7, FR-026) -> `terrain_refiner_infer.py`.
  - The core `terrain_shadow_256 -> height_257` lane this spec designed was carried forward and
    actually executed (with real CUDA receipts) under spec 139 (see below) rather than under 126's
    own plan.
- Not implemented: US6/US8 texture-tier decode (FR-019..FR-022) — no `tier_0..tier_4` /
  `TextureTier` code found anywhere in `data-harvester`. Multi-tile seam continuity (US5, FR-010) not
  verified as a distinct module. The <=200M-parameter multi-head model with per-signal ablation
  (FR-023/024/025, the constitution-v2.0.0 gate this plan was written to satisfy) not verified as a
  single unified artifact — the actual implementation split into the narrower single-signal v60
  lane instead.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: essentially all SC-001..SC-014 (this plan's Phase 0 experiments and further
  phases were absorbed into spec 139's differently-scoped plan rather than run as specified here).
- Open residue (spec-stated only): US6/US8 texture tier decode (FR-019-022, entirely unbuilt); US5
  multi-tile/seam stitching (FR-010) unverified.
- Superseded by / overlaps: 125 (US4 shading decode literally lives in a "Spec 125" file), 139 (the
  actual funded height-reconstruction lane with real training receipts), 140/141 (texture/motif and
  method-translation split out as separate concerns).
- Disposition: ARCHIVE-SUPERSEDED
- Proposed epic theme: minimap-decomposition-and-codec-parity
- Confidence: medium — this spec's research.md is a design document; the actual build happened
  under 139's differently-named contract, so "implemented" here means "the idea shipped elsewhere."

---

### 127 Weak-Signal & White-Plate Tile Explorer
- Stated status: Draft | Tasks: no tasks.md.
- Scope: Viewer-only feature — inspect degenerate tiles (self-normalized relief, raking-light shade,
  normals view, true min/max/range), neighbour-derived auto-amplification (not just era constant),
  and a listing/browser of all 361 weak/white-plate tiles with camera-jump.
- Verified implemented:
  - `WeakSignalDetector.EstimateFactorFromRanges` exists and is called internally by
    `EstimateAmplificationFactor` (`src/core/WowViewer.Core.Runtime/World/Terrain/WeakSignalDetector.cs:44-72`)
    — the spec's premise that it has "zero callers" is now stale at the detector-internals level.
  - A "Weak Signal Amplifier" ImGui window exists (`ViewerApp_Sidebars.cs:3854` `DrawWeakSignalWindow`,
    slider, auto/manual toggle, WDL-guide fallback via `TryGetTerrainWeakSignalWdlTile`) — this is the
    pre-existing spec-062 amplifier UI, not a spec-127 build.
- Partial: The amplifier UI's "auto" path only consults the coarse WDL guide tile
  (`TryGetTerrainWeakSignalWdlTile`) — `EstimateAmplificationFactor` (the neighbour-range-derived
  path FR-008 requires) has **no caller anywhere in the viewer**, confirming the spec's own diagnosis
  ("the auto-factor path consults only a coarse WDL guide tile... the smarter behaviour is written
  and unwired") is still true today for the neighbour-derived factor specifically.
- Not implemented: FR-002/003/004/005/006 (self-normalized relief view, raking-light shading view,
  scaled normals view, precision min/max/range display, flat-tile labeling) — no
  `raking`/`SelfNormal`/`WhitePlate` hits anywhere in `ViewerApp_Sidebars.cs` or
  `ViewerApp_TerrainInspection.cs`. FR-010 (weak/white-plate tile listing with camera-jump) — no
  listing/browser UI found. FR-008 (neighbour-derived auto factor) — unwired per above.
- Checkbox accuracy: N/A (no tasks.md).
- Operator gates owed: SC-001..SC-006 all require the UI above to exist first; none are close.
- Open residue (spec-stated only): essentially the entire spec — US1 (inspection views) and US3
  (listing) are unbuilt; US2 (neighbour auto-amplify) has the underlying math but no UI wiring
  (FR-008/FR-009 partial).
- Superseded by / overlaps: 132 (three-tier signal classification supersedes some of the
  "weak/normal/strong" framing), 194/196 (the stratigraphy workbench is a more sophisticated
  successor UI for inspecting/amplifying weak-signal terrain and likely obsoletes this spec's
  narrower ask).
- Disposition: FOLD
- Proposed epic theme: weak-signal-terrain-archaeology
- Confidence: high

---

### 132 Terrain Brush Signature Classification
- Stated status: Draft, tasks table marks Phase 1 "In progress", Phases 2-6 "Pending" | Tasks:
  matches code exactly — spot-checked T001-T003/T007 present, T009+ (all later phases) absent.
- Scope: Three-tier (strong/normal/weak) classification (US1), nested weak-signal tiers (US2),
  brush-scar height<->alpha correlation (US3), cross-map fragment alignment (US4), Nov-2001 rescale
  boundary detection (US5), predictive texture-from-heightmap model (US6).
- Verified implemented (US1 only):
  - `data-harvester/src/harvester/v50/classify.py` (127 lines) — `SignalTier` enum,
    `compute_signal_tier()`, published deterministic criteria (`WEAK_MAX_RANGE`, `NORMAL_MAX_RANGE`),
    explicitly handles missing-alpha as `None` rather than fabricating a score (FR-007).
  - `data-harvester/scripts/v50_tile_classify.py` (161 lines) CLI.
  - `tile_inventory.py` carries `signal_class`/`signal_class_evidence` fields (line 282-283, 382).
  - `tile_composite.py` draws a green outline for normal-tier tiles (`TIER_OUTLINE`, line 349).
  - `tests/v50/test_classify.py` (104 lines).
- Not implemented: US2 nested-signal detector (`nested_signal.py`), US3 brush-scar correlator
  (`brush_correlate.py`), US4 cross-map fragment alignment (`fragment_align.py`), US5 rescale-boundary
  detector (`rescale_boundary.py`), US6 predictive model (`brush_model.py`) — none of these files, nor
  their CLI/test counterparts, exist. This matches the tasks.md status table exactly.
- Checkbox accuracy: accurate — tasks.md's own status column ("In progress" / "Pending") correctly
  reflects the code, and no task is checked `[x]` despite T001-T003/T007 existing (understated, not
  overstated).
- Operator gates owed: none for the shipped US1 slice (deterministic, no training). US6 would need a
  GPU training run once built.
- Open residue (spec-stated only): US2-US6 (FR-003, FR-004, FR-005, FR-008, FR-009) — the DeadminesInstance/
  Westfall lineage-recovery narrative that motivated the whole spec is entirely unbuilt.
- Superseded by / overlaps: 127 (weak-tile explorer), 140 (motif/paste archaeology — a broader,
  differently-designed successor to US3/US4/US6's ambitions), 194 (stratum classification enum
  overlaps US1's tiering intent at a different layer).
- Disposition: FOLD
- Proposed epic theme: terrain-brush-and-motif-archaeology
- Confidence: high

---

### 133 Unbaked Minimap Decomposition
- Stated status: Draft | Tasks: no tasks.md.
- Scope: Emit `terrain_shadow_256` (textureless Lambert+ambient+cast-shadow signal) as a first-class
  harvest/store signal alongside `minimap_rgb`/`normal_xyz`/`height_257` (US1), build a row-aligned
  decomposed-signal curriculum (US2), train a shadow->height model (US3).
- Verified implemented — this is the most cleanly delivered spec in the batch:
  - `TerrainShadow256` property -> `src/core/WowViewer.Core/Maps/TerrainTileTensorPack.cs:333`
    (plus a companion `ObjectifiedTerrainShadow256` at :311, an extension beyond spec).
  - Serialization -> `NpzTileSerializer.cs`, `RawArraySerializer.cs`.
  - Compositor emission explicitly commented "Spec 133" at two sites in
    `tools/harvest/WowViewer.Tool.Harvest/Program.cs` (lines ~2915, ~3445) and in
    `src/core/WowViewer.Core.IO/Maps/TerrainMinimapCompositor.cs`.
  - Curriculum + training (US2/US3) -> the entire `data-harvester/src/harvester/v60/` package keys
    off `terrain_shadow_256` by name (`clean_signal_corpus.py`, `store.py`,
    `real_terrain_synthetic.py`, `real_terrain_synthetic_zarr.py`, `terrain_models.py`,
    `terrain_method_translation.py`, etc.), and `data-harvester/scripts/v60_train_shadow_height.py`
    is literally the shadow->height trainer US3 asked for, with matching `tests/v60/` coverage.
- Partial: FR-004 (V50 store manifest declaring `terrain_shadow_256`) — the signal was found live in
  the **v60** store/curriculum, not the v50 store package (`data-harvester/src/harvester/v50/*.py`
  has no `terrain_shadow_256` hits); the spec named v50 but delivery landed one version later, which
  is a naming drift rather than a gap.
- Not implemented: nothing material — all three user stories have real code and tests.
- Checkbox accuracy: N/A (no tasks.md), but spec is functionally complete.
- Operator gates owed: SC-003 (shadow->height beats tile-mean by >=5%) needs a real run — see spec
  139's embedded receipts, which report this exact experiment already executed with real numbers.
- Open residue (spec-stated only): none identified — treat as complete/superseded-by-successor.
- Superseded by / overlaps: 125 (US6/US4 textureless-residual + shadow-decode content duplicates
  this spec almost exactly), 139 (owns the actual trained shadow->height model with real receipts).
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: minimap-decomposition-and-codec-parity
- Confidence: high

---

### 134 V60 Controlled Terrain Reconstruction Experiment (a.k.a. "v60 unified dataset model")
- Stated status: "terrain-only control learning is active; object lanes are parked" (spec.md header,
  itself an operator-authored status note) | Tasks: 51/95 checked (`grep -c '\[x\]'`=51, `'\[ \]'`=44).
- Scope: Very large spec (54 FRs) — synthetic control corpus (US1), object identification/marking +
  sieve (US2, explicitly deferred per spec header), limited control-data model experiment (US3),
  albedo normalization + textureless gate (US4), tiny real-data transfer decision (US5), later-client
  adapters (US6, P3), real-tile observation intake (US7).
- Verified implemented:
  - US1 control corpus -> `data-harvester/src/harvester/v60/control_corpus.py`,
    `control_experiment.py`, with `tests/v60/test_control_corpus.py`,
    `test_control_experiment.py`, `scripts/v60_visualize_control_corpus.py`.
  - US2 object sieve/library/marker (nominally "parked") -> code exists anyway:
    `object_sieve.py`, `object_library_sieve.py`, `object_marker.py`, with
    `test_object_sieve.py`, `test_object_library_sieve.py`, `test_object_marker.py`,
    `test_real_object_mask_model.py` — contradicts the "parked" framing at the code level, though
    the spec header's "active vs parked" is a scope-prioritization statement, not a claim the code
    doesn't exist.
  - US3 limited experiment -> `clean_signal_corpus.py`/`clean_signal_train.py` (shared with spec 139)
    plus `real_synthetic_pairs.py`, `rgb_method_benchmark.py`.
  - US7 real-tile observation intake / viewer catalog (Phase 9/10) ->
    `src/core/WowViewer.Core/Maps/DatasetVersionCatalog.cs`,
    `src/viewer/WoWViewer/ViewerApp_DatasetCatalog.cs`,
    `tests/WowViewer.Core.Tests/DatasetVersionCatalogTests.cs` — real, wired into the viewer.
- Partial: US2's marker export contract (FR-044, `known_object_marker_256` + identity table) — only
  one file (`object_marker.py`) implements this; depth of coverage (does it actually emit the
  sidecar identity table per FR-044) not independently verified beyond the file existing. Spec's own
  2026-08-09 "parked" note records that the marker retrieval result was evaluated and failed
  (held-out top-1 ~zero) — i.e. this sub-lane was tried, measured, and explicitly shelved by the
  operator, which is a real (negative) result, not an unstarted task.
- Not implemented: US6 later-client adapters (P3, explicitly out of scope for now per spec's own
  assumption #7) — no evidence of a second-era adapter.
- Checkbox accuracy: roughly consistent (51/95, and the spec header itself narrates which lanes are
  active vs. parked) — did not exhaustively verify all 51 checked items individually given batch time
  budget; spot-checks (control corpus, dataset catalog) confirm real code.
- Operator gates owed: this spec explicitly documents user-run CUDA/marker results already (the
  marker-retrieval failure, sieve corrections) — further GPU runs for US3/US4/US5 expansion decisions
  remain owed per the spec's own "expand" gate (FR-017).
- Open residue (spec-stated only): US6 later-client adapters (P3); US2 marker lane is measured-failed
  and explicitly deferred, not something to silently re-open without new evidence.
- Superseded by / overlaps: 139 (shares `clean_signal_*` modules directly — 134 and 139 are two specs
  describing overlapping/adjacent parts of the same `v60/` codebase), 133 (shares `terrain_shadow_256`
  contract), 140/141 (object/motif evidence layers this spec's US2 anticipated).
- Disposition: KEEP-ACTIVE
- Proposed epic theme: v60-controlled-terrain-experiment
- Confidence: medium (spec is large; verified representative slice per user story, not all 54 FRs)

---

### 139 V7-Inspired Clean-Signal Terrain Reconstruction
- Stated status: "Phase 5 implementation and minimap-observable raw-RGB diagnostic preparation
  complete; promotion and albedo-normalized transfer remain held pending user-run evidence" —
  spec.md contains an unusually thorough inline evidence log (real CUDA run numbers, dates, MAE
  values, regressions) written by the operator/agent as work progressed. | Tasks: 34/48 checked.
- Scope: Reproduce v7's coarse+detail multi-scale structure idea without its WDL/leakage-prone input
  contract; deployment input is exactly 4 channels (luma, x/y gradient, albedo confidence); train and
  ablate architectures (`pyramid_cnn`, `segformer_b0`, `unet_lite_v2`) and v7-style structural losses;
  gate real-minimap transfer behind an explicit hold/diagnose/expand decision.
- Verified implemented: every module named in the spec's own "Implementation checkpoint" section is
  present in `data-harvester/`: `src/harvester/v60/clean_signal_inputs.py`, `clean_signal_targets.py`,
  `clean_signal_corpus.py`, `clean_signal_model.py`, `clean_signal_losses.py`, `clean_signal_train.py`,
  and CLIs `scripts/v60_build_clean_signal_corpus.py`, `v60_validate_clean_signal_corpus.py`,
  `v60_visualize_clean_signal.py`, `v60_train_clean_signal.py`. Real training receipts are embedded
  directly in spec.md (best cell `pyramid_cnn`+`v7_structural_v1`, final-height MAE 0.145868 vs.
  0.181995 tile-mean baseline within-family; complete-family run 0.173904 vs 0.191047, 8.97%
  improvement; reflect-padding confirmation run 0.137891 vs 0.191047, 27.82% improvement, with named
  regressions on `cross_tile_lightning`/`cross_tile_burn`) — this is the strongest real-run evidence
  trail in the whole batch, self-reported with both successes and failures (real-bridge transfer
  scored **worse** than baseline, -106% and -185%, and is reported as such rather than hidden).
- Partial: US4 real-transfer promotion (FR-012) is explicitly held — the spec's own status line says
  so; this is accurate self-reporting, not a gap to flag as new.
- Not implemented: nothing claimed as done that isn't; the spec is unusually honest about what
  remains blocked (cross-tile family regressions, real-domain transfer still net-negative).
- Checkbox accuracy: accurate — 34/48 matches the phase structure (Phases 1-5 done, Phase 6 real
  transfer and part of Phase 7 polish outstanding).
- Operator gates owed: SC-004 (held-out-family >=5% with no bucket regression >5%) — not yet met per
  the embedded receipts (cross-tile buckets regress far more than 5%). SC-005/006 transfer gates —
  explicitly held.
- Open residue (spec-stated only): cross-tile family regression must be fixed before promotion
  (FR-009/SC-004); real albedo-normalized transfer (US4/FR-012) remains blocked pending the
  albedo-normalization operation from spec 134's US4.
- Superseded by / overlaps: 126 (this is where 126's core height-reconstruction ambition actually
  landed), 133 (consumes `terrain_shadow_256`), 134 (shares the same `v60/clean_signal_*` files —
  134 and 139 should probably be reconciled into one epic owner), 140/141 (named as the consumer of
  their guidance bundles, not yet wired).
- Disposition: KEEP-ACTIVE
- Proposed epic theme: v60-controlled-terrain-experiment
- Confidence: high

---

### 140 Terrain Paste and Fractal Motif Archaeology
- Stated status: Draft | Tasks: 0/51 checked.
- Scope: Motif atlas (US1), recurring-paste retrieval across tile/chunk boundaries with transform
  detection (US2), tileset-identity vs. geometry separation (US3), feeding bounded guidance into
  spec 139 (US4), optional object-placement evidence (US5), paint-order/sculpt-intent inference from
  ordered alpha layers (US6), plus an extensive alpha-evidence-fan-out and brush-scale-join design
  (FR-017..FR-028).
- Verified implemented: none. Grepped for `motif`, `paste_family`/`PasteFamily`, `paint_order`,
  `alpha_evidence`/`AlphaEvidence`, `brush_scale`/`BrushScaleRecord` across
  `data-harvester/src/` — the only hits are unrelated pre-existing files (`chunk_motifs.py`,
  `terrain_feature_labels.py`, `spec103/prefab_curation.py`), none of which implement this spec's
  designed entities (`MotifCandidate`, `PasteFamily`, `TilesetProfile`, `AlphaEvidenceBundle`,
  `BrushScaleRecord`, `GuidanceBundle`, `DifficultyGuidance`).
- Not implemented: the entire spec (all FR-001..FR-028), matching its own tasks.md (0/51).
- Checkbox accuracy: accurate (nothing checked, nothing built).
- Operator gates owed: all SC-001..SC-014 require this to be built first; no GPU/training gate
  reached yet since the code doesn't exist.
- Open residue (spec-stated only): the whole spec is residue if this research direction is still
  wanted — most notably US6's paint-order/sculpt-intent hypothesis (FR-017..FR-020, FR-026..FR-028)
  and US2's cross-tile paste-family retrieval (FR-006..FR-008), since these are the parts most
  clearly differentiated from spec 132's overlapping (also-unbuilt) US3/US4/US5.
- Superseded by / overlaps: 132 (US3 brush-texture correlation, US4 cross-map alignment, US5
  rescale-boundary detection are near-duplicates of 140's US2/US3/US6 and are equally unbuilt — these
  two specs should be merged, not both carried forward separately), 139 (declared consumer via
  "GuidanceBundle").
- Disposition: FOLD
- Proposed epic theme: terrain-brush-and-motif-archaeology
- Confidence: high (clean negative grep result across the full designed vocabulary)

---

### 141 Terrain Method Translation and Evidence Gates
- Stated status: Draft | Tasks: 23/30 checked.
- Scope: Maintain an external-method evidence ledger (US1: DSM2DTM, ResDepth, SMRF, CSF, Prithvi,
  etc.), enforce an RGB-only/height-prior/point-cloud/combined modality boundary with fail-closed
  forbidden-signal audits (US2), compare object-aware RGB terrain completion with no/predicted/
  withheld object masks (US3), and preserve novel research leads for follow-up (US4, P3).
- Verified implemented:
  - US1/US2 -> `data-harvester/src/harvester/v60/terrain_method_translation.py` (559 lines):
    `ExternalMethodRecord`, `InputContract`, `TranslationDecision`, `validate_method_records`,
    `initial_method_records` (the DSM2DTM/ResDepth/SMRF/CSF/Prithvi ledger), `build_rgb_only_contract`/
    `build_height_prior_contract`/`build_point_cloud_contract`/`build_combined_contract`,
    `audit_input_reads` (the forbidden-signal fail-closed audit, FR-003). Tests:
    `tests/v60/test_terrain_method_translation.py`,
    `test_terrain_method_translation_contract.py`, plus a fixtures file
    `terrain_method_translation_methods.json`.
  - US3 -> `data-harvester/src/harvester/v60/rgb_method_benchmark.py` (355 lines):
    `_authored_source_report`, `_object_library_source_report`, `_condition_reports`,
    `build_rgb_method_benchmark_plan`, `_require_no_forbidden_reads`. Test:
    `tests/v60/test_rgb_method_benchmark.py`.
- Not implemented: US4 research-lead ledger (FR-010, `ResearchLead` entity) — grepped
  `ResearchLead`/`research_lead` across `v60/`, zero hits. This matches the 7 unchecked tasks
  (Phase 6 in tasks.md is the research-lead phase).
- Checkbox accuracy: accurate — 23/30 matches Phases 1-5 done, Phase 6 (US4) and part of Phase 7
  outstanding.
- Operator gates owed: SC-003/SC-005 (real RGB-only benchmark training run with all three mask
  conditions) — the dry-run plan builder exists; whether a real CUDA run has executed against it was
  not separately confirmed in this pass (no embedded receipt like spec 139's).
- Open residue (spec-stated only): US4 research-lead record system (FR-010, SC-007).
- Superseded by / overlaps: 139 (declared owner of the clean-signal model this feeds), 140 (declared
  co-owner of the archaeology side).
- Disposition: FOLD
- Proposed epic theme: v60-controlled-terrain-experiment
- Confidence: high

---

### 194 Temporal Stratigraphy & Weak Signal Development Mesh Restoration
- Stated status: (no explicit status line; spec is terse/structured) | Tasks: 16/16 checked.
- Scope: C#/viewer feature — `StratigraphyLevelAnalyzer`, `TemporalStratumClassifier` (as an enum +
  classification, see below), `SeamDiscontinuityProfiler`, SIMD `TemporalMeshRestorer`, an in-viewer
  Stratigraphy workbench, and CLI `terrain-stratigraphy-scan`/`terrain-stratigraphy-patch`.
- Verified implemented:
  - `src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy/StratigraphyLevelAnalyzer.cs`,
    `SeamDiscontinuityProfiler.cs`, `TemporalMeshRestorer.cs`, `FastTerrainNormalSolver.cs` — all
    exist as named classes.
  - `TemporalStratum.cs` implements the classification as an `enum TemporalStratum` (Active_1x,
    LateRevision_4x_8x, ClassicErasure_33x, DeepProto_64x_512x, Holed_DevMesh_1x,
    Submerged_OceanFloor, BitExact_Flat per spec) — the spec names a class
    `TemporalStratumClassifier`; the actual code folds classification into the enum + analyzer rather
    than a separate class. Functionally equivalent, naming variance only.
  - CLI -> `tools/inspect/WowViewer.Tool.Inspect/TerrainStratigraphyScanCommand.cs`,
    `tools/converter/WowViewer.Tool.Converter/TerrainStratigraphyPatchCommand.cs`, both wired into
    their respective `Program.cs`.
  - Viewer workbench -> `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` and
    `ViewerApp_TerrainInspection.cs` reference `WeakSignal`/stratigraphy controls extensively.
  - Test: `tests/WowViewer.Core.Tests/StratigraphyLevelAnalyzerTests.cs`.
- Partial: NFR-001 (zero-allocation, <2ms render-thread path) and NFR-004 (core/UI separation) are
  architectural claims not independently benchmarked in this pass — code exists in the right layers
  (`WowViewer.Core.Runtime` for logic, `WoWViewer` for UI), satisfying NFR-004 by inspection; NFR-001
  performance was not measured (would require a runtime/profiling gate).
- Not implemented: nothing identified as missing.
- Checkbox accuracy: accurate (16/16 checked, all verified present).
- Operator gates owed: NFR-001's <2ms/zero-allocation claim and NFR-003 (client read-only safety in
  a real run) are runtime proofs, not verifiable from static code alone.
- Open residue (spec-stated only): none functional; only the two runtime NFR proofs above.
- Superseded by / overlaps: 196 (direct sequel spec, extends the same `Stratigraphy` namespace),
  127/132 (this spec is functionally the mature successor to both's weak-signal-inspection ambitions).
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: weak-signal-terrain-archaeology
- Confidence: high

---

### 196 WDL Lattice Magnetization, Multi-Anchor Polarity Inversion & Neighbor-Mesh Auto-Fitting
- Stated status: (no explicit status line) | Tasks: 14/14 checked.
- Scope: Polarity inversion + multi-anchor baselines (US1), neighbour-mesh auto-fit boundary solver
  (US2), WDL macro-lattice magnetization + `.wdl` file export (US3), async non-blocking tile
  restoration pipeline (US4).
- Verified implemented:
  - `src/core/WowViewer.Core.Runtime/World/Terrain/Stratigraphy/NeighborMeshHeightSolver.cs`,
    `WdlLatticeMagnetizer.cs`, `WdlFileWriter.cs` — all present, all referenced from
    `ViewerApp_Sidebars.cs`.
  - `TemporalStratigraphyOptions` and `TemporalMeshRestorer.cs` updated for polarity/anchor per T002/T003
    (file exists and was already found as a shared file with spec 194).
  - Tests: `tests/WowViewer.Core.Tests/NeighborMeshHeightSolverTests.cs`,
    `WdlLatticeMagnetizerTests.cs`. `WdlFileWriter.Write(...)` is exercised from inside
    `WdlLatticeMagnetizerTests.cs` rather than its own dedicated test file — T013 claims tests for
    all three classes; coverage exists but is consolidated into one file, not split three ways as the
    task literally describes.
- Partial: none material — minor test-file-organization variance only (see above).
- Not implemented: nothing identified as missing against the 5 ACs.
- Checkbox accuracy: accurate (14/14, all verified present); one cosmetic overstatement (T013 implies
  3 separate test files, code has 2).
- Operator gates owed: AC-002 (boundary RMSE <=0.05m on real data), AC-005 (>=60 FPS / no >16ms UI
  stall) are runtime/perf proofs not verifiable from source alone.
- Open residue (spec-stated only): none functional; AC-002/AC-005 runtime proof only.
- Superseded by / overlaps: 194 (direct predecessor, same namespace and file set).
- Disposition: ARCHIVE-COMPLETE
- Proposed epic theme: weak-signal-terrain-archaeology
- Confidence: high

---

## Batch summary

| id | disposition | residue count | theme |
|----|--------------|---------------|-------|
| 125 | ARCHIVE-SUPERSEDED | 3 (super-resolution, restoration hallucination gate, residual-extractor confidence) | minimap-decomposition-and-codec-parity |
| 126 | ARCHIVE-SUPERSEDED | 2 (texture-tier decode US6/US8, multi-tile seam stitching US5) | minimap-decomposition-and-codec-parity |
| 127 | FOLD | 3 (inspection views US1, tile listing US3, neighbour-auto-amplify UI wiring US2) | weak-signal-terrain-archaeology |
| 132 | FOLD | 5 (US2 nested tiers, US3 brush-scar correlation, US4 fragment alignment, US5 rescale-boundary, US6 predictive model) | terrain-brush-and-motif-archaeology |
| 133 | ARCHIVE-COMPLETE | 0 | minimap-decomposition-and-codec-parity |
| 134 | KEEP-ACTIVE | 1 (US6 later-client adapters; US2 marker lane is measured-failed/parked, not open residue) | v60-controlled-terrain-experiment |
| 139 | KEEP-ACTIVE | 2 (cross-tile family regression fix, real albedo-normalized transfer promotion) | v60-controlled-terrain-experiment |
| 140 | FOLD | 6 (US1 atlas, US2 paste retrieval, US3 tileset separation, US4 guidance feed, US6 paint-order inference, alpha-evidence fan-out) | terrain-brush-and-motif-archaeology |
| 141 | FOLD | 1 (US4 research-lead ledger) | v60-controlled-terrain-experiment |
| 194 | ARCHIVE-COMPLETE | 0 (2 runtime-only NFR proofs owed) | weak-signal-terrain-archaeology |
| 196 | ARCHIVE-COMPLETE | 0 (2 runtime-only AC proofs owed) | weak-signal-terrain-archaeology |
