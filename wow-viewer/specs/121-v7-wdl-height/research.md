# Research: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Created**: 2026-07-24
**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)

All Technical Context rows resolved against existing code. No unknowns remain.

## D-01: Stage A backbone = SegFormer-B0 encoder + lattice heads (new class, reused loader)

- **Decision**: New `MitB0LatticeNet` in `harvester/spec121/` wrapping `SegformerModel` at B0
  config with two heads (outer 17×17, inner 16×16) emitting the 545-value lattice under the
  Spec 117 masked contract. Optional pretrained encoder via `SegformerModel.from_pretrained(
  "nvidia/mit-b0")`, reusing the exact loader pattern in
  `harvester/v50/direct_geometry_model.py` (lines around `load pretrained encoder weights`).
- **Rationale**: Spec 117 proved from-scratch 675K `LatticeNet` v2 plateaus above tile-mean
  (val 0.2307 vs 0.1277) — capacity/inductive-bias failure, not data failure. Spec 114 proved
  the same B0 encoder clears a numeric geometry gate. B0 ≈ 3.3M encoder + small heads lands
  mid-band of the user's 3–30M requirement.
- **Alternatives considered**: (a) Bigger from-scratch U-Net — Spec 117 v2 already showed the
  from-scratch localization failure at 679 tiles; scaling params without pretrained priors
  repeats it. (b) `WdlPriorNet` (spec103, used by `v50/wdl_prior_train.py`) — derives its target
  synthetically from `height_257` instead of the real `wdl_outer_17/inner_16` arrays; keeps the
  older unmasked contract. Kept as-is; not extended. (c) DINOv2/DAV2 backbones — tainted by the
  Spec 120 retrieval failure and DAV2's earlier rejection; out of the user's requested band of
  trust.

## D-02: Stage A target = Spec 117 real-lattice contract, NOT the spec103 synthetic target

- **Decision**: Stage A trains against `wdl_outer_17`/`wdl_inner_16` + present flags via
  `encode_lattice_target`/`decode`/`select_lattice_rows` from `harvester/spec117/lattice_model.py`.
- **Rationale**: The real lattice keeps per-sample presence (gaps never fabricated); the
  `height_257`-derived target interpolates gaps. The v7 idea is specifically "predict the WDL
  prior", and the real WDL lattice is that prior.
- **Alternatives considered**: spec103 `build_wdl_target(height_257)` — simpler, fully dense, but
  trains a heightmap-downsampler, not a minimap→WDL correlation. Rejected as primary; acceptable
  as an auxiliary ablation later (not in this spec).

## D-03: Stage B = existing detailer + prior-as-coarse bridge; zero trainer rewrite

- **Decision**: Stage A predictions are materialized into the materialized-coarse-store schema
  (`index.parquet` + zarr attrs) that `geometry_detailer_train.py --coarse-store` already
  validates via `validate_coarse_store`, mirroring `direct_geometry_materialize.py`. The detailer
  consumes the predicted prior exactly as it consumed generated coarse relief — contract-wise the
  prior IS a generated coarse field.
- **Rationale**: Spec 118's T027 analogue already proved this acceptance path end-to-end with a
  feature store; the coarse-store path is even more direct. Zero trainer changes = zero parity
  risk; the residual contract (`truth − prior`, zero-init head starts at prior passthrough) is
  the literal v7 architecture.
- **Alternatives considered**: (a) Feeding the prior via `--feature-store` — rejected; the
  feature-store slot is for class-channel priors (terrain feature map) and the coarse channel is
  the semantically correct home for a height prior. (b) New detailer trainer — rejected; violates
  "favor minimal fixes" and duplicates a validated trainer.

## D-04: Detailer trunk option = `detailer_mit_b0_v1` behind `--architecture`, U-Net stays default

- **Decision**: Add `DetailerMitB0Net` (B0 trunk + residual head, same 257 alignment, zero-init
  head) to `geometry_detailer_model.py`; `--architecture` flag on `geometry_detailer_train.py`
  defaults to `detailer_unet_v1` (parity). Document the U-Net `--base` that lands ≥3M for
  within-band comparisons.
- **Rationale**: User explicitly allows a HuggingFace/SegFormer backbone and wants 3–30M. The
  default U-Net at base 32 is 1.56M — below band; the flag makes the band-compliant trunk a
  config choice, not a rewrite. Default-off preserves bit-parity of existing runs.
- **Alternatives considered**: Widening U-Net only — keeps everything from-scratch; the B0 option
  exists because pretrained features are the whole point of the user's ask.

## D-05: Object-mask loss semantics differ per stage (pixel-level vs tile-level)

- **Decision**: Stage B uses Spec 118's existing pixel-level `--object-mask-weight`
  (`1 − w·mask` per point, touched/untouched MAE). Stage A (lattice, no pixels) uses a per-tile
  coverage scale: tile loss multiplied by `1 − w·coverage_fraction` where coverage is the mean of
  `object_geometry_visible_mask_257`; missing array → warn + disable + record, mirroring
  `object_loss.py` behavior.
- **Rationale**: The mask's information content for a 545-point lattice target is "how much of
  this tile's minimap is object-contaminated" — a trust weight on the tile, not a per-point
  weight. Pixel-level application to sparse lattice points would require an arbitrary
  point→pixel mapping.
- **Alternatives considered**: Mapping each lattice vertex to its nearest mask pixel — precise
  but the lattice is the model's OUTPUT space; contamination lives in the INPUT image. Tile-level
  trust weight is the honest summary. Row-dropping (spec103 `filter_deployable_rows`) — too
  blunt; discards data instead of down-weighting.

## D-06: Run-record stage names reused, not widened

- **Decision**: Stage A reuses stage `"lattice_prior"` (added by Spec 117); Stage B reuses the
  detailer stage. Architecture identity (`mit_b0_lattice` / `detailer_mit_b0_v1` vs
  `lattice_net_v2` / `detailer_unet_v1`) distinguishes runs; `architecture.config_sha256`
  already carries the shape.
- **Rationale**: Spec 117's lesson — `config_sha256` hashes but does not carry config — is
  handled by explicitly saving `lattice_config.base`-style fields + backbone id/revision in the
  checkpoint (FR-003/FR-009). No schema change = no migration.
- **Alternatives considered**: New stage names per backbone — multiplies schema surface for zero
  validation value.

## D-07: Specs 119/120 closed as negative evidence, artifacts kept read-only

- **Decision**: Move both spec dirs to `specs/archived/`; keep their trained checkpoints and the
  retrieval PoC sheet as reference; never retry minimap-scale object identity.
- **Rationale**: Spec 119's own PoC (p50=10px, max=29px instances; ~0.99 cosine to unrelated
  blobs) is a decisive, measured negative. Spec 120's DINOv2 pivot inherits the same scale
  physics. The user has called the outcome; the spec records it so no future session relitigates.
- **Alternatives considered**: Scale-matched re-render of the library at 8–32px (noted in Spec
  119 as the only viable retrieval path) — explicitly out of scope; the user redirected the masks
  to loss-side use.

## D-08: Substrate release named v50.2 (user decision, 2026-07-24)

- **Decision**: this lane's store release is **v50.2** = v50.1 signals + Spec 117 WDL lattice
  arrays + Spec 118 object-mask arrays (built with `--stream-profile full`; the v22 profile omits
  the strict object arrays). Trainer `--release` defaults to `v50.2`.
- **Rationale**: user wants zero naming overlap with the failed v24/v25 convergence lanes and an
  explicit marker that the dataset — not a new model version — is what changed. The spec number
  (121) is a speckit sequence, not a model version; model ids are `mit_b0_lattice` /
  `detailer_mit_b0_v1`.
- **Compatibility**: the v50.1 store (`curriculum-0_5_3_3368-dual_v3.zarr`) already carries the
  lattice arrays, so unweighted Stage A runs work today with `--release v50.1`; only the
  mask-weighted comparison runs require the v50.2 rebuild (USERGUIDE.md Phase 1).

## D-09: Within-map completion reframe (B-reframe, user decision 2026-07-24)

- **Decision**: Stage A reframed from cross-region WDL prediction (failed: −73% vs tile-mean on
  the Spec 116 region-isolated split) to **within-map WDL completion**: train on WDL-covered
  tiles, predict missing WDL tiles of the SAME map. This is v7's actual deployment constraint
  (v7 worked where WDL coverage existed locally) and matches the diagnostic evidence (train tiles
  +18.9% vs tile-mean — the model can learn zone-local color↔height mapping).
- **Split**: new ``v121-within-map-split-v1`` schema (separate from ``v50-held-out-split-v1``
  which hard-requires adjacency isolation). Per-map random held-out fraction; adjacent tiles
  allowed in both splits (deployment reality for completion). Optional ``--buffer-rings 1`` for
  a stricter eval. The trainer auto-detects the split schema via ``detect_split_schema`` and
  dispatches to ``apply_within_map_split`` vs ``apply_held_out_split`` — zero new CLI flags.
- **Replaces**: the original US1 cross-region prediction. The cross-region record stands as a
  measured negative (research.md R-1); this reframe is the lane going forward.

## Open Risks

- **R-1**: SC-001 may still fail if minimap RGB genuinely lacks elevation signal at WDL scale
  (Spec 117's V7-doc reframe: v7 used WDL as INPUT, never predicted it from RGB). Mitigation:
  G1 stops the lane cheaply; previews distinguish underfit vs no-signal; a valid negative closes
  the question permanently.
- **R-1 EVIDENCE (2026-07-24, `lattice-mit_b0-v1` epoch-54 checkpoint diagnosis,
  `output/runs/lattice-mit_b0-v1/diagnosis_epoch_snapshot.json`)**: the failure is NOT signal
  absence — it is **non-transfer across the region-isolated split**. On 120 TRAIN tiles the model
  beats tile-mean by **+18.9%** (0.1061 vs 0.1308; 65.8% of rows win) — RGB→lattice mapping IS
  learnable for seen regions. On 240 held-out tiles it is **73% WORSE** than tile-mean (0.2186 vs
  0.1263; only 31.3% of rows win). Per-map: Azeroth 0.2295 vs baseline 0.1341, Kalimdor 0.1851 vs
  0.1026 — both maps fail, so it is not a single-zone quirk. Predictions are not collapsed
  (pred_std 0.14 vs target_std 0.17). This is the third architecture with the identical wall
  (LatticeNet v2 0.2427, v5 0.2307, MiT-B0 0.2125 — val vs 0.1277 baseline), which rules out
  capacity/inductive bias as the cause. Interpretation: the color→elevation relationship is
  zone-local (texture palette ↔ height statistics differ per region); the Spec 116 8-neighbour
  isolation split demands exactly the cross-region transfer that authored minimaps do not support
  at WDL scale. Options: (A) record G1 negative and stop; (B) reframe Stage A as WITHIN-map WDL
  completion (train on WDL-covered tiles, fill missing WDL tiles of the SAME map — v7's actual
  deployment constraint; evaluation becomes within-map held-out, where transfer plausibly holds);
  (C) one discriminating synthetic-source run (exact height↔image correspondence) to test whether
  the transfer failure is authored-minimap contamination (baked lighting/objects) rather than
  fundamental. Await user decision; the run itself continues to its natural early-stop for the
  final relief-stratified record.
- **R-1 UPDATE (2026-07-24, user)**: option C is BLOCKED — synthetic rows are not valid training
  input until the curriculum records `synthetic_lighting_contract='NoonWhiteGlobal'` (the older
  synthetic renders predate the corrected lighting ownership). `--source authored` is the only
  valid bootstrap source on the current store. The fork is therefore A (record negative + stop)
  vs B (within-map WDL completion reframe); C is deferred until a NoonWhiteGlobal curriculum
  exists (a v50.2-era curriculum decision, not this lane).
- **R-2**: Store may lack the Spec 118 arrays until the user rebuilds it (Full profile).
  Mitigation: graceful warn+disable paths already exist; dry-runs verify array presence.
- **R-3**: HuggingFace download unavailable offline. Mitigation: from-scratch default;
  pretrained load is an explicit flag with hub id + revision recorded.
