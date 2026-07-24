# Implementation Plan: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Feature Branch**: `121-v7-wdl-height`
**Created**: 2026-07-24
**Status**: Draft
**Spec**: [spec.md](spec.md)

## Summary

Rebuild the v7 two-stage shape on the v50 signal store with everything the failed lines taught us:

- **Stage A** — `MitB0LatticeNet`: a SegFormer-B0 encoder (HuggingFace `transformers`, pretrained
  `nvidia/mit-b0` optional) with two lattice heads predicting the 545-point WDL lattice under the
  Spec 117 masked target contract. The from-scratch `LatticeNet` v2 (Spec 117) stays as the
  selectable fallback architecture. ~3.4M params at default config.
- **Stage B** — the existing `GeometryDetailerNet` residual detailer, fed the **predicted WDL
  prior** as its coarse input via a new bridge that materializes Stage A outputs into the exact
  `--coarse-store` schema the detailer trainer already validates. A `mit_b0` trunk variant for the
  detailer is added behind `--architecture` (default stays the proven U-Net at a widened base so
  both options sit inside the 3–30M band).
- **Object masks** — loss-side only: Stage A gets a per-tile coverage-scaled
  `--object-mask-weight`; Stage B already has the pixel-level `--object-mask-weight` from Spec 118.
  No segmentation/classification/retrieval task is created anywhere.

Almost no new machinery: the plan is wiring + one new model class + one new bridge + one new
trainer, all in `wow-viewer/data-harvester/src/harvester/spec121/` with thin CLIs in
`wow-viewer/data-harvester/scripts/`. No C#. All training is user-run (RULE 0); all CLIs are
dry-run-first.

## Technical Context

| Concern | Decision | Where it lives today |
|---|---|---|
| Substrate store | **v50.2 release** = v50.1 Full-profile Zarr + Spec 117 WDL lattice arrays + Spec 118 object-mask arrays (minimap_rgb, wdl_outer_17, wdl_inner_16, wdl_*_present, object_geometry_visible_mask_257, height_257) | `harvester/v50/signal_catalog.py` |
| Stage A target contract | Spec 117 masked lattice encode/decode (per-tile min-max, RANGE_FLOOR, present-mask never fabricated) | `harvester/spec117/lattice_model.py` |
| Stage A backbone | SegFormer-B0 via `SegformerModel` / `SegformerForSemanticSegmentation`; pretrained load optional | `harvester/v50/direct_geometry_model.py` (`MitB0RegressionNet`, pretrained loader) |
| Stage A fallback arch | `LatticeNet` v2 (675K) selectable via `--architecture` | `harvester/spec117/lattice_model.py` |
| Stage A trainer shape | Mirror `lattice_train.py`: required `--held-out-split`, tile-mean baseline, previews, onecycle warmup-aware stale counter | `harvester/spec117/lattice_train.py`, `harvester/v50/lr_schedule.py` |
| Stage B detailer | `GeometryDetailerNet` (residual, zero-init head, 257 alignment) — unchanged contract | `harvester/v50/geometry_detailer_model.py` |
| Stage B coarse input | Materialized coarse store schema (index.parquet + attrs), produced by new prior bridge | `harvester/v50/geometry_detailer_train.py` (`validate_coarse_store`), `direct_geometry_materialize.py` |
| Object-mask loss (B) | `--object-mask-weight` pixel weighting + touched/untouched MAE — already implemented | `harvester/spec118/object_loss.py` |
| Object-mask loss (A) | New per-tile coverage-scaled weight (lattice has no pixels) | new, `harvester/spec121/` |
| Spectral losses | V7 terms ported loss-only, flags default 0 | `harvester/v50/spectral_guidance.py` |
| Split | Spec 116 held-out split (8-neighbour isolation, leakage refusal) | `harvester/spec116/held_out_split.py` |
| Run records | `v50-model-stage-run-v1`, `promotion_verdict=pending` | `harvester/v50/model_stage_contract.py` |
| Deps | `transformers>=4.52`, `timm>=1.0` already pinned | `data-harvester/pyproject.toml` |

## Constitution Check

| Principle | Verdict | Note |
|---|---|---|
| I. Repo Independence | PASS | All new code inside `wow-viewer/data-harvester/`. |
| II. Library-First | PASS | Logic in `harvester/spec121/`; scripts are thin CLIs. |
| III. Real-Data Validation | PASS | User-run gates on the `H:\CLIENTS` v50 store; validation sheets are the acceptance surface. |
| IV. Residual Model Chain | PASS | This plan IS the residual chain: prior model → residual detailer, independent checkpoints, no shared weights. |
| V. Streaming-First Dataset | PASS | No new harvest; consumes existing store. |
| VI. No Client Path Assumptions | PASS | Store paths are CLI args; no client root in code. |
| Read-Only Reference | PASS | `gillijimproject_refactor` untouched. |
| Frozen Surfaces | PASS | No C# at all; `AlphaWdtWriter` etc. untouched. |
| Training Script Changes | PASS | New trainer + one bounded flag addition; defaults preserve parity; documented here. |
| One Phase at a Time | PASS | Phases below are sequential with validation gates; user runs all training. |

## Phase 0 — Research (research.md)

All technical-context rows resolved against existing code; no NEEDS CLARIFICATION remains.
Decisions recorded in [research.md](research.md).

## Phase 1 — Design

- [data-model.md](data-model.md) — entities: lattice target, prior coarse store, object-mask loss
  weighting, run record binding.
- [contracts/cli-contract.md](contracts/cli-contract.md) — the three new CLIs + the two reused
  trainer CLIs with exact flags.
- [quickstart.md](quickstart.md) — user-run command sequence (PowerShell), gates, verdict
  recording.

## Phase 2 — Implementation Phases (each ≤10 steps, one concern each)

### Phase 1: Stage A model + contract reuse (US1, FR-001/003/009)

1. `harvester/spec121/lattice_backbone_model.py`: `MitB0LatticeNet` — SegFormer-B0 encoder +
   outer/inner heads emitting 545 values; config-only reconstructable; param count recorded;
   from-scratch default with optional `from_pretrained("nvidia/mit-b0")` loader reusing the
   `direct_geometry_model` helper pattern.
2. Architecture registry: `--architecture {lattice_net_v2, mit_b0_lattice}`; identity hash carries
   arch id + config.
3. Unit tests: shapes, masked loss parity with Spec 117 contract, config reconstruction, param
   band 3–30M, pretrained-load path (mocked hub call), absent-sample exclusion.

### Phase 2: Stage A trainer + CLI (US1, FR-004/005/007/008/011/012)

4. `harvester/spec121/lattice_backbone_train.py` mirroring `lattice_train.py`: required
   `--held-out-split`, tile-mean baseline always recorded, onecycle + warmup-aware stale counter,
   per-epoch previews + final sheets, `--gradient-weight` passthrough.
5. `--object-mask-weight` (default 0.0 = parity): per-tile loss scale `1 - w·coverage` from
   `object_geometry_visible_mask_257`; missing-array warn+disable; touched/untouched tile-level
   MAE in record.
6. `scripts/spec121_train_lattice_prior.py` thin CLI, dry-run-first, `--confirm-run` gate; run
   record `v50-model-stage-run-v1` stage `"lattice_prior"`, `promotion_verdict=pending`.
7. Tests: dry-run plan contents, baseline presence, mask-weight math, missing-array degradation,
   refusal without split.

### Phase 3: Prior → coarse-store bridge (US2, FR-010/014)

8. `harvester/spec121/prior_coarse_bridge.py` + CLI: frozen Stage A checkpoint → batch inference →
   dense 257×257 prior field (bilinear over outer/inner grids, Spec 117 bridge rule) → coarse
   store in the exact schema `validate_coarse_store` accepts (mirror
   `direct_geometry_materialize.py` attrs + index.parquet), checkpoint sha256 in attrs.
9. Integration proof: fixture store + random-init checkpoint → bridge `--write` → dry-run
   `geometry_detailer_train.py --coarse-store` accepts it with zero trainer changes.

### Phase 4: Stage B runs + optional mit_b0 trunk (US2, FR-002/003)

10. `DetailerMitB0Net` (SegFormer-B0 trunk, residual head, 257 alignment) added to
    `geometry_detailer_model.py` behind `--architecture {detailer_unet_v1, detailer_mit_b0_v1}`;
    U-Net default unchanged; U-Net `--base` documented at the value that lands ≥3M for parity
    comparisons.
11. Tests: both archs constructable from config, residual contract (zero-init start = prior
    passthrough), param band.

### Phase 5: Paired mask-loss comparison + chain materializer (US3/US4, FR-006/007/014)

12. `scripts/spec121_materialize_chain.py`: minimap-only → Stage A → bridge → Stage B → final
    height sheets (fixed rows + worst cases + OOD mode); provenance = both checkpoint sha256s.
13. Quickstart records the paired-run matrix (mask-weight 0 vs 1 per stage) and verdict table;
    null result = valid close.

### Phase 6: Bookkeeping

14. Move `specs/119-object-library-classifier` + `specs/120-minimap-placement-retrieval` to
    `specs/archived/` with a pointer note; memory-bank sync (activeContext/progress); feature.json
    pointer to this spec.

## Validation Gates (user-run, RULE 0)

- G1 (after Phase 2 user training): SC-001 — Stage A ≥15% below tile-mean. Fail = stop lane,
  record negative result.
- G2 (after Phase 4 user training): SC-002 — Stage B ≥9% below prior-only baseline; GT-prior
  ablation reported.
- G3 (Phase 5): SC-003 paired verdict recorded; SC-005 visual sheets → user verdict.
- Promotion stays `pending` until user visual gate at every stage.
