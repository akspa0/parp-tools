# Progress

## Completed / Landed

- PM4 `MSHD.Field04` corpus analysis was tightened:
  - packed tile-id hypotheses were ruled out on the development corpus
  - cross-tile follow-up showed most cross-tile `CK24` objects span multiple `Field04` buckets
  - docs/tests/analyzers now treat `Field04` as a reusable scene/group bucket instead of a per-tile id
- PM4 `MSLK.TypeFlags` now has partial guided semantics recorded in docs/analyzers:
  - `0x03` = M2 tops
  - `0x10` = interior WMO floors
  - `0x12` = exterior WMO solid surfaces
- viewer shell usability planning and scene-graph planning were split into separate future specs:
  - `044-viewer-shell-usability`
  - `045-scene-graph-workbench`
- M2 continuity already landed in recent slices:
  - classic chunked `MDLX` support path under spec `043`
  - non-throwing invalid animation selection path
  - `3.0.1` layer-0 missing-texture no-draw fix

## New Planning Surface

- **PM4 asset-matching automation planning pack (2026-06-03)**
  - created spec `046-pm4-asset-matching`
  - bounded owner:
    - replace freeze-prone `Export PM4 Obj Set` with a library/CLI export lane
    - segment PM4 object candidates for downstream automation
    - use Zarr-backed signal corpora for PM4 segments and staged WMO/M2 references
    - generate ranked candidate matches and proposal-grade replacement placements for missing development tiles
  - explicitly out of scope in this planning pass:
    - no runtime regrouping change landed
    - no direct map writeback landed
    - no attempt to rescue the old manual PM4 matcher as the primary owner

- **V18 focused two-build minimap-to-terrain reset (2026-06-04)**
  - spec `047-v18-distill-corpus-open-source-loop` remains the owner, but the active contract was reset to the most basic useful lane:
    - focus on `0_5_3_3368` and `3_3_5_12340` only
    - use `minimap_rgb` only as model input
    - train height with plain L1
    - train normals with masked cosine only
  - explicitly dropped from active signoff:
    - renderer-truth capture
    - object-mask / roof-mask / liquid weighting
    - synthesized-input generation
    - distillation
    - open-source student release
  - trainer simplification landed in `wow-viewer/data-harvester/scripts/train_v16_1_common.py`:
    - `_height_loss(...)` now uses direct `F.l1_loss(pred, target)`
    - `_normal_loss(...)` now uses masked cosine against `normal_mask`
    - `_combined_loss(...)` follows the same plain-loss contract
    - default normal route is now `v16_1_1_base`
    - active normal input contract is logged as `minimap_rgb -> normals_xyz`
  - `wow-viewer/data-harvester/scripts/train_v18.py` now defaults `--dataset-dir` to `wow-viewer/output/datasets/v18`, so the V18 entrypoint no longer quietly points at the V16 corpus unless explicitly overridden
  - focused datastore honesty remains in place from the earlier audit:
    - both focused stores still carry zero renderer-truth coverage
    - carry-over PNG trees are explicitly not current proof
    - roof/object mismatches may remain in the stores, but they are no longer blockers for the active minimap-only lane unless a trainer path consumes them again
  - first bounded proofs landed:
    - height: `wow-viewer/models/v18/height/runs/v18_height_focus_minimap_smoke_20260604_r2/`
      - 1 epoch, batch size 4, train/val = 32/8, balanced across the two focused builds
      - best `val_loss = 0.6626`
    - normal: `wow-viewer/models/v18/normal/runs/v18_normal_focus_minimap_smoke_20260604_r2/`
      - 1 epoch, batch size 4, train/val = 32/8, balanced across the two focused builds
      - best `val_loss = 0.2251`
  - one leftover bug was exposed and fixed during the first normal proof:
    - `_preview_normal(...)` still expected old weighted-loss tensors (`terrain_valid_mask`, hard-region outputs, etc.)
    - active simplified normal runs now preview from `base_mask`, `train_mask`, and `invalid_mask` instead

## Next Likely Steps

- implement the first `046` slice: deterministic non-freezing PM4 segment export
- validate `ck24ObjectId`-rooted segmentation against known reference tiles before broad automation
- build the staged-asset Zarr signal corpus and deterministic candidate scorer
- keep PM4 grouping/spec docs in sync if segmentation ownership changes again
- spec `047` follow-ups:
  - validate the focused stores specifically for minimap/height/normal readiness
  - scale the bounded minimap-only height run beyond smoke budget
  - scale the bounded minimap-only normal run beyond smoke budget
