# Active Context

## Workspace Guardrails

- `gillijimproject_refactor` stays read-only reference code except bounded continuity/doc updates.
- all new code lands in `wow-viewer`.
- staged clients under `I:/parp/parp-tools/output/tmp/wowarchive-clients/` are the only trusted client roots.

## Current wow-viewer Lanes

- PM4 research remains library-first in `wow-viewer/src/core/WowViewer.Core.PM4`.
- `MSHD.Field04` is no longer treated as packed tile `XX_YY`.
- current PM4 evidence says `Field04` is a reusable scene/group bucket:
  - `0/502` direct packed `TileX/TileY` matches in the development corpus
  - `73` distinct `Field04` values are reused across multiple tiles
  - `204/266` cross-tile `CK24` values bridge multiple `Field04` buckets
- treat `Field04` as useful debug/group metadata, not the primary cross-tile stitch key.
- current PM4 `MSLK.TypeFlags` partial semantics from guided real-data inspection:
  - `0x03` = M2 top surfaces
  - `0x10` = interior WMO floors
  - `0x12` = exterior WMO solid surfaces
- keep `MSLK.TypeFlags` distinct from `MSLK.GroupObjectId`.

## Planned PM4 Follow-Ups

- spec `046-pm4-asset-matching` now owns the future PM4 automation lane:
  - replace freeze-prone `Export PM4 Obj Set`
  - export deterministic PM4 object segments
  - use Zarr-backed PM4 and staged-asset signal corpora
  - automate WMO/M2 candidate ranking
  - synthesize proposal-grade replacement placements for missing development tiles
- broken manual PM4 matching UI is explicitly not the workflow owner for that lane.

## Viewer Shell / UI

- spec `044-viewer-shell-usability` owns the dockable shell cleanup and menu/sidebar fixes.
- spec `045-scene-graph-workbench` owns the future Blender-style scene outliner for terrain, objects, and PM4.

## M2 / Runtime Continuity

- standalone classic `MDLX` / chunked M2 support is tracked under spec `043`.
- bad runtime animation selections now fail soft instead of crashing the viewer.
- pre-release `3.0.1.8303` embedded-profile M2 no-draw was reduced to the layer-0 missing-texture fallback seam and fixed in the viewer path.
- **2026-06-05 Ghidra trace against WoW 1.12.1 (Build 5875) Win32**: 1.12.1 `.mdx` files are NOT chunked `MDLX` — they use the `MD20` magic (flat pointer-table format) with the legacy `.mdx` extension. The native cache loader normalizes `.mdl`/`.mdx`/`.m2` to `.m2` and dispatches to the same `MD20` parser regardless. The 1.12.1 `MD20` header layout, view table offset (`0x3c/0x40` instead of `0x44/0x48`), and per-record strides (sequence `0x6c`, light `0xc`, camera `0x2c`, ribbon `0x7c`, particle `0xdc`, light `0xc`) are all different from 3.3.5. The current `M2ModelReaderDispatcher` routes 1.12.1 `.mdx` to the 3.3.5 `M2ModelReader`, which silently misreads the strides. The chunked `M2Chunked` reader is therefore currently a dead branch for 1.12.1 (correctly) but the M2 path is the actual bug. Research doc: `wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md`. This finding should drive a follow-up spec to add an era-aware MD20 reader or a 1.12.1-specific stride table.

## Data Harvester / Training Continuity

- V16.1.1 remains a bounded normal-terrain lane.
- prefer startup autotune and larger batch ladders when VRAM headroom allows.
- validation artifacts are only authoritative when regenerated after behavior changes.

## V18 Focused Two-Build Terrain Reconstruction System (spec 047)

- spec `047-v18-distill-corpus-open-source-loop` remains the active owner for the focused 0.5.3 + 3.3.5 V18 lane, but the contract is now the final terrain-system design rather than the earlier smoke-only framing.
- architecture doc: `wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md`.
- active design owner:
  - focused corpus only: `0_5_3_3368` and `3_3_5_12340`
  - build a focused curation manifest from the V18 Zarr stores
  - train `minimap_rgb -> normalized height_257`
  - train `minimap_rgb -> normal_xyz`
  - keep quilt-level stitching and later ADT writeback as downstream follow-through
- explicitly out of scope for the active iteration:
  - renderer-truth capture as training truth
  - precise object-mask / roof-mask loss lanes
  - liquid-derived weighted loss stacks
  - synthesized-input generation
  - distillation
  - open-source student release
- focused corpus boundary remains:
  - `0_5_3_3368`
  - `3_3_5_12340`
  - other four build stores stay in place but are out of scope
- landed in the 2026-06-04 simplification pass:
  - `wow-viewer/data-harvester/scripts/train_v16_1_common.py`
    - height loss is plain `F.l1_loss(pred, target)`
    - normal loss is masked cosine only
    - default normal route is `v16_1_1_base`
    - default normal contract logs as `minimap_rgb -> normals_xyz`
  - `wow-viewer/data-harvester/scripts/train_v18.py`
    - now defaults `--dataset-dir` to `wow-viewer/output/datasets/v18`
    - active V18 entrypoint no longer quietly falls back to the V16 dataset root
  - focused renderer-truth state stays explicitly non-authoritative:
    - both focused stores remain cleared to `has_object_visibility_mask = 0`
    - both focused stores remain cleared to `has_no_object_minimap = 0`
    - carry-over PNG trees are not active signoff evidence
- landed in the 2026-06-05 final-design/operator pass:
  - spec pack was rewritten in place:
    - `wow-viewer/specs/047-v18-distill-corpus-open-source-loop/spec.md`
    - `plan.md`
    - `tasks.md`
    - `research.md`
    - `data-model.md`
    - `quickstart.md`
    - `contracts/`
  - focused wrappers now exist:
    - `wow-viewer/data-harvester/scripts/build_v18_curation_manifest.py`
      - defaults to `wow-viewer/output/datasets/v18`
      - defaults to builds `0_5_3_3368` + `3_3_5_12340`
      - writes under `wow-viewer/output/datasets/v18/curation/<run-name>/`
    - `wow-viewer/data-harvester/scripts/train_v18_focus.py`
      - defaults to `wow-viewer/output/datasets/v18`
      - defaults to builds `0_5_3_3368` + `3_3_5_12340`
      - auto-picks the latest focused `kept_tiles.parquet` when present
  - `.specify/feature.json` still points at stale spec `011`; use the `047` directory directly until that pointer is explicitly reopened
  - bounded proof now exists:
    - height smoke run: `wow-viewer/models/v18/height/runs/v18_height_focus_minimap_smoke_20260604_r2/`
      - 1 epoch, batch size 4, 32 train tiles, 8 val tiles
      - `val_loss = 0.6626`
    - normal smoke run: `wow-viewer/models/v18/normal/runs/v18_normal_focus_minimap_smoke_20260604_r2/`
      - 1 epoch, batch size 4, 32 train tiles, 8 val tiles
      - `val_loss = 0.2251`
    - focused curation manifest: `wow-viewer/output/datasets/v18/curation/v18_focus_terrain_v1/`
      - audited rows: `6763`
      - kept rows: `4096`
      - keep ratio: `0.6056`
      - dominant reject causes:
        - `blank_minimap_blank_normals = 2396`
        - `blank_what_plate_tile = 221`
        - `normal_minimap_edge_mismatch = 36`
        - `wmo_loss_wipeout_tile = 14`
      - kept difficulty mix:
        - `easy = 8`
        - `medium = 30`
        - `hard = 3070`
        - `pathological = 988`
  - one leftover seam was exposed and fixed during that proof:
    - `_preview_normal(...)` still expected old weighted-loss tensors and crashed after a successful epoch
    - it now falls back cleanly to simplified-lane preview tensors (`base_mask`, `train_mask`, `invalid_mask`)
- landed in the 2026-06-05 focused-mask/tuning pass:
  - the earlier simplification had gone too far for active focused training:
    - curation and dataset tensors still carried liquid/terrain-valid context
    - but active `height` used full-tile `L1`
    - and active `normal` used cosine over `normal_mask` only
  - active focused loss path is now terrain-valid again in `wow-viewer/data-harvester/scripts/train_v16_1_common.py`:
    - `_height_loss(...)` now masks `abs(pred-target)` by `terrain_valid_mask_257`
    - `_normal_loss(...)` now masks cosine by `normal_mask * terrain_valid_mask_257`
    - this keeps liquid-hidden and object-hidden regions out of the loss without restoring a large auxiliary liquid-weight stack
  - one more terrain-valid seam was then reopened and fixed:
    - harvested `object_roof_mask_256` / `object_roof_weight_257` already existed
    - but active `terrain_valid_mask_257`, focused curation `trainable_cov`, and the height preview still ignored that roof/top-geometry layer
    - `wow-viewer/data-harvester/src/harvester/v16_1_dataset.py`
      - now composes terrain-valid masks through a shared helper that includes roof/top-geometry occlusion
    - `wow-viewer/data-harvester/scripts/build_v16_curation_manifest.py`
      - now uses the same roof-aware terrain-valid logic for `terrain_valid_cov` / `trainable_cov`
      - now records `roof_cov`
    - `wow-viewer/data-harvester/scripts/train_v16_1_common.py`
      - height preview now shows the actual combined masked weight, not the stale basement-only `weight_257`
      - height/normal preview outputs can surface `object_roof_weight`
  - focused curation is now stricter in `wow-viewer/data-harvester/scripts/build_v16_curation_manifest.py`:
    - rows with `trainable_cov < 0.20` are rejected as `insufficient_trainable_terrain`
    - this catches liquid-hidden wipeout rows even when they are not WMO-dominated
  - focused wrapper/operator defaults are now pointed at the actual 8 GB lane:
    - `wow-viewer/data-harvester/scripts/train_v18_focus.py`
      - defaults `--target-vram-gb 8`
      - defaults startup `--autotune-batch-size`
      - defaults `--strict-build-balance`
  - focused sampling is now explicitly corrected:
    - prior `build_balanced=True` behavior was only approximate and could still let the 3.3.5 pool dominate
    - `wow-viewer/data-harvester/scripts/train_v16_1_common.py`
      - now supports strict near-equal per-build balancing without replacement
      - oversized pool/epoch requests auto-cap to the largest feasible balanced subset
      - focused epoch logging/config now records the effective balanced epoch size
    - this is the active owner fix for the discovered `700` vs `2636` train skew in the focused `v2` manifest/run
  - focused docs now steer away from the earlier smoke budget:
    - quickstart/README examples now use larger tile pools and `40` epochs instead of `20`
  - targeted Python proof now exists:
    - `wow-viewer/data-harvester/src/harvester/test_v18_focus_masks.py`
      - proves height loss ignores non-trainable regions
      - proves normal loss honors terrain-valid masking
      - proves terrain-valid composition includes roof/top-geometry masks
      - proves curation rejects low-trainable tiles
      - proves strict build-balance equalizes/caps focused subsets as designed
- the next real proof is:
  - rerun focused curation so the active `kept_tiles.parquet` drops low-trainable liquid-hidden rows
  - scale the height run beyond smoke budget through `train_v18_focus.py height`
  - scale the normal run beyond smoke budget through `train_v18_focus.py normal`
  - confirm live losses fall below the previous liquid-poisoned plateaus after the new manifest + mask path
