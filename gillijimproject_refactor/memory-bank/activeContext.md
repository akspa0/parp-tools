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

## Data Harvester / Training Continuity

- V16.1.1 remains a bounded normal-terrain lane.
- prefer startup autotune and larger batch ladders when VRAM headroom allows.
- validation artifacts are only authoritative when regenerated after behavior changes.

## V18 Focused Two-Build Minimap-to-Terrain Loop (spec 047)

- spec `047-v18-distill-corpus-open-source-loop` is still the active owner for the focused 0.5.3 + 3.3.5 V18 lane, but the active contract was reset again on 2026-06-04.
- architecture doc: `wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md`.
- active proof owner:
  - use `minimap_rgb` only as model input
  - train `height_257` with plain L1
  - train `normal_xyz` with masked cosine against `normal_mask`
- explicitly out of scope for the active iteration:
  - renderer-truth capture as training truth
  - object-mask / roof-mask / liquid-derived loss weighting
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
- bounded proof now exists:
  - height smoke run: `wow-viewer/models/v18/height/runs/v18_height_focus_minimap_smoke_20260604_r2/`
    - 1 epoch, batch size 4, 32 train tiles, 8 val tiles
    - `val_loss = 0.6626`
  - normal smoke run: `wow-viewer/models/v18/normal/runs/v18_normal_focus_minimap_smoke_20260604_r2/`
    - 1 epoch, batch size 4, 32 train tiles, 8 val tiles
    - `val_loss = 0.2251`
  - one leftover seam was exposed and fixed during that proof:
    - `_preview_normal(...)` still expected old weighted-loss tensors and crashed after a successful epoch
    - it now falls back cleanly to simplified-lane preview tensors (`base_mask`, `train_mask`, `invalid_mask`)
- the next real proof is:
  - validate the focused stores for minimap/height/normal readiness
  - scale the height run beyond smoke budget
  - scale the normal run beyond smoke budget
