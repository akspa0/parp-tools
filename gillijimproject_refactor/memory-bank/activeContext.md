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

## V18 Distill Corpus and Open-Source Release Loop (spec 047)

- spec `047-v18-distill-corpus-open-source-loop` is the active owner for the focused 0.5.3 + 3.3.5 corpus and the open-source release loop.
- architecture doc: `wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md`.
- key decisions:
  - trim the harvest to `0_5_3_3368` and `3_3_5_12340` only — the other four builds stay out of scope.
  - promote renderer-truth object-mask coverage to a first-class V18 signal (reuses `WowViewer.Tool.ValidationCapture capture-batch`).
  - keep the V16.1 / V18 model line as the teacher; no architecture changes.
  - procedural synthesizer → main-model distillation → small open-source student under MIT/Apache 2.0.
- the main model and the focused real-data corpus stay in-repo under the Bring Your Own Data policy; only the student and the labeled synthesized corpus are distributable.
- superseded drafts now reroute to spec 047:
  - `015-v16-1-2-height-derived-normal-refiner` (refiner failed)
  - `017-v16-1-4-combined-normal-height-model` (combined head not used)
  - `022-v17-unified-normal-height-refiner` (V17 hybrid folded)
  - `023-v17-1-global-minimap-signal-reconstruction` (V17.1 contract is V16.1 + V18)
