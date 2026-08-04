# Progress — wow-viewer

Last updated: 2026-08-04

**This file is a dated ledger of what shipped, newest first.** One entry per session, a few lines
each. Findings and how-it-works go in the workstream file; this only records *that* it happened and
what the evidence was. See "Memory bank layout" in `coding_standards.md`.

Current state and open work: [activeContext.md](activeContext.md).

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

## Before 2026-08-01

Condensed into [archive/2026-08-01-progress-detail.md](archive/2026-08-01-progress-detail.md) at the
feature-complete declaration and spec audit. Older session history is in
[archive/](archive/README.md).
