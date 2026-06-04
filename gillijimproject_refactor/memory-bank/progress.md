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

- **V18 distill corpus and open-source release loop planning pack (2026-06-04)**
  - created spec `047-v18-distill-corpus-open-source-loop`
  - architecture doc: `wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md`
  - focused owner:
    - trim the V18 harvest to `0_5_3_3368` and `3_3_5_12340` only
    - expand renderer-truth object-mask capture from one anchor tile per build to the full focused corpus
    - keep the existing V16.1 / V18 model line as the teacher (no architecture changes)
    - procedural synthesizer → main-model distillation → small open-source student under MIT/Apache 2.0
  - explicit non-goal: do not redesign the V18 model or expand to all six builds.
  - superseded drafts reroute to spec 047:
    - `015-v16-1-2-height-derived-normal-refiner`
    - `017-v16-1-4-combined-normal-height-model`
    - `022-v17-unified-normal-height-refiner`
    - `023-v17-1-global-minimap-signal-reconstruction`

## Next Likely Steps

- implement the first `046` slice: deterministic non-freezing PM4 segment export
- validate `ck24ObjectId`-rooted segmentation against known reference tiles before broad automation
- build the staged-asset Zarr signal corpus and deterministic candidate scorer
- keep PM4 grouping/spec docs in sync if segmentation ownership changes again
- start spec `047` Phase A1: focused two-build V18 build (`build_focused_two_build_corpus.py`) against staged `0_5_3_3368` and `3_3_5_12340`
