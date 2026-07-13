# Active Context — wow-viewer

Last updated: 2026-07-13

## Current target — Spec 102 strict 3.3.5 M0 reset

- Initial M0 is **3.3.5 only** and must cover every readable staged-client terrain map in provenance. A Northrend-only corpus is not a substitute; it may be a later ablation only after the complete corpus passes its gate. 0.5.3 is a separate future model.
- M0 is one small 3,043,041-parameter RGB-minimap → one strict object-visibility-mask model. It has no DepthAnything, extra heads, shared weights, or numeric side inputs. Its output is only a deterministic minimap cleaner input; it is not terrain reconstruction.
- The target is `object_geometry_visible_mask_257`, never `object_precise_mask_257`, reduced masks, bounds, circles, MCRF coverage, centroids, or a whole-placement erase. The C# source now emits `strict-geometry-terrain-liquid-fragment-trace-v3`: a variable-length numeric audit sidecar preserving transformed object XYZ, placement/asset/triangle identity, raw-MCVT three-node interpolation facts, liquid facts, classification, overlaps, unresolved placement facts, and a content hash. It is not a model input or image-shaped supervision.
- Each object fragment is compared with raw MCVT terrain elevation. Below-terrain fragments are removed individually; above-ground fragments remain. Liquid evidence uses WL → MCLQ → MH2O per pixel, with explicit failure for unreadable/unknown liquid. Initial M0 is dry-only: any liquid state/coverage rejects the tile until a valid-loss-mask phase exists.
- C# focused proof: 29 strict tests passed, including raw MCVT fragment tracing, stale-hash rejection, incomplete-trace preservation, water/unknown rejection, and WL/MCLQ/MH2O per-pixel precedence. Harvest Debug build passed.

## Full-map source gate

- Read-only staged discovery at `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft` reports 125 maps, 52 terrain-ready maps, and 5,471 occupied WDT locations. Raw V18 has 5,134 rows across the 52 terrain-ready maps; 367 locations lack required source signals.
- Eight readable maps are not tensor-pack-ready because canonical minimap RGB is absent: `ArgentTournamentDungeon`, `ArgentTournamentRaid`, `DalaranArena`, `development_nonweighted`, `ExteriorTest`, `OrgrimmarArena`, `QA_DVD`, and `WintergraspRaid`. Six production maps in that group also lack MCLY/MCAL, so deterministic texture/alpha composition cannot create canonical RGB. This is a staged-client source gap, not a parser defect or a reharvest fix.
- The M0 coverage audit must bind discovery → raw V18 → strict numeric store → curation → split and fail when any readable map/row is absent, target provenance is legacy/fallback-derived, water is non-dry/unknown, or a source gap remains. Therefore no CUDA, training, calibration, validation render, cleaner materialization, W1, H2, or 0.5.3 work is authorized now.

## Next proof owner

1. Complete the Python v3 trace-sidecar transport and all pre-CUDA gates.
2. Reharvest a real strict 3.3.5 corpus only when the staged-output route is available; inspect numeric target/trace facts before curation.
3. Supply canonical RGB for all eight readable source-gap maps, or explicitly revise the product input contract. Only then can full-map coverage authorize one fresh three-epoch M0 decision.

## Terrain reconstruction after M0

- Terrain supervision remains numeric mesh data: raw MCVT vertex topology/Z and paired 17×17 + 16×16 WDL lattice values. `height_257` and normals are projections/validation facts, not terrain images or model-forward inputs.
- Validation PNG/OBJ/mesh renders are post-inference observability only. Fixed-light viewer terrain-shadow captures can be training-time diagnostic/guidance after numeric H2 proof; they are not deployment input or replacement terrain labels.
- `ef99e715` is only a control-flow reference for the older simple trainer. The unified V25 trainer and all historical M0/H0/H1 metrics are invalid architecture/data-contract evidence.

## Other active boundaries

- New work belongs in `wow-viewer`; `gillijimproject_refactor` is read-only reference. Never use `H:\CLIENTS`; use staged clients only.
- Spec 080 remains the UI release owner. Spec 089/V23 and legacy V24/V25 lanes are historical/paused unless explicitly reopened; they cannot be used to bypass the Spec 102 target or full-map gates.
