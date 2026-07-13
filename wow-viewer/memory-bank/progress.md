# Progress — wow-viewer

Last updated: 2026-07-13

## 2026-07-13 — Spec 102 strict target and all-map gate

- Reset the initial model decision to **3.3.5 only, full readable-map provenance required**. No Northrend-only substitution and no 0.5.3 mixture; later ablations/era models are separate work.
- Confirmed the permitted M0 architecture is small and single-purpose: 3,043,041 parameters, raw RGB minimap input, one strict object-visibility-mask output, no DepthAnything, no extra heads, and no numeric side inputs.
- Implemented strict C# geometry labeling from transformed M2/WMO triangles. It compares every raster fragment against raw MCVT interpolation, removes only individually below-terrain fragments, retains above-ground fragments/overlaps, and fails a tile rather than inventing a fallback for unresolved geometry.
- Repaired strict liquid evidence to compose WL → MCLQ → MH2O per pixel. Unknown or unreadable liquid is explicit failure; initial M0 curation is dry-only.
- Added the `strict-geometry-terrain-liquid-fragment-trace-v3` numeric sidecar: transformed world coordinates, placement/asset/triangle identity, raw MCVT three-node evidence, liquid/terrain elevations, classification, asset table, unresolved placements, and a content hash. Serializers reject materialized targets without it and reject mutated sidecars.
- Local C# proof: 29 focused tests passed; `WowViewer.Tool.Harvest` Debug build passed.
- Read-only staged discovery at `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft`: 125 map records, 52 terrain-ready maps, 5,471 WDT locations, and 5,134 raw-V18 rows. Eight readable maps lack canonical minimap RGB (`ArgentTournamentDungeon`, `ArgentTournamentRaid`, `DalaranArena`, `development_nonweighted`, `ExteriorTest`, `OrgrimmarArena`, `QA_DVD`, `WintergraspRaid`); six production maps also lack MCLY/MCAL. There are 367 missing-required-source locations.
- This is a fixed staged-client source gap, not a parser bug and not reharvestable. The strict coverage gate must therefore block CUDA/training until canonical RGB is supplied or the product input contract is consciously changed.
- Python transport of the v3 variable-length trace into V18/numeric/curation/coverage is in progress. It must stay a provenance sidecar, never become an image tensor or M0 forward input.

## Durable historical boundary

- Commit `ef99e715` is useful only as a simple trainer-control-flow reference. It does not restore the invalid V25 target/data/model contract.
- Old `object_precise_mask_257`, numeric-v3, 2,059-row curation, old split/audit, checkpoints, calibration, and validation panels are transport/history evidence only. They cannot authorize M0, a cleaner, W1, H2, or a CUDA run.
- H0/H1/V23/V24/V25 results remain historical and non-comparable to the reset. Terrain targets going forward are numeric raw-MCVT vertex/lattice data; dense height/normal images and validation renders are derived observability, not the terrain model representation.
- Separate viewer/UI, PM4, and legacy data lanes are unchanged. New implementation remains in `wow-viewer`; `gillijimproject_refactor` is reference-only.
