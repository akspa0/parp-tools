# Parity Samples: M2 Render Parity Recovery

**Initial scope**: 3.3.5.12340 world M2 instances (trees, cutout/transparent doodads).

## Sample Table

| SampleId | Build | Map | TileX | TileY | ModelPath | ExpectedVisible | Notes |
|----------|-------|-----|-------|-------|-----------|-----------------|-------|
| elwynn-tree-01 | 3.3.5.12340 | Elwynn | 27 | 57 | WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/TREES/ELWYNNTREECANOPY03.M2 | True | Tree canopy, cutout alpha |
| elwynn-tree-02 | 3.3.5.12340 | Elwynn | 27 | 57 | WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/TREES/ELWYNNTREECANOPY02.M2 | True | Tree canopy variant |
| elwynn-tree-03 | 3.3.5.12340 | Elwynn | 27 | 57 | WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/TREES/ELWYNNTREE01.M2 | True | Tree trunk, opaque |
| tirisfal-tree-01 | 3.3.5.12340 | Tirisfal | 36 | 57 | WORLD/AZEROTH/TIRISFAL/PASSIVEDOODADS/TREES/TIRISFALTREE01.M2 | True | Undead tree, cutout |
| elwynn-sign-01 | 3.3.5.12340 | Elwynn | 27 | 57 | WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/ROADSIGN01.M2 | True | Signpost, transparent |
| elwynn-lamppost-01 | 3.3.5.12340 | Elwynn | 27 | 57 | WORLD/AZEROTH/ELWYNN/PASSIVEDOODADS/LAMPPOST01.M2 | True | Lamppost, opaque |

## Adding Samples

1. Add a row to the table above with all fields populated.
2. Verify the model exists in staged `3_3_5_12340` client data.
3. Run adapter probe and record output in `wow-viewer/output/tmp/m2-parity/`.
4. Update `LastResult` field once validated.