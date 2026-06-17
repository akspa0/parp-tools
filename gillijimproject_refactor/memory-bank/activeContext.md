# Active Context

## Direction
WoW viewer. Libraries bridge to Unreal Engine. No Vulkan/WebGL/Museums/BASE.

## 2026-06-17 — PM4 fingerprint→WMO identity matching proven

**`pm4 identify-models` command built and validated.** Matches PM4 fingerprint groups against WMO archive local bounds (MOHD BoundsMin/BoundsMax) using sorted-dimension similarity.

**Results from 616 PM4s vs 506 WMO roots**: 1223 matches (score >= 0.30), 545 matches (score >= 0.95), 972 matches (score >= 0.90). Top match: GoldshireInn.wmo at 0.996 score — exact 30x32x60 dimension match.

**Key discoveries**:
- Sorted dimension ratio matching works — WMO local AABB dimensions are rotation-invariant identifiers
- Type pairs confirmed: 0x40/0x41 = same M2 (collision+visual), 0x42/0x43 = same WMO (exterior+interior)
- 0xC0/0xC1/0xC2/0xC3 also match WMOs — navmesh/interior collision variants sharing same model bounds
- Multi-tile objects need separate handling — fingerprint changes per tile for same OID
- 506/1985 WMOs scanned — archive enumeration missed ~75%. Need listfile-based enumeration for full coverage.
- 304 unique WMOs matched across all types

**Ck24ObjectId is a global object identifier spanning tiles.** Same ObjectId on multiple tiles = same physical object (OID 52202 on 8 tiles = one large WMO). Different ObjectIds with the same (surfaces, indices, vertices) fingerprint = different instances of the same model.

**`pm4 fingerprint-scan` command added.** Reads all 616 development PM4s, extracts 1604 CK24 groups with fingerprints. Key discovery:
- **611/616 PM4s use WorldSpace (absolute) coordinates** — only 5 use TileLocal. `Pm4PlacementMath.IsLikelyTileLocal()` detects this correctly.
- **1162 distinct fingerprints** across all groups. Most common: `(35, 144, 90)` appears 97 times (common WMO wall segment).
- **272 ObjectIds appear on 2+ tiles**. Top: OID 52202 spans 8 tiles, OID 43196 spans 8, OID 44166 spans 7.
- **Multi-tile reconstruction**: combining per-tile bounding boxes across tiles reconstructs the full model bounds.

**CK24 type distribution (616 PM4s)**: 0x40(M2-a)=80, 0x41(M2-b)=161, 0x42(WMO-a)=584, 0x43(WMO-b)=466, 0xC0=77, 0xC1=100, 0xC2=66, 0xC3=38, 0x3E=5, 0x3F=10, plus rare 0x3D/0xB6/0xBD/0xBE/0xBF.

**Next step**: Match fingerprint groups against WMO archive by local bounding box dimensions (sorted, rotation-invariant). Same model at different placements produces same fingerprint + same sorted dimensions.

## What's Done
012, 014, 024, 025, 033, 037, 041, 043, 044 (P1), 048, 054, 058, 059, 060, 061, 062

## What's Not Started
001, 029, 030/031/032 (research), 038/040 (research), 042, 045, 049, 053, 055, 056, 057

## Biggest Unproven Gap (046)
Full WMO enumeration (1985 WMOs instead of 506) to improve identity coverage. Multi-tile OID tracking to reconstruct full model bounds from per-tile fragments. M2 type (0x40/0x41) collision vertex reading.

## Staged Clients
Only `output/tmp/wowarchive-clients/` paths are valid.

## Known Issues
- Viewer click-freeze on dense PM4 (timing shipped, numbers pending)
- Some 0.5.3 Alpha `.wdt.MPQ` fail to parse
- pm4 identify-models only scans 506/1985 WMO roots — archive enumeration misses ~75% of WMOs
- pm4 correlate-models only produces hits on tiles where ADT placements overlap PM4 geometry (1/50 tiles)
- Alpha WDT write fails on placement-heavy tiles (>14999 bytes)
- 14 pre-existing test failures (stale ChunkedFileReader fixtures)