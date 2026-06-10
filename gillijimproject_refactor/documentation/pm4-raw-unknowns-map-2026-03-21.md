# PM4 Raw Unknowns Map (Mar 21, 2026)

This document is the standalone-reader counterpart to the active viewer contract in [documentation/pm4-current-decoding-logic-2026-03-20.md](documentation/pm4-current-decoding-logic-2026-03-20.md).

Its job is narrower:

- record what the raw PM4 corpus actually proves today
- separate verified links from partial fits and weak hypotheses
- keep the open unknowns explicit instead of letting them hide inside reconstruction heuristics

The data below comes from the standalone PM4 research path and the fixed development corpus at `test_data/development/World/Maps/development`.

## Commands

- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-audit --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mslk-refindex-classifier --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-msur-geometry --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-structure-confidence --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-linkage --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-mscn --input test_data/development/World/Maps/development`
- `dotnet run --project src/Pm4Research.Cli/Pm4Research.Cli.csproj -- scan-unknowns --input test_data/development/World/Maps/development`
- `dotnet run --project src/WoWMapConverter/WoWMapConverter.Cli/WoWMapConverter.Cli.csproj -- pm4-validate-coords --input test_data/development/World/Maps/development --json output/pm4_reports/development_pm4_coordinate_validation_report.json`

## Corpus Shape

- total PM4 files scanned: `616`
- non-empty PM4 files: `309`
- typed chunk walk diagnostics: none found in the current corpus run
- unknown chunk signatures after current typed coverage: none found in the current corpus run

Chunk population summary:

- recurring general PM4 structure is carried by `MSHD`, `MSLK`, `MSPV`, `MSPI`, `MSVT`, `MSVI`, `MSUR`, `MSCN`, `MPRL`, and `MPRR`
- `MDBI` and `MDBF` are one-tile only in the current development corpus
- `MDBH`, `MDOS`, and `MDSF` appear as chunk containers in many files, but only one tile currently carries populated destructible-building payloads
- the trusted `development_00_00.pm4` tile is still the main destructible/Wintergrasp reference tile, not the general PM4 baseline

## Decode-Trust Snapshot

The new standalone `scan-structure-confidence` report is the guardrail against inherited GPT-era PM4 lore being mistaken for proven structure.

Current corpus result:

- audited chunk-layout confidence is strong across the active typed set: `13` tracked chunk families currently land in `high` layout confidence, with zero stride-remainder files in the fixed corpus for those layouts
- field semantics are much weaker than the stable strides make them look:
  - `2` tracked fields currently land in `high` semantic confidence
  - `4` land in `medium`
  - `9` land in `low`
  - `4` land in `very-low`
- current strongest byte+semantic anchors are:
  - `MSPV`, `MSPI`, `MSVT`, `MSVI`
  - `MSUR.Normal + signed plane term`
  - `MSUR.MsviFirstIndex + IndexCount -> MSVI`
  - `MDSF -> {MSUR, MDOS}` on the populated destructible tile
- current highest hallucination-risk fields are:
  - `MSLK.RefIndex`
  - `MPRR.Value1`
  - `MPRL.Unk04`, `Unk14`, `Unk16`
  - `MDOS.buildingIndex`

Practical reading of that result:

- current standalone PM4 decoders are mostly reading the byte layouts correctly
- the main remaining hallucination vector is semantic naming and over-closure, not raw stride parsing
- future PM4 work should explicitly distinguish:
  - byte layout confidence
  - derived/verified reference behavior
  - field names that are still only provisional

## Explicit Source Conflicts

The structure-confidence report now keeps a live conflict inventory for places where older notes or inherited field names overstate certainty.

Current conflict set includes:

- `MSLK.LinkId`:
  - older constant-field lore is too strong
  - current corpus shows a stable data-bearing tile-link envelope, not a meaningless constant placeholder
- `MSLK.RefIndex`:
  - current standalone decoder keeps the byte position, but the semantics are not closed as a universal `MSUR` index because `4553` entries fail that mapping
- `MSUR.Normal + Height`:
  - current code names the floats, but no standalone corpus report yet proves those bytes are geometric normals plus height
- `MSUR.MdosIndex`:
  - strong link into `MSCN`, but not a fully closed scene-node contract because `6201` references still miss
- `MSUR.Normal + Height`:
  - older notes treated this as uncertain naming
  - current standalone geometry audit now shows that bytes `4..15` behave like true unit surface normals and the trailing float behaves like the negative plane-distance term along that normal, not a generic axis height
- MSCN coordinate frame:
  - current standalone evidence does not support re-importing the old blanket “swap XY and you are done” rule
- `MPRR.Value1`:
  - still fits multiple target domains heavily and remains semantically open

## Geometry-Validated `MSUR` Plane Fields

Current standalone corpus findings from `scan-msur-geometry`:

- analyzed surfaces: `518092`
- degenerate surfaces: `0`
- stored normal magnitude is unit-like on every analyzed surface: `518092 / 518092`
- stored normal alignment against geometry-derived polygon normal is strong and positive on every analyzed surface: `518092 / 518092`
- best height candidate is not a raw centroid axis; it is the negative plane-distance term along the stored normal:
  - candidate: `storedPlane.-`
  - mean absolute error: `0.00367829`
  - fits within `0.1`: `517960`
  - fits within `1.0`: `518092`

Current interpretation:

- current standalone evidence now strongly supports treating `MSUR` bytes `4..15` as true surface normals
- the trailing float currently named `Height` is much better described as a signed plane-distance term, with the sign inverted relative to `dot(normal, centroid)`
- this closes one major hallucination-risk seam in the raw decoder, while also showing that the existing property name `Height` overstates the wrong semantic shape

## `MSLK.RefIndex` Family Classifier

Current standalone corpus findings from `scan-mslk-refindex-classifier`:

- files with mismatches: `150`
- total mismatch rows: `4553`
- classified/resolved families: `505`
- ambiguous families still remaining: `344`
- resolved mismatch rows covered by classified families: `2651`

Important baseline result:

- `MPRR` still fits almost every mismatch row by raw size alone (`4481 / 4553`, `98.4%` coverage)
- because of that, raw fit counts are too biased to classify mismatch families directly
- the classifier instead scores family-local coverage above corpus baseline so oversized domains do not win just by being big

Current classified-family shape:

- largest resolved family population: `probable-MSVT` (`293` families)
- other resolved slices include:
  - `possible-MSPV`: `59`
  - `probable-MPRL`: `36`
  - `probable-MSPI`: `28`
  - `possible-MSVI`: `26`
  - `probable-MSVI`: `20`
  - `probable-MSCN`: `10`
  - `probable-MSPV`: `10`

Current interpretation:

- the unresolved `MSLK.RefIndex` population is no longer one undifferentiated blob
- a large share now clusters into likely geometry/path/vertex-facing families, with a smaller but real `MPRL`-specific slice
- this still does **not** close the field semantically, but it is much stronger than treating every mismatch row as equally open

## Placement-Truth Validation For `MPRL`

Current real-data result from the existing active `pm4-validate-coords` validator against fixed `_obj0.adt` placements:

- tiles scanned: `616`
- tiles validated: `206`
- skipped without `_obj0.adt`: `217`
- skipped without placements: `193`
- total `_obj0.adt` placements: `59723`
- total `MPRL` refs checked: `114301`
- refs inside expected tile bounds: `114301 / 114301` (`100.0%`)
- refs within `32` units of nearest placement: `107907 / 114301` (`94.4%`)
- average nearest placement distance: `10.98`

Current interpretation:

- this materially strengthens `MPRL.Position` as real placement/footprint data against actual map-object placements on the fixed development dataset
- this validation is about `MPRL`, not `MPRR`; it does not by itself close `MPRR` graph semantics

## Verified Relationships

These edges are currently strong enough to treat as verified in the raw decode path:

- `MSUR.MsviFirstIndex + IndexCount -> MSVI` fits all observed surfaces
- `MSVI -> MSVT` fits all observed mesh indices
- `MSLK.MspiFirstIndex + MspiIndexCount -> MSPI` fits all active link windows in indices-mode
- `MSPI -> MSPV` fits all observed path indices
- `MDSF.MsurIndex -> MSUR` fits all populated destructible mappings
- `MDSF.MdosIndex -> MDOS` fits all populated destructible mappings
- `LinkId` uses the `0xFFFFYYXX` tile-link pattern everywhere in the current corpus run

These are the strongest current raw “plug into each other” links.

## Partial Relationships

These edges are real but not semantically closed yet:

### `MSLK.RefIndex`

- `MSLK.RefIndex -> MSUR` fits most links but fails on `4553` entries across `150` files
- `development_00_00.pm4` has zero `RefIndex -> MSUR` mismatches
- mismatch candidate domains on those bad values:
  - `MSPI`: `3006`
  - `MSVI`: `2902`
  - `MSCN`: `2670`
  - `MSLK`: `2632`
  - `MSPV`: `2406`
  - `MSVT`: `2055`
  - `MPRR`: `4481`
  - `MPRL`: only `86`

Current interpretation:

- `RefIndex` is not a generally reliable raw `MSUR` index
- the unresolved mismatch population is very unlikely to be “secretly MPRL” in the general case
- stronger next candidates are `MSPI`, `MSVI`, `MSCN`, and `MSLK`

### `MSCN`

Current standalone corpus findings:

- `MSUR.MdosIndex -> MSCN` fits `511891` times and misses `6201`
- `1886 / 1895` CK24 groups have at least one valid MSCN-backed `MdosIndex`
- the same `1886` CK24 groups also have both mesh-side and MSCN-side geometry in the current corpus run
- raw MSCN bounds overlap CK24 mesh bounds much more often than swapped-XY MSCN bounds:
  - raw overlap: `1162` fits, `724` misses
  - swapped XY overlap: `10` fits, `1876` misses
- `MSLK.GroupObjectId` does **not** behave like full CK24 ownership:
  - low 24 bits -> `CK24`: `0` fits, `1272796` misses
  - low 16 bits -> `CK24ObjectId`: `399` fits, `1272397` misses

Current interpretation:

- in the standalone raw decode path, `MSCN` is strongly linked into CK24 families through `MSUR.MdosIndex`
- current corpus evidence does **not** support the old broad claim that MSCN should be XY-swapped to align with mesh-side PM4 geometry in this raw path
- current corpus evidence also does **not** support treating `MSLK.GroupObjectId` as a direct full-CK24 identifier
- `MSCN` still looks like a real collision/ownership layer, but not one that can be collapsed into a simple “world-space swapped copy” rule without more proof

### `Ck24ObjectId`

Current raw decoding rule:

- the UI `Ck24ObjectId` is not a separate PM4 field
- it is the low 16 bits of `MSUR.PackedParams -> Ck24`

Current standalone corpus findings:

- distinct full `CK24` values: `1229`
- distinct `Ck24ObjectId` values: `1215`
- analyzed non-zero object-id groups: `1601`
- object-id groups reused across multiple full `CK24` values in the same file: `2`
- object-id groups reused across multiple `CK24` type bytes in the same file: `2`
- both current reuse cases are on tile `36_24`:
  - `obj=47331` across `0xBFB8E3` and `0x3FB8E3`
  - `obj=43690` across `0x3EAAAA` and `0xBEAAAA`

Current interpretation:

- `Ck24ObjectId` is usually a near one-to-one slice of full `CK24` inside a file
- the two reuse cases are still important because they show the low16 layer can survive when the high/type bits change
- this is enough to treat `Ck24ObjectId` as a derived identity slice, but not enough to claim it is the hidden general hierarchy system by itself

### `MSLK.GroupObjectId`

Current standalone corpus findings on the unresolved `MSLK.RefIndex` mismatch population (`4553` entries):

- `GroupObjectId low16 -> Ck24ObjectId`: only `16` fits
- `GroupObjectId low24 -> CK24`: only `15` fits

Current interpretation:

- `GroupObjectId` does not currently behave like the missing direct ownership key for the unresolved `RefIndex` population
- it may still represent a narrower member/group/hierarchy layer, but the current corpus does not justify calling it a direct CK24 object id or full CK24 key

### `MSLK.GroupObjectId -> MPRL.Unk04`

- direct entry-level overlap exists (`65819` fits), but the majority of link entries do not have a same-file `MPRL` key match
- this is still consistent with a real relationship because many files lack populated `MPRL` payload or have links outside the linked-footprint subset

Current interpretation:

- treat this as a real relationship, but not as a one-to-one entry mapping claim yet

### `MPRR.Value1`

- non-sentinel `Value1` fits `MPRL` often, but also fits `MSVT` even more often
- current counts do not close `MPRR` as “purely MPRL” or “purely geometry”

Current interpretation:

- `MPRR` remains mixed or overloaded until clustered by tile/family/value patterns

## Field Findings

### `MSLK`

- `TypeFlags` has `10` observed values in the current corpus run
- dominant values: `0x01`, `0x02`, `0x0A`, `0x04`, `0x0C`
- `Subtype` has `19` observed values, heavily concentrated in `0`, `1`, `2`, `3`
- `SystemFlag` is always `0x8000`

Current interpretation:

- `SystemFlag` is effectively constant in current data
- `TypeFlags` and `Subtype` are still open semantic layers and should be correlated with mismatch families, floors, and object classes instead of being guessed from names alone

### `MSUR`

- `IndexCount` is dominated by `3` and `4`, with real use of `5`, `6`, `7`, and above
- `AttributeMask` has `25` observed values
- `GroupKey` has `7` observed values, dominated by `3`, `18`, `16`, and `19`

Current interpretation:

- `IndexCount` behaves like real polygon/surface membership, not noise
- `AttributeMask` and `GroupKey` remain open and need correlation against object family or scene role

### `MPRL`

- `Unk02` is always `-1`
- `Unk06` is always `0x8000`
- `Unk14` spans `-1..15`, concentrated in `-1`, `0`, `1`, `2`, `3`
- `Unk16` has only two observed values: `0x0000` and `0x3FFF`

Current interpretation:

- `Unk02` and `Unk06` behave like constants in the current corpus
- `Unk14` looks floor- or level-like
- `Unk16` looks class- or terminator-like

### `MSHD`

- `Field00`, `Field04`, and `Field08` all have large distinct-value counts
- `Field00` and `Field08` often share dominant values, but are not universally identical

Current interpretation:

- `MSHD` is still structurally decoded but semantically open
- these fields should be correlated with tile shape/population and not named prematurely

### MSCN Coordinate Space

Current standalone corpus counts:

- total MSCN points: `1342410`
- swapped-world tile fits: `0`
- raw-world tile fits: `0`
- ambiguous tile fits: `0`
- tile-local-like points: `9990`
- neither bucket: `1332420`

Current interpretation:

- the standalone raw PM4 path does not currently support a simple “MSCN is already world-space” rule across the development corpus
- it also does not support a broad “MSCN is simple tile-local 0..533 data” rule outside one tile-local-like subset
- use this as a guardrail against re-importing old rollback-era coordinate claims without revalidating them in the standalone path

### `MSUR.MdosIndex` Failure Families

Current standalone corpus findings:

- `58` files carry bad `MSUR.MdosIndex` references
- the bad population is not only `CK24=0` aggregate data; large non-zero families also fail heavily, for example:
  - tile `0_0`, `CK24=0x40AA0A`, `invalid=725`, `valid=171`
  - tile `35_23`, `CK24=0x428AF0`, `invalid=331`, `valid=2512`
  - tile `36_19`, `CK24=0x428AD9`, `invalid=327`, `valid=2516`
  - tile `16_48`, `CK24=0x433C94`, `invalid=182`, `valid=2`

Current interpretation:

- the bad `MdosIndex` population is not just terrain/noise from `CK24=0`
- some object-like non-zero CK24 families are carrying real mixed valid/invalid MSCN linkage, which makes them strong candidates for alternate scene-node semantics or partial decode drift

## Ambiguities Still Open

### `MSLK.MspiIndexCount`

Current corpus counts:

- active links: `598882`
- indices-only valid: `399183`
- triangles-only valid: `0`
- both modes valid: `199699`
- neither valid: `0`

Current interpretation:

- the current data does not support a triangles-only model
- ambiguity remains because many links are valid under both interpretations
- indices-mode is the stronger current default, but this is not final semantic closure

### Destructible Payload

- `MDBH` / `MDOS` / `MDSF` are only meaningfully populated on one tile in the current development corpus
- `MDOS.buildingIndex -> MDBH` fits only `1` entry and misses `24`

Current interpretation:

- `MDOS.buildingIndex` is not yet proven to be a direct `MDBH` slot
- likely alternatives include indirection through `MDBI`, a hashed/object key, or another compact identifier

## Current Unknown List

- exact alternate semantics of the mismatch population in `MSLK.RefIndex`
- exact raw/local/world coordinate ownership of `MSCN`
- exact semantics of the `MSUR.MdosIndex` miss population
- final meaning of `MSLK.TypeFlags`
- final meaning of `MSLK.Subtype`
- exact role of `MSLK.GroupObjectId`
- exact semantics of `MSLK.MspiIndexCount` in the overlap cases where both interpretations fit
- exact semantics of `MPRL.Unk14`
- exact semantics of `MPRL.Unk16`
- exact semantics of `MSHD` fields
- exact semantics of `MPRR.Value1` and `MPRR.Value2`
- exact destructible-building key mapping in `MDOS`
- final PM4 local/world frame ownership relative to ADT/object truth

## Strong Negative Results

- the general unresolved `MSLK.RefIndex` population is not well explained by `MPRL`
- the standalone raw corpus does not support the older blanket claim that MSCN must be XY-swapped to line up with PM4 mesh geometry
- the general unresolved `MSLK.RefIndex` mismatch population is not well explained by `MSLK.GroupObjectId -> CK24` identity matches either
- `00_00` is not the main tile for the `MSLK.RefIndex` mismatch problem
- chunk presence alone is not proof of populated destructible-building payload

## Next Proof Tasks

1. Cluster bad `MSLK.RefIndex` entries by `LinkId`, `TypeFlags`, and repeated value bands.
2. Compare mismatch families against `MSPI`, `MSVI`, `MSCN`, and `MSLK` domains directly, but keep `GroupObjectId` as a weaker secondary hint rather than a presumed identity field.
3. Break the `6201` bad `MSUR.MdosIndex` cases down by CK24 family and tile to see whether they are placeholder nodes, alternate scene layers, or decode drift.
4. Inspect the two reused `Ck24ObjectId` cases on tile `36_24` and determine whether the type-byte flip there is a real hierarchy/state transition or only signed/high-bit variation.
5. Correlate `MPRL.Unk14` and `Unk16` against trusted floor/object truth on `00_00` and other externally validated tiles.
6. Resolve `MDOS.buildingIndex` using the Wintergrasp/destructible reference tile and external placement truth.
7. Use the verified raw edges above as constraints during PM4/ADT correlation, not as post-hoc support for viewer heuristics.