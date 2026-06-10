# Development Repair Pipeline Spec

Created: 2026-03-19

## Purpose

Define the command surface, data contract, routing rules, and outputs for reconstructing the original 4.x `development` map into a reproducible 3.3.5-compatible repair set.

## Fixed Inputs

Primary inputs:

- `test_data/development/World/Maps/development`
- `test_data/minimaps/development`
- `pm4-adt-test12/wmo_library.json`
- `pm4-adt-test12/modf_reconstruction/`

Reference inputs:

- `test_data/WoWMuseum/335-dev/World/Maps/development`
- `PM4ADTs/clean/`
- `PM4ADTs/wdl_generated/`

## Command Surface

### Active Audit Command

Command:

```text
wowmapconverter development-analyze [--input-dir <dir>] [--tile-limit <n>] [--json <path>]
```

Behavior:

- scans the original development source directory
- correlates root ADTs, `_obj0`, `_tex0`, PM4, and `WL*` files by tile
- inspects root terrain structure
- reports chunk health and likely repair route

Expected JSON output shape:

- map-level presence of `WDT` and `WDL`
- counts for root ADTs, sidecars, PM4, and `WL*`
- tile-level health for each discovered tile

### Planned Repair Orchestration Command

Target command surface:

```text
wowmapconverter development-repair \
  [--input-dir <dir>] \
  [--output-dir <dir>] \
  [--minimap-dir <dir>] \
  [--wmo-library <json>] \
  [--modf-reconstruction <dir>] \
  [--mode audit|repair|package] \
  [--tile <x_y>] \
  [--skip-pm4] \
  [--skip-wl] \
  [--skip-wdl-generate] \
  [--skip-uniqueid-fix] \
  [--manifest <path>]
```

Modes:

- `audit`: classify only, no output map changes
- `repair`: produce repaired ADTs/WDT/manifests
- `package`: produce the repaired set plus debug artifacts and summary docs

## Tile Classification Contract

Each tile must be assigned one repair class before repair begins.

### Classes

- `healthy-split`
  - root ADT has usable terrain chunks
  - merge split sidecars and preserve terrain
- `index-corrupt`
  - root terrain exists but chunk indices are wrong or duplicated
  - repair `MCNK.IndexX/IndexY`
- `scan-only-root`
  - no trustworthy `MCIN`, but scan-order `MCNK` recovery exists
  - preserve terrain with scan-order fallback and normalize on output
- `wdl-rebuild`
  - no trustworthy terrain root remains
  - generate a 3.3.5 ADT from WDL and fill supporting data from other sources
- `manual-review`
  - insufficient source evidence for automatic repair

### Minimum Tile Signals

Per tile, record:

- root ADT present and byte length
- `_obj0` present and byte length
- `_tex0` present and byte length
- PM4 present and byte length
- `WLW`, `WLM`, `WLQ`, `WLL` presence and byte lengths
- root status:
  - `missing`
  - `zero-byte`
  - `mcin-valid`
  - `mcin-partial`
  - `scan-only`
  - `no-mcnk`
- `MCNK` chunk counts and header-index mismatch counts
- recommended action string

## Repair Routing Rules

### Rule 1: Preserve Original Terrain When It Is Structurally Usable

Use original root terrain first when:

- the tile has usable `MCNK` data
- chunk count is sufficient to reconstruct a monolithic tile
- any `MCNK.IndexX/IndexY` damage can be repaired deterministically

Required actions:

- merge split tile
- normalize or repair chunk index metadata
- preserve existing texture and placement sidecars when trustworthy

### Rule 2: Use WDL Generation Only For Missing Or Unusable Terrain

Use WDL-generated ADTs when:

- root ADT is missing
- root ADT is zero-byte
- root ADT has no usable terrain chunks
- original terrain is too corrupted to preserve confidently

Constraints:

- generated tiles must be explicitly marked as synthetic in the manifest
- minimap-derived `MCCV` paint may be used only on generated tiles lacking trustworthy texture/color data

### Rule 3: PM4 Terrain Refinement Is Secondary, Not The Base Terrain Source

PM4 data should refine terrain, not replace the terrain generation contract.

Phase order:

1. validated `MPRL` refinement
2. later `MSCN` refinement after active-core validation exists

### Rule 4: WL Files Feed Liquids, Not Terrain Heights

Use `WL*` data to synthesize 3.x liquid chunks.

Required behavior:

- convert `WLW`, `WLM`, `WLQ`, and `WLL` to `MH2O`
- avoid overwriting better existing liquid data without an explicit precedence decision
- record liquid provenance in the manifest

### Rule 5: UniqueID Stratification Must Be Preserved As Debug Metadata

The pipeline must preserve or regenerate a stable placement `UniqueID` story.

Required outputs:

- final placement `UniqueID`
- original/reconstructed source of that `UniqueID`
- optional range bucket or chronology layer label for viewer/debug filtering

## Output Layout

Target repair output root:

```text
<output>/
  development.wdt
  development.wdl
  World/Maps/development/
    development_X_Y.adt
  manifests/
    summary.json
    tiles/
      development_X_Y.json
  debug/
    terrain-diffs/
    pm4/
    ck24/
    liquids/
  docs/
    repair-summary.md
```

## Tile Manifest Contract

Each tile manifest must include:

- tile coordinates
- repair class
- source files used
- whether split merge ran
- whether `MCNK` index repair ran
- whether WDL generation ran
- whether PM4 `MPRL` patching ran
- whether PM4 `MSCN` patching ran
- whether `WL*` conversion ran
- whether minimap `MCCV` paint ran
- warnings or unresolved issues

## Viewer Follow-Up Contract

The viewer is not the repair engine, but it must be able to consume repair metadata.

Required viewer capabilities after the repair pipeline exists:

- show the repair class for the active tile
- toggle generated terrain vs preserved terrain
- filter placements by `UniqueID` range
- show PM4 debug layers independently:
  - `MPRL`
  - `MSCN`
  - CK24 groups
  - reconstructed WMO proxies

## Existing Code Seams To Reuse

### Active Codebase

- `src/WoWMapConverter/WoWMapConverter.Core/Services/DevelopmentMapAnalyzer.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/WlToLiquidConverter.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4CoordinateService.cs`
- `src/WoWMapConverter/WoWMapConverter.Core/Formats/PM4/Pm4CoordinateValidator.cs`

### Rollback Reference Implementations

- `WoWRollback/WoWRollback.PM4Module/SplitAdtMerger.cs`
- `WoWRollback/WoWRollback.PM4Module/Services/WdlToAdtGenerator.cs`
- `WoWRollback/WoWRollback.PM4Module/GlobalUniqueIdFixer.cs`
- `WoWRollback/WoWRollback.Core/Services/PM4/Pm4ModfReconstructor.cs`
- `WoWRollback/WoWRollback.Cli/Commands/DevelopmentRepairCommand.cs`
- `WoWRollback/WoWRollback.Cli/Commands/RepairMcnkIndicesCommand.cs`

## Explicit Non-Claims

This spec does not claim:

- full 4.x texture parity is already solved
- MSCN semantics are fully validated in active core
- all museum 3.3.5 exports are clean references
- WMO v17 generation from CK24 is already production-grade for every object class

## First Delivery Target

The first production-worthy delivery from this spec is:

- an auditable repair manifest for the full original development dataset
- a deterministic repaired 3.3.5 map set with explicit synthetic/generated tile labeling
- enough metadata for the viewer to later filter by repair provenance and `UniqueID` range