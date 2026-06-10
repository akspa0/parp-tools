# Development Repair Plan

Created: 2026-03-19

## Goal

Build a non-GUI reconstruction pipeline that takes the original 4.x `development` dataset and produces the best possible 3.3.5-compatible repair set in one repeatable pass.

This is not just a terrain converter.

The required end state is:

- audit all original source tiles and sidecars from `test_data/development/World/Maps/development`
- repair usable split ADTs into valid 3.3.5-style monolithic ADTs
- reconstruct missing or unusable terrain from WDL where no trustworthy ADT terrain remains
- use PM4 `MPRL` and later `MSCN` evidence to refine terrain and object reconstruction
- convert `WL*` files into proper 3.x liquid data
- reconstruct object layers with stable `UniqueID` stratification
- emit a repaired `WDT`, keep the original `WDL`, and document every fallback used per tile

## Source Of Truth

Primary fixed source paths:

- `test_data/development/World/Maps/development`
- `test_data/minimaps/development`
- `pm4-adt-test12/wmo_library.json`
- `pm4-adt-test12/modf_reconstruction/`

Reference-only downstream artifacts:

- `test_data/WoWMuseum/335-dev/World/Maps/development`
- `PM4ADTs/clean/`
- `PM4ADTs/wdl_generated/`

The museum 3.3.5 tiles and earlier generated outputs are useful comparison targets, not the source of truth.

## Current Facts

- The original 4.x `development` dataset includes split ADTs, PM4 files, and `WL*` liquid files.
- A subset of tiles are structurally bad in ways that are not all the same:
  - zero-byte root placeholders
  - roots without `MCIN`
  - repaired 3.3.5 exports with broken `MCNK.IndexX/IndexY`
  - tiles where only sidecars or WDL-era data remain useful
- Existing rollback tooling already covers many repair primitives:
  - split ADT merge
  - `MCNK` index repair
  - WDL-generated ADTs
  - minimap-to-`MCCV` painting
  - PM4 `MPRL` terrain patching
  - PM4 CK24/WMO reconstruction
  - global `UniqueID` reassignment
- Active core code now also has a first explicit PM4 coordinate-validation seam and a new development-dataset audit seam.

## Architecture Direction

The repair pipeline should be staged, deterministic, and tile-manifest driven.

Each tile should be classified first, then routed through the minimum necessary repair path.

### Tile Classes

- `healthy-split`: root + sidecars can be merged directly into a valid 3.3.5 ADT
- `index-corrupt`: terrain exists but `MCNK.IndexX/IndexY` or chunk ordering is broken
- `scan-only-root`: root terrain exists but cannot rely on `MCIN`
- `wdl-rebuild`: no trustworthy terrain root remains; generate terrain from WDL
- `manual-review`: no trustworthy terrain plus no reliable fallback inputs

### Repair Inputs By Priority

1. original root ADT terrain if structurally usable
2. `_tex0` and `_obj0` split sidecars
3. WDL-generated terrain when root terrain is absent or unusable
4. PM4 `MPRL` terrain intersections for terrain refinement
5. PM4 `MSCN` collision geometry for later terrain/object reconstruction passes
6. `WL*` liquids converted to `MH2O`
7. minimap-derived `MCCV` only on WDL-generated tiles that otherwise lack trustworthy texture/color data

## Output Contract

The repaired output set should contain:

- a valid 3.3.5 `development.wdt`
- one repaired monolithic `development_X_Y.adt` per output tile
- the original `development.wdl` copied alongside the repaired map
- per-tile manifests describing:
  - source files used
  - repair class
  - whether split merge was used
  - whether `MCNK` indices were repaired
  - whether WDL terrain generation was used
  - whether PM4 terrain patching was applied
  - whether `WL*` liquids were converted
  - whether minimap `MCCV` paint was applied

## Execution Phases

### Phase 1: Dataset Audit

Use the active CLI audit to inventory the source dataset.

Required outcomes:

- tile-by-tile presence of root ADT, `_obj0`, `_tex0`, PM4, and `WL*`
- identify zero-byte or chunkless roots
- identify roots missing `MCIN`
- identify tiles with broken `MCNK.IndexX/IndexY`
- flag tiles that have fallback inputs available for reconstruction

Current active command:

- `wowmapconverter development-analyze --json <report.json>`

### Phase 2: Merge And Normalize Usable Split Tiles

For tiles with usable split data:

- merge root + `_tex0` + `_obj0` into monolithic 3.3.5 ADTs
- normalize chunk ordering
- repair `MCNK.IndexX/IndexY` when mismatched
- keep existing texture layers and placements whenever they are valid

Reference implementation currently lives in rollback tooling:

- `WoWRollback.PM4Module/SplitAdtMerger.cs`
- `WoWRollback.Cli/Commands/RepairMcnkIndicesCommand.cs`

### Phase 3: WDL Terrain Generation For Missing Tiles

For tiles with unusable or absent terrain roots:

- generate 3.3.5 ADTs from WDL heights
- preserve holes where possible
- copy the original WDL alongside the final repaired output
- mark these tiles explicitly as synthetic terrain in the manifest

Reference implementation:

- `WoWRollback.PM4Module/Services/WdlToAdtGenerator.cs`

### Phase 4: PM4 Terrain Refinement

Refine generated or repaired terrain using PM4 evidence.

Required order:

1. apply validated `MPRL` terrain intersections first
2. add `MSCN`-driven refinement only after the active-core coordinate contract is validated

This phase should be opt-in at first and manifest every changed chunk/vertex range.

### Phase 5: WL To MH2O Liquids

Convert `WLW`, `WLM`, `WLQ`, and `WLL` inputs into 3.x-compatible liquid chunks.

Required outcomes:

- synthesize `MH2O` payloads for repaired/generate tiles
- do not overwrite better existing liquid data without an explicit precedence rule
- record liquid-source provenance in the manifest

Reference implementation:

- `src/WoWMapConverter/WoWMapConverter.Core/Formats/Liquids/WlToLiquidConverter.cs`

### Phase 6: PM4 CK24 Object Reconstruction

Object reconstruction should be layered, not all-or-nothing.

Targets:

- reconstruct WMO placements from CK24 groups where confidence is high
- allow WMO v17 outputs built from CK24 geometry when asset matching is unavailable
- preserve unmatched PM4 geometry as debug artifacts instead of dropping it
- stratify placements by `UniqueID` ranges to preserve approximate chronology/debug layering

Reference implementations:

- `WoWRollback.Core/Services/PM4/Pm4ModfReconstructor.cs`
- `WoWRollback.PM4Module/Program.cs` `convert-ck24-to-wmo`
- `WoWRollback.PM4Module/GlobalUniqueIdFixer.cs`

### Phase 7: WDT, Manifest, And Packaging

Finalize the repaired map set.

Required outputs:

- repaired `development.wdt`
- copied `development.wdl`
- full tile manifest JSON
- build summary markdown
- optional debug exports:
  - PM4-derived geometry
  - CK24 grouping dumps
  - terrain diff images

### Phase 8: Viewer Debug Integration

Do not block the repair CLI on viewer work, but plan for the viewer to consume the manifests.

Required viewer debug features:

- show tile repair class and provenance
- toggle repaired vs generated terrain
- `UniqueID` range slider/filter for placements
- PM4 layer toggles:
  - `MPRL`
  - `MSCN`
  - CK24 objects
  - reconstructed WMO/WMOv17 proxies

## Immediate Implementation Slice

The next concrete slice should be:

1. keep the new `development-analyze` command as the audit front door
2. add a first active CLI orchestration command that wraps:
   - split merge
   - `MCNK` index repair
   - WDL generation for missing tiles
   - `WL*` to `MH2O`
3. port only the stable rollback primitives needed for that path into active core or call them cleanly from a dedicated repair tool project
4. leave CK24/WMO reconstruction and viewer `UniqueID` range filtering as the next phase, not hidden scope

## Validation Requirements

- validate against the original `test_data/development` source set, not only museum exports
- do not claim terrain repair is correct from builds alone
- keep per-tile manifests so visual problems can be traced back to the exact fallback path used
- explicitly report when a tile is WDL-generated or PM4-refined rather than original-terrain-preserving

## Non-Goals For The First Slice

- perfect later-era texture fidelity for every damaged 4.x `_tex0` tile
- final polished PM4 rendering inside the viewer
- claiming MSCN semantics are fully solved before active-core validation exists
- silently replacing broken originals without provenance