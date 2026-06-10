## Overview

- **Goal**
  - Prototype the `WoWRollback` tool suite for selectively stripping placement unique IDs from converted ADTs, enabling "time-travel" map views.
- **Key Observations**
  - Existing exports (`AlphaWdtAnalyzer`) already emit `id_ranges_by_map.csv` summarizing placement unique ID spans per map.
  - ADTs currently written include fully patched area IDs; rollback should focus on placement records while preserving area data.
- **Constraints**
  - Must not mutate source `AlphaWdtAnalyzer`; instead, build a separate workflow under `WoWRollback/`.
  - Support map-specific configuration (ranges to keep/suppress) and batch processing.

## Inputs & Data Sources

We need to generate the UniqueID ranges for every alpha WDT that we have access to, even without decoding the data, base things off of AlphaWDTAnalysisTool, but build it as a standalone tool.

- **Converted ADTs**
  - Located under `output_wdt-export/` (or user-specified export root).
  - Contain placement `uniqueId` values within `M2` and `WMO` chunks.
- **Range Summary CSV**
  - `output_wdt-export/csv/id_ranges_by_map.csv` currently overwritten per run.
  - Action item: emit per-map CSV (e.g., `id_ranges_by_map_<map>.csv`) to retain history.
- **User Configuration**
  - Proposed formats:
    - YAML/JSON manifest listing `map`, `includeRanges`, `excludeRanges`.
    - CLI flags (`--keep-range`, `--drop-range`) supporting quick experiments.
- **Reference DBC Data (optional)**
  - Map names & localization for user-friendly CLI output.

## High-Level Architecture

- **Folder**: `WoWRollback/`
  - `RollbackCli/`: .NET console entry point.
  - `Core/`: shared libraries (ADT parsing/writing utilities, config loader, range filtering).
  - `Docs/`: specifications, usage examples.
- **Pipeline**
  1. Load user config + per-map range summaries.
  2. Enumerate ADT files for a target map.
  3. For each ADT:
     - Parse placement chunks (reuse/adapt logic from `AlphaWdtAnalyzer.Core.Export.AdtWotlkWriter` to locate MCNK/placement tables, but in read-only mode initially).
     - Filter `uniqueId` entries per configured keep/drop rules.
     - Option A: Write a modified ADT copy alongside original (e.g., `*.rollback.adt`).
     - Option B: Emit patch instructions (diff-like) for manual application.
  4. Produce summary report (counts per map, removed IDs, first/last removed).

## Feature Breakdown

1. **Range Catalog Enhancements (AlphaWdtAnalyzer)**
   - Update exporter to write `id_ranges_by_map_<map>.csv` per run.
   - Include metadata: placement type (`M2`, `WMO`), min/max per tile.
2. **Rollback Config Schema**
   - `Map` identifier (name or ID).
   - `Mode`: whitelist (`keep`) or blacklist (`drop`), with mutual exclusivity.
   - Ranges expressed as `[min, max]`, allow open-ended (`>=`, `<=`).
   - Optional presets ("alpha", "beta", etc.) referencing known historical layers.
3. **ADT Reader/Writer Module**
   - Leverage existing ADT parsing utilities (consider factoring reusable code from `AdtWotlkWriter.cs`).
   - Ensure zero data loss: copy untouched sections verbatim, modify only placement arrays.
4. **CLI UX**
   - `rollback-cli --map Kalimdor --config configs/kalimdor_alpha.yml --input-dir output_wdt-export/ --output-dir rollback_alpha/`
   - Verbose mode listing removed `uniqueId` ranges per tile.
5. **Validation Tooling**
   - Diff viewer comparing original vs. rollback ADT stats (counts, bounding boxes).
   - Optional preview: render placement positions via existing visualization pipeline.

## Risks & Mitigations

- **Risk**: Incomplete ADT parsing for placements may corrupt files.
  - **Mitigation**: Start with "dry-run" mode reporting would-be removals; add unit tests using known ADTs.
- **Risk**: Range definitions overlapping essential assets (e.g., doodads for navigation).
  - **Mitigation**: Provide reversible workflow (output copies). Encourage config presets curated by experts.
- **Risk**: Performance bottlenecks processing thousands of ADTs.
  - **Mitigation**: Parallel processing with safeguards; incremental runs per map.

## Open Questions

- Should the rollback operate on raw Alpha ADTs, converted 3.3.5 ADTs, or both?
- Do we need to support per-placement annotations (e.g., label ranges by build number)?
- How to best integrate with existing visualization/export logs for user guidance?

## Next Steps

1. Align on configuration schema and storage layout (`WoWRollback/`).
2. Prototype enhanced range export in `AlphaWdtAnalyzer` to produce persistent per-map summaries.
3. Build skeleton `RollbackCli` project with dry-run capability (report-only).
4. Implement safe ADT writer that clones files with filtered placements.
5. Validate on sample maps (e.g., `Kalimdor`, `EasternKingdoms`) and document workflow.