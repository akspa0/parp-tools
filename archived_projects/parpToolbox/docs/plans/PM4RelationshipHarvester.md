# PM4 Relationship Harvester
MSPI vs MSVI/MSUR/MSLK/MSCN Analysis Plan

Last updated: 2025-08-20

## Purpose
Determine whether MSPI represents an alternate or distinct detail layer relative to MSVI/MSUR by harvesting structured relationships and producing CSV outputs suitable for downstream analysis. This “Relationship Harvester” runs within the existing PM4MscnAnalyzer framework, using `Pm4Adapter` to load tiles/regions with the right flags and output comprehensive CSVs.

## Scope
- Load one or more PM4 tiles (merged into a global scene) with MSCN remapping.
- Compute per-tile statistics for MSPI/MSVI.
- Validate and correlate MSLK ranges against MSPI.
- Compare MSUR surfaces (MSVI triangle ranges) against MSPI triangle sets.
- Run spatial proximity tests against MSCN vertices to see which source (MSPI vs MSVI) aligns more closely.
- Emit CSVs for overview, coverage, overlap, proximity, index validation, and region rollups.

## References and Context
- Loader/CLI code:
  - `src/PM4MscnAnalyzer/Program.cs` (CLI entrypoint)
  - `src/parpToolbox/Services/PM4/Pm4Adapter*.cs` (loading, analysis flags)
- Proposed analysis class:
  - `src/PM4MscnAnalyzer/MspiVsMsviAnalysis.cs` (new)
- Critical fields (per prior analysis):
  - High priority: `PackedParams`, `GroupKey`, `MsviFirstIndex`, `IndexCount`
  - Grouping: `MSLK.ParentIndex_0x04` (object grouping key)
  - Uncertain semantics: `Nx/Ny/Nz`, `Height` (avoid assumptions)
  - Deprecated/fabricated: Bounds* fields

## CLI
Command: mspi-analyze

Flags:
- -i, --input: Root directory containing PM4 tiles
- -p, --pattern: Glob for tile selection (e.g., "region/*/*.pm4")
- -o, --out: Output directory base
- -s, --session: Optional session name; used to create a timestamped subfolder
- --with-mscn: Include MSCN remap and proximity analysis (default: true)
- --proximity-samples: Number of centroid samples per source for proximity (default: 1000; 0 disables)
- --emit-tri-sets: Dump canonicalized per-triangle keys to JSONL (heavy; default: false)
- --max-tiles: Safety limit on analyzed tiles (optional)
- --verbose: Extra logging

Loader options:
- Pm4LoadOptions:
  - CaptureRawData = true
  - AnalyzeIndexPatterns = true
  - ValidateData, VerboseLogging toggled via flags
- Region loader merges tiles and applies MSCN remapping, yielding globally comparable vertex/index spaces.

## Outputs
Base folder: `OutputLocator.GetSessionFolder(outBase, sessionName)`

Emitted files:
- tiles_overview.csv
- mslk_mspi_coverage.csv
- msur_vs_mspi_overlap.csv
- mscn_proximity_summary.csv
- index_oob.csv
- coverage_summary.csv
- Optional: tri_sets.jsonl (guarded by --emit-tri-sets)

### CSV Schemas

1) tiles_overview.csv
- Columns:
  - region, tile_x, tile_y, tile_id
  - mspi_index_width (16|32|unknown)
  - mspi_index_count, mspi_tri_count
  - mspi_min_index, mspi_max_index
  - mspi_oob_index_count
  - msvi_index_count, msvi_tri_count
  - msvi_min_index, msvi_max_index
  - msvi_oob_index_count
  - msur_surface_count
  - mslk_entry_count
  - mscn_vertex_count
  - notes
- Purpose: per-tile snapshot across MSPI/MSVI and related chunks.

2) mslk_mspi_coverage.csv
- One row per MSLK entry with `MspiFirstIndex >= 0`.
- Columns:
  - region, tile_id, link_id
  - parent_index_0x04
  - mspi_first_index, mspi_index_count, mspi_index_end
  - mspi_index_width
  - in_bounds
  - oob_index_count
  - tri_count
  - overlap_msvi_tri_count
  - overlap_ratio_vs_msvi
  - unique_mspi_tri_count
  - notes
- Purpose: validate MSLK→MSPI ranges and relate to MSVI.

3) msur_vs_mspi_overlap.csv
- Compare MSUR surfaces (MSVI ranges) vs MSPI tri sets.
- Columns:
  - region, tile_id, surface_id
  - group_key
  - msvi_first_index, msvi_index_count
  - msvi_tri_count
  - overlap_mspi_tri_count
  - overlap_ratio_vs_msvi
  - unique_msvi_tri_count
  - unique_mspi_tri_count
  - notes
- Purpose: quantify duplication/augmentation/divergence between MSUR/MSVI and MSPI.

4) mscn_proximity_summary.csv
- Spatial alignment summary of MSCN points to MSPI vs MSVI.
- Columns:
  - region, tile_id
  - samples_requested, samples_effective
  - mspi_wins_count, msvi_wins_count, ties_count
  - mean_dist_mspi, mean_dist_msvi
  - median_dist_mspi, median_dist_msvi
  - p90_dist_mspi, p90_dist_msvi
  - notes
- Purpose: check which source is spatially closer to MSCN.

5) index_oob.csv
- Detailed index validation diagnostics.
- Columns:
  - region, tile_id, source (MSPI|MSVI|MSLK), ref_id
  - index_width (16|32)
  - first_index, index_count
  - min_index, max_index, vertex_count
  - oob_index_count
  - oob_first_occurrence
  - notes

6) coverage_summary.csv
- Region-level rollups.
- Columns:
  - region
  - tiles, tiles_with_mspi, tiles_with_msvi
  - total_mspi_tris, total_msvi_tris
  - total_overlap_tris
  - overlap_ratio
  - mspi_unique_tris, msvi_unique_tris
  - notes

7) tri_sets.jsonl (optional)
- One JSON object per tile:
  - { region, tile_id, tri_keys_msvi: [[i0,i1,i2], ...], tri_keys_mspi: [...] }
- Canonicalize triangle keys by sorting each triple ascending to normalize sets.

## Analysis Methods

- Globalization:
  - Use merged scene for globally comparable vertex/index references across tiles.
- Index width detection:
  - Record MSPI index width (16/32/unknown) based on parsed chunks; proceed using parsed model even if “unknown.”
- Triangle set construction:
  - MSVI: Use `MsviFirstIndex` + `IndexCount` per MSUR surface.
  - MSPI: Use `MSLK.MspiFirstIndex/Count` when available; include other explicit MSPI buffers if present.
  - Canonicalize each triangle key by sorting vertex indices ascending.
- Overlap metrics:
  - Compute set intersections and unique counts per tile and per-surface.
- Proximity (MSCN):
  - Sample N triangle centroids from each source. For each, find nearest MSCN vertex and record distances.
  - Aggregate wins (smaller distance), ties, mean/median/p90.
- Index validation:
  - Use `AnalyzeIndexPatterns` outputs; cross-check per-range during set construction to populate `index_oob.csv`.

## Implementation

- New class: `src/PM4MscnAnalyzer/MspiVsMsviAnalysis.cs`
  - Methods:
    - RunAsync(options, scene)
    - BuildTileStats()
    - BuildMslkCoverage()
    - CompareSurfacesVsMspi()
    - RunMscnProximity()
    - WriteCsvs()
    - DumpTriSetsJsonl() [optional]
- CLI wiring: `src/PM4MscnAnalyzer/Program.cs`
  - Add `mspi-analyze` command
  - Parse flags → load region via `Pm4Adapter` → run `MspiVsMsviAnalysis` → write CSVs.

## Performance & Safety
- Default sampling: 1000 proximity samples per source per tile (configurable).
- Avoid heavy JSONL dumps unless `--emit-tri-sets` is set.
- Stream CSV writers to limit memory usage.
- Respect `--max-tiles` to bound runtime.

## Acceptance Criteria
- CSVs generated for tile 00_00 and a small region without exceptions.
- OOB counts populated and consistent with index pattern analysis.
- Overlap and proximity numbers stable across repeated runs.
- Clear rollups in `coverage_summary.csv` to inform MSPI semantics decision.

## Example Usage
- Single tile:
  - pm4mscn-analyzer mspi-analyze -i i:/pm4/region -p "**/00_00.pm4" -o i:/out -s mspi-vs-msvi
- Small region with proximity:
  - pm4mscn-analyzer mspi-analyze -i i:/pm4/region -p "region/*/*.pm4" -o i:/out -s r1 --proximity-samples 1000 --with-mscn

## Deliverables
- CLI command `mspi-analyze`.
- CSVs:
  - tiles_overview.csv
  - mslk_mspi_coverage.csv
  - msur_vs_mspi_overlap.csv
  - mscn_proximity_summary.csv
  - index_oob.csv
  - coverage_summary.csv
  - tri_sets.jsonl (optional)

## Open Questions
- Is 1000 the right default for `--proximity-samples`?
- Should `PackedParams` and `GroupKey` appear in more CSVs (beyond `msur_vs_mspi_overlap.csv`)?
