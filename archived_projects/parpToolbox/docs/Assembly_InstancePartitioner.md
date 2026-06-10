# Composite-Hierarchy Instance Partitioner

This assembler refines per-object export by splitting each CK24 group (CompositeKey >> 8, top 24 bits) into vertex-connectivity-based components. It optionally applies soft MSLK-based gating to suppress unrelated surfaces when a majority of surfaces have no MSLK hits.

- Strategy: `--assembly composite-hierarchy-instance`
- Base: Builds on the correctness of `composite-hierarchy` while separating repeated instances to reduce over-merging.
- Output naming: `CK_<CK24_HEX>_inst_<COMPONENT_INDEX>` (CK24 is 6 hex digits; the dataset’s low DD byte is known to be 0).

## How it Works
1. Group MSUR surfaces by CK24 (CompositeKey & 0xFFFFFF00 >> 8).
2. Within each group, build connectivity via shared global vertices (from MSVI indices).
3. Each connected component becomes an instance and is exported as its own OBJ.
4. Optional: Soft MSLK gating keeps surfaces with any parent hits; if the proportion of unlinked surfaces exceeds a threshold, filter to the linked subset (never filters to empty; reverts if it would).

## Flags
- `--ck-instance-min-tris <N>`
  - Minimum triangle count required to emit an object (default: 600)
- `--ck-instance-no-mslk`
  - Disable soft MSLK gating (default: enabled)
- `--ck-instance-allow-unlinked-ratio <0..1>`
  - Allow this fraction of unlinked surfaces before filtering to linked-only (default: 0.15)

## Diagnostics
When `--csv-diagnostics` is enabled:
- `instance_partition.csv`
  - Columns: index, name, vertexCount, triangleCount, ck24, component_index, component_surface_count_pre, component_surface_count, mslk_hits, mslk_unlinked, unlinked_ratio
  - Only rows from this strategy are included.
- `surface_parent_hits.csv`
  - Columns: surfaceIndex, compositeKey, ck24, groupKey, msviFirstIndex, indexCount, parentHitCount
  - Helps validate MSLK coverage vs. surfaces.
- `assembly_coverage.csv`
  - Unchanged: per-object vertex/triangle counts.

## Tips
- Start with defaults, then tune:
  - Lower `--ck-instance-min-tris` (e.g., 400) if complete objects are being filtered out.
  - Increase `--ck-instance-allow-unlinked-ratio` (e.g., 0.30) if soft gating is too aggressive.
  - Turn off gating via `--ck-instance-no-mslk` if needed.
- Keep `--include-adjacent` for cross-tile completeness. Ensure global MSCN remapping is enabled (default path).

## Known Limitations
- Uses shared-vertex connectivity; extremely thin 1-vertex contacts can glue components. A future enhancement may require shared-edge connectivity.
- MSLK gating relies on tile-index offsets being correct; if link tile coordinates are wrong, gating may undercount hits.
