# Feature Specification: Heightmap Pattern Miner

**Feature Branch**: `092-heightmap-pattern-miner`
**Created**: 2026-07-06
**Status**: Draft
**Input**: User wants simple sampling of heightmap data to detect repeated terrain patterns that may improve understanding of the V23 height model.

## User Scenarios and Tests

### Scenario 1 - Mine repeated chunk-cell motifs

An ML operator can scan V18/V22-style `height_257` tiles and find repeated sampled patch shapes that span multiple terrain cells and align to the MCNK chunk-cell grid.

**Acceptance**:
- The operator can select build, maps, terrain-cell spans, chunk alignment, sample limit, and output directory.
- The output includes ranked repeated motifs with counts, distinct tile coverage, and example locations.
- The output includes a visual atlas of the top repeated motifs.

### Scenario 2 - Feed findings back into V23 analysis

An ML operator can use the mined motifs to inspect whether V23 validation failures cluster around recurring terrain shapes.

**Acceptance**:
- The summary JSON records stable pattern IDs and source locations.
- Pattern grouping is shape-focused and does not depend on absolute world elevation alone.
- The first slice does not change V23 training behavior.

## Requirements

- **REQ-001**: The tool MUST live under `wow-viewer/data-harvester/scripts/`.
- **REQ-002**: The tool MUST read Zarr stores with `height_257` and `index.parquet`.
- **REQ-003**: The tool MUST support filtering by map name.
- **REQ-004**: The tool MUST support configurable terrain-cell spans and chunk-aligned sampling.
- **REQ-005**: The tool MUST group normalized low-resolution patch signatures so repeated shapes can be found despite absolute height offsets.
- **REQ-006**: The tool MUST output `summary.json`.
- **REQ-007**: The tool MUST output a visual `pattern_atlas.png`.
- **REQ-008**: The tool MUST avoid changing V23 training code in the first slice.
- **REQ-009**: The tool SHOULD provide filters for low-variance and over-saturated patch artifacts so repeated void/plateau regions do not dominate every run.
- **REQ-010**: The tool MUST reject windows smaller than the configured minimum terrain-cell span; the default minimum is 32 cells.
- **REQ-011**: The tool MUST report `cell_span`, `chunk_x`, and `chunk_y` for motif examples so a match is tied to meaningful terrain geometry.
- **REQ-012**: The default signature SHOULD be coarse enough to group terrain families, not only byte-near-identical local patches.

## Non-Goals

- Training V23 with pattern weights.
- Adding a new loss term.
- Proving hidden steganographic payloads.
- Replacing the V18 curation manifest.

## Risks

- Flat areas, empty regions, and water-adjacent plateaus can dominate repeated-pattern counts.
- Coarse quantization may merge different terrain shapes across large cell spans.
- Fine quantization may miss useful repeated motifs.
