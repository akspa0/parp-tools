# Feature Specification: Heightmap Pattern Miner

**Feature Branch**: `092-heightmap-pattern-miner`
**Created**: 2026-07-06
**Status**: Draft
**Input**: User wants simple sampling of heightmap data to detect repeated terrain patterns that may improve understanding of the V23 height model.

## User Scenarios and Tests

### Scenario 1 - Mine repeated height motifs

An ML operator can scan V18/V22-style `height_257` tiles and find repeated sampled patch shapes across maps and tiles.

**Acceptance**:
- The operator can select build, maps, patch sizes, stride, sample limit, and output directory.
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
- **REQ-004**: The tool MUST support configurable patch sizes and stride.
- **REQ-005**: The tool MUST group normalized low-resolution patch signatures so repeated shapes can be found despite absolute height offsets.
- **REQ-006**: The tool MUST output `summary.json`.
- **REQ-007**: The tool MUST output a visual `pattern_atlas.png`.
- **REQ-008**: The tool MUST avoid changing V23 training code in the first slice.
- **REQ-009**: The tool SHOULD provide filters for low-variance and over-saturated patch artifacts so repeated void/plateau regions do not dominate every run.

## Non-Goals

- Training V23 with pattern weights.
- Adding a new loss term.
- Proving hidden steganographic payloads.
- Replacing the V18 curation manifest.

## Risks

- Flat areas, empty regions, and water-adjacent plateaus can dominate repeated-pattern counts.
- Coarse quantization may merge different terrain shapes.
- Fine quantization may miss useful repeated motifs.
