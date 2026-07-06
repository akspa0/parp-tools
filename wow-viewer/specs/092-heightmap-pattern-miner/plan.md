# Implementation Plan: Heightmap Pattern Miner

## Scope

Add a standalone data-harvester analysis script that mines repeated heightmap patch signatures from Zarr height tiles. This is diagnostic tooling for V23, not a trainer change.

## Phase 1 - Repeated motif atlas

1. Create `scripts/mine_heightmap_patterns.py`.
2. Load `height_257` from a selected build store and filter rows from `index.parquet`.
3. Sample configurable patch sizes and strides.
4. Normalize each patch locally, downsample to a small signature grid, quantize, and hash.
5. Filter low-variance and over-saturated patches before ranking.
6. Rank repeated signatures by occurrence count and distinct tile coverage.
7. Write `summary.json` and `pattern_atlas.png`.
8. Validate with py_compile and a bounded real-data run.

## Phase 2 - V23 diagnostics

1. Add optional join against V23 validation/error artifacts.
2. Report motif-level mean absolute error if prediction outputs are supplied.
3. Export a motif curriculum manifest only after the diagnostic signal is useful.

## Validation

- `uv run python -m py_compile scripts/mine_heightmap_patterns.py`
- Bounded run against a real V18 Zarr store with a small tile limit.
