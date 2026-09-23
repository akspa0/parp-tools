# Research — Spec 232 Cartography Composition

## T015 lattice rotation

**Decision**: rotate `AlphaTileData` full-tile lattices before `ToTileLoadResult` slices MCNKs.

**Rationale**: MCNKs share height-edge vertices. Rotating each chunk independently creates incompatible seams even if chunk slots move correctly. A single 257x257 source-to-target index map preserves shared edges by construction. The same mapping must rotate all channel planes and the 16x16 chunk metadata lattice.

**Alternatives considered**:

- Retain `TransformChunksForTarget`: rejected because the operator observed the broken seams.
- Resample rotated terrain after slicing: rejected because it manufactures data, duplicates policy, and risks a different live/export result.
- Free-angle resampling: out of scope; only quarter-turn and mirrors are specified.

## Project persistence

**Decision**: serialize one JSON project per base map through `PhaseLayerProjectFile`, placed below `output/projects/cartography/` by `CartographyProjectStore`.

**Rationale**: this is project-managed, human-editable and diffable; it holds disabled and unresolved layers rather than treating current map loading as project truth.

**Alternative considered**: write state into client data. Rejected: client content is not settings storage, and output must be loose/project-owned.

## Export parity

**Decision**: export will consume `PhaseCompositionPolicy` and the adapter's compose path rather than duplicate coordinate or channel rules.

**Rationale**: exact parity is SC-5. A separate exporter map transform could pass simple tests while disagreeing at borders, locks, or presence gates.

**Alternative considered**: export only changes/diffs. Rejected by FR-3; each output is a complete client map set.

