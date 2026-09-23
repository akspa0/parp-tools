# Specification Quality Checklist: Weak-Signal & White-Plate Tile Explorer

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-03
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

Validation performed 2026-08-03. Three iterations of review; issues found and corrected:

1. **Implementation leakage (fixed)**: The first draft named `WeakSignalDetector`,
   `EstimateFactorFromRanges`, `TryGetTerrainWeakSignalWdlTile`, specific file paths, and the
   `ClassicCompressionFactor=33.334` constant throughout the requirements. These were moved to the
   Context section as background and the requirements reworded behaviourally ("the coarse guide
   reference", "the era constant"). Naming the unwired function is motivation, not a requirement.

2. **Untestable requirement (fixed)**: An early FR read "the amplifier should be smarter". Replaced
   with FR-008/FR-009, which specify the reference and the fallback order and are checkable against
   the 45 tiles measured to have a strong neighbour.

3. **Unbounded runtime dependency (fixed)**: The first draft had the viewer consume the generated
   inventory directory at `output/datasets/v50/v50.1/tile-inventory`. That would have limited the
   feature to harvested maps and coupled a viewer feature to a dataset artifact. Recorded in
   Assumptions instead: the viewer classifies from terrain it has already loaded, and the offline
   inventory is a validation reference (FR-013, SC-003) rather than a runtime input.

All success criteria are grounded in measurements taken 2026-08-03 rather than estimates: 1756 tiles,
361 degenerate, 205 carrying relief, 156 bit-exact flat, 45 with a strong neighbour. SC-002 and SC-003
are therefore checkable against a known-correct answer key.

Two criteria are deliberately qualitative-but-verifiable rather than numeric: SC-005 (30 seconds to
inspect a named tile) and SC-006 (browsing without perceptible wait). Both are observable from a user
session; neither depends on knowing the implementation.
