# Specification Quality Checklist: Ground-Up v50 Terrain Height Model — Real WDL Prior + Residual Detailer

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-07-30
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

- The spec names specific existing code (`TerrainWdlLattice.FromTerrainVertices`,
  `WowViewer.Core.IO.Maps.WdlSummaryReader`, the `wdl_outer_17`/`wdl_inner_16` signal,
  `terrain_refiner_train.py`'s `--wdl-prior-dropout`) — these are root-cause evidence and reuse
  precedent for the Problem Statement/Governing Principle, not a prescribed implementation. Matches
  the precedent set by Spec 122's own checklist notes.
- FR-009 (synthetic-fidelity-gap handling) intentionally requires the concrete decision to be made
  and recorded during the planning phase rather than fixing it here — the spec states a strong
  expected direction (Assumptions) without pre-deciding a question the user has flagged genuine
  interest in exploring further. This is a deliberate planning-phase deferral, not an unresolved
  [NEEDS CLARIFICATION] gap.
- **Revision 1 (2026-07-30, same day)**: the user correctly flagged that the original draft
  implicitly assumed a real WDL prior is always available at deployment, which is false for
  genuinely novel inputs (hand-painted minimaps, content with no surviving `.wdl`) — exactly the
  deployment case this model lineage exists to serve. Revised to add User Story 3 (graceful
  degradation via prior-dropout, reusing `terrain_refiner_train.py`'s already-implemented
  `V7TileDataset` pattern), FR-016/FR-017, SC-007, and updated edge cases/assumptions accordingly.
- **Revision 2 (2026-07-30, same day)**: the user then correctly flagged that prior-absent
  (minimap-alone) prediction also needs protection from object/road color contamination, and
  explicitly did not want that solved by reviving Specs 119/120's confirmed-dead object-instance
  detection. Revised the Out of Scope, Problem Statement, and Assumptions to distinguish that dead
  instance-retrieval line from Spec 115's proven, different, semantic surface-family classifier
  (reused via FR-018), added object/road-coverage-stratified reporting (FR-019, SC-008, US3
  Scenario 5), and explicitly named Spec 118 US3's segmenter as unvalidated rather than assumed
  usable.
- Re-validated after both revisions; all checklist items still pass.
