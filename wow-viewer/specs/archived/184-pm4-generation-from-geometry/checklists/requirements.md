# Specification Quality Checklist: PM4/PD4 generation from source geometry

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-23
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

- **Domain vocabulary vs. implementation detail.** Chunk names (MSUR, MSLK, MSPV/MSPI, MSCN) appear
  in the Context and evidence sections because they are the names of the artefacts under study, not
  implementation choices. The requirements and success criteria are written in structural terms
  ("walkable surface", "adjacency record", "wall quad", "blocked connection") so they stay testable
  without naming code.
- **FR-003 is the gate on the whole lane.** It requires the conformance check to be shown to separate
  real files from the current generator's output before its scores count as evidence. This is the
  repo's standing rule that a measurement unable to distinguish the interpretations it tests is not
  evidence — the same failure mode that produced a zero-by-construction counter in the earlier
  connective-geometry work.
- **FR-004/FR-005 deliberately forgo coverage for correctness.** Similarity-based pairing is
  available and would yield far more pairs, but asset identity is unsolved at P@1 ≈ 1.3%; admitting
  those pairs would score the generator against wrong targets. The ambiguity-rejection count is
  required output so the cost of that choice stays visible.
- **Open, and deliberately not resolved in the spec**: the 1.24% of real adjacency edges that do not
  reciprocate. Cross-tile neighbours are the obvious candidate. US4's boundary-policy scenario forces
  the generator to state a policy rather than silently produce dangling references, which is the
  correct handling while the cause is unmeasured.
- **MSCN is now unowned.** The window finding removed its only claimed index consumer, so the "MSCN
  is the per-object exterior boundary" reading has no surviving index evidence. US6 carries it as a
  named gap rather than the spec inventing a generation rule for it.
