# Specification Quality Checklist: PM4 Negative-BSP Object Matching

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

Validation performed 2026-08-03. Three review iterations; issues found and corrected:

1. **Implementation leakage (fixed)**: The first draft named `Pm4AssetMatchScorer`,
   `Pm4SegmentHeightStats`, `MinimumMatchedScore = 0.45`, `AmbiguousScoreWindow = 0.03`, the
   `pm4-asset-reference-signal-v1` identifier and the MSLK flag byte values throughout the
   requirements. All moved to Context/Assumptions as background; requirements reworded behaviourally
   ("classify as confident, ambiguous, or unmatched" rather than naming the constants).

2. **Untestable central requirement (fixed)**: An early FR read "match on negative BSP structure".
   That names a technique, not a checkable outcome. Replaced with FR-001/FR-002 (derive a structural
   description for BOTH sides so they are comparable) and SC-002, which is checkable against a
   specific confusable-pair scenario.

3. **Missing failure-mode requirement (added)**: The draft had no way to tell "scored low because
   the object genuinely differs" from "scored low because an input was missing". FR-012 now requires
   per-segment recording of unavailable signals, without which SC-001 comparisons could be measuring
   data coverage instead of matching quality.

Deliberately left open for planning: the spec does NOT prescribe how walkable-surface structure is
formulated (graph, descriptor, spatial hash, or otherwise). That is a design decision with several
reasonable answers and is recorded as an Assumption. The requirement is that structure is compared
and that the comparison is evidenced, versioned and deterministic.

FR-007 (keep the existing scalar signals alongside) is the guard that makes SC-001 meaningful: with
the old signals retained, the structural contribution is measured rather than asserted, and small
segments that cannot carry structural signal still score.

The historical context in the spec's Context section is recorded knowledge, not requirements. It is
present because it explains why the negative-space reframe is plausible (the WMO/Q3 BSP lineage) and
because the 1999 single-world artifacts are otherwise undocumented anywhere in the repo.
