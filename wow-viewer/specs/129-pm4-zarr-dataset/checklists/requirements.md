# Specification Quality Checklist: PM4 Zarr Dataset

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

Validation performed 2026-08-03. Two review iterations.

Every number in this spec is measured, not estimated, using the project's own C# tooling
(`pm4 inspect`, `pm4 cross-tile`) rather than an independent parse: 616 files, 309 non-empty,
1229 distinct CK24, 266 cross-tile (21.6%), CK24=0 spanning 291 tiles as a sentinel, the
MSVT/MSCN/MPRL axis-order divergence, and the 0_0 vs 0_1 asymmetry. SC-001 and SC-003 are therefore
checkable against a known answer key rather than being aspirational.

**FR-001 exists because of a mistake made during this spec's own research.** MPRL was hand-parsed
outside `Pm4CoordinateService`, the axis order was assumed rather than read, and the result was a
confident and entirely wrong conclusion that tile data was stacked above and below the map. The
canonical decoder had the transform. That failure is why the coordinate space is a required stored
property (FR-002) rather than a reader convention, and why re-implementing chunk parsing is
prohibited outright.

**The object-primary layout is recorded as an Assumption, not a requirement**, because it is a
structural decision that is expensive to reverse. The 21.6% cross-tile measurement justifies it and
is stated so the decision can be re-argued against evidence rather than taste. FR-003 states the
behaviour required (an object spanning tiles stays one object, all three levels are reachable)
without mandating the physical layout that achieves it.

**FR-005/FR-006 exist because the decode is partly assumption by its author's own account.** The
inspector already publishes confidence levels and caveats — MSLK TypeFlags "medium" and "partial,
not corpus-closed", MSUR GroupKey and AttributeMask "low", MSLK GroupObjectId explicitly not a
confirmed object identity. A store that flattened these into equally-authoritative columns would
launder assumption into fact, and every downstream conclusion would inherit that.

FR-007 separates three states that are easy to conflate and that caused repeated wrong findings in
the sibling terrain work: genuinely absent, not extracted, and sentinel. CK24=0 spanning 291 tiles
is the concrete case — it is not an object spanning the map, it is a null key.

Deliberately not specified: the physical zarr layout for variable-length per-object data, and how
coordinate-space metadata is encoded. Both have several reasonable answers and belong in planning.
