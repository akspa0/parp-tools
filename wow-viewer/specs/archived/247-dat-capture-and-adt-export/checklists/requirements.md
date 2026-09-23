# Specification Quality Checklist: DAT Capture & LK ADT Export

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-20
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

Two operator decisions are already baked in and did not need clarification markers:

1. **Phase order** — the v22 alpha codec goes first, chosen by the operator over relaxing the guard
   first or shipping lossy output now. Recorded as FR-001/FR-002 blocking.
2. **Capture output** — PNG per tile plus a stitched overview, chosen over client-layout BLP or both.
   Recorded as FR-006. The spec states the outcome ("images", "stitched overview") rather than the
   file format, to keep the criteria technology-agnostic; the concrete format belongs in `plan.md`.

Named code symbols (`AhdrTerrainAdapter`, `LkAdtWriter`, `ScreenshotRenderer`, chunk tags, byte
offsets) are intentionally confined to the Context, Assumptions and Dependencies sections as
**measured facts and reuse pointers**, not as requirements. Every FR and SC is stated as observable
behaviour.

Two assumptions carry real risk and are flagged in the spec rather than resolved here, because
resolving them needs work the spec is requesting:

- v22 `ASHD` → LK `MCSH` is a same-size guess, unverified.
- Area IDs 3519/3520 are unconfirmed against the area table; no zone name may be asserted until they
  are.
