# Specification Quality Checklist: PM4/PD4 format documentation and terminology restoration

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

- **FR-002 is the whole point of the spec.** The drift being corrected is not that names are ugly but
  that they *assert semantics the data does not have*. `AttributeMask` is a scalar count, and that
  name kept it from being read as one. `_0x18` was documented as an MSCN index and the connective-
  geometry search went to the wrong chunk as a result. A name that claims knowledge the project does
  not have is a defect, not a style preference.
- **Two catalog entries were corrected in place immediately** (`MSUR._0x02`, `MSUR._0x18`) rather than
  waiting for implementation, because leaving a measurably false claim in a shared catalog propagates
  it to every reader. Only the evidence and confidence text changed; the C# field renames are FR-003
  work and are deliberately not done yet, so FR-012's "figures identical before and after" gate
  still has a clean baseline.
- **FR-012 exists because renames are exactly where silent behaviour changes hide.** The corpus
  analyzers publish figures; those figures are the regression test for a rename.
- **FR-014 is deliberately two-directional.** wowdev may be wrong — it is wrong about `_0x18` — and
  the project must be able to say so with evidence rather than either deferring silently or
  substituting a private name without argument.
- **US3 is allowed to fail.** SC-007 explicitly accepts zero surviving MPRR candidates as an outcome.
  The value-domain sweep has already run and discriminates nothing; the deliverable is a smaller,
  recorded search space, not a guaranteed decode. Recording eliminations is what stops the next
  session repeating the sweep.
- **The per-model format is the control for the CK24 question.** Its end-of-record value is zero in
  3,447 of 3,447 surfaces while the per-tile format packs identity there. Any proposed reading has to
  explain both populations, which is a much stronger constraint than the per-tile corpus alone.
