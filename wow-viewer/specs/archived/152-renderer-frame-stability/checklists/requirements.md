# Specification Quality Checklist: Renderer frame-time stability and per-era terrain lighting

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-15
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

Note: the Context section names two existing source files. That is deliberate — it records the
measured evidence that motivates the ordering constraint, and it is the reason the defect survived
earlier passes. It prescribes no implementation.

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

## Deliberate deferrals

These are intentionally left to planning rather than fixed here, because fixing them without
measurement would be inventing numbers:

- The hitch threshold, the noise-floor figure, and the SC-004 improvement margin are stated as
  "stated" values to be established from the first baseline run, not guessed in the spec.
- The SC-005 brightness tolerance is left to the comparison method chosen in planning.
- Era boundaries for the lighting profiles are named as a range (0.5.3 through 4.0.x); the exact cut
  points are a research output, since the repo already treats 0.5.3 / 0.6.0 / 1.0.0 as behaviorally
  distinct elsewhere.

## Notes

- The ordering constraint (detector power before optimization) is the spec's central risk control.
  US1 must complete before US4 begins; US3 is independent and may proceed in parallel.
- Prior "no regression found" results from the existing stationary harness must be treated as false
  nulls, not as evidence.
