# Specification Quality Checklist: Unified Format Version Profiles

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-07-15
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain — resolved 2026-07-15 (see Resolved Clarifications)
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

### Deliberate deviations from the default checklist posture

**"Written for non-technical stakeholders" is interpreted as "written for the project owner."**
This is an internal architecture spec whose stakeholder is the repo owner. Byte offsets and function
addresses appear in the Assumptions and Edge Cases sections **on purpose**: this project's recorded
history is that *every* bug fixed in this area was caused by a guessed layout, so the evidence trail
is the load-bearing content, not incidental implementation detail. Stripping it would make the spec
prettier and less useful. Requirements themselves (FR-001..FR-018) stay behaviour-level.

**SC-004 is an inverted success criterion.** It requires the code to *shrink*. This is unusual but is
the explicit point of the feature: two competing schemes already exist, and the failure mode being
guarded against is a third. Recorded in the user's standing guidance that "simplify must shrink, not
grow."

### Validation findings addressed during authoring

- **Premise correction surfaced, not silently dropped.** The original request asked for Warcraft.NET
  adapters. Code inspection showed the premise inverted (our M2 code has zero Warcraft.NET
  dependency). Rather than quietly omitting it, the spec records the finding (Finding 1) and lists
  it in Out of Scope, so the decision is auditable.
- **Prior art surfaced.** `FormatProfileRegistry` is a 717-line prior attempt at this exact
  architecture. A spec that did not name it would have caused it to be reinvented. It is now the
  spec's anchor (Finding 2) and the source of FR-009/FR-010.
- **Constitution conflicts checked.** Principle II ("one canonical owner per format surface") is the
  governing authority for FR-008/FR-009. Principle I (core must not reference the viewer) is what
  makes the 1.0.0-vs-1.12.1 disambiguation non-trivial and is flagged to the plan rather than
  hand-waved.

### Resolved Clarifications

1. **Question 1 (FR-018) — scope of non-M2 format surfaces. RESOLVED 2026-07-15: M2 only, designed
   for the rest.** The user chose to migrate only the M2 surface and delete its inert profile
   records, leaving ADT/WMO/MDX on `FormatProfileRegistry` untouched, with the new system shaped so
   they can migrate later under their own spec. Recorded as FR-018/FR-019/FR-020.
   **Consequence for planning**: `FormatProfileRegistry` survives this feature. It is not a failure
   of SC-003 ("exactly one owner") — that criterion is per *format surface*, and after this change
   the M2 surface has exactly one owner where today it has two. Do not "helpfully" migrate the other
   surfaces to make the file disappear; FR-020 forbids it.

### Risks carried into planning (not spec blockers)

- ~~**Animation index provenance is unverified.**~~ **RETIRED 2026-07-15** by tracing `FUN_0070f960`.
  The interpolation-range index is the sequence index; our existing `sequenceIndex` is correct.
  Evidence appended to `specs/104-legacy-m2-rendering/evidence/1.0.0-ghidra/m2_track_sampler.c`.
- **NEW RISK, found while retiring the above: the sequence time base is era-dependent (FR-005a).**
  1.0.0 sequences carry `start`/`end` into a shared global timeline; Wrath+ carries `duration` and
  is sequence-local. `M2SequenceDefinition` models `duration` only — it **cannot represent 1.0.0's
  start/end**. This is a second expressiveness gap of the same class as FR-001, and it was invisible
  until the caller was traced. Implementing FR-001 alone would still produce wrong keys. Planning
  must treat era-dependent *time base* as a first-class concern alongside era-dependent *track
  addressing*, and must check whether the era-100 reader even captures start/end today.
- **SC-001 cannot be self-certified.** Per AGENTS Rule 0 the user runs render validation. The spec
  is not signed off on the strength of a passing build or a green test run.
- **Baseline capture ordering is a real hazard.** FR-016 requires the 3.x/4.x baseline to be
  captured *before* any shared type is touched. If planning reorders this, the no-regression gate
  measures nothing.
