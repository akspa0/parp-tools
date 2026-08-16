# Specification Quality Checklist: WMO Doodad Inventory and Asset Chronology

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-16
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

## Validation Notes

**The finding that reordered the spec.** The request was "inventory every doodad in every WMO". The
first thing measurement showed is that the corpus itself cannot currently be enumerated: the earliest
staged build holds **532** world objects and the existing enumeration reports **1**. Inventory was
therefore demoted from P1 to P2 and corpus discovery promoted, because an inventory over a 500×
under-counted corpus would have produced a confident, wrong timeline in which every asset appeared to
arrive later than it did. SC-001 pins the 1-versus-532 gap as the first thing to close.

**How that gap was established, and the caveat it carries.** The 532 count was taken by direct
inspection of the build's own tree rather than through the project's inspection tooling — because the
tooling is the thing under test here, and it reports 1. A raw count is legitimate as the ground truth a
tool is measured against, but it is *not* a substitute for the tooling and must not become the
feature's method. FR-001 requires discovery to be a reported capability of the system, and the planning
phase should route every subsequent observation through the inspection surface, extending it where it
comes up short. The 532 figure itself should be re-derived through that surface once it exists; until
then it is an external check, not a product of the system.

**Zero [NEEDS CLARIFICATION] markers.** Every candidate ambiguity had either a defensible default or a
measurable answer:

- *How finely can assets be dated?* Between builds, by set difference — defensible now. Finer
  granularity is US5's hypothesis, explicitly allowed to return null.
- *Which asset classes?* Doodad references from world objects, as asked. Textures and sounds are out,
  recorded as an assumption rather than a question.
- *What about `uniqueId`?* Recorded as out of scope with the reason: it dates placements, not assets.
  The user's framing already treats this feature as the signal *beyond* `uniqueId`.

**Standing risks to carry into planning:**

- **"This asset does not exist" is the feature's most dangerous output.** The measured index gap means
  absence-from-index and absence-from-build are different facts. FR-004, FR-005 and SC-003 exist for
  this; a plan that resolves references against an index alone reintroduces the defect the spec was
  written to avoid.
- **US5 must be allowed to fail.** The within-file ordering hypothesis is the kind of claim this
  project has already seen test null once, on a related chronology question, because it was the wrong
  clock. FR-009 requires both the result and a statement of whether the test could have detected the
  effect — a null from an underpowered test is not a finding.
- **Renames are indistinguishable from removal-plus-introduction** without further evidence. The
  timeline must say so rather than choose; a plan that silently treats one as the other will
  manufacture introductions that never happened.
- **Repair modifies data.** FR-011 through FR-013 and SC-008/SC-009 are the guardrails. Planning must
  keep analysis and repair separable, so that no analysis path can mutate anything.
- **US7 is a survey before a promise.** "Make the converters work properly" cannot be scoped before the
  current state is recorded. This project has twice been caught by a capability that was documented or
  assumed but not real; the survey is the cheap defence.
