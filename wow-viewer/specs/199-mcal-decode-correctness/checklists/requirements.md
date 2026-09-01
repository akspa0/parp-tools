# Specification Quality Checklist: MCAL Alpha Map Decode Correctness

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-01
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

Validation run 2026-09-01, one iteration.

**"No implementation details"** — the Context section names four concrete files and a
function. Retained deliberately: the duplication *is* the problem being specified, and it
cannot be stated without naming what is duplicated. No requirement or success criterion names
a type or API.

**Priority note**: US1 and US2 are both P1, which the template does not normally encourage.
This is deliberate and load-bearing. Consolidating four decoders (US1) without establishing
the correct rule (US2) would produce a single authoritative guess — arguably worse than four
visible disagreements, because it would look settled. They ship together or the spec has not
delivered.

**Sequencing note**: US3 (delete the fabrication) is P2 despite being the visible symptom the
operator reported. Removing it first would replace wrong terrain with missing terrain and
destroy the only signal currently indicating that decode fails at all. This is recorded so
planning does not "fix the screenshot" first.

**Evidence honesty**: FR-005 and SC-006 exist because the native decoder is not yet isolated
(research.md R5). If it cannot be, the rule is backed by file-side proof rather than by the
client, and the spec says so rather than implying stronger evidence than it has. This project
has shipped `MCXH` and `Ck24Type` on exactly that kind of unstated inference.
