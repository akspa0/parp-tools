# Specification Quality Checklist: PM4 Remaining Decode

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

Every figure was produced by running the project's own `pm4 unknowns` analyzer over the 616-file
development corpus, not estimated: the six verified zero-miss relationships, the nine open
questions with their evidence strings, and the fit/miss counts for every partial edge. SC-001 is
therefore checkable against a concrete baseline (65,819 / 1,206,977) rather than being aspirational.

**The spec's central claim is that two reported problems are one problem.** The user described the
viewer selecting a surface instead of an object as a UI annoyance, separate from the missing
connective geometry. The measurement says otherwise: `MSLK.GroupObjectId -> MPRL.Unk04` is ~5%
resolved, so the viewer selects surfaces because surfaces are the largest unit the decode can
justify. US1 and US2 are therefore both P1 and are the same work stated as a UI outcome and as a
decode outcome. This reframing is the main thing to check if the spec is revisited.

**MSPV/MSPI is recorded as a LEAD, not a conclusion.** It is the strongest candidate for the
connective geometry on two measured grounds: it is a second geometry stream larger than the decoded
surface mesh (2,418,205 vs 1,930,146 index fits) and it attaches to the same link records. The
Assumptions section states explicitly that it may be eliminated, so the spec does not quietly
harden a hunch into a premise.

**FR-001 and FR-002 exist because of a specific failure.** While researching the sibling specs, a
PM4 chunk was hand-parsed outside the canonical decoder with an assumed axis order, producing a
confident and entirely wrong conclusion about data being stacked above and below the map. Both the
prohibition on reimplementing the decoder and the requirement that every claim be corpus-wide rather
than single-file trace directly to that.

**FR-008 and FR-012 make negative results first-class.** Nine questions have been open long enough
that some may be unanswerable, and MPRR may simply have no single target domain. Without a way to
record "eliminated, with evidence" and "no semantic meaning, with evidence", the same searches get
repeated indefinitely and the open list never shrinks.

SC-006 is the only criterion requiring a physical rather than statistical result: a grouping rule
can fit the corpus and still produce absurd objects, so at least one reconstruction must be measured
against a real asset. That guards against optimising a metric instead of decoding a format.
