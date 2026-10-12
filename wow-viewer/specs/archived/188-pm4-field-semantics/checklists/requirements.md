# Specification Quality Checklist: PM4 field semantics and a grouping surface that tests them

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-24
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

- **FR-004 is the load-bearing requirement**, and it exists because both halves of the trap have now
  been sprung within one session. `MSUR.GroupKey` is 100% pure with 9 corpus values and 0%
  distinctness — purity implying structure that is not there. `MSLK.GroupObjectId` was recorded as a
  doodad identity on 99.9% distinctness, when it is near-unique per link at 1.622 links per distinct
  value — distinctness implying structure that is not there. Reporting either alone produces a
  confident wrong answer; the pair is what discriminates.
- **FR-005's reference grouping is chosen for independence, not correctness.** Connected components
  of the adjacency graph are derived from stored topology rather than from a guessed field, so
  comparing a candidate against them cannot be circular. The spec deliberately does not claim
  components are the right object boundary — they are measurably *finer* than objects (299 components
  against 16 objects on one tile).
- **US3's population is the majority of the corpus and currently has no working grouping at all.**
  186,060 surfaces across 283 of 309 files carry a uniformly zero placement key, and the field long
  used as their identity does not group. That is the gap most likely to be mistaken for a decode
  failure when it is really an absence — the identity may simply not be stored, which is why FR-007
  requires a working grouping rather than a found key.
- **US4 is a sweep, not a decoder, and the distinction is deliberate.** Characterising a field by
  behaviour is cheap and safe; naming it is where the evidence rules bite, and those live in spec
  185. SC-007's re-derivation requirement is the detector-power gate: a sweep that cannot rediscover
  the three fields already settled has no business pronouncing on the ones that are not.
- **FR-012 exists because this work is cosmetic-looking and is not.** Retiring a grouping mode
  touches overlay construction; the guarantee that rendered geometry and published corpus figures are
  unchanged is what separates a relabel from a regression.
- **Open, deferred to planning**: whether falsified modes are deleted or kept with corrected labels.
  FR-002 permits either. Saved reports and older exports may reference them by name, which argues for
  relabelling, but that is a compatibility question best answered against the actual export formats.
