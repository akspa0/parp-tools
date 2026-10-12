# Specification Quality Checklist: M2 Reader Era Parity (1.x – 3.0.1)

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-15
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

**Iteration 1 findings, resolved in the spec as written:**

- *Class and constant names leaked into the first draft.* Reader type names, field offsets, and byte
  strides were named directly. Rewritten so the Context section describes observed behaviour
  ("locates the bone array at a different header position and steps at 88 bytes per bone") and the
  requirements describe outcomes. The concrete anchors survive in the measured-state table and defect
  list, which is where evidence belongs — not in the requirements.
- *Zero [NEEDS CLARIFICATION] markers, deliberately.* Every candidate ambiguity is answerable by
  measurement rather than preference:
  - Whether the late-3.x/4.x route is genuinely sound → US1 / SC-002.
  - Where the real boundary between "working" and "broken" builds sits, given that 3.0.1 and 3.3.5
    likely share a version word → US1 / recorded as an Assumption.
  - How many staged builds actually reach the refusing range → US1, and explicitly gates US3's effort.
  Asking the user to choose would substitute opinion for evidence on questions the survey settles.

**Iteration 2 — corrections from measurement (2026-08-15, same day):**

- *A stated assumption was wrong and has been replaced.* The first draft assumed 3.0.1 and 3.3.5
  declared the same version word, and on that basis called the range "1.x through 3.0.1" approximate.
  Measured: 3.0.1.8303 declares `0x107`, 3.3.0.10958 declares `0x108`. The range is **exact** and the
  boundary is crisp. The spec now carries the measurement and an explicit note that the earlier
  assumption was wrong — negative results stay recorded rather than being quietly overwritten.
- *The reference was over-stated.* "3.3.5 through 4.0.0 works" is now narrowed to `0x108`, measured at
  3.3.0.10958 (151 bones, geometry available). The 4.0.0 beta still fails, so `0x109`+ is not the
  yardstick.
- *US4 was mis-sequenced.* It was written as blocked on US2/US3. It is not: `MDLX` and `0x108` both
  work today, so a first High Elf ↔ Blood Elf comparison is reachable now. US2/US3 widen which builds
  can participate rather than unblocking the story.
- *A structural constraint was added.* Rig comparison cannot be a bone-count check — 54 against 151
  across the two working routes. The comparison must ask whether the earlier bone set appears within
  the later one with corresponding parents and pivots.

**Standing risks to carry into planning:**

- US1 is a prerequisite, not a parallel track. Sequencing US2 or US3 ahead of it repeats the mistake
  that produced D3 — a confident claim about an era that measurement contradicts.
- SC-003 asserts bone counts match the file's own declaration. A file whose declaration is itself
  wrong would satisfy the letter and not the intent; the plausibility checks in Edge Cases are what
  cover that, and planning must keep them.
- FR-009 permits the reference route's behaviour to change. Any such change needs its own evidence,
  or "parity with the working path" quietly becomes parity with a new guess.
- **The unit of support is the build.** Structurally significant changes land in `0.0.1` patch
  releases without a version-word bump, so two builds one patch apart can differ in ways that break a
  reader written against the other. Planning must not reintroduce expansion-shaped reasoning ("this is
  TBC, so it must be…") *or* neighbour-shaped reasoning ("8334 is between 8303 and 8391, so it must
  be…"). FR-002 and FR-011 forbid both; the wrong assumption corrected in iteration 2 is what it looks
  like when that reasoning creeps back in. The three staged 3.0.1 builds are the standing test of
  whether the survey respects this.
- **Scope ceiling is 4.0.0 and it is not a phasing decision.** Later formats are not a future
  extension of this work. SC-008 makes the ceiling checkable. Any planning artifact that reasons about
  post-4.0.0 formats — even to defer them — is out of bounds.
