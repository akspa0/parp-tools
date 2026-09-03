# Specification Quality Checklist: 5.0.1 Physics

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-02
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [~] Written for non-technical stakeholders
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

**Grounding**: written against the live binary. Domino, the source-tree layout, the assertion pivot,
the ~90-function floor, the `0x00c26e50`-`0x00c4e012` range and the physics-culling CVar strings are
measured and recorded in
[`workstream-atmosphere-501-ghidra.md`](../../../memory-bank/workstream-atmosphere-501-ghidra.md).

**Iteration 2 (2026-09-02) — operator redirected the approach.** The first draft specified writing
the full solver. Operator direction: use an existing permissively licensed C# physics library, and
keep copyrighted engine code out of the tooling. Reworked throughout:

- Added a **"The solver is licensed in, not reimplemented"** section drawing the boundary explicitly:
  data formats and *observable behaviour* are in scope, engine algorithms are not.
- Added FR-005 (contract limited to layouts/behaviour/budget), FR-006 (third-party library required),
  FR-007 (license verified at the version used and recorded), FR-008 (no proprietary engine code in
  any form, including machine-translated), plus SC-011 and US1 AC5 and US3 AC0.
- Candidate libraries are named as *candidates only* with selection and license verification
  deferred to `plan.md`. The spec requires the check rather than asserting any library's license,
  because licenses change by version and an assumption here would be exactly the kind of unverified
  inherited claim this project keeps getting burned by.
- **This resolves the scope concern the first draft raised.** That concern is now historical and the
  paragraph stating it has been replaced rather than left to contradict the new direction.

**Iteration 1 — issues found and fixed:**

1. *US3 said "bodies behave correctly".* Untestable. Replaced with the properties a wrong mapping
   violates first: rests without sinking, gains no energy, does not tunnel, is deterministic.
2. *The ~90 asserting functions were treated as the size of Domino.* Corrected to a **floor**
   throughout — only asserting functions are visible to the pivot.
3. *Physics culling was missing.* The binary proves a cull distance exists. Added as US5.
4. *SC-007 said "no noticeable cost".* Replaced with a frame-time-distribution measurement, because
   this repository's profiler uses a static camera and produces false nulls.
5. *Era gating was an assumption line.* Promoted to US7 at P1. Domino does not exist in 0.5.3.

**Deliberate deviations:**

- *"No implementation details"* passes for Requirements and Success Criteria. **Context** names
  addresses and source files because the evidence *is* the scope for a decode feature. Candidate
  libraries are named in Context as options with the decision explicitly deferred, not chosen.
- *"Written for non-technical stakeholders"* stays partial; the reader is the operator.

**Open risks carried into planning:**

- **Library selection is now the pivotal decision**, and it is a `plan.md` deliverable: license,
  cloth capability, determinism, and whether native interop is acceptable against this repository's
  cross-platform targets.
- **Cloth may not be first-class** in every candidate. If the chosen library lacks it, US4 — the
  visible flag motion that motivated the whole request — is affected, so cloth capability must be a
  selection criterion rather than a discovery.
- **Physics data location is unmeasured.** The spec deliberately says "the physics description
  associated with a model" rather than naming a container. Planning must establish it from the
  binary, not from outside assumptions.
- **Behaviour matching, not algorithm matching.** Where the library and the client differ, the spec
  requires recording a finding. Planning should say how close is close enough.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.
