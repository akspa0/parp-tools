# Specification Quality Checklist: Zarr-First Asset Residency

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-01
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) — **in the requirement and success
      sections**. FR-004 says "in the language the viewer is written in"; SC-002 says "the reference
      implementation". See note 1 for the deliberate exception.
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders — the *evidence* preamble is not, by house convention
      (note 1)
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain — three open decisions were put to the operator
      directly instead (note 2)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable — every SC has a number, a comparison, or a zero
- [x] Success criteria are technology-agnostic
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded — Out of Scope names 204, 199 and 205 explicitly so the boundaries are
      not re-litigated
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification — see note 1

## Notes

1. **The "Read this first" and "storage situation today" sections cite code and file:line
   deliberately.** Every recent spec in this repo does (183, 199, 201, 204, 205), because this
   project's failure mode is specs written against inherited names rather than measured behaviour
   (`feedback_a_name_stops_the_looking`). The requirements and success criteria themselves are kept
   technology-agnostic; the evidence that motivates them is not, and should not be.

2. **Three decisions were put to the operator and are now RESOLVED (2026-09-01):**
   - **Residency scope** → *everything the renderer touches*. Coverage is a completeness
     requirement; SC-003a is a zero, and partial coverage is not a success state.
   - **Store authority** → *derived and rebuildable*. The client stays the source of truth
     (FR-017a). This is what makes it safe to store output from decoders known to be wrong today
     (199 MCAL, 205 MH2O) — a bad store is discarded, not repaired.
   - **Texture form** → *both*. Portable pixels keep SC-009 intact; a derived block-compressed
     array makes upload a memcpy (FR-008a). Cost: ~2x texture storage plus the consistency
     obligation in FR-008b, accepted deliberately.

   Note the interaction the operator's answers create: **full coverage** and **both texture forms**
   together make this the largest version of the store, while **derived** keeps that affordable by
   making the whole thing disposable. The plan should treat store size as a first-class number.

3. **The premise correction is recorded in the spec body, not buried here.** The operator's stated
   cause was "the storage bottleneck, or how we read data out of the MPQs"; spec 204 measured the
   read as *not* the cost. The spec keeps the operator's solution (Zarr) and re-aims it at the
   measured cause (in-frame decode), rather than either dismissing the request or accepting a
   premise the evidence contradicts.

4. **US2 is a hard gate and is not optional.** `ZarrTileDatasetLoader.LoadTile` throws
   `NotImplementedException` today. Every other story that involves the viewer reading store data is
   untestable until it exists.
