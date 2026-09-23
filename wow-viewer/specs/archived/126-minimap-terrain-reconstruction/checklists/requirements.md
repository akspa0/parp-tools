# Specification Quality Checklist: Minimap-to-Terrain Reconstruction Stack

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-02
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

### Deliberate deviations from a strict reading of "no implementation details"

Three items name specific technology. Each is a **constraint the work must honour**, not a design
choice being smuggled into the spec, and removing them would lose real information:

1. **SC-013 caps the stack at 200 million parameters.** This is a hard budget the user set. It is
   stated as a size limit rather than as an architecture.
2. **The Assumptions section excludes depth-foundation-model families.** This is a standing project
   exclusion with a recorded history, not a preference. Leaving it implicit invites the excluded
   approach to be reproposed.
3. **The Why section notes the codec round-trip is bit-exact across the two implementations.** This
   is a statement about *data validity* — training input matching deployment input — which is the
   premise the whole feature rests on.

Domain vocabulary (DXT1, MCVT, MCLY, MCAL, WDL, 256x256, 257x257) is treated as the subject matter's
own terminology rather than implementation detail. These are properties of the file formats and the
game data being reconstructed; they cannot be paraphrased away without making requirements untestable.

### Resolved without a clarification marker

Two questions had multiple reasonable readings. Both were resolved with a documented default rather
than a blocking marker, and both are flagged for confirmation at planning time:

- **Tier 4 (per-layer MCAL/MCLY) in scope now or deferred?** Resolved as **in scope, P3, gated on
  tier 3 passing** (User Story 8). Rationale: specifying it now means the tier 1-3 datasets and
  evaluation are designed to support it, which avoids rebuilding them later. Gating it prevents the
  most speculative output from consuming effort first. The alternative — defer to a follow-on feature
  — remains viable and costs only a later data migration.
- **What "a single large image" means.** Resolved as: a stitched image over a **known** tile grid is
  the primary case; an image with unknown grid or scale is supported at relative-relief-only with
  scale and elevation reported unavailable (FR-009, User Story 5 scenario 3).

### Risk carried into planning

- **User Story 1 is a hard prerequisite.** The unlit-albedo render pass does not exist. Until it does,
  SC-001 cannot be measured and the albedo-removal ordering question stays open.
- **Two gating measurements from the archived relational-layers spec were never run**, in particular
  whether layer masks derive from terrain shape. That answer materially changes User Story 6's
  difficulty and should be obtained early rather than assumed.
- **User Story 3 has been deferred in prior specs.** Its acceptance scenarios are written to be
  independently verifiable specifically so it cannot be quietly dropped again.
