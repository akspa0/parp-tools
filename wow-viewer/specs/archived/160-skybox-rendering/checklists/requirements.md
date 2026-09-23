# Specification Quality Checklist: Skybox Rendering

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-17
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

### Iteration 1 — issues found and fixed

1. **Implementation detail in requirements.** First-pass FR text named specific symbols
   (`_skyDome.UpdateFromLighting`, `NightVisibility`, `WmoSummary.HasSkybox`). Rewritten as
   behavioural statements. Symbol-level evidence is confined to the Context defect table, where it
   documents *what was measured*, not *what to build*.

2. **Unmeasurable success criteria.** "Sky looks correct" replaced throughout with differential
   tests — change an authored value, observe a corresponding rendered change. SC-001 and SC-003 are
   now verifiable without knowing the implementation.

3. **Unbounded scope.** An explicit Out of Scope section was added. Clouds, weather, glare, and
   post-effects sit adjacent to sky work and would otherwise creep in; the LIT cloud fields are
   named specifically because the loader already exposes them.

4. **Open question left dangling.** The dome/model layering contract was carried in as an open
   research item. It is resolved in Context under "Research: the dome/model layering contract" with
   the evidence that settled it, so no [NEEDS CLARIFICATION] marker was needed.

5. **Missing non-regression requirements.** Three active specs (151, 152, 153) own renderer
   frame-time work. Sky changes touch the per-frame path, so FR-022 through FR-024 and SC-008/SC-009
   were added to keep this spec from silently regressing theirs.

### Deliberate spec decisions

- **Source-agnostic scope was chosen by the user over era-gating.** The project has a precedent
  (`MinimapEraProfile`) where 0.5.3 / 0.6.0 / 1.0.0 genuinely differ. The risk is accepted, and
  mitigated by making provenance mandatory (FR-003, SC-004) so build-specific divergence surfaces as
  a reportable fact rather than a silent wrong colour.

- **The no-silent-mixing rule (FR-002) is inherited, not invented.** Project research states LIT
  tracks and Light\* DBC records are separate sources that must not be mixed; the existing code
  already carries this rule for fog. FR-002 extends the same rule to sky, which is what makes
  "source-agnostic" safe.

- **Five user stories, not one.** Each maps to one confirmed defect and is independently testable.
  US1 is sequenced first because the overwrite defect makes every other story invisible until it is
  fixed — US2 through US5 cannot be visually validated ahead of it.

- **No concrete frame-cost budget is stated.** FR-022 requires a budget; the number belongs against
  a measured pre-change baseline in planning, not asserted here.

## Status

All checklist items pass. Spec is ready for the speckit-plan phase.
