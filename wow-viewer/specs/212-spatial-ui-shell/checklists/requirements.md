# Specification Quality Checklist: 3D Spatial UI Shell

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

**Iteration 1 — issues found and fixed:**

1. *Success criteria were not measurable.* An early draft of SC-001 read "the scene has more room".
   Replaced with a comparison against the 2D shell at the same window size and a threshold (35%).
2. *SC-005 originally said "no noticeable frame cost".* That is unverifiable, and the memory bank
   records that this repository's renderer profiler uses a static camera and produces false null
   results. SC-005 now names a budget and explicitly requires measurement against the frame-time
   distribution rather than a static-camera average.
3. *US4 had no failure mode for a missing toolchain.* Added AC and FR-017/SC-007 requiring a clean
   clone to build with no OpenSCAD installed and no MCP server reachable. Without this the repository
   would acquire a hidden build dependency on an external server.

**Deliberate deviations from the generic checklist:**

- *"No implementation details"* passes for the Requirements and Success Criteria sections, which are
  the ones that bind. The **Context & Motivation** and **Assumptions** sections do name existing code
  (`TryGetSceneViewportRect`, `OffGeometry`, `ProceduralMeshLoader`, `_useTabUi`). This is deliberate
  and follows `AGENTS.md` §1 and the precedent of spec 210: a spec in this repository is read by an
  agent that needs to know which existing seam the feature attaches to. The grounding is confined to
  the narrative sections and never appears in an FR or SC.
- *"Written for non-technical stakeholders"* is marked partial and will stay partial. The sole
  stakeholder is the operator of a reverse-engineering toolchain. Terms like "field of view", "pointer
  ray" and "workspace task" are that reader's vocabulary; removing them would cost precision and buy
  nothing.

**Open risk carried into planning (not a spec defect):**

- The pointer-to-content mapping on curved surfaces (US4) is the feature's main technical risk. The
  spec manages it by ordering — US1 must fully pass on flat surfaces before US4 is attempted, and US4
  re-runs the entire US1 suite. `plan.md` must not reorder these.
- FR-022 (screen-anchored constructs) names the constraint but not the resolution. Deciding what a
  modal *does* in a spatial shell is a design decision that belongs in `plan.md`.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.
