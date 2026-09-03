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

**Added 2026-09-02 — US6, selection outlines.** Operator-reported annoyance: selection highlights are
wireframe boxes, which at small scales are larger than the object they describe. Added here rather
than as a new spec because it is the same thesis as the rest of the feature — stop approximating,
make the UI follow the real geometry — and because spec 211 already established selection visual
treatment as an owned concern rather than an ad-hoc one.

Deliberately marked **P1 with no dependency on US1–US5**, so it is separately shippable and is not
gated behind the shell work. If the shell is deferred or descoped, US6 still lands. `plan.md` must
preserve that independence.

Related, already fixed outside this spec (2026-09-02): the box overlay's accent inflate had a *fixed*
0.75 yd floor and its dash length a 6.0 yd floor, so a small object got a halo six times its own size
and a second solid-looking box. Both are now proportional. That was a mitigation of the symptom;
US6 removes the cause.

**Open risk carried into planning (not a spec defect):**

- The pointer-to-content mapping on curved surfaces (US4) is the feature's main technical risk. The
  spec manages it by ordering — US1 must fully pass on flat surfaces before US4 is attempted, and US4
  re-runs the entire US1 suite. `plan.md` must not reorder these.
- FR-022 (screen-anchored constructs) names the constraint but not the resolution. Deciding what a
  modal *does* in a spatial shell is a design decision that belongs in `plan.md`.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.

**Added 2026-09-02 (second pass) — US7 museum profile, US8 3D tool controls.** Operator direction:
a profile that "dials it all back" to a camera-locked floating HUD over the world — "more of a museum
than an in-your-face data explorer" — and controls shaped like what they control, the time-of-day
slider becoming an interactive clock.

- **US7 is P1 and independently shippable.** A minimal camera-locked HUD over a full-window scene
  delivers the experience without US3's context rig or US4's authored shells. It is the reason the
  spatial shell is worth building, so it must not sit behind the harder stories.
- **FR-032 is the load-bearing constraint**: a HUD element must invoke the *same* underlying action as
  its full-shell equivalent. The failure mode is obvious and expensive — two implementations of one
  operation drift apart, and the museum profile silently becomes a fork with its own bugs.
- **US8's clock is grounded**: the current control is `ImGui.SliderFloat("Time of Day", 0..1)` in
  `DrawTimeOfDayControl`. ImGui has no built-in circular control, so this has to be drawn and
  hit-tested; SC-016 pins correctness to producing identical lighting to the existing linear control,
  including across the midnight wrap, so the new control cannot quietly disagree with the old one.
- **Connects to spec 216**: reaching 3am to light a scene by torchlight is the concrete task the clock
  exists for. 216 requires only that time can be set — the existing slider satisfies that — so neither
  spec blocks the other.
