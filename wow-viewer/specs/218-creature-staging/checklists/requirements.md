# Specification Quality Checklist: Creature Staging

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

**Grounding — four facts measured in the existing code before writing:**

1. **Spawning already exists**: `WorldSpawnRecord`, `WorldScene.SetExternalSpawns`, and
   `AlphaCoreDbReader` resolving `creature_template.display_id1 -> CreatureDisplayInfo.ModelID ->
   mdx_models_data.ModelName`.
2. **Attachment points are parsed and never rendered.** `MdxAttachment`, `MdxAttachmentFile` and
   `MdxAttachmentSummary` exist in Core; **nothing in `src/viewer/WoWViewer/Rendering/` references
   any of them.** This is the gap and the reason a torch cannot be put in a hand.
3. **Equipment display resolution does not exist.** The catalog resolves creature displays only;
   there is no item-to-appearance path.
4. **Capture automation already exists**: a capture queue, `CameraShotPoint` presets persisted to
   `camera_shot_points.json`, batch handling and chrome suppression.

Facts 2 and 3 are the whole feature. Without checking them this would have been scoped as "add a
paper doll", which would have produced a UI over machinery that does not exist.

**Iteration 1 — issues found and fixed:**

1. *US4 (saved scenes) was initially P2 convenience.* Promoted to **P1**. Spec 216's value depends on
   re-running the comparison every time the lighting model changes; without a replayable arrangement
   each re-test is a manual reassembly and the results are not comparable. That is the difference
   between a demo and an instrument, so it is not optional.
2. *Automation was going to be a new path.* FR-022 and SC-008 now forbid it: this drives the existing
   capture queue and camera-preset machinery. A second automation path would diverge from the first
   and double the surface that has to keep working.
3. *Equipment was assumed to be "attach a model".* Not all of it is — some appearances are geoset
   visibility or texture changes on the body. FR-012 requires those to be handled or explicitly
   reported, never silently ignored, because silent omission would look like a working paper doll
   that quietly drops half of what it is given.
4. *Era gating was an assumption line.* Promoted to US6 at P1 for a reason specific to this feature:
   the reconstruction's entire value is that it uses **era-correct assets**. An era-mismatched model
   would make the lighting comparison meaningless while looking completely plausible.
5. *Reproducibility was missing from automation.* FR-020 (wait until loaded and settled) and FR-021
   (repeated capture is identical) were added — a capture taken mid-stream would silently compare a
   half-loaded scene against the reference.

**Deliberate deviations:**

- *"No implementation details"* passes for Requirements and Success Criteria. **Context** names
  existing types because the measured gap in them is the justification for the feature.
- *"Written for non-technical stakeholders"* stays partial; the reader is the operator.

**Open risks carried into planning:**

- **Item-to-appearance resolution is entirely unmeasured.** It is the largest unknown here, and
  planning must establish it from the data rather than assume it mirrors the creature-display path.
- **"The night colour profile is already correct" is an operator judgement, not a measurement.** It
  is recorded as an assumption. The reconstruction is partly a way of checking it, so planning should
  not treat it as settled input.
- **Attachment identity across eras.** Whether attachment identifiers are stable between 0.5.3-era
  and later data is unknown; the spec requires enumeration and reporting rather than assuming a fixed
  table.
- **Effect budget when attachments carry their own effects.** An attached torch adds emitters to a
  host that already has some, and this project has a measured per-object cost problem.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.
