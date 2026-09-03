# Specification Quality Checklist: Model Cursor as a Scene Light Source

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

1. `areatest.lit` is already probed by `LitSourcePathResolver` and named in `LitLoader`'s
   no-variant status. Nothing needs inventing to reach that profile.
2. Particles already render — `ModelRenderer` builds emitters from `MdxParticleEmitter2` and draws
   them through `ParticleRenderer` in the transparent pass.
3. MDX lights are parsed (`MdxLightSummary`, `MdxLightType`, `LITE`) and consumed by
   `UploadMdxLights`.
4. **But `UploadMdxLights` uploads a model's lights as uniforms into that model's own shader
   program**, capped at `MaxMdxLocalLights = 8`. A model's light illuminates only itself.

Fact 4 is the whole feature. Without measuring it, this spec would have been written as "wire up the
cursor model" — a day of work that would have produced a torch glowing on its own in an unchanged
dark room, which is exactly not the screenshot. It is called out in Context, in US3's rationale, and
in the Assumptions as something planning must not assume away.

**Iteration 1 — issues found and fixed:**

1. *The first draft treated the 2001 screenshot as a target to match.* Reframed as **evidence**:
   SC-007 requires a mismatch to be recorded as a lighting finding. Tuning values until a picture
   matches, without understanding why, is how this project has produced unverified results before.
2. *"The torch lights the scene" had no measurable form.* SC-002 now requires a measurable
   brightness difference on surrounding surfaces with the light on versus off — a light that only
   lights its own model passes a visual smell test and fails this.
3. *Light-count limits were unaddressed.* The renderer already caps local lights at 8. FR-010
   requires the over-limit selection rule to be stated and deterministic rather than arbitrary.
4. *Ambient-type lights were going to be treated as point lights.* `MdxLightType` distinguishes
   them; FR-011 requires the declared type to be honoured.
5. *The clock control was nearly pulled in here.* It belongs to spec 212. This spec requires only
   that time of day can be set, which the existing slider already satisfies, so neither blocks the
   other.

**Deliberate deviations:**

- *"No implementation details"* passes for Requirements and Success Criteria. **Context** names
  existing types and methods because the measured gap in them *is* the justification for the feature.
- *"Written for non-technical stakeholders"* stays partial; attenuation and LIT profiles are the
  operator's vocabulary.

**Open risks carried into planning:**

- **Scene-wide model lighting is a shader-side change across terrain, WMO and model paths.** Three
  render paths must agree on how an external light reaches them. That is the bulk of the work and is
  not a cursor concern at all.
- **Frame cost is unmeasured.** A dynamic light touching every nearby surface is not free, and this
  project already has a measured per-object cost problem.
- **Generalising to all doodads is explicitly out of scope** but will be the obvious next request;
  planning should not paint itself into a cursor-only corner.
- **LIT chain ownership** stays with spec 143. This feature selects and reports; it must not grow a
  second LIT path.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.
