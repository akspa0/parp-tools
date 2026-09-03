# Feature Specification: Model Cursor as a Scene Light Source

**Feature Branch**: `216-model-cursor-light-source` (authored on `v0.5.3`; specs 193–215 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "we need to be able to set the cursor to use any mdx or m2 with particle effects or lights, to act as the mouse cursor. The idea is to use the `areatest.lit` lighting profile with a torch.mdx/torch.m2, as the cursor, to see the game from a totally foreign point of view that was only shown in a single screenshot from 2001, where a thief has a torch in one hand, and it's the only light source in the dark scene at around 3am on the time of day slider."

## Context & Motivation

There is one 2001 screenshot of a thief holding a torch in a dark scene, the torch the only light in
it. That image is a record of how the alpha engine's lighting actually behaved — not a mock-up, and
not something any current view of this data reproduces. Reconstructing the conditions that produce it
is a measurement, not a decoration: if the scene can be made to look like the screenshot, the
lighting model is right; if it cannot, something in it is wrong and the screenshot says so.

Spec 210 put a cursor into the scene as real geometry and established the asset path for it. This
feature generalises that cursor from a fixed set of styles to **any model**, and then makes the
model's own lights and particles real — which is what turns "a torch model follows the pointer" into
"a torch lights the room".

Four facts were measured in the existing code before this spec was written:

- **`areatest.lit` is already a probed LIT variant.** `LitSourcePathResolver` looks for
  `World\<map>\areatest.lit` and `World\Maps\<map>\areatest.lit`, and `LitLoader` names it in its
  no-variant-found status alongside `lights.lit` and `light.lit`. The profile the operator wants is
  already reachable; nothing needs inventing to select it.
- **Particles already render.** `ModelRenderer` builds `ParticleEmitter`s from
  `MdxParticleEmitter2` and draws them through `ParticleRenderer` in the transparent pass. A torch
  cursor's flame should work through the existing path.
- **MDX lights are parsed and partly used** — `MdxLightSummary`, `MdxLightType`, the `LITE` chunk,
  and `UploadMdxLights` in `ModelRenderer`.
- **But those lights light only their own model.** `UploadMdxLights` uploads a model's `LITE` lights
  as uniforms into *that model's own shader program*, capped at `MaxMdxLocalLights = 8`. A torch's
  flame currently illuminates the torch and nothing else. **This is the gap.** Terrain, WMOs and
  other doodads receive nothing from it, so the defining property of the 2001 screenshot — one small
  light source picking out the world around it — cannot happen today no matter which model is used
  as a cursor.

The time-of-day control needed to reach 3am exists as `ImGui.SliderFloat("Time of Day", 0..1)` in
`DrawTimeOfDayControl`. Replacing that linear slider with an interactive clock is a UI concern and is
owned by **spec 212**, not by this feature; the two meet only in that this feature needs *a* way to
set the hour, and one already exists.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Any model can be the cursor (Priority: P1)

The operator picks any MDX or M2 from the loaded client data and it becomes the pointer — a torch, a
lantern, a weapon, anything. It is positioned, scaled and oriented the way the existing scene cursor
is, and it animates.

**Why this priority**: It is the entry point to everything else here, and it delivers on its own:
being able to hold any asset at the pointer and move it through the world is a genuinely useful way
to look at models in context, independent of any lighting work.

**Independent Test**: Select several models of differing size and complexity as the cursor and
confirm each loads, renders at the pointer, animates, and can be swapped without restarting.

**Acceptance Scenarios**:

1. **Given** any loadable MDX or M2 in the client data, **When** the operator selects it as the
   cursor model, **Then** it renders at the pointer with the existing cursor's positioning and
   scaling behaviour.
2. **Given** a cursor model with animations, **When** it renders, **Then** its animation plays.
3. **Given** a model that fails to load or is not a valid model, **When** it is selected, **Then**
   the previous cursor remains active and the failure is reported.
4. **Given** a very large or very small model, **When** it is used as the cursor, **Then** it is
   scaled to a usable on-screen size and the operator can adjust that scale.
5. **Given** a cursor model selection, **When** the viewer is closed and reopened, **Then** the
   selection is restored.

---

### User Story 2 - The cursor model's particles play (Priority: P2)

A torch cursor has a flame. Particle emitters on the cursor model run and are drawn.

**Why this priority**: It is most of the visual identity of the torch, and the machinery already
exists — this is wiring the cursor into a working path, not building one. It follows US1 because
there must be a model before it can emit.

**Independent Test**: Use a model with particle emitters as the cursor and confirm emission,
movement and lifetime look correct as the pointer moves.

**Acceptance Scenarios**:

1. **Given** a cursor model with particle emitters, **When** it renders, **Then** its particles are
   emitted and drawn.
2. **Given** the pointer moving quickly, **When** particles are emitted, **Then** they behave
   plausibly relative to the motion rather than teleporting with it as a rigid block.
3. **Given** a cursor model with no emitters, **When** it renders, **Then** nothing is emitted and no
   cost is incurred.

---

### User Story 3 - The cursor model's lights illuminate the scene (Priority: P1)

The torch lights the world. Its `LITE` lights contribute to terrain, WMOs and other models around
the pointer, falling off with distance, so moving the pointer moves a pool of light through the
scene.

**Why this priority**: This is the feature. It is also the only part that does not already exist —
model lights currently illuminate only the model that owns them. Without this, every other story
here produces a torch that glows on its own in an unchanged dark room, which is precisely not the
screenshot.

**Independent Test**: Place a light-emitting cursor model in a dark scene and confirm that surfaces
near the pointer are lit, that lighting tracks the pointer, and that it falls off with distance.

**Acceptance Scenarios**:

1. **Given** a cursor model carrying one or more lights, **When** the scene renders, **Then**
   terrain, world objects and other models near the pointer are illuminated by them.
2. **Given** the pointer moving, **When** the scene renders, **Then** the illuminated region follows
   it continuously.
3. **Given** a lit surface at increasing distance from the light, **When** rendered, **Then**
   illumination falls off according to the light's declared attenuation rather than cutting off
   abruptly.
4. **Given** a cursor model with more lights than the renderer supports at once, **When** rendered,
   **Then** the selection of which lights apply is deterministic and stated, not arbitrary.
5. **Given** a cursor model with no lights, **When** rendered, **Then** scene lighting is unchanged
   from before this feature.
6. **Given** the cursor light active, **When** frame cost is measured, **Then** it stays within the
   feature's declared budget.

---

### User Story 4 - The 2001 thief scene can be reproduced (Priority: P1)

The operator can put the viewer into the configuration the screenshot shows: the `areatest.lit`
profile, time set to roughly 3am, ambient and directional light at their night values, and a torch
cursor as the only meaningful light. The result is judged against the screenshot.

**Why this priority**: This is the acceptance test for the whole feature and the operator's actual
goal. It is also the measurement: a mismatch is evidence about the lighting model, not merely a
disappointing picture.

**Independent Test**: Configure the scene as described and compare against the 2001 screenshot.

**Acceptance Scenarios**:

1. **Given** a map with an `areatest.lit` profile, **When** the operator selects it, **Then** it is
   loaded and applied, and the UI states which LIT source is active.
2. **Given** the time set to approximately 3am, **When** the scene renders, **Then** it is dark
   enough that a single small light source is the dominant illumination.
3. **Given** a torch cursor in that scene, **When** it moves, **Then** it reads as the only light
   source, revealing surfaces as it passes.
4. **Given** the reproduced scene, **When** compared against the 2001 screenshot, **Then** the
   operator can state which aspects match and which do not, and any mismatch is recorded as a
   lighting finding rather than dismissed.
5. **Given** this configuration, **When** the operator wants to return to it, **Then** it can be
   saved and restored as a named preset.

---

### User Story 5 - Era behaviour is gated and provenance is carried (Priority: P1)

Cursor model lighting respects era differences, and every lighting contribution records where it came
from.

**Why this priority**: P1 because it constrains the other work rather than following it. Light
attenuation, ambient handling and LIT semantics are era-scoped in this project — the same lesson
MCNR component order and minimap generation both taught — and the 2001 screenshot is alpha-era
evidence that must not be reproduced with a later era's lighting model and then believed.

**Independent Test**: Load content of each supported era with a lighting cursor and confirm each uses
its own profile, with the decision recorded.

**Acceptance Scenarios**:

1. **Given** content of any supported era, **When** the cursor light is applied, **Then** the era's
   own lighting rules are used.
2. **Given** an unrecognised build, **When** it loads, **Then** it is flagged rather than silently
   defaulted.
3. **Given** any lighting result, **When** inspected, **Then** it carries the era profile and the LIT
   source that produced it.

---

### Edge Cases

- **The cursor is over UI, or off-window.** A light that stays lit at a stale world position when the
  pointer is not in the scene.
- **The cursor points at the sky.** There is no surface hit, so the light has no anchor distance.
- **More scene lights than the renderer supports.** The cursor light must compete with, or take
  precedence over, existing scene lights by a stated rule.
- **A model whose lights are animated**, including ones that flicker or are keyed off an animation
  that is not playing.
- **A model with an ambient-type light**, which is not a point source and must not be treated as one.
- **Enormous attenuation ranges.** A light declaring a range that would illuminate the whole map.
- **A cursor model with thousands of particles**, at the near plane, filling the screen.
- **Deferred or missing textures** on a cursor model chosen before its textures resolve.
- **The map has no `areatest.lit`.** Most will not; the failure must be clear rather than silent.
- **Capture runs.** A cursor light in a recorded frame changes the recording; capture behaviour must
  be defined.

## Requirements *(mandatory)*

### Functional Requirements

**Cursor model**

- **FR-001**: Operators MUST be able to select any loadable MDX or M2 in the client data as the
  cursor model.
- **FR-002**: The system MUST render the selected model at the pointer using the existing scene
  cursor's positioning, scaling and depth behaviour.
- **FR-003**: The system MUST play the cursor model's animations.
- **FR-004**: The system MUST keep the previous cursor active and report the failure when a selected
  model cannot be loaded.
- **FR-005**: Operators MUST be able to adjust cursor model scale, and the selection and scale MUST
  persist across sessions.

**Particles**

- **FR-006**: The system MUST run and draw the cursor model's particle emitters.
- **FR-007**: The system MUST incur no particle cost for a cursor model that has none.

**Scene lighting**

- **FR-008**: The system MUST apply the cursor model's lights to surrounding scene geometry —
  terrain, world objects and other models — not only to the cursor model itself.
- **FR-009**: The system MUST apply each light's declared attenuation.
- **FR-010**: The system MUST apply a stated, deterministic rule when more lights are present than
  can be applied at once.
- **FR-011**: The system MUST handle non-point light types according to their declared type rather
  than treating every light as a point source.
- **FR-012**: The system MUST leave scene lighting unchanged when the cursor model carries no lights.
- **FR-013**: The system MUST define cursor-light behaviour when the pointer is outside the scene
  viewport or has no surface hit.
- **FR-014**: Cursor lighting MUST stay within a declared frame budget.

**The reproduction scenario**

- **FR-015**: Operators MUST be able to select the `areatest.lit` profile where a map provides one,
  and the UI MUST state which LIT source is active.
- **FR-016**: The system MUST report clearly when a requested LIT variant is not present for a map.
- **FR-017**: Operators MUST be able to set the time of day to reach night values. (The control's
  form is spec 212's concern; this feature requires only that the capability exists.)
- **FR-018**: Operators MUST be able to save and restore a named preset capturing cursor model, LIT
  source, time of day and lighting settings.

**Era and capture**

- **FR-019**: The system MUST apply era-appropriate lighting rules and MUST flag unrecognised builds
  rather than defaulting them.
- **FR-020**: Every lighting result MUST carry its era profile and LIT source as provenance.
- **FR-021**: The system MUST define whether the cursor and its light appear in capture output, and
  MUST leave existing capture behaviour unchanged by default.

### Key Entities

- **Cursor Model**: The model currently acting as the pointer — its asset identity, scale, animation
  state, particle emitters and lights.
- **Cursor Light Contribution**: The illumination the cursor model contributes to the scene: position,
  colour, intensity, attenuation and type.
- **Scene Light Set**: The lights applied to a surface when it is shaded, and the rule that selects
  them when there are more than can be applied.
- **LIT Profile Selection**: Which LIT source is active for the current map, and why.
- **Scene Preset**: A saved, restorable combination of cursor model, LIT source, time of day and
  lighting settings.
- **Era Profile**: The per-build statement of lighting rules, carried as provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Any loadable model in the client data can be made the cursor, and swapping cursor
  models requires no restart.
- **SC-002**: A light-emitting cursor model illuminates terrain, world objects and other models —
  demonstrated by a measurable brightness difference on surfaces near the pointer with the light on
  versus off.
- **SC-003**: Illumination tracks the pointer continuously, with no visible lag or stepping.
- **SC-004**: With a cursor model carrying no lights, scene output is identical to the pre-feature
  baseline.
- **SC-005**: Cursor model rendering, particles and lighting together stay within the declared frame
  budget, measured against the frame-time distribution rather than a static-camera average.
- **SC-006**: The 2001 thief scene can be reproduced and the operator can state, point by point,
  which aspects match the screenshot and which do not.
- **SC-007**: Any mismatch found in SC-006 is recorded as a lighting finding with its evidence, not
  discarded.
- **SC-008**: The scene preset restores an identical configuration across sessions.
- **SC-009**: Every lighting result carries an era profile and LIT source; unrecognised builds are
  flagged, never defaulted.

## Assumptions

- **The measured gap is scene-wide model lighting.** `UploadMdxLights` applies a model's lights only
  within that model's own shader program. Making a model's light illuminate *other* geometry is the
  substantive work of this feature; everything else is selection and wiring. Planning must not assume
  the existing local-light path can be reused unchanged.
- **Particles already work.** The existing `ParticleRenderer` path is expected to serve; if it does
  not, that is a finding about the particle path rather than new scope here.
- **`areatest.lit` already resolves.** `LitSourcePathResolver` probes it today. What is missing is
  operator selection and clear reporting, not a new loader.
- **The screenshot is evidence, not a target to fake.** If the scene cannot be made to match, the
  correct outcome is a recorded lighting finding. Tuning values until a picture matches, without
  understanding why, would produce exactly the kind of unverified result this project has been burned
  by before.
- **The clock control is spec 212's.** This feature needs only that the time of day can be set; the
  existing slider satisfies that.
- **The LIT chain belongs to spec 143.** This feature selects a profile and reports it; it does not
  re-implement LIT parsing or the lighting chain.
- **Testability follows the standing constraint.** No test project references the viewer, so light
  selection rules, attenuation, preset serialisation and era gating belong in `WowViewer.Core*`.
- **Visual judgement is operator work.** The screenshot comparison is not automatable.

## Out of Scope

- The interactive clock control and any other UI form work (spec 212).
- Re-implementing LIT parsing or the day/night lighting chain (spec 143; see also 106, 160).
- A general dynamic-lights system for arbitrary world objects. This feature makes the *cursor* model
  a scene light; generalising that to every doodad is a larger change and a separate decision.
- Shadow casting from cursor lights.
- Weather and fog interaction (spec 215).
