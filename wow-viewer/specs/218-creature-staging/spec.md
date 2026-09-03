# Feature Specification: Creature Staging — Spawn, Equip, Pose, Reconstruct

**Feature Branch**: `218-creature-staging` (authored on `v0.5.3`; specs 193–217 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "If we could spawn creatures and give them attachments properly, with some sort of paper doll style ui, then we could very easily test this with all the right objects, in the renderer. We should be able to automate that too, and do it via the existing automation in the viewer, so we could try to recreate the screenshot with the 2003 era assets and areatest.lit with the time set to around 3am."

## Context & Motivation

The reference screenshot for spec 216 shows a rogue at night holding a lit torch, the flame throwing
warm light onto a stone well, the wooden frame beside it, his own arm, and the ground. Everything
else is in cool night colour.

The operator's reading of the current viewer against that image is precise and narrows the problem
usefully: **the night colour profile is already right. What is missing is the point light.** That
makes reconstruction a genuine test rather than an aesthetic exercise — nearly every variable is
already correct, so a difference in the result is attributable.

But the scene cannot be assembled today, because there is no way to put a torch in a character's
hand. That is this feature. Spec 216 makes an effect cast light; **this spec builds the rig that
holds the effect in the right place, on the right model, in the right scene** — and then makes that
arrangement replayable so the comparison can be repeated whenever the lighting changes.

Four things were measured in the existing code before this spec was written:

- **Spawning already exists.** `WorldSpawnRecord` and `WorldScene.SetExternalSpawns` place creature
  instances in the world, and `AlphaCoreDbReader` resolves them through
  `creature_template.display_id1 → CreatureDisplayInfo.ModelID → mdx_models_data.ModelName`.
- **Attachment points are parsed and never rendered.** `MdxAttachment`, `MdxAttachmentFile` and
  `MdxAttachmentSummary` exist in Core, and **nothing in `src/viewer/WoWViewer/Rendering/` references
  any of them.** No model is ever attached to another model's attachment point. **This is the gap** —
  the reason a torch cannot be placed in a hand.
- **Equipment display is not resolved.** The catalog reads `CreatureDisplayInfo`; there is no
  equivalent path from an item to the model and texture it displays as.
- **Capture automation already exists**, with a capture queue, `CameraShotPoint` presets persisted to
  `camera_shot_points.json`, batch handling, and UI-chrome suppression. A scene reconstruction should
  drive that rather than grow a second automation path.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A model can be attached to another model's attachment point (Priority: P1)

A torch is placed in a character's hand and stays there — following the hand through animation,
inheriting its position, orientation and scale.

**Why this priority**: This is the measured gap and the thing every other story here depends on.
It also stands alone: being able to attach any model to any attachment point on any other model is
directly useful for inspecting how assets were meant to fit together, independent of lighting.

**Independent Test**: Attach a model to a named attachment point on an animated model and confirm it
tracks the point through a full animation cycle without drift or lag.

**Acceptance Scenarios**:

1. **Given** a model with attachment points, **When** it is inspected, **Then** its attachment points
   are enumerated with their identities.
2. **Given** an attachment point and a second model, **When** the second is attached, **Then** it
   renders at that point with the correct position, orientation and scale.
3. **Given** an attached model on an animating host, **When** the host animates, **Then** the
   attachment follows the point every frame, without drift or a frame of lag.
4. **Given** an attached model that has its own animation or effects, **When** it renders, **Then**
   those play — an attached torch burns.
5. **Given** an attachment point that a model does not have, **When** attachment is requested,
   **Then** it is reported rather than silently placed at the origin.
6. **Given** an attached model, **When** it is detached, **Then** it is removed cleanly and the host
   is unchanged.

---

### User Story 2 - Creatures can be spawned and placed deliberately (Priority: P1)

The operator chooses a creature, places it at a chosen point in the world, and orients it. It can be
moved, re-posed and removed.

**Why this priority**: The existing spawn path populates the world from data; this is about placing a
*chosen* subject deliberately, which is what a reconstruction needs. It is P1 because a torch with
nobody holding it does not reproduce the screenshot.

**Independent Test**: Spawn a chosen creature at a chosen position and orientation, move it, and
remove it, confirming each step.

**Acceptance Scenarios**:

1. **Given** the creature catalog, **When** the operator searches it, **Then** creatures are findable
   by name and by identifier.
2. **Given** a chosen creature, **When** it is spawned, **Then** it appears at the requested position
   and orientation with its correct display model.
3. **Given** a spawned creature, **When** the operator moves, rotates or rescales it, **Then** it
   updates in place.
4. **Given** a spawned creature, **When** an animation is selected, **Then** it plays that animation,
   including holding a chosen pose.
5. **Given** spawned creatures, **When** the operator clears them, **Then** they are removed and the
   scene returns to its prior state.

---

### User Story 3 - A paper-doll panel equips a subject (Priority: P2)

The operator equips items onto a spawned creature through a panel that shows the equipment slots, and
the model updates to display what was equipped.

**Why this priority**: This is the operator's stated way of assembling "all the right objects"
quickly. It sits after US1 and US2 because it is an interface over them — a paper doll with no
attachment rendering behind it can equip nothing.

**Independent Test**: Equip an item into a slot and confirm the subject displays it at the correct
attachment point; unequip and confirm it is removed.

**Acceptance Scenarios**:

1. **Given** a spawned subject, **When** the paper-doll panel is opened, **Then** its equipment slots
   are shown.
2. **Given** a slot and an item, **When** the item is equipped, **Then** the subject displays it at
   the attachment point that slot maps to.
3. **Given** an equipped item, **When** it is unequipped, **Then** it is removed from the subject.
4. **Given** an item whose display data cannot be resolved, **When** it is equipped, **Then** the
   failure is reported and the subject is left in a valid state.
5. **Given** an item that is not a model — a texture-only or geoset-only appearance — **When** it is
   equipped, **Then** it is either applied correctly or explicitly reported as unsupported, never
   silently ignored.

---

### User Story 4 - A scene arrangement is saved and replayed (Priority: P1)

The whole arrangement — subject, equipment, pose, position, the LIT profile, the time of day, and the
camera — is saved as a named scene and can be restored exactly.

**Why this priority**: This is what turns a one-off screenshot attempt into a **repeatable
measurement**. Spec 216's value depends on being able to re-run the comparison every time the
lighting model changes; without this, every re-test is a manual reassembly and the results are not
comparable. P1 because it is the difference between a demo and an instrument.

**Independent Test**: Save an arrangement, change the scene entirely, restore it, and confirm every
element returns to its saved state.

**Acceptance Scenarios**:

1. **Given** an assembled scene, **When** it is saved, **Then** subject, equipment, pose, placement,
   LIT profile, time of day and camera are all captured.
2. **Given** a saved scene, **When** it is restored, **Then** every captured element returns to its
   saved value.
3. **Given** a saved scene, **When** it is restored in a later session, **Then** it restores
   identically.
4. **Given** a saved scene referencing content that cannot be loaded, **When** it is restored,
   **Then** the failure names what is missing rather than silently restoring a partial scene.

---

### User Story 5 - Reconstruction runs through the existing automation (Priority: P2)

A saved scene can be captured by the automation the viewer already has, producing an image for
comparison against the reference — without a human reassembling anything.

**Why this priority**: The operator asked for it explicitly, and it is what makes the comparison
cheap enough to repeat. It follows US4 because there must be a saved scene before it can be
automated.

**Independent Test**: Run a saved scene through the capture automation unattended and obtain the
image.

**Acceptance Scenarios**:

1. **Given** a saved scene, **When** it is submitted to the existing capture automation, **Then** it
   is set up, captured, and the result written, with no manual step.
2. **Given** a batch of saved scenes, **When** it runs, **Then** each is captured and the viewer
   returns to its prior state afterwards.
3. **Given** an automated capture, **When** it runs, **Then** UI chrome is suppressed as it is for
   existing capture paths.
4. **Given** the same saved scene captured twice, **When** the results are compared, **Then** they
   are identical apart from deliberately animated elements.
5. **Given** this feature, **When** it is implemented, **Then** it drives the existing capture queue
   and camera-preset machinery rather than introducing a second automation path.

---

### User Story 6 - Era behaviour is gated and provenance is carried (Priority: P1)

Staging uses each era's own creature, display and item data, and every resolved model records where
it came from.

**Why this priority**: P1 because it constrains the rest. The reconstruction's whole value is that it
uses **era-correct assets** — an era-mismatched model would make a lighting comparison meaningless
while looking perfectly plausible. Display and item data differ across eras, and this project has
paid for era-blind resolution before.

**Independent Test**: Stage the same subject against content of different eras and confirm each
resolves through its own chain, with the decision recorded.

**Acceptance Scenarios**:

1. **Given** content of any supported era, **When** a creature or item is resolved, **Then** that
   era's data chain is used.
2. **Given** an unrecognised build, **When** it is used, **Then** it is flagged rather than
   defaulted.
3. **Given** any staged subject, **When** inspected, **Then** it carries the era profile and the
   tables that produced each resolution.
4. **Given** a saved scene, **When** it is restored, **Then** it records which era's assets it was
   built from, so a later comparison cannot silently mix eras.

---

### Edge Cases

- **A model with no attachment points**, or with attachment identifiers this data era does not use.
- **Nested attachments** — something attached to an already-attached model.
- **An attachment whose host is culled, faded, or not yet loaded.**
- **An attached effect and the host's own effects** competing for the same budget.
- **A creature whose display record resolves to a missing or invalid model.**
- **An item that occupies two slots**, or a slot that accepts two models.
- **Equipment that changes the body** — geoset visibility or texture rather than an attached model.
- **A saved scene whose creature, item, map, or LIT profile no longer exists.**
- **A saved scene captured on a different display size or aspect ratio** than it was authored at.
- **Automated capture while assets are still streaming in**, which would capture a half-loaded scene.
- **Time of day advancing during an automated capture**, making it non-reproducible.

## Requirements *(mandatory)*

### Functional Requirements

**Attachment**

- **FR-001**: The system MUST enumerate a model's attachment points with their identities.
- **FR-002**: The system MUST render a model attached to a named attachment point on a host model,
  with correct position, orientation and scale.
- **FR-003**: An attachment MUST follow its point through host animation without drift or a frame of
  lag.
- **FR-004**: An attached model's own animations and effects MUST play.
- **FR-005**: The system MUST report a requested attachment point that does not exist rather than
  placing the model at a default position.
- **FR-006**: The system MUST support detaching cleanly.

**Staging**

- **FR-007**: Operators MUST be able to find creatures by name and identifier and spawn a chosen one
  at a chosen position and orientation.
- **FR-008**: Operators MUST be able to move, rotate, rescale, animate, pose and remove a staged
  subject.
- **FR-009**: The system MUST resolve a creature's display model through the era's own data chain.

**Equipment**

- **FR-010**: The system MUST present the subject's equipment slots and allow equipping and
  unequipping.
- **FR-011**: The system MUST resolve an item to its displayed appearance and apply it at the
  attachment point its slot maps to.
- **FR-012**: The system MUST handle non-model appearances — geoset or texture changes — or report
  them as unsupported; it MUST NOT ignore them silently.
- **FR-013**: The system MUST report an unresolvable item and leave the subject in a valid state.

**Scenes**

- **FR-014**: The system MUST save a named scene capturing subject, equipment, pose, placement, LIT
  profile, time of day and camera.
- **FR-015**: The system MUST restore a saved scene exactly, across sessions.
- **FR-016**: The system MUST name what is missing when a saved scene cannot be fully restored,
  rather than restoring a partial scene silently.
- **FR-017**: A saved scene MUST record the era of the assets it was built from.

**Automation**

- **FR-018**: The system MUST allow a saved scene to be captured through the viewer's existing
  capture automation, unattended.
- **FR-019**: The system MUST support capturing a batch of saved scenes and restoring prior state
  afterwards.
- **FR-020**: Automated capture MUST wait until the scene is fully loaded and settled before
  capturing.
- **FR-021**: Repeated capture of one saved scene MUST produce identical output apart from
  deliberately animated elements.
- **FR-022**: This feature MUST drive the existing capture queue and camera-preset machinery and MUST
  NOT introduce a second automation path.

**Era**

- **FR-023**: The system MUST resolve creature, display and item data through the content's era chain.
- **FR-024**: The system MUST flag unrecognised builds rather than defaulting them.
- **FR-025**: Every resolution MUST carry its era profile and source table as provenance.

### Key Entities

- **Attachment Point**: A named location on a model where another model can be mounted, following the
  host's animation.
- **Attachment**: A model mounted at a point on a host, with its own animation and effects.
- **Staged Subject**: A deliberately placed creature — its display model, placement, scale, animation
  or held pose, and equipment.
- **Equipment Slot**: A place on a subject that an item occupies, and the attachment point or
  appearance change it maps to.
- **Item Appearance**: What an item displays as — an attached model, a geoset change, a texture, or a
  combination.
- **Staged Scene**: The saved, restorable arrangement — subjects, equipment, poses, placements, LIT
  profile, time of day, camera, and the era it was built from.
- **Era Profile**: The per-build statement of which data chains apply, carried as provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A model attached to an animated host tracks its attachment point for a full animation
  cycle with no measurable drift and no frame of lag.
- **SC-002**: An attached torch burns — its effects play while attached.
- **SC-003**: A chosen creature can be found, spawned, placed, posed and removed without restarting.
- **SC-004**: Every equipment slot the paper doll presents either equips correctly or reports its
  limitation — zero silently ignored.
- **SC-005**: A saved scene restores every captured element exactly, across sessions.
- **SC-006**: A saved scene captured twice produces identical output apart from deliberately animated
  elements.
- **SC-007**: A batch of saved scenes runs unattended through the existing automation and leaves the
  viewer in its prior state.
- **SC-008**: This feature adds zero new automation paths — the existing capture queue and camera
  presets are what run.
- **SC-009**: The reference scene can be assembled from era-correct assets and captured, and the
  operator can state point by point how it differs from the reference image.
- **SC-010**: Every staged resolution carries an era profile and source table; unrecognised builds are
  flagged, never defaulted.

## Assumptions

- **The measured gap is attachment rendering.** Attachment points are parsed in Core and referenced
  nowhere in the renderer. That is the substantive work; spawning already exists and the automation
  already exists.
- **Equipment display resolution does not exist yet.** The catalog resolves creature displays only.
  How an item becomes an appearance is unmeasured, and planning must establish it from the data
  rather than assume it mirrors the creature path.
- **The night colour profile is already correct**, per the operator's reading of the current viewer
  against the reference image. That is what makes reconstruction a useful test: if the assembled
  scene still differs, the difference is attributable rather than lost among many wrong variables.
  It is an operator judgement, not a measurement, and the reconstruction is partly a way of checking
  it.
- **This is the test rig for spec 216.** This spec puts a torch in a hand; 216 makes the torch's
  effect light the scene. Neither is much use without the other, but they are separable: attachment
  and staging are worth having on their own, and 216 can be exercised on a bare model.
- **Automation is extended, not replaced.** The existing capture queue, `CameraShotPoint` presets and
  chrome suppression are the machinery. FR-022 forbids a parallel path.
- **Era-correct assets are the point.** An era-mismatched model would make a lighting comparison
  meaningless while looking entirely plausible, which is the worst kind of wrong.
- **Testability follows the standing constraint.** No test project references the viewer, so
  attachment resolution, slot-to-point mapping, scene serialisation and era gating belong in
  `WowViewer.Core*`.
- **Visual comparison is operator work.** Judging the reconstruction against the reference is not
  automatable here.

## Out of Scope

- Making effects cast light (spec 216).
- The clock control and UI shell work (spec 212).
- Character customisation — faces, hair, skin — beyond what equipping requires.
- Gameplay behaviour, AI, movement, or combat for staged creatures.
- Authoring or editing creature, item or display data.
- Server-driven population; this is deliberate placement by the operator.
