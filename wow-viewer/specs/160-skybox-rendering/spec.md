# Feature Specification: Skybox Rendering

**Feature Branch**: `160-skybox-rendering`

**Created**: 2026-08-17

**Status**: Draft

**Input**: User description: "Get skyboxes working in the wow-viewer renderer." Five confirmed defects, source-agnostic era scope with mandatory provenance, layering contract resolved by research.

## Context

The viewer already contains most of the parts a working sky needs: a procedural sky dome, a
client-skybox-model render path, a LIT loader that decodes sky colour bands, a Light\* DBC chain
that resolves a `LightSkybox` model name, and a WMO reader that parses `MOSB`. None of it reaches
the screen correctly. The sky the user sees today is, in every case, a hardcoded two-colour
gradient with no client-authored input.

### Confirmed defects

| # | Defect | Evidence |
|---|---|---|
| D1 | Client-authored sky colours are computed, assigned, then discarded every frame. `_skyDome.UpdateFromLighting(...)` runs *after* the LIT/DBC branches assign `ZenithColor`/`HorizonColor`, and unconditionally overwrites both with hardcoded day/night lerps | `WorldScene.cs:10566-10572`, `SkyDomeRenderer.cs:69-99` |
| D2 | The client skybox model renders only at night. The draw is gated behind `_skyDome.NightVisibility > 0.001f`, so no skybox model appears in daylight | `WorldScene.cs:11882` |
| D3 | Two of five authored sky bands are used. The source authors Sky Top / Sky Upper / Sky Middle / Sky Lower / Sky Horizon; the dome shader is a single two-colour `mix` between zenith and horizon | `SkyDomeRenderer.cs:214`, `LitLoader.cs:24-28`, `lit-draft.md` colour-track table (indices 2-6) |
| D4 | WMO-declared skyboxes never reach the renderer. `MOSB` is parsed, but the summary the renderer consumes reduces it to a boolean `HasSkybox` and drops the name; the name-preserving reader has no renderer consumer | `WmoSummary.cs:94`, `WmoSummaryReader.cs:54-72`, `WmoSkyboxSummaryReader.cs` |
| D5 | Skybox identification is filename string-matching (`contains("skybox")`, `contains("skybowl")`, `contains("environments/stars/")`) rather than a declared relationship from client data | `WorldSkyboxBackdropClassifier.cs:22-25` |

### Research: the dome/model layering contract

The open question was whether a resolved client skybox model should *replace* the procedural
gradient or *composite over* it. Client research resolves this: the day/night system carries a
sky-gradient state (`LightDataSky`) **and** a separate skybox model reference as two distinct
things, and the canonical model asset is a star field, which is only meaningful drawn over a
darkened gradient rather than instead of one.

**Resolved contract**: the gradient is the base layer; resolved skybox models composite over it;
both stay behind all world geometry. A fully opaque skybox model naturally hides the gradient
without needing a special case, and a sparse one (stars) correctly shows gradient through it. This
also means a missing or unresolvable skybox model degrades to a correct gradient rather than to a
hole.

### Source selection constraint (carried from existing project rules)

Map-scoped LIT tracks and the Light\* DBC chain are **separate sources that must not be silently
mixed** — mixing them produces a sky profile no client file ever authored. Source-agnostic in this
spec therefore means *whichever single coherent source resolves for the loaded build wins*, never
*blend fields from both*.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Authored sky colours reach the screen (Priority: P1)

A user loads a map whose client data authors sky colours for the time of day. The sky they see is
the sky the client data describes. When they scrub the time-of-day control, the sky changes the way
the authored data says it should, not the way a hardcoded curve says it should.

**Why this priority**: This is the single defect that invalidates every other sky feature. While
authored colours are discarded each frame, correct band decoding, correct model selection, and
correct WMO swapping are all invisible. Nothing else in this spec can be visually validated until
this is fixed, and it is the smallest change with the largest visible delta.

**Independent Test**: Load a map with an authored sky profile, note the rendered sky colour, then
alter the authored source value and reload. The rendered sky must change correspondingly. Today it
does not change at all.

**Acceptance Scenarios**:

1. **Given** a map whose resolved profile authors sky colours, **When** the frame renders, **Then**
   the sky colours on screen are the authored values, not hardcoded constants.
2. **Given** an authored sky profile, **When** the user advances the time of day, **Then** the sky
   follows the authored timed samples for that source.
3. **Given** a map with no resolvable sky profile, **When** the frame renders, **Then** a documented
   fallback sky renders and reports itself as the fallback — never a black or untextured sky.
4. **Given** any rendered frame, **When** the user inspects sky diagnostics, **Then** each sky colour
   value names the source and record that supplied it.

---

### User Story 2 - The skybox model is visible across the whole day (Priority: P1)

A user flies around a map at noon and sees the client's skybox model, not an empty gradient. The
same model remains correctly present at dawn, dusk, and midnight, changing appearance with the time
of day rather than appearing and disappearing.

**Why this priority**: This is what "skyboxes are not working" most directly names. The model is
resolved, loaded, and drawn correctly — but only during a narrow night window, so for most of the
day cycle the feature is simply absent. Fixing it is a gate removal plus correct animation timing,
independent of the colour path in US1.

**Independent Test**: Set the time of day to noon on a map with a resolvable skybox model and
confirm the model renders. Sweep the full day cycle and confirm it is continuously present with no
pop-in or pop-out at the day/night boundary.

**Acceptance Scenarios**:

1. **Given** a map with a resolvable skybox model, **When** the time of day is midday, **Then** the
   skybox model renders.
2. **Given** the same map, **When** the user sweeps the full day cycle, **Then** the skybox model is
   continuously present and never blinks out at a threshold.
3. **Given** a skybox model that encodes time-varying appearance, **When** the time of day advances,
   **Then** the model's appearance advances with the world clock rather than free-running or holding
   a single frame.
4. **Given** a skybox model and world geometry, **When** the frame renders, **Then** all world
   geometry draws in front of the skybox and the skybox never occludes terrain or objects.

---

### User Story 3 - The full authored sky gradient is reproduced (Priority: P2)

A user looking from horizon to zenith sees the full authored vertical gradient — the distinct bands
the client data describes — rather than a straight blend between two endpoint colours.

**Why this priority**: Depends on US1 to be observable at all. Once authored colours survive, using
only two of five bands is the remaining visible fidelity gap: the middle of the sky is interpolated
rather than authored, which flattens dawn and dusk in particular.

**Independent Test**: Alter one authored mid-sky band value in isolation and confirm the rendered
sky changes in that band's region while the zenith and horizon stay put.

**Acceptance Scenarios**:

1. **Given** a profile authoring five sky bands, **When** the frame renders, **Then** all five bands
   contribute to the rendered gradient.
2. **Given** the same profile, **When** one mid-sky band is changed in isolation, **Then** the change
   is visible in that band's region of the sky and the other bands are unaffected.
3. **Given** a source that authors fewer than five bands, **When** the frame renders, **Then** the
   available bands are used and the shortfall is reported rather than silently zero-filled.
4. **Given** the rendered gradient, **When** viewed across band boundaries, **Then** the transitions
   are continuous with no visible banding seam.

---

### User Story 4 - WMO interiors use their declared skybox (Priority: P3)

A user walks into a building or instance whose WMO declares its own skybox. The sky visible through
its windows and openings becomes that declared skybox. Walking back out restores the outdoor sky.

**Why this priority**: A real client behaviour and the reason `MOSB` is parsed at all, but it
affects a bounded set of locations rather than every frame everywhere, and it needs a working
outdoor skybox path (US1/US2) underneath it before it can be validated.

**Independent Test**: Enter a WMO known to declare a skybox and confirm the visible sky changes;
leave and confirm it reverts.

**Acceptance Scenarios**:

1. **Given** a WMO that declares a skybox, **When** the camera is inside it, **Then** the declared
   skybox is the sky that renders.
2. **Given** the same WMO, **When** the camera leaves it, **Then** the outdoor sky is restored.
3. **Given** a WMO whose declared skybox name cannot be resolved to a loadable asset, **When** the
   camera is inside it, **Then** the outdoor sky continues to render and the unresolved reference is
   reported.
4. **Given** a WMO that declares no skybox, **When** the camera is inside it, **Then** the outdoor
   sky continues to render unchanged.
5. **Given** a transition between outdoor and WMO-declared skies, **When** the camera crosses the
   boundary, **Then** the change does not flicker or oscillate on repeated boundary crossings.

---

### User Story 5 - Skybox identification comes from client data (Priority: P3)

A user loading any map gets the skyboxes the client data declares, including ones whose asset paths
contain none of the words the current heuristic looks for, and does not get ordinary world models
misclassified as sky merely because of their filename.

**Why this priority**: A correctness and generality fix rather than a visible-today fix. The
heuristic happens to work for the common cases, so the payoff is avoiding silent wrong behaviour on
maps and builds outside those cases — worth doing, but only after the visible path works.

**Independent Test**: Identify an asset that the declared client data marks as a skybox but whose
path does not contain the heuristic's keywords, and confirm it is treated as a skybox. Then confirm
a non-sky asset whose path happens to contain a keyword is not treated as one.

**Acceptance Scenarios**:

1. **Given** an asset declared as a skybox by client data with a non-matching filename, **When** the
   scene loads, **Then** it is classified as a skybox.
2. **Given** a non-sky asset whose filename contains a heuristic keyword, **When** the scene loads,
   **Then** it is not classified as a skybox.
3. **Given** any classified skybox, **When** the user inspects diagnostics, **Then** the classifier
   reports which declaration classified it.
4. **Given** a build where no declaration source is available, **When** the scene loads, **Then** the
   previous filename-based behaviour is used as an explicitly-reported fallback rather than
   classifying nothing.

---

### Edge Cases

- **Both a LIT profile and a Light\* DBC chain resolve for the same build.** One coherent source is
  selected and reported; fields are never blended across sources.
- **The profile resolves but authors no sky bands.** The fallback sky renders and reports itself.
- **The skybox model name resolves but the asset is missing or fails to load.** The gradient renders
  alone; the unresolved reference is reported, and the failure is not retried every frame.
- **The skybox asset is still streaming in.** The gradient renders alone until the asset is ready;
  no black frame, no stall waiting on the load.
- **Multiple skybox placements are in range.** A single active skybox is selected deterministically,
  so the choice does not flicker as the camera moves between equidistant candidates.
- **The camera is inside nested or overlapping WMOs with different declared skyboxes.** Selection is
  deterministic and does not oscillate frame to frame.
- **The camera crosses a WMO boundary repeatedly.** The sky does not strobe.
- **The time of day wraps past midnight.** The gradient and the model's time-driven appearance both
  wrap continuously with no discontinuity at the wrap point.
- **Sky rendering is disabled by the user.** No sky source is evaluated and no sky work is submitted.

## Requirements *(mandatory)*

### Functional Requirements

#### Sky colour source and provenance

- **FR-001**: The renderer MUST resolve sky colours from whichever sky profile source is available
  for the loaded build, without an era gate on which source is consulted.
- **FR-002**: The renderer MUST select exactly one coherent sky source per evaluation and MUST NOT
  blend individual sky fields across different sources.
- **FR-003**: Every resolved sky value MUST carry provenance identifying the source and the record
  that supplied it, and that provenance MUST be inspectable at runtime.
- **FR-004**: Client-authored sky colours MUST reach the rendered frame. No later stage may
  overwrite a resolved authored value with a hardcoded value.
- **FR-005**: When no sky profile resolves, the renderer MUST render a documented fallback sky and
  MUST report that the fallback — not authored data — is in use.
- **FR-006**: Sky colours MUST follow the world time-of-day clock using the authored timed samples
  of the selected source.

#### Sky gradient

- **FR-007**: The rendered sky gradient MUST reproduce every sky band the selected source authors,
  positioned at its authored place in the vertical gradient.
- **FR-008**: When a source authors fewer bands than the full band set, the renderer MUST use the
  bands present and report the shortfall rather than silently substituting values.
- **FR-009**: Band-to-band transitions in the rendered gradient MUST be continuous, with no visible
  seam at a band boundary.

#### Skybox models

- **FR-010**: A resolved skybox model MUST render at every point in the day/night cycle. Visibility
  MUST NOT be gated on a night-only or day-only condition.
- **FR-011**: A skybox model's time-varying appearance MUST be driven by the world time-of-day clock.
- **FR-012**: Skybox models MUST composite over the sky gradient, and both MUST render behind all
  world geometry. World geometry MUST never be occluded by the sky.
- **FR-013**: When multiple skybox candidates are in range, the renderer MUST select the active one
  deterministically so the selection does not oscillate between frames at equal distance.
- **FR-014**: A skybox model that is missing, unresolvable, or still loading MUST NOT prevent the sky
  gradient from rendering, and its failure MUST be reported once rather than retried every frame.

#### WMO-declared skyboxes

- **FR-015**: The WMO read path the renderer consumes MUST preserve the declared skybox name, not
  only whether one exists.
- **FR-016**: When the camera is inside a WMO that declares a skybox, that skybox MUST become the
  active sky; when the camera leaves, the outdoor sky MUST be restored.
- **FR-017**: WMO skybox selection MUST be deterministic and stable under nested WMOs and repeated
  boundary crossings.
- **FR-018**: An unresolvable WMO-declared skybox MUST fall back to the outdoor sky and report the
  unresolved reference.

#### Classification

- **FR-019**: Skybox identification MUST be driven by client-declared relationships rather than by
  matching keywords in asset filenames.
- **FR-020**: When no declaration source is available for a build, the renderer MUST fall back to the
  existing filename-based identification and MUST report that the fallback is in use.
- **FR-021**: Each classified skybox MUST report which declaration classified it.

#### Non-regression

- **FR-022**: Sky rendering MUST NOT introduce new frame-time hitches or increase steady-state frame
  cost beyond an agreed budget, measured with the project's existing frame-time instrumentation.
- **FR-023**: When the user disables sky rendering, no sky source evaluation and no sky draw
  submission MUST occur.
- **FR-024**: Sky asset resolution and loading MUST NOT block the render thread.

### Key Entities

- **Sky Profile**: The resolved, coherent set of sky values for the loaded build and current map —
  the band colours, the timed samples that drive them, and the skybox model reference. Carries the
  identity of the source it came from.
- **Sky Band**: One authored colour at a defined height in the vertical sky gradient. An ordered set
  of these defines the gradient from horizon to zenith.
- **Skybox Model Reference**: A named client asset to be drawn as sky, together with where the name
  came from — an outdoor profile record, a WMO declaration, or a fallback discovery.
- **Sky Provenance Record**: For any rendered sky value, the source and record that supplied it, plus
  whether it is authored data or a fallback. Attached to every value, inspectable at runtime.
- **Active Sky Selection**: The single sky in effect for the current frame — the chosen gradient
  source, the chosen skybox model, and the reason each was chosen (outdoor, WMO interior, fallback).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Changing an authored sky colour in the client data changes the rendered sky
  correspondingly. Today, 0% of authored sky colour changes affect the rendered frame; the target is
  100%.
- **SC-002**: A resolvable skybox model is visible at 100% of sampled points across a full day cycle,
  up from the current night-only window.
- **SC-003**: Every authored sky band the source provides is independently observable in the rendered
  gradient — changing any one band alone produces a visible change confined to its region.
- **SC-004**: 100% of rendered sky values report the source and record that supplied them, including
  values that come from a fallback.
- **SC-005**: Entering a WMO that declares a skybox changes the visible sky, and leaving restores it,
  on every crossing, with no flicker across repeated crossings.
- **SC-006**: A skybox declared by client data but not matching the old filename keywords is
  correctly treated as sky; a non-sky asset that does match the keywords is not.
- **SC-007**: Every failure mode — no profile, missing asset, still-loading asset, unresolvable WMO
  reference — still renders a sky. Zero black or untextured sky frames across the failure matrix.
- **SC-008**: Frame-time distribution shows no new hitches attributable to sky work, and steady-state
  sky cost stays within its agreed budget, measured on the project's existing instrumentation.
- **SC-009**: With sky rendering disabled, sky evaluation and sky draw cost measure zero.

## Assumptions

- **Layering is settled by research, not left open.** The gradient is the base layer and skybox
  models composite over it, both behind world geometry — grounded in the client carrying sky-gradient
  state and a skybox model reference as separate things, and in the canonical model being a star
  field that requires a gradient behind it.
- **Source-agnostic means single-source selection, not field blending.** Where both a map-scoped LIT
  profile and a Light\* DBC chain resolve, one is selected wholesale and recorded. Absent other
  direction, the more specific map-scoped source is preferred over the global one, and the choice is
  reported rather than assumed.
- **The existing sky dome and backdrop render paths are the foundation.** This is repair and
  extension of the current paths, not a replacement sky renderer.
- **The camera-inside-WMO determination already available in the runtime is the trigger for
  US4**; this spec does not introduce a new interior-detection mechanism.
- **The project's existing frame-time instrumentation is the measurement tool for FR-022 and
  SC-008**; this spec does not add a new profiler.
- **Real-client visual proof is user-run.** Per project rules, build and test success is not
  rendering proof; the visual acceptance scenarios here are validated by the user against a real
  client, with commands prepared for them.
- **The concrete frame-cost budget for FR-022/SC-008 is set during planning**, against a baseline
  captured before any change, rather than asserted in this spec.
- **No new client formats are decoded.** All sources named here — LIT tracks, the Light\* DBC chain,
  and WMO `MOSB` — already have readers in the project.

## Out of Scope

- Clouds, cloud density, and cloud masks, including the LIT cloud fields.
- Weather systems and weather-driven sky changes.
- Sun and moon disc rendering beyond what already exists in the dome.
- Light shafts, sun glare, bloom, and other atmospheric post effects.
- New client-format decoding or new file readers.
- Underwater and other volumetric sky replacements.
- Reworking the terrain, object, or fog lighting paths, except where fog colour is already shared
  with the sky gradient.
