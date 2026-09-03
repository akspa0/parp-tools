# Feature Specification: 5.0.1 Weather System — Decode and Implement

**Feature Branch**: `215-mop-weather-system` (authored on `v0.5.3`; specs 193–214 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "snow storm weather effects that we also need to implement... very important that we plan for that stuff now, with real data from the real client binary."

## Context & Motivation

The viewer has no weather. Scenes render in permanent still, clear conditions regardless of what the
zone declares, so a snowstorm zone looks like a calm one and the atmosphere that most defines a place
is simply absent.

Weather is also the missing input to spec 214. Flags whip because *wind* drives them; the physics
solver decides how cloth responds, but not how hard it is blowing. Those two specs meet at exactly
that boundary and nowhere else.

Reconnaissance against the real binary (recorded in
[`memory-bank/workstream-atmosphere-501-ghidra.md`](../../memory-bank/workstream-atmosphere-501-ghidra.md))
establishes the anchors:

- `MapWeather.cpp` (`0x00debaae`) — the map-level weather driver.
- `DBFilesClient\Weather.dbc` (`0x00e059f0`) — the data table.
- `Lightning.cpp` (`0x00e26da8`) — storm lightning, a separate system from the precipitation.
- `EffectSwirlingFog.cpp` (`0x00e0ff54`) — a weather-driven fog effect distinct from the base fog
  model that spec 147 owns.
- `ParticleSystem2.cpp` (`0x00d74f84`) — the particle engine precipitation is drawn with.

Weather does not stand alone in the client: it modulates lighting and fog. Those systems already have
owning specs (**143** lighting, **147** fog, **160** sky), so this feature **drives** them through a
declared interface rather than reimplementing them.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The weather contract is recovered from the binary (Priority: P1)

An engineer can read a written contract for how the client selects, blends and applies weather —
what `Weather.dbc` holds, how a zone's weather is chosen, how transitions work, and what each weather
type changes — with every claim citing the address it came from.

**Why this priority**: Everything else depends on it, and it is independently valuable: it settles
what weather *is* in this engine, which is currently unknown to the project.

**Independent Test**: Select any claim in the contract at random; it names an address, and inspecting
that address supports it.

**Acceptance Scenarios**:

1. **Given** the `Weather.dbc` load path, **When** it is decoded, **Then** the table's fields, types
   and meanings are documented with the addresses they were read at.
2. **Given** `MapWeather.cpp`'s functions, **When** they are decompiled, **Then** the selection and
   transition logic is documented, including how weather changes over time.
3. **Given** a field whose purpose is inferred rather than measured, **When** documented, **Then** it
   is marked unverified and named for what it demonstrably is.
4. **Given** the recovered contract, **When** written, **Then** it states explicitly which lighting
   and fog values weather modulates, so specs 143/147/160 can consume it.

---

### User Story 2 - Precipitation renders (Priority: P1)

Snow and rain fall in the scene, in the density and character the zone's weather declares.

**Why this priority**: This is the visible result and the reason for the request. It is independently
testable and delivers value on its own — a snowing zone that snows is worth having before wind,
lightning or fog blending exist.

**Independent Test**: Load a zone with declared weather and observe precipitation matching its
declared type and intensity.

**Acceptance Scenarios**:

1. **Given** a zone declaring snow, **When** it renders, **Then** snow falls at the declared
   intensity.
2. **Given** a zone declaring no weather, **When** it renders, **Then** no precipitation appears and
   cost is unchanged from before this feature.
3. **Given** the camera moving through precipitation, **When** it renders, **Then** particles remain
   distributed around the camera rather than trailing, clumping, or popping in.
4. **Given** the camera indoors or underwater, **When** it renders, **Then** precipitation is
   suppressed where the client suppresses it.

---

### User Story 3 - Weather transitions rather than switching (Priority: P2)

Moving between zones with different weather produces a blend over time, not an instant change.

**Why this priority**: An instant switch reads as a bug and would make US2 look broken in normal use.
It depends on US1's transition contract.

**Independent Test**: Cross a boundary between two differently-weathered zones and observe the blend.

**Acceptance Scenarios**:

1. **Given** two zones with different weather, **When** the camera crosses between them, **Then**
   intensity blends over time rather than switching instantly.
2. **Given** a transition in progress, **When** the camera reverses, **Then** the blend reverses
   smoothly without snapping.
3. **Given** a transition, **When** it completes, **Then** the resulting state matches the
   destination zone's declared weather exactly.

---

### User Story 4 - Weather drives wind, and wind drives physics (Priority: P2)

Weather produces a wind value that spec 214's cloth simulation consumes, so a flag in a storm behaves
differently from a flag in calm air.

**Why this priority**: This is the join between the two specs and the whole reason the operator
raised them together. It is P2 because it needs precipitation working first to be judged, and because
spec 214 must exist to consume it.

**Independent Test**: With physics active, compare flag motion in a declared storm against calm
conditions.

**Acceptance Scenarios**:

1. **Given** active weather, **When** wind is queried at a world position, **Then** a wind value is
   produced whose strength corresponds to the weather's declared intensity.
2. **Given** no weather or physics disabled, **When** wind is queried, **Then** a defined neutral
   value is returned and nothing depends on weather being present.
3. **Given** a weather transition, **When** wind is queried during it, **Then** wind follows the
   blend rather than jumping.

---

### User Story 5 - Weather modulates lighting and fog through their owners (Priority: P2)

Overcast conditions darken and desaturate the scene and thicken fog, by driving the existing lighting
and fog systems rather than by bypassing them.

**Why this priority**: Weather without this reads as particles pasted over an unchanged sunny world.
It is P2 because it requires interfaces from specs 143/147/160 and must not be built by duplicating
what they own.

**Independent Test**: Toggle weather and confirm the lighting and fog values the owning systems
produce change, with the change attributable to weather.

**Acceptance Scenarios**:

1. **Given** active weather, **When** the scene renders, **Then** lighting and fog reflect it, via
   the systems that own them.
2. **Given** weather modulating a value, **When** that value is inspected, **Then** its provenance
   shows weather as the contributing source.
3. **Given** weather removed, **When** the scene renders, **Then** lighting and fog return exactly to
   their unmodulated values.
4. **Given** this feature, **When** it is implemented, **Then** it adds no second lighting or fog
   model of its own.

---

### User Story 6 - Storm lightning (Priority: P3)

Storms produce lightning: flashes that light the scene, on the client's timing.

**Why this priority**: A distinct system in the binary (`Lightning.cpp`), genuinely separate from
precipitation, and the least essential of the visible effects.

**Independent Test**: Observe a declared storm over time and confirm flashes occur and illuminate.

**Acceptance Scenarios**:

1. **Given** a storm, **When** it runs, **Then** lightning occurs at intervals matching the recovered
   contract.
2. **Given** a lightning flash, **When** it occurs, **Then** it briefly affects scene lighting through
   the owning lighting system.
3. **Given** no storm, **When** the scene renders, **Then** no lightning occurs.

---

### User Story 7 - Era behaviour is gated and provenance is carried (Priority: P1)

Weather behaves per era. Alpha content is unaffected by 5.0.1-only mechanisms, and every weather
decision records which era profile produced it.

**Why this priority**: P1 because it constrains all the other work rather than following it. The
project's standing rule is that eras differ structurally — this is the same lesson MCNR component
order and minimap generation both taught — and 0.5.3's weather, if any, is not 5.0.1's.

**Independent Test**: Load alpha and 5.0.1 content and confirm each uses its own profile, with the
decision recorded.

**Acceptance Scenarios**:

1. **Given** 0.5.3 content, **When** it loads, **Then** no 5.0.1-only weather mechanism is applied
   and rendering is unchanged from before this feature.
2. **Given** 5.0.1 content, **When** it loads, **Then** the decoded weather path applies.
3. **Given** an unrecognised build, **When** it loads, **Then** it is flagged rather than defaulted.
4. **Given** any weather result, **When** inspected, **Then** it carries its era profile and evidence
   source.

---

### Edge Cases

- **A zone declaring weather that has no data.** A referenced weather row that is missing or empty.
- **Camera transitions indoors, underwater, and between them** mid-transition.
- **Precipitation and the camera's near plane.** Particles at the eye must not fill the screen.
- **Very fast camera movement**, including teleports, which can strand or streak particles.
- **Overlapping weather sources** — a zone override and a map default disagreeing.
- **Weather during a phase change**, where the terrain underneath is being swapped.
- **Wind consumed before physics exists.** US4's interface must be defined so that spec 214's absence
  is not a blocker.
- **Absence in the alpha client.** 0.5.3 having no comparable system is the expected finding.
- **Capture and automation.** Deterministic runs need weather to be reproducible or suppressible.
- **Cost when weather is off.** No weather must mean no cost, not a disabled system still ticking.

## Requirements *(mandatory)*

### Functional Requirements

**Evidence**

- **FR-001**: The system MUST produce a written weather contract in which every field, selection rule
  and transition claim cites the binary address it was measured at.
- **FR-002**: The system MUST distinguish measured facts from inferences and MUST name fields for
  what they demonstrably are.
- **FR-003**: The contract MUST state which lighting and fog values weather modulates, in a form
  specs 143/147/160 can consume.
- **FR-004**: Reverse-engineering sessions MUST be read-only with respect to the Ghidra program
  unless the operator authorises otherwise.

**Selection and transition**

- **FR-005**: The system MUST resolve the active weather for the camera's location from the decoded
  data.
- **FR-006**: The system MUST blend between weather states over time rather than switching instantly,
  and MUST handle a reversed or interrupted transition without snapping.
- **FR-007**: The system MUST resolve conflicting weather sources by a stated, deterministic rule.
- **FR-008**: The system MUST handle a missing or empty weather reference without failing the frame.

**Presentation**

- **FR-009**: The system MUST render precipitation of the declared type at the declared intensity.
- **FR-010**: The system MUST keep precipitation distributed around the camera without trailing,
  clumping or popping, including during fast movement.
- **FR-011**: The system MUST suppress precipitation where the client suppresses it, including
  indoors and underwater.
- **FR-012**: The system MUST produce storm lightning on the recovered timing, affecting scene
  lighting through the owning lighting system.

**Integration**

- **FR-013**: The system MUST expose a wind value queryable at a world position, following weather
  intensity and its transitions.
- **FR-014**: The system MUST return a defined neutral wind value when no weather is active, and MUST
  NOT require any consumer to exist.
- **FR-015**: The system MUST modulate lighting and fog only through the systems that own them, and
  MUST NOT introduce a second lighting or fog model.
- **FR-016**: Modulated values MUST carry provenance identifying weather as a contributing source.
- **FR-017**: Removing weather MUST return lighting and fog exactly to their unmodulated values.

**Cost, determinism and era**

- **FR-018**: With no active weather, the system MUST cost no more than the pre-feature baseline.
- **FR-019**: The system MUST be suppressible and reproducible for capture and validation runs.
- **FR-020**: The system MUST gate weather behaviour by era profile and MUST NOT apply 5.0.1-only
  mechanisms to earlier eras.
- **FR-021**: The system MUST flag unrecognised builds rather than defaulting them.
- **FR-022**: Every weather result MUST carry its era profile and evidence source.

### Key Entities

- **Weather Contract**: The address-cited description of weather data, selection, transition and
  effects — the deliverable of US1.
- **Weather Type**: A declared kind of weather with its intensity range and the effects it drives.
- **Weather State**: What is active now — type, intensity, and any transition in progress.
- **Precipitation Field**: The particle presentation of falling weather, positioned relative to the
  camera.
- **Wind Field**: The queryable wind at a world position; the interface spec 214 consumes.
- **Atmospheric Modulation**: The set of lighting and fog adjustments weather contributes, applied by
  the systems that own those values.
- **Lightning Event**: A timed flash within a storm and its lighting contribution.
- **Era Profile**: The per-build statement of which weather capabilities exist, carried as provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of structural claims in the weather contract cite a binary address; a reviewer can
  select any claim at random and confirm it.
- **SC-002**: Every field of the decoded weather table is either documented with a measured meaning
  or explicitly marked unverified — none silently omitted.
- **SC-003**: A zone declaring snow renders snow at its declared intensity, judged a match against
  client reference footage by the operator.
- **SC-004**: Crossing between differently-weathered zones produces a blend with no instant switch and
  no snap on reversal.
- **SC-005**: Wind queried during a transition follows the blend continuously — zero discontinuities.
- **SC-006**: Toggling weather changes lighting and fog only through their owning systems, and this
  feature adds zero additional lighting or fog models.
- **SC-007**: Removing weather returns lighting and fog to bit-identical unmodulated values.
- **SC-008**: With no active weather, frame cost is indistinguishable from the pre-feature baseline,
  measured against the frame-time distribution rather than a static-camera average.
- **SC-009**: Capture runs with weather suppressed reproduce their pre-feature output exactly.
- **SC-010**: 0.5.3 content renders identically to the pre-feature baseline — zero observable change.

## Assumptions

- **The binary and Ghidra project are available**, as for spec 214. Loss of access blocks this
  feature.
- **Decode from 5.0.1, gate by era.** 5.0.1 is the complete implementation and is not evidence about
  0.5.3.
- **Lighting, fog and sky are not owned here.** Specs 143, 147 and 160 own them. This feature drives
  them through declared interfaces; where those interfaces do not yet exist, defining them is
  coordination work with the owning spec, not a licence to build a parallel model.
- **Spec 214 is the wind consumer, not a dependency.** The wind interface is defined here and is
  useful with or without a physics solver behind it. Neither spec blocks the other.
- **Precipitation uses the existing particle path** (`ParticleSystem2`) rather than a new engine.
- **`EffectSwirlingFog` is weather-driven and distinct** from the base fog model spec 147 owns; which
  side implements it is settled during planning with that spec.
- **Testability follows the standing constraint.** No test project references the viewer, so weather
  selection, transition/blend, the wind field and era gating belong in `WowViewer.Core*`.
- **Visual judgement is operator work.** Reference footage comparison is not automatable here.

## Out of Scope

- The physics solver and cloth simulation (spec 214).
- The base lighting, fog and sky models (specs 143, 147, 160).
- Server-driven weather, gameplay effects of weather, and weather scripting.
- Authoring or editing weather data.
- Establishing what 0.5.3 did instead; that is separate work against the 0.5.3 binary.
