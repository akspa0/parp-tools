# Feature Specification: Alpha Demo Restoration — WTF Commands, Camera Follow, and Torchlight

**Feature Branch**: `158-alpha-demo-restoration`

**Created**: 2026-08-16

**Status**: Draft

**Input**: User description: "we should try to handle the 'demo' wtf files that place the camera/player
at specific areas in the game world - but we would need to be able to also handle worldport versus
teleport commands, teleport has no mapID, while worldport has mapID's. I'd love to be able to restore all
the 'demo' functionality that the early alpha offered, which was there to show to investors how far along
development had come. There are a TON of demo*.wtf files, as well as various random wtf files that do
various things like setting up the client with specific profiles of settings. We should try to build our
UI to use similar keybindings for things (Their profiling/performance stuff was activated with commands
like Alt+P, which displayed similar stats as our runtime stats and perf graph, over top of the game
window, for instance. We would like to also be able to set a character model as the camera, and equip
items that give off their own point lights, like torches. Now that we have the areatest.lit lights
working, we have the 2001 era lights, which means we need a camera with a torch, or the ability to equip
a torch to the camera that acts as a point light. the attached screenshot hails from 2001, the area
exists in the maps we have, but we have never seen the version of the game with the torches equipped and
in action. We can recreate that experience very very easily, now, since the files are all in there, it's
just a matter of reading them all and understanding them properly."

## Scope Note — read before the user stories

**SUPERSEDED (2026-08-16) — see [Spec 159](../159-wtf-command-inspection/spec.md).** The paragraph below
was written after a search that, even in its corrected/broadened form, still only read one file's full
content (`Config.wtf` from 0.5.3.3368) and inferred every other file's content from its filename alone —
it never actually opened `SandBox.wtf`, any `config-cache.wtf`/`bindings-cache.wtf`, and never dug
seriously into 2.0.0 specifically. The user identified 2.0.0 directly as the first client with real
demonstration-point content tied to promotional screenshots Blizzard released, and corrected that WTF
files are a general command-scripting surface the client's interpreter executes, not a settings-only
format. One data point already on hand and worth carrying forward rather than re-deriving: a full
recursive search of `H:\CLIENTS` found only a root-level `realmlist.wtf` for 2.0.0.5610/5665 — no
`WTF\Config.wtf` at all, unlike every other staged build — which may mean that staged copy has never
actually been launched (`Config.wtf` is normally written by the client on first run), not that it lacks
demo content by design. That is exactly the kind of thing a shallow filename search cannot tell you and a
real inspection tool (Spec 159) is built to resolve properly. Treat the paragraph below as **retracted
pending Spec 159's actual findings**, not as established fact.

### Original (superseded) reasoning

"Demo\*.wtf" in the original request is shorthand for "a file with any name" containing worldport,
teleport, and camera/setting commands — not a literal filename requirement. Checking for that properly
means checking file *content*, not matching a filename pattern, and checking every staged client, not a
sample of two. Every `.wtf` file found across all ten staged client installs was one of `Config.wtf`,
`realmlist.wtf`, `SandBox.wtf`, or a `WTF\Account\<name>\...\config-cache.wtf` / `bindings-cache.wtf` —
these are legitimate, unaltered client data, exactly as staged. **No file containing worldport, teleport,
or camera-placement content exists in any client this project currently has staged** was the conclusion
drawn — since retracted, per the note above, because the search behind it was not as thorough as it was
presented to be.

One real possibility raised at the time: a scripted investor demo walking through several zones may never
have existed as a saved `.wtf` file at all. GM commands like `worldport`/`teleport` are normally typed
live through a console, not persisted to a settings file. This remains a possibility worth keeping in
mind, but it is not a conclusion to rest on before Spec 159's real inspection has actually run.

None of this makes the request worthless. `Config.wtf` itself — present in every staged client — uses the
exact same `SET <name> "<value>"` text syntax a command file would use. A reader for that syntax is
immediately useful against real files available right now, and is ready the moment a real source turns up
(the community file-list catalogs thousands of `WTF\<name>.wtf` entries across all WoW versions
generally, e.g. `WTF\AhnQiraj.wtf` — a possible future lead, not a confirmed or accessible source today).
So this spec builds the general capability (User Stories 1–2) and keeps the specific "replay a captured
investor demo" experience as its own story (User Story 6), stated plainly as blocked on a source that
does not currently exist in this project's possession, rather than silently dropped or silently promised.
If a real source file surfaces later — from the user's own records, a community archive, or anywhere else
— User Story 6 needs nothing rebuilt, only that file.

The camera-follow and torchlight stories (User Stories 4–5) are independent of the WTF/command work —
they deliver "we have never seen the torches in action" regardless of whether any demo file is ever
found, using a character model and its own attachment/lighting-relevant data that already exists in
staged clients today.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read a real WTF file's settings (Priority: P1)

A user points the viewer at a real `.wtf` file — starting with `Config.wtf`, which every staged client
already has — and gets back every setting the file declares, exactly as written, with any line the reader
could not interpret called out rather than silently dropped.

**Why this priority**: Foundational and immediately useful against files that exist today, independent of
whether a demo file is ever found. Every other WTF-driven story in this spec depends on this reader
existing first.

**Independent Test**: Point the reader at a real `Config.wtf` from a staged client; confirm every
`SET name "value"` line produces a matching setting, and that the file is fully accounted for (every line
either became a setting or was explicitly reported as not understood — never silently skipped).

**Acceptance Scenarios**:

1. **Given** a real `Config.wtf` file, **When** it is read, **Then** every `SET name "value"` line
   produces a setting whose name and value match the file's plain text exactly.
2. **Given** a line in the file that is not a recognizable `SET` statement (a comment, blank line, or
   unexpected syntax), **When** the file is read, **Then** that line does not abort the read and is
   recorded as not understood rather than silently discarded.
3. **Given** the same `Config.wtf` read from two different staged clients, **When** compared, **Then** any
   difference in their settings is visible in the result, not hidden by only reporting one file's shape.

---

### User Story 2 - Execute worldport and teleport commands (Priority: P1)

A user runs a worldport command (which names a target map) and the camera ends up at the right position
on that map, loading it first if it wasn't already loaded. A user runs a teleport command (which never
names a map) and the camera moves to a new position on whatever map is already loaded — it never tries to
load a different map, because the command never said which one.

**Why this priority**: This is the concrete, valuable half of "place the camera/player at specific areas"
— and it does not require a real demo file to be useful: a hand-written or future-found command source
can drive it today. Tied with Story 1 as foundational because it is the reason the WTF reader matters.

**Independent Test**: Construct a small set of worldport and teleport commands (hand-written is
sufficient — this does not require a real demo file). Run a worldport command targeting a map that is not
currently loaded; confirm the map loads and the camera ends at the specified position on it. Run a
teleport command while a map is loaded; confirm only the camera position changes and no map load is
attempted.

**Acceptance Scenarios**:

1. **Given** a worldport command naming a map and a position, **When** it is executed, **Then** that map
   is loaded (if not already) and the camera is positioned as specified once loading completes.
2. **Given** a teleport command naming only a position (no map), **When** it is executed, **Then** the
   camera moves to that position on the currently loaded map and no map load is attempted.
3. **Given** a worldport command naming a map that cannot be loaded, **When** it is executed, **Then** the
   camera is not repositioned and the failure is reported — it never silently repositions on the wrong map.
4. **Given** a command that is neither a recognizable worldport nor teleport instruction, **When** it is
   encountered, **Then** it is reported as unrecognized rather than silently ignored.

---

### User Story 3 - Toggle the performance overlay with Alt+P (Priority: P2)

A user presses Alt+P and the existing performance overlay (frame stats, perf graph) appears or
disappears, exactly as it does today from its toolbar buttons — matching the keybind the real 2001-era
client used for the equivalent view.

**Why this priority**: Small, fully independent of every other story in this spec, and delivers
immediate, low-risk parity with the remembered Alpha-era experience. Nothing else needs to exist first.

**Independent Test**: Press Alt+P with the overlay hidden; confirm it appears. Press it again; confirm it
disappears. Confirm every other keybind (including the existing bare "P") behaves exactly as before.

**Acceptance Scenarios**:

1. **Given** the performance overlay is hidden, **When** the user presses Alt+P, **Then** it appears,
   identical to what the existing toolbar toggle already shows.
2. **Given** the performance overlay is visible, **When** the user presses Alt+P, **Then** it disappears.
3. **Given** any other existing keybind (including plain "P"), **When** used after this change, **Then**
   its behavior is unchanged.

---

### User Story 4 - Attach the camera to a character model (Priority: P3)

A user selects a loaded, animating character model and switches the camera to follow it — the camera
tracks a specific point on the model (e.g. its head or eye position) every frame, moving and turning with
the model's current animation, instead of the ordinary free-flying camera.

**Why this priority**: A real, independently valuable capability — but it is the setup for Story 5's
torchlight, not the payoff itself, so it is ordered after the smaller, immediately-useful Stories 1–3.

**Independent Test**: Load a scene with an animating character model. Switch the camera to follow it;
confirm the camera tracks the model's position and orientation smoothly through its animation, with no
perceptible drift or lag. Switch back to free-fly; confirm normal camera control resumes cleanly.

**Acceptance Scenarios**:

1. **Given** a loaded, animating character model, **When** the user attaches the camera to it, **Then**
   the camera's position and orientation follow that model's current animated pose every frame.
2. **Given** the camera is attached to a model, **When** the model's animation changes (e.g. walk to
   idle), **Then** the camera continues tracking smoothly without a visible jump or lag.
3. **Given** the camera is attached to a model, **When** the user detaches it, **Then** ordinary free-fly
   control resumes from the camera's last followed position, not a jarring reset.

---

### User Story 5 - Equip a torch that casts real light (Priority: P3)

A user equips a torch (or similar lit item) onto a character model — attached at the model's hand, using
the model's own attachment data — and the torch visibly lights up nearby terrain and objects as it moves,
recreating the documented 2001 screenshot experience that has never been seen "in action" in this viewer.

**Why this priority**: The actual payoff the user described — but it is the largest, riskiest piece of
this spec (new attachment-point parsing plus genuinely new dynamic-lighting rendering, where today's
lighting is entirely static-per-scene), so it is ordered last among the "new capability" stories, after
its prerequisites are in place.

**Independent Test**: Attach a torch model to a character's hand attachment point in a dark or dim scene.
Confirm nearby terrain/objects visibly brighten near the torch, and that the brightening moves as the
torch (or the character carrying it) moves. This can be demonstrated with the camera in ordinary free-fly
mode looking at the character — it does not strictly require Story 4's camera-attach to be built first,
though the two together are what reproduces the referenced screenshot most directly.

**Acceptance Scenarios**:

1. **Given** a character model with a torch attached at its hand attachment point, **When** the scene
   renders, **Then** the torch visibly emits light that brightens nearby terrain and/or objects, not just
   a flame visual with no lighting effect.
2. **Given** the torch-carrying character moves, **When** the scene renders each frame, **Then** the
   light's effect on nearby geometry moves with it, with no lag or detachment.
3. **Given** the torch is un-equipped or removed, **When** the scene next renders, **Then** its light
   contribution is gone immediately — nothing keeps glowing after its source is removed.
4. **Given** a scene with existing static LIT-file lighting already in place, **When** a torch is added,
   **Then** the torch's light combines with the existing lighting rather than replacing or breaking it.

---

### User Story 6 - Replay a captured investor demo (Priority: P4 — blocked)

A user loads a real captured `demo*.wtf` file and watches the viewer step through the same sequence of
worldport/teleport commands and setting changes an early-Alpha investor demo would have shown.

**Why this priority**: This is the literal "restore all the demo functionality" ask, but it cannot be
delivered today: no such file exists in any client this project has staged (Scope Note, above). Kept as
its own story, at the bottom, explicitly blocked rather than quietly folded into Story 2 or dropped
entirely — the moment a real source file is found, this story needs nothing from Stories 1–5 to change,
only a real file to point them at.

**Independent Test**: Cannot be independently tested today — there is no real source file to test
against. This story's "test" is finding one. Once found, its test is: load it, and confirm the viewer's
resulting sequence of map/camera changes matches what the file specifies, using the machinery already
built in Stories 1–2 unmodified.

**Acceptance Scenarios**:

1. **Given** no real `demo*.wtf` file exists in this project's possession, **When** this feature is
   evaluated, **Then** it is reported as blocked-on-data, not as done, not as silently descoped.
2. **Given** a real demo file is found in the future, **When** it is read with Story 1's reader and
   executed with Story 2's command handling, **Then** no new parsing or command logic is needed beyond
   what those stories already deliver — only the file itself was missing.

---

### Edge Cases

- A WTF file with a setting name repeated more than once: the later value is not silently assumed to
  win — this behavior must be decided and stated, not left to whatever the parser happens to do.
- A worldport command targeting the map that is *already* loaded: no unnecessary reload occurs, only the
  camera repositions.
- A teleport command issued before any map is loaded at all: reported as a failure (there is no "current
  map" to teleport within), not silently ignored or misinterpreted as a worldport.
- The camera is mid-follow (Story 4) when the followed model is removed from the scene or its asset fails
  to reload: the camera falls back to free-fly at its last known position rather than freezing, tracking
  nothing, or crashing.
- Two torches (or a torch plus another light-emitting equipped item) attached at once: their light
  contributions combine rather than one silently overriding the other.
- A torch attached to a model that is very far from the camera or off-screen: its light still affects
  nearby geometry correctly (no light that only works near the camera specifically) but does not need to
  be computed at a cost that scales with total scene size — only nearby geometry is actually affected.
- Alt+P pressed while a text input field (e.g. a search box) has focus: does not fire the toggle if doing
  so would interrupt typing, matching how this project's other modifier-combo bindings already behave.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST parse a `.wtf` file's `SET <name> "<value>"` lines into named settings whose
  values match the file's plain text exactly.
- **FR-002**: System MUST tolerate a line it cannot interpret as a `SET` statement without aborting the
  rest of the file, and MUST record that the line was not understood rather than silently dropping it.
- **FR-003**: Every parsed WTF setting MUST carry which file it came from.
- **FR-004**: System MUST distinguish a worldport command (names a target map) from a teleport command
  (names no map) and apply each according to that distinction.
- **FR-005**: A worldport command MUST load its target map first (if not already loaded) and only then
  position the camera; if the target map cannot be loaded, the camera MUST NOT be repositioned and the
  failure MUST be reported.
- **FR-006**: A teleport command MUST reposition the camera on the currently loaded map only, and MUST
  NOT attempt to load a different map under any circumstance.
- **FR-007**: An unrecognized command MUST be reported, never silently ignored.
- **FR-008**: System MUST toggle the existing performance overlay when Alt+P is pressed, with behavior
  identical to its existing toolbar toggle.
- **FR-009**: The Alt+P binding MUST NOT alter the behavior of any existing keybind, including plain "P".
- **FR-010**: System MUST let the camera track a specific bone of a loaded, animating character model
  every frame, following that bone's current world position and orientation.
- **FR-011**: Detaching the camera from a followed model MUST return it to ordinary free-fly control from
  its last followed position, without a jarring reset or inconsistent state.
- **FR-012**: System MUST attach an item model to a character model's hand attachment point, using the
  character model's own declared attachment data rather than an approximated or hardcoded offset.
- **FR-013**: An attached torch (or similar item) MUST emit a point light whose position follows its
  attachment point every frame.
- **FR-014**: A torch's point light MUST visibly affect the shading of nearby terrain and/or objects, and
  MUST combine with, not replace or break, any existing static lighting already present in the scene.
- **FR-015**: Removing or un-equipping a light-emitting item MUST remove its light contribution
  immediately, with no residual glow.
- **FR-016**: This feature MUST NOT claim or require that a real `demo*.wtf` file exists — that capability
  (User Story 6) is explicitly blocked on finding one and MUST be reported as such, not delivered as if
  data were available when it is not.

### Key Entities

- **WTF Document**: the parsed contents of one `.wtf` file — its recognized settings (name/value pairs)
  plus any lines it could not interpret, and which file it came from.
- **Command**: one interpretable instruction extracted from a WTF document (or another future source) —
  its kind (worldport, teleport, or unrecognized), a target map identifier when present, and a target
  position.
- **Camera Follow Target**: a reference to a specific loaded model and bone the camera is currently
  tracking, replacing ordinary free-fly control while active.
- **Attached Light Source**: a point light bound to a specific attachment point on a model, whose
  position, color, and radius are derived from the attached item and which is removed the moment that
  item is un-equipped.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A real `Config.wtf` file from a staged client is read and every setting it produces matches
  the file's plain-text contents exactly, spot-checked against the raw file.
- **SC-002**: A worldport command changes both the loaded map and the camera's position; a teleport
  command changes only the camera's position on the map already loaded, and never triggers a map load.
- **SC-003**: Alt+P toggles the performance overlay with no other observable side effect, matching the
  existing toolbar toggle exactly.
- **SC-004**: A camera following a model's bone stays visually attached to it throughout a played
  animation, with no perceptible drift or lag, across at least one full animation cycle.
- **SC-005**: A lit torch attached to a character visibly brightens nearby terrain or objects, and that
  brightening moves with the character with no observable lag, in a scene dim enough for the effect to be
  clearly visible (recreating the referenced 2001 screenshot's conditions).
- **SC-006**: No part of this feature is delivered by claiming a `demo*.wtf` file exists when it does not
  — User Story 6 is explicitly reported as blocked-on-data rather than silently omitted or silently
  marked done.

## Assumptions

- No `.wtf` file of any name containing worldport, teleport, or camera-placement content exists in any
  client this project currently has staged — verified by content across all ten staged client installs,
  not by filename pattern and not by sampling. The community file-list's `WTF\<zonename>.wtf` entries are
  a possible future lead, not a confirmed or currently accessible source. It is also possible the
  original "demo" was never saved to a file at all (live-typed console commands during a session) — if
  so, no amount of further searching this project's staged clients will find it, and the way forward is a
  real source the user or a community archive can provide, not a broader automated search.
- A minimal way to trigger a parsed command (at minimum: run all commands found in a given WTF file) is
  in scope as part of Story 2. A full interactive console/command-line UI is not required by this spec —
  that would be a separate, larger feature if wanted later.
- "Character model as the camera" (Story 4) means the camera's position and orientation are driven by the
  model's bone data. It does not require rendering the game world from a body-occluded first-person
  perspective (hiding the character's own head/body from view) unless refined later — the camera simply
  tracks the bone's transform.
- The torch's flame visual (e.g. a particle effect) is a property of the attached item model itself,
  reusing this project's existing particle rendering. This spec's new work is specifically the point-light
  emission from that attachment point, not authoring a new visual effect.
- Attachment-point parsing is scoped to the Alpha-era (M2Era100) model format specifically, since its
  layout is already fully documented in this codebase and a real decompiled-client reference algorithm
  already exists to validate against. Attachment support for other model eras is not required here.
- A torch's point light needs to visibly and correctly affect nearby shading; it does not need to
  reproduce the real client's exact falloff/attenuation curve to the last detail — a reasonable,
  visibly-correct approximation satisfies this spec.
- Repeated settings within one WTF file (edge case, above): last-write-wins is the assumed default absent
  another reasonable convention found during planning, but this must be stated explicitly in the
  implementation, not left implicit.
