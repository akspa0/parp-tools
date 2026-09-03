# Feature Specification: Audio Playback Lifecycle and Correctness

**Feature Branch**: `217-audio-lifecycle` (authored on `v0.5.3`; specs 193–216 follow the same convention)

**Created**: 2026-09-02

**Status**: Draft

**Input**: User description: "get our audio functioning properly, like 5.0.1's audio system, such that we don't have forever looping annoying sounds that are not even the midi or mp3 music, but random horrible triggers that should only fire once. That's why audio is not allowed to autoplay, because it's very broken and not based on anything we observed in any client besides bits of 0.5.3!"

## Context & Motivation

Audio in the viewer is disabled by default because it is wrong in a specific way: sounds that should
fire once fire forever, the music that should play does not, and the result is unusable. Autoplay
being off is a workaround for a defect, not a design choice, and it means an entire dimension of the
data is invisible.

The operator's diagnosis is that the current behaviour is not modelled on any client — it is
assembled from fragments of 0.5.3 observation. Reconnaissance against the 5.0.1 binary (recorded in
[`memory-bank/workstream-audio-501-ghidra.md`](../../memory-bank/workstream-audio-501-ghidra.md))
explains the symptom exactly.

**A playing sound in the client is a state machine over six explicit lists.** Every sound is a
`SoundKitObject` that lives on exactly one of them and is explicitly unlinked from one and linked to
another at each transition — `WaitingForDownloadList`, `Loading`, `GoGoGo` (ready), the playing list,
`FadeList`, and `DeleteList`, which `ProcessSoundKitObjectDeleteList` reclaims. The log strings name
every edge.

**A sound that never reaches the delete list never stops.** That is the mechanism behind a one-shot
trigger playing forever. It is a *lifecycle* defect, not a decoding or playback one, which is why it
has survived attention to the decoding.

Four more findings bear directly on the reported symptoms:

- **There are three repeat modes, not two.** `"StopSound: Periodic sound between tweets"` establishes
  a **periodic** mode — repeating with silence between repetitions — distinct from both one-shot and
  continuous loop, and stopped by its own path. Conflating periodic with looping produces continuous
  noise where the client produces occasional chirps. This is a strong candidate for the specific
  "forever looping annoying sounds" complaint.
- **The client suppresses duplicates at the play call.** The exposed entry point is
  `PlaySoundKitID(ID, optional["SFX","Music","Ambience" or "Master"], optional[forceNoDuplicates])`.
  Without that suppression a repeatedly-fired trigger stacks on itself.
- **Channels are finite and prioritised.** `"No More Valid Channels, linking to
  SoundKitObjects_DeleteList"` shows that failing to get a channel is a *normal, handled* outcome
  that routes to deletion. An implementation with unbounded voices does not behave like the client
  under load — it accumulates.
- **Variation selection is weighted.** Asserts on `m_pSoundKitRec->m_Freq[i]` show a per-variation
  frequency array, matching what the 0.5.3 workstream already measured (`BuildSoundFilesRec` proves
  ten filename pointers paired with ten frequency values). Ignoring the weights makes selection
  uniform, so rare variations play as often as common ones.

**The audio backend is era-split** and this must be stated before any "make it like the client" work
begins: 0.5.3 is DirectSound + DirectMusic with MIDI+DLS ambience; 5.0.1 is FMOD. The *lifecycle
discipline* and the *concepts* transfer between them. The backend does not.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A one-shot sound plays once and stops (Priority: P1)

A sound that is meant to fire once fires once, ends, and releases whatever it was holding. It does
not repeat, and it does not accumulate.

**Why this priority**: This is the reported defect and the reason audio is switched off. Fixing it
alone makes audio usable, which is the whole ask. Everything else here is fidelity on top of a
system that stops when it should.

**Independent Test**: Trigger a one-shot sound, confirm it plays once and stops, and confirm the
resources it used are released rather than retained.

**Acceptance Scenarios**:

1. **Given** a one-shot sound, **When** it is triggered, **Then** it plays to completion exactly once
   and stops.
2. **Given** a sound that has finished, **When** the system next updates, **Then** it is reclaimed and
   no longer occupies a playback slot.
3. **Given** the same trigger fired repeatedly, **When** duplicate suppression is requested, **Then**
   overlapping copies of the same sound are prevented.
4. **Given** a long session with many triggers, **When** it runs, **Then** the number of live sounds
   does not grow without bound.
5. **Given** a sound that fails to start, **When** the failure occurs, **Then** it is released rather
   than left in a pending state forever.

---

### User Story 2 - Repeat modes are distinguished (Priority: P1)

One-shot, periodic and looping sounds each behave as their own kind. A periodic sound repeats with
silence between repetitions; a loop plays continuously; a one-shot plays once.

**Why this priority**: The evidence identifies this conflation as a likely direct cause of the
reported "forever looping" noise, and no amount of lifecycle correctness helps if a periodic sound is
started as a continuous loop. P1 alongside US1 because the two together are the reported bug.

**Independent Test**: Play one sound of each mode and observe the timing: single, gapped-repeating,
and continuous respectively.

**Acceptance Scenarios**:

1. **Given** a periodic sound, **When** it plays, **Then** it repeats with silence between
   repetitions rather than continuously.
2. **Given** a looping sound, **When** it plays, **Then** it repeats continuously until stopped.
3. **Given** a one-shot sound, **When** it plays, **Then** it does not repeat.
4. **Given** a sound whose repeat mode cannot be determined from the data, **When** it is
   encountered, **Then** it is reported rather than assumed to be a loop.
5. **Given** a periodic sound, **When** it is stopped, **Then** it stops on its own path and does not
   restart.

---

### User Story 3 - Music and ambience play (Priority: P2)

Zone music and ambience play, on their own category buses, at the volumes those categories are set
to.

**Why this priority**: This is the audio the operator *wants* to hear and currently does not. It sits
after US1/US2 because playing more audio through a broken lifecycle would make things worse, not
better.

**Independent Test**: Enter zones with declared music and ambience and confirm each plays, and that
category volume controls affect the intended sounds.

**Acceptance Scenarios**:

1. **Given** a zone with declared music, **When** it is entered, **Then** that music plays.
2. **Given** a zone with declared ambience, **When** it is entered, **Then** that ambience plays.
3. **Given** the category buses, **When** a volume is changed, **Then** only sounds in that category
   are affected.
4. **Given** a zone change, **When** it occurs, **Then** the outgoing music and ambience stop or fade
   rather than continuing underneath the new zone.
5. **Given** era-appropriate music data, **When** it is resolved, **Then** the correct table chain is
   used for that era rather than a single assumed path.

---

### User Story 4 - Playback is bounded and prioritised (Priority: P2)

There is a finite number of simultaneous sounds. When more are requested than can play, priority
decides, and the overflow is discarded cleanly.

**Why this priority**: The binary proves the client works this way and that channel exhaustion is a
handled outcome. Unbounded playback is both unlike the client and a slow accumulation problem.

**Independent Test**: Request far more simultaneous sounds than the limit and confirm the limit holds,
priority is respected, and nothing accumulates.

**Acceptance Scenarios**:

1. **Given** more sounds requested than the limit, **When** they are requested, **Then** the number
   playing never exceeds the limit.
2. **Given** a sound that cannot obtain a slot, **When** that happens, **Then** it is released
   cleanly and is not treated as an error condition.
3. **Given** sounds of differing priority competing, **When** they compete, **Then** higher priority
   wins by a stated rule.
4. **Given** sustained over-subscription, **When** it continues, **Then** memory and slot usage stay
   bounded.

---

### User Story 5 - Variation selection is weighted (Priority: P3)

A sound with several variations picks among them using the declared weights, so common variations are
common and rare ones are rare.

**Why this priority**: Genuine fidelity, and cheap once the data is read, but the audio is usable
without it. It also directly explains a class of "that sound is wrong" impressions that are otherwise
hard to pin down.

**Independent Test**: Trigger a multi-variation sound many times and compare the observed
distribution against the declared weights.

**Acceptance Scenarios**:

1. **Given** a sound with weighted variations, **When** it is triggered many times, **Then** the
   distribution of variations reflects the declared weights.
2. **Given** a variation with zero weight, **When** the sound is triggered, **Then** that variation is
   not selected.
3. **Given** a sound with a single variation, **When** it is triggered, **Then** that variation plays.

---

### User Story 6 - Audio can be trusted enough to autoplay (Priority: P1)

Audio is on by default because it behaves. The operator can also see what is playing and why.

**Why this priority**: This is the operator's actual goal — the workaround removed. It is P1 because
it is the acceptance test for the whole feature: audio is only allowed back on when US1, US2 and US4
demonstrably hold. It also requires the inspection surface that makes any future audio defect
diagnosable rather than mysterious.

**Independent Test**: Enable audio by default and run a long session across several zones, confirming
nothing accumulates, nothing repeats that should not, and the inspection view explains what is
playing.

**Acceptance Scenarios**:

1. **Given** the defects in US1, US2 and US4 are fixed and demonstrated, **When** the viewer starts,
   **Then** audio is enabled by default.
2. **Given** audio playing, **When** the operator inspects it, **Then** they can see each live sound,
   its source, its category, its repeat mode and its lifecycle state.
3. **Given** any live sound, **When** it is inspected, **Then** it carries provenance identifying the
   data that produced it.
4. **Given** the operator wants silence, **When** they mute, **Then** audio stops immediately and
   completely.
5. **Given** a capture or validation run, **When** it executes, **Then** audio does not alter its
   output.

---

### User Story 7 - Era behaviour is gated and provenance is carried (Priority: P1)

Audio uses each era's own model. 0.5.3's tables and 5.0.1's are not treated as the same system.

**Why this priority**: P1 because it constrains everything else. The backend and the data chain are
both era-split — 0.5.3 is DirectSound/DirectMusic with MIDI+DLS, 5.0.1 is FMOD with a different table
set — and the operator's stated complaint is precisely that current behaviour is an unattributed
mixture. Repeating that would reproduce the bug in a new form.

**Independent Test**: Load content of each supported era and confirm each resolves audio through its
own chain, with the decision recorded.

**Acceptance Scenarios**:

1. **Given** content of any supported era, **When** audio is resolved, **Then** that era's data chain
   is used.
2. **Given** an unrecognised build, **When** it loads, **Then** it is flagged rather than defaulted.
3. **Given** any audio decision, **When** inspected, **Then** it carries its era profile and the
   table or file that produced it.
4. **Given** a mechanism observed only in one era, **When** it is applied, **Then** it is not applied
   to another era without its own evidence.

---

### Edge Cases

- **A sound whose asset is missing or fails to decode**, after it has already been placed in a
  pending state.
- **A zone change during a fade**, and a change back before the fade completes.
- **A sound triggered by an object that is unloaded** while the sound is still playing.
- **Positional sounds with no valid position**, including the unverified MCSE emitter frame — see
  the standing note that the chunk-local assumption has never been measured.
- **Emitters entering and leaving range repeatedly** at a boundary, which can restart a sound every
  frame.
- **Very many emitters in range at once**, in a dense zone.
- **Muting or disabling audio mid-playback**, including during a fade.
- **Time compression or paused simulation**, where periodic timers may still be running.
- **The same sound legitimately requested twice** — duplicate suppression must not silence intended
  overlaps.
- **Capture runs**, which must be unaffected.

## Requirements *(mandatory)*

### Functional Requirements

**Lifecycle**

- **FR-001**: Every sound MUST progress through explicit lifecycle states and MUST reach a terminal
  state that releases its resources.
- **FR-002**: A sound that has finished, failed to start, or failed to obtain a playback slot MUST be
  released rather than left pending.
- **FR-003**: The number of live sounds MUST stay bounded over an arbitrarily long session.
- **FR-004**: The system MUST support suppressing duplicate instances of the same sound at the point
  of triggering.
- **FR-005**: The system MUST support fading a sound in and out over a specified duration, and a fade
  MUST complete into a defined terminal state.

**Repeat modes**

- **FR-006**: The system MUST distinguish one-shot, periodic and looping sounds and MUST play each
  according to its own mode.
- **FR-007**: The system MUST report, rather than assume, a sound whose repeat mode cannot be
  determined from the data.
- **FR-008**: Each repeat mode MUST have a stop path that leaves the sound stopped.

**Content**

- **FR-009**: The system MUST play zone music and ambience.
- **FR-010**: The system MUST route sounds to distinct category buses with independent volume
  control.
- **FR-011**: The system MUST stop or fade outgoing zone audio on a zone change rather than layering
  it under the new zone.
- **FR-012**: The system MUST select among a sound's variations using the declared weights, and MUST
  never select a zero-weight variation.

**Budget**

- **FR-013**: The system MUST enforce a maximum number of simultaneously playing sounds.
- **FR-014**: The system MUST resolve competition for playback slots by a stated priority rule.
- **FR-015**: Failure to obtain a playback slot MUST be a handled outcome, not an error state.

**Operability**

- **FR-016**: Audio MUST be enabled by default once the lifecycle, repeat-mode and budget
  requirements are demonstrated.
- **FR-017**: The system MUST provide an inspection view showing each live sound with its source,
  category, repeat mode and lifecycle state.
- **FR-018**: Every audio decision MUST carry provenance identifying the data that produced it.
- **FR-019**: Muting MUST stop audio immediately and completely.
- **FR-020**: Audio MUST NOT alter capture or validation output.

**Era**

- **FR-021**: The system MUST resolve audio through the data chain belonging to the content's era.
- **FR-022**: The system MUST NOT apply a mechanism observed in one era to another era without
  evidence for that era.
- **FR-023**: The system MUST flag unrecognised builds rather than defaulting them.
- **FR-024**: Every audio result MUST carry its era profile.

### Key Entities

- **Sound Instance**: One playing or pending sound — its source data, category, repeat mode,
  priority, lifecycle state and position if any.
- **Lifecycle State**: Where an instance is in its life, including the terminal state that releases
  it.
- **Repeat Mode**: One-shot, periodic (repeating with gaps), or looping.
- **Sound Definition**: The data describing a sound — its variations, their weights, volume, priority
  and distance behaviour.
- **Category Bus**: An independently controllable group such as effects, music, ambience, and the
  master.
- **Playback Budget**: The maximum simultaneous sounds and the priority rule that allocates them.
- **Emitter**: A positioned source in the world that can start and stop sounds as it enters and
  leaves range.
- **Era Profile**: The per-build statement of which audio data chain and mechanisms apply.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A one-shot sound plays exactly once — measured across repeated triggering, zero
  unintended repetitions.
- **SC-002**: Over a long session across multiple zones, the number of live sounds returns to its
  resting level and does not trend upward — zero unbounded growth.
- **SC-003**: Periodic sounds repeat with measurable silence between repetitions, and looping sounds
  do not; every sound in the test set is classified into the correct one of the three modes.
- **SC-004**: Zone music and ambience play in zones that declare them.
- **SC-005**: Simultaneous playback never exceeds the declared limit, and sustained over-subscription
  leaves memory and slot usage flat.
- **SC-006**: Observed variation distribution over many triggers matches the declared weights.
- **SC-007**: Audio is enabled by default, and a long unattended session produces no runaway or
  repeating sound.
- **SC-008**: Every live sound is inspectable with its source, category, repeat mode and lifecycle
  state.
- **SC-009**: Capture and validation output is unchanged with audio enabled.
- **SC-010**: Every audio result carries an era profile and data source; unrecognised builds are
  flagged, never defaulted.

## Assumptions

- **The defect is lifecycle, not decoding.** The evidence points at missing state transitions and a
  missing periodic mode. Planning should confirm this against the current implementation before
  reworking the decoders — and if the diagnosis is wrong, that finding is the outcome, not a
  detour.
- **The backend is era-split and the concepts are what transfer.** 0.5.3 is DirectSound/DirectMusic
  with MIDI+DLS; 5.0.1 is FMOD. This feature adopts the *discipline* the 5.0.1 engine demonstrates —
  explicit lifecycle, distinct repeat modes, bounded prioritised channels, weighted variation,
  duplicate suppression — and does not assume either era's backend or table set applies to the other.
- **Existing 0.5.3 evidence stands and is reused.** The `SoundEntries` layout, the MIDI/DLS pairing
  rule, and the ZoneMusic-is-not-SoundEntries finding are already recorded in the 0.5.3 workstream
  note and are not re-derived here.
- **The MCSE emitter frame is still unverified.** The chunk-local assumption has no evidence. This
  feature must not build positional behaviour on it without measuring first.
- **Autoplay is the outcome, not the starting point.** Enabling it before the lifecycle is fixed
  would restore the reported problem.
- **Testability follows the standing constraint.** No test project references the viewer, so
  lifecycle state, repeat-mode classification, budget/priority rules, weighted selection and era
  gating belong in `WowViewer.Core*`. A state machine is unusually well suited to unit tests, and
  most of SC-001 through SC-006 can be tested without a sound device.
- **Listening is operator work.** Whether audio sounds right is an operator judgement.

## Out of Scope

- Replacing the audio backend or adopting a specific audio library, unless planning shows the current
  one cannot meet the requirements.
- Combat, spell, and gameplay-driven sound triggering.
- Voice-over, vocal error sounds, and UI sounds.
- Reverb and environmental DSP.
- Authoring or editing audio data.
- Establishing what 0.5.3 did where the existing workstream note does not already say; that is
  separate work against the 0.5.3 binary.
