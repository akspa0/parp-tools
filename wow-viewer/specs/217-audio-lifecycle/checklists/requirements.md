# Specification Quality Checklist: Audio Playback Lifecycle and Correctness

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

**Grounding**: written against the live 5.0.1 binary. The six-list lifecycle, the periodic mode, the
`PlaySoundKitID` signature with its category buses and `forceNoDuplicates`, finite prioritised
channels, and the weighted variation array are all measured and recorded in
[`workstream-audio-501-ghidra.md`](../../../memory-bank/workstream-audio-501-ghidra.md).

**The diagnosis came out of the evidence, not the other way round.** The reported symptom is
"one-shots that fire forever". The binary shows every sound explicitly unlinking from one list and
linking to another, terminating at a delete list that a dedicated pass reclaims. A sound that never
reaches that list never stops. That reframes the work from *decoding* to *lifecycle*, which is why
US1 is a state-machine story rather than a parser story — and it is stated as a diagnosis to confirm
(see Assumptions), not as an established fact about our code.

**Iteration 1 — issues found and fixed:**

1. *The first draft had two repeat modes.* The binary has three — `"StopSound: Periodic sound between
   tweets"` proves a **periodic** mode with silence between repetitions, stopped by its own path.
   Conflating periodic with looping produces continuous noise where the client produces occasional
   chirps, which matches the reported complaint closely enough to be a prime suspect. Promoted to its
   own P1 story.
2. *Channel limits were missing.* `"No More Valid Channels, linking to SoundKitObjects_DeleteList"`
   shows exhaustion is a **handled outcome routing to deletion**, not an error. FR-015 states that
   explicitly, because treating it as an error is how sounds get stranded in a pending state — the
   same class of bug as the reported one.
3. *Autoplay was going to be a requirement.* Inverted: FR-016 makes enabling it **conditional on the
   lifecycle, repeat-mode and budget requirements being demonstrated**, and US6 is written as the
   acceptance test for the feature. Turning autoplay on before the lifecycle is fixed would just
   restore the reported problem.
4. *Era handling was an assumption line.* The operator's own complaint is that current behaviour is
   an unattributed mixture of fragments. Promoted to US7 at P1, with FR-022 forbidding cross-era
   application without evidence. The backend itself is era-split — DirectSound/DirectMusic in 0.5.3,
   FMOD in 5.0.1 — so "make it like the client" is meaningless without naming the era.
5. *"Audio sounds right" was a success criterion.* Unverifiable. Replaced with counting: zero
   unintended repetitions, live-sound count not trending upward, every sound classified into the
   correct repeat mode, observed variation distribution matching declared weights.

**Deliberate deviations:**

- *"No implementation details"* passes for Requirements and Success Criteria. **Context** quotes
  client log strings and names the `PlaySoundKitID` signature because that evidence *is* the
  diagnosis. No FR or SC names an audio library or API.
- *"Written for non-technical stakeholders"* stays partial; the reader is the operator.

**Open risks carried into planning:**

- **Confirm the diagnosis first.** The spec asserts the defect is lifecycle rather than decoding,
  from client evidence plus the reported symptom — not from an audit of our own audio code. Planning's
  first task is to check it against the current implementation. If it is wrong, that is the finding.
- **MCSE emitter frame is still unverified.** [[project_mcse_emitter_frame_unverified]] stands: the
  chunk-local assumption has no evidence and must be measured before positional behaviour is built on
  it. Listed in Edge Cases so it cannot be quietly assumed.
- **Backend adequacy is unassessed.** Out of Scope excludes replacing the backend *unless* planning
  shows it cannot meet these requirements. That determination has not been made.
- **Most of this is unit-testable.** A lifecycle state machine, repeat-mode classification, priority
  and weighted selection can be tested in Core without a sound device, which matters given no test
  project references the viewer. Planning should exploit that rather than defer everything to
  listening.

**Status**: All items pass or are deliberately partial with documented rationale. Ready for
speckit-plan.
