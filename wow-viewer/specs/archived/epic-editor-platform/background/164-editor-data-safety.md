# Feature Specification: Editor Data Safety

**Feature Branch**: `164-editor-data-safety`

**Created**: 2026-08-19

**Status**: Draft

**Input**: User description: "All intermediate data should be in-flight in zarr, so no data is ever lost if the program suddenly crashes while someone is editing data. That's a big issue with the existing noggit — it also randomly corrupts ADT's for no reason, due to random bugs reading/dealing with improperly ported assets ('fuckported' assets… you really cannot go backwards from later versions of the data very easily — we already do some really hacky things to re-fit WMO's with more than 384 groups into 0.5.3 WMO's, which I'm not sure even works right)." Scope confirmed in session: crash-safe journal **and** integrity gate in one spec; integrity handling goes to **detect, refuse, and attempt repair**.

**Depends on**: [161 — Editor Plugin Host](../161-editor-plugin-host/spec.md) for the host, bridge,
Editor Operations, and dirty state. Applies to every plugin including
[162 — World Authoring](../162-world-authoring-plugin/spec.md).

## Context

This spec exists because of two specific, named ways the incumbent tool fails its users:

1. **It loses work.** A crash mid-edit discards everything staged since the last save.
2. **It corrupts output.** ADTs come out damaged "for no reason" — in practice, because a reader
   accepted a malformed input, produced plausible-looking garbage, and a writer serialized it.

These are one problem seen from two sides: **the editor's staged state is not durable, and its
inputs are not verified.** Both are answered before an editor with real users exists, because both
get much harder to retrofit once people have work in flight.

### Why the second failure mode is structural, not sloppiness

The corrupting inputs have a name here: **"fuckported" assets** — assets ported *backwards* from a
later client to an earlier one. The term is a joke about a real constraint: the data genuinely does
not go backwards easily. Later formats carry structure earlier ones cannot express, so a downport is
lossy by necessity, and the tooling that performs it is doing something the format was never designed
to support.

This repo already owns a concrete instance. `WmoV17ToV14Converter` — **1,585 lines** — downports v17
WMOs to the v14 the 0.5.3 client wants, against two hard legacy budgets:

| Constraint | Value | Handling |
|---|---|---|
| `LegacyMaxGroupCount` | 384 | Overflow groups are **merged** into the final legacy group ([`MergeOverflowGroups`](../../src/core/WowViewer.Core.IO/Wmo/WmoV17ToV14Converter.cs)) |
| `LegacyMaxGroupVertexCount` | `0xBFFF` | Batches are split; a single batch exceeding the budget throws |

The overflow path is not untested — `Convert_WhenSourceExceedsLegacyGroupLimit_MergesOverflowIntoFinalLegacyGroup`
exists in a 764-line test file. **What it lacks is real-data validation**: synthetic fixtures only, no
verification against a fuckported WMO from a real client, and no evidence the 0.5.3 client accepts
the merged result. Under Constitution III that is not "works"; it is "compiles and satisfies a
fixture." The user's own read — *"I'm not sure it even works right"* — is the correct confidence
level, and this spec's job is to replace it with a measurement.

The general rule this yields is stronger than any single fix: **a downport is not a conversion, it is
a lossy re-fit, and every lossy re-fit must record what it lost.** An asset that silently drops
structure is indistinguishable from one that dropped it correctly — until it reaches a writer.

### Why Zarr for the journal

Zarr is already load-bearing here. The harvester writes Zarr v3 stores
([`zarr_io.py`](../../data-harvester/src/harvester/zarr_io.py)), the viewer reads them
([`ZarrTileDatasetLoader`](../../src/viewer/WoWViewer/Terrain/ZarrTileDatasetLoader.cs)), and
`DatasetVersionCatalog` already models `DatasetSourceKind.ZarrStore`. Editing state is exactly the
shape Zarr is good at — chunked arrays (heightmaps, alpha layers, shadow masks) plus group attributes
for the structured records that accompany them.

The properties that matter for crash safety come free: chunk writes are individual files, so a crash
damages at most the chunk being written, not the store; and a reader can open a partially-written
store and see what completed.

### The journal is a work product, not only insurance

The user asked for staged edits to be "saved in temp zarr data, and then **saved for later use**."
That second clause is the more interesting requirement. A journal that only exists to survive crashes
is discarded on every clean exit. A journal that persists becomes a durable, inspectable,
resumable record of an editing session — one that can be reopened tomorrow, handed to someone else,
diffed against another session, or replayed. This spec treats persistence as the default and crash
recovery as one thing that falls out of it.

### The validation corpus already exists

`H:\CLIENTS\WoW335\modernwow\` is a 3.3.5 client **full of fuckported assets that crash the
viewer today**. It is the right fixture for this spec for three reasons:

- **The symptom is stronger than the one specified.** "Crashes the viewer" is unambiguous and
  measurable in a way "corrupts ADTs for no reason" is not. A crash on read is the integrity gate's
  easiest win and its clearest pass/fail.
- **It is real data, from a real workflow.** The client root contains `modernwow.noggitproj` — it is
  literally a Noggit project, produced by the tool whose failure modes this spec exists to avoid.
- **It exercises the write path, not just the read path.** These assets are what an author would be
  editing, so they are exactly the inputs that would reach a writer unverified.

**Reframing that follows**: every asset in this client that currently crashes the viewer is a test
case. The bar is not "the Editor survives" — it is that each one is *identified*, with the specific
violated constraint named, and either quarantined or repaired with its losses recorded. A crash
converted into a silent skip would be a regression, not a fix.

### Out of scope

- **Journaling anything the user did not change.** The journal holds staged edits and their
  provenance, never a copy of the client.
- **Repairing the client install or rewriting source assets in place.**
- **A general asset-conversion framework.** This spec validates and repairs at the boundaries the
  Editor actually writes through; it does not restructure the existing converters.
- **Multi-user or networked collaboration on a journal.** Single local user.
- **Any Blizzard container as a journal or repair-output format.** The journal is Zarr; repaired
  assets are loose client-content files (Constitution VII).
- **The datastore itself** — multi-version packaging and the builder fix are
  [165](../165-unified-zarr-datastore/spec.md).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A crash loses nothing (Priority: P1)

The user is mid-edit — chunks pasted, placements moved, table cells changed — and the process dies.
On relaunch, the Editor offers the session back, with every staged edit intact up to the last one
completed.

**Why this priority**: It is the headline requirement and the one the incumbent tool fails at. Every
other story here is worth less if this one does not hold.

**Independent Test**: Stage a mixed set of edits across plugins, kill the process without warning at
several points including mid-write, relaunch, and confirm every completed edit is recoverable and no
partial edit is presented as complete.

**Acceptance Scenarios**:

1. **Given** a user stages an edit, **When** the operation completes, **Then** it is durable in the
   journal before the Editor reports it as staged.
2. **Given** staged edits exist, **When** the process is killed without warning, **Then** relaunching
   detects the session and offers to resume it.
3. **Given** a session is resumed, **When** the user inspects their work, **Then** every completed
   edit is present with its original values, and the undo history is restored.
4. **Given** the process is killed **during** a journal write, **When** the session is reopened,
   **Then** the incomplete edit is identified as incomplete and discarded or offered separately —
   never silently presented as complete.
5. **Given** a session is recovered, **When** the user saves, **Then** output is identical to what
   the same edits would have produced with no crash.
6. **Given** multiple plugins had staged edits, **When** recovery runs, **Then** each plugin's state
   is restored independently, and one plugin's unrecoverable state does not block the others.

---

### User Story 2 - Nothing is written from input that did not verify (Priority: P1)

Before the Editor writes a file, everything that file's contents derive from has been checked. If any
input failed verification, the write does not happen and the user is told exactly what failed.

**Why this priority**: This is the corruption failure mode directly. A durable journal that faithfully
preserves the path to a corrupt ADT has solved the wrong half of the problem.

**Independent Test**: Feed the Editor a known-malformed asset — including a real fuckported WMO — and
confirm the failure is identified specifically and no output file is produced.

**Acceptance Scenarios**:

1. **Given** an asset is read, **When** it fails structural validation, **Then** it is quarantined
   with a specific diagnostic naming the constraint violated, not a generic parse error.
2. **Given** a write is requested, **When** any contributing input is quarantined or unverified,
   **Then** the write is refused, the blocking inputs are named, and no partial file is left behind.
3. **Given** a write proceeds, **When** it completes, **Then** the file is re-read and verified before
   being reported as saved; a file that does not read back correctly is not reported as success.
4. **Given** an asset passes validation, **When** it is used, **Then** the fact that it was verified —
   and against what — is recorded with the output.
5. **Given** validation cannot reach a verdict, **When** the input is used, **Then** it is treated as
   unverified and blocks the write, rather than being assumed good.

---

### User Story 3 - Known-bad patterns can be repaired, provably (Priority: P2)

For specific, named defects — group-count overflow, vertex-budget overflow, index truncation — the
user can apply a repair. The repair never touches the source, is verified by re-reading its own
output, and records exactly what it changed.

**Why this priority**: Refusal alone would block real work on assets the user needs, and the >384
downport already exists precisely because these assets must be usable. It is P2 because refusal (US2)
is the safety property; repair is capability on top of it.

**Note on risk (raised in session, and the reason for the constraints below)**: repair logic layered
on formats that are only partly understood is itself a source of new corruption. Every requirement
here is written to contain that: repairs are per-pattern rather than general, opt-in rather than
automatic, never in-place, and must prove themselves against a re-read before being offered.

**Independent Test**: Take a real WMO exceeding 384 groups, repair it, and confirm the result loads in
the 0.5.3 client *and* renders correctly — the check the existing merge path has never been held to.

**Acceptance Scenarios**:

1. **Given** a quarantined asset matching a known repair pattern, **When** the user views it, **Then**
   the specific defect, the proposed repair, and what the repair will lose are stated before anything
   is applied.
2. **Given** a repair is applied, **When** it completes, **Then** the source asset is unmodified and
   the repaired asset is a new artifact.
3. **Given** a repair produces an artifact, **When** it is verified, **Then** it is re-read through
   the same validation the original failed; a repair whose output does not validate is reported as
   failed and is not offered for use.
4. **Given** a repaired asset is used in a write, **When** the output's provenance is inspected,
   **Then** the repair, its pattern, and its losses are recorded.
5. **Given** a defect matches no known repair pattern, **When** the user asks, **Then** they are told
   no repair exists — the system does not attempt a general fix.
6. **Given** a repair pattern is applied to an asset it was not designed for, **When** the mismatch is
   detected, **Then** it is refused rather than applied approximately.

---

### User Story 4 - Sessions persist and resume by choice (Priority: P2)

An editing session is a durable artifact. The user closes the Editor cleanly, comes back later, and
reopens the session with staged work intact — and can keep, name, or discard sessions deliberately.

**Why this priority**: This is the "saved for later use" half of the request, and it is what turns
the journal from crash insurance into a working practice. P2 because US1 delivers the safety on its
own.

**Independent Test**: Stage edits, exit cleanly, relaunch, reopen the session, continue editing, and
save — confirming the result matches doing it all in one sitting.

**Acceptance Scenarios**:

1. **Given** staged edits, **When** the user exits cleanly, **Then** the session persists by default
   and is not discarded.
2. **Given** persisted sessions, **When** the user opens the Editor, **Then** they can list, name,
   resume, or delete them, with what each contains summarized.
3. **Given** a resumed session, **When** the user continues, **Then** editing and undo behave as if
   never interrupted.
4. **Given** a session was recorded against a client that is no longer loaded, **When** it is
   resumed, **Then** the mismatch is detected and reported; edits are never silently applied to a
   different build.
5. **Given** sessions accumulate, **When** storage grows past a configured bound, **Then** the user is
   told which sessions are consuming space and can act; nothing is deleted without consent.

---

### User Story 5 - Every written file can be traced (Priority: P3)

For any file the Editor produced, the user can find out what it came from — source client and build,
which operations touched it, what was validated, what was repaired and how, what was lost.

**Why this priority**: Diagnostic value grows with time and matters most when something *does* go
wrong. It is P3 because the safety properties do not depend on it.

**Independent Test**: Produce a file through a session involving a repaired asset, then reconstruct
its full history from provenance alone, without the session that made it.

**Acceptance Scenarios**:

1. **Given** a written file, **When** its provenance is inspected, **Then** source build, contributing
   operations, validation results, and repairs are all present.
2. **Given** a repaired input contributed, **When** provenance is read, **Then** the repair pattern
   and its recorded losses are identifiable.
3. **Given** a file was written by a resumed session, **When** provenance is read, **Then** the full
   history spans the interruption.
4. **Given** provenance cannot be recorded, **When** a write is attempted, **Then** it is refused —
   an untraceable write is not permitted.

### Edge Cases

- Disk fills mid-journal-write.
- The journal store is corrupted or partially deleted between sessions.
- The user resumes a session whose source files changed underneath it.
- Two sessions target the same output file.
- An edit that is durable in the journal but whose plugin no longer exists on relaunch.
- A repair that succeeds structurally but produces a visually wrong asset — caught only by the render
  check, which is why one is required.
- An asset that validates in isolation but is invalid in combination with another.
- Journal growth during a long session of large terrain edits.
- Killing the process during the post-write verification read.
- A fuckported asset whose defect is *within* budget limits and therefore structurally invisible.

## Requirements *(mandatory)*

### Functional Requirements

**Durability**

- **FR-001**: Every completed Editor Operation MUST be durable in the journal before the Editor
  reports it as staged.
- **FR-002**: The journal MUST use the repo's existing Zarr store conventions, not a new format, and
  MUST NOT use any Blizzard container (Constitution VII).
- **FR-003**: A crash MUST NOT damage previously journaled operations; at most the in-flight one is
  lost.
- **FR-004**: An incompletely written operation MUST be detectable as incomplete and MUST NOT be
  presented as complete.
- **FR-005**: Recovery MUST restore staged edits **and** the undo history.
- **FR-006**: Per-plugin state MUST recover independently; one plugin's unrecoverable state MUST NOT
  block others.
- **FR-007**: Journaling MUST NOT stall the frame loop or make editing perceptibly slower.

**Sessions**

- **FR-008**: Sessions MUST persist across clean exits by default.
- **FR-009**: Users MUST be able to list, name, resume, and delete sessions, with contents summarized.
- **FR-010**: A session MUST record the client build it was made against, and resuming against a
  different build MUST be detected and reported, never silently applied.
- **FR-011**: Nothing in the journal is deleted without user consent; growth MUST be surfaced with
  what is consuming space.

**Integrity gate**

- **FR-012**: Assets MUST be structurally validated on read; failures MUST name the violated
  constraint specifically.
- **FR-013**: A write MUST be refused if any contributing input is quarantined or unverified, naming
  the blocking inputs. "Unverified" MUST be treated as failing, not as passing.
- **FR-014**: Every written file MUST be re-read and verified before being reported as saved.
- **FR-015**: A refused or failed write MUST leave no partial file.
- **FR-016**: Lossy downport paths MUST record what was lost, per asset. A downport that cannot state
  its losses MUST NOT be treated as verified.

**Repair**

- **FR-017**: Repairs MUST be per-named-pattern. No general-purpose repair.
- **FR-018**: Repairs MUST be opt-in, with the defect, the proposed change, and the expected losses
  stated before application.
- **FR-019**: Repairs MUST NOT modify the source; output is a new artifact.
- **FR-020**: A repair's output MUST pass the same validation the original failed. A repair that
  cannot prove itself MUST be reported as failed and MUST NOT be offered for use.
- **FR-021**: A repair pattern applied to a non-matching asset MUST be refused, never approximated.
- **FR-022**: The existing 384-group overflow merge MUST be validated against a real fuckported WMO
  and confirmed to load **and render** in the target client. If it does not, that MUST be reported as
  a finding rather than quietly corrected.

**Provenance**

- **FR-023**: Every written file MUST carry provenance: source build, contributing operations,
  validation results, repairs applied, losses recorded.
- **FR-024**: A write that cannot record provenance MUST be refused.

**Validation**

- **FR-025**: Crash recovery MUST be validated by actual process kills at multiple points including
  mid-write — not by simulated failures alone.
- **FR-026**: Integrity and repair MUST be validated against `H:\CLIENTS\WoW335\modernwow\`, the
  known fuckported 3.3.5 corpus, with commands, build identity, and hashes recorded.
- **FR-027**: Every asset in that corpus that currently crashes the viewer MUST, after this spec, be
  identified with its specific violated constraint and either quarantined or repaired. Converting a
  crash into a silent skip does not satisfy this.
- **FR-028**: The corpus MUST be swept end to end and its results published as a census — how many
  assets crash today, how many are classified afterwards, by defect class, and how many have a repair
  pattern. A sampled check does not satisfy this.

### Key Entities

- **Edit Journal**: The durable Zarr-backed record of a session's staged operations, written as they
  complete.
- **Editing Session**: One named, resumable body of staged work — its journal, source build, plugins
  involved, and status.
- **Validation Verdict**: The outcome of checking one asset — passed, failed with named constraint,
  or unable to determine (which counts as failing).
- **Quarantined Asset**: An asset that failed validation, retained with its verdict so it can be
  inspected or repaired, and blocked from contributing to writes.
- **Repair Pattern**: One named, narrow fix for one known defect — what it matches, what it changes,
  what it loses, and how its output is proven.
- **Loss Record**: What a lossy re-fit discarded, attached to the resulting artifact.
- **Output Provenance**: The full traceable history of a written file.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Killing the process at 10 different points during a mixed editing session loses no
  completed operation in any trial.
- **SC-002**: No process kill, at any point including mid-write, produces a recovered session in
  which a partial edit appears complete.
- **SC-003**: A recovered session saves output byte-identical to the same edits performed without
  interruption.
- **SC-004**: Journaling adds no perceptible latency to an edit and no frame-loop stall.
- **SC-005**: A session survives a clean exit and resumes days later with staged work intact.
- **SC-006**: Every known-malformed test asset is refused with a diagnostic naming the specific
  violated constraint — zero generic parse-error refusals.
- **SC-007**: No output file is ever produced from a quarantined or unverified input, across the full
  validation suite.
- **SC-008**: Every file the Editor reports as saved reads back and validates — zero
  reported-success-but-unreadable outputs.
- **SC-009a**: Zero assets in `H:\CLIENTS\WoW335\modernwow\` crash the viewer after this spec —
  every previously-crashing asset instead produces a named verdict.
- **SC-009b**: A published census of that corpus states, by defect class, how many assets crash
  today, how many are classified afterwards, and how many have a repair pattern.
- **SC-009**: The 384-group overflow merge is validated against a real fuckported WMO and its
  behavior stated as measured fact, replacing "not sure it even works right" with a result either
  way.
- **SC-010**: Every repair's output passes the validation its input failed; a repair that cannot is
  never offered.
- **SC-011**: Any written file's full history is reconstructible from provenance alone, without the
  session that produced it.
- **SC-012**: No source asset or game-install file is modified by any repair, verified by hashing
  before and after.

## Assumptions

- Spec 161 has landed; Editor Operations are data with a reverse, which is what makes them journalable
  at all. Journaling reuses that model rather than introducing a second change representation.
- Zarr conventions follow the repo's existing stores. This spec does not design a new on-disk format.
- The journal holds staged edits and provenance only — never a copy of client data.
- "Unverified" and "failed" are treated identically for write-blocking. Optimistic defaults are how
  the incumbent corrupts files.
- Repair coverage is deliberately narrow and grows only as specific defects are understood. Breadth
  here is a liability, not a feature.
- The 384-group merge is assumed **unproven, not broken**. This spec measures it; the result decides
  whether it needs fixing, and that fix would be its own work.
- Post-write verification re-reads the file. It does not attempt to prove semantic correctness beyond
  what validation covers — except for repairs, where a render check is required.
- Session storage is local and single-user.
- `H:\CLIENTS\WoW335\modernwow\` is the primary integrity corpus and is available now. It defines
  the defect classes worth writing repair patterns for — coverage is driven by what is actually in it,
  not by what is theoretically possible.
- A crash in that corpus is a *finding*, not a blocker. The census records it; the repair patterns
  follow the census rather than preceding it.
- A "fuckported" asset is any asset re-fitted from a later client to an earlier one, whether by this
  repo's tooling or by a third party before it arrived.
