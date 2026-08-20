# Feature Specification: Editor Edit Journal — Crash Recovery and Resumable Sessions

**Feature Branch**: `172-editor-edit-journal`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**.
**Depends on**: [168](../168-editor-session-undo/spec.md) — the session and operation history this
spec makes durable.

## Scope

Every completed Editor Operation is written to a Zarr-backed journal as it happens. A crash loses at
most the operation in flight. Sessions **persist by default** and can be resumed later, named, or
deleted — the journal is a work product, not only crash insurance.

## User Story - A crash loses nothing, and sessions resume (Priority: P1)

The user is mid-edit and the process dies. On relaunch, the Editor offers the session back with every
completed edit intact. The same mechanism lets them close cleanly on Monday and resume on Thursday.

**Independent Test**: Stage a mixed set of edits across plugins, kill the process without warning at
several points **including mid-write**, relaunch, and confirm every completed edit is recoverable and
no partial edit is presented as complete.

**Acceptance Scenarios**:

1. **Given** a user stages an edit, **When** the operation completes, **Then** it is durable in the
   journal **before** the Editor reports it as staged.
2. **Given** staged edits, **When** the process is killed without warning, **Then** relaunching
   detects the session and offers to resume it.
3. **Given** a session is resumed, **When** inspected, **Then** every completed edit is present with
   its original values, and the undo history is restored.
4. **Given** the process is killed **during** a journal write, **When** reopened, **Then** the
   incomplete edit is identified as incomplete and discarded or offered separately — **never** silently
   presented as complete.
5. **Given** a recovered session, **When** saved, **Then** output is identical to what the same edits
   would have produced with no crash.
6. **Given** multiple plugins had staged edits, **When** recovery runs, **Then** each plugin's state
   restores independently; one plugin's unrecoverable state does not block the others.
7. **Given** staged edits, **When** the user exits cleanly, **Then** the session persists by default
   and can be listed, named, resumed, or deleted.
8. **Given** a session recorded against a client that is no longer loaded, **When** resumed, **Then**
   the mismatch is detected and reported; edits are **never** silently applied to a different build.

### Edge Cases

- Disk fills mid-journal-write.
- The journal store is corrupted or partially deleted between sessions.
- A session resumed after its source files changed underneath it.
- An edit durable in the journal whose plugin no longer exists on relaunch.
- Journal growth during a long session of large terrain edits.

## Requirements

### Functional Requirements

- **FR-001**: Every completed Editor Operation is durable in the journal before being reported as
  staged.
- **FR-002**: The journal uses the repo's existing Zarr store conventions, not a new format, and
  **not** any Blizzard container (**Constitution VII**).
- **FR-003**: A crash must not damage previously journaled operations; at most the in-flight one is
  lost.
- **FR-004**: An incompletely written operation is detectable as incomplete and never presented as
  complete.
- **FR-005**: Recovery restores staged edits **and** the undo history.
- **FR-006**: Per-plugin state recovers independently.
- **FR-007**: Journaling must not stall the frame loop or make editing perceptibly slower.
- **FR-008**: Sessions persist across clean exits by default; users can list, name, resume, and delete
  them with contents summarized.
- **FR-009**: A session records the client build it was made against; resuming against a different
  build is detected and reported.
- **FR-010**: Nothing in the journal is deleted without user consent; growth is surfaced with what is
  consuming space.
- **FR-011**: The journal holds staged edits and their provenance only — **never** a copy of client
  data.

## Success Criteria

- **SC-001**: Killing the process at 10 different points during a mixed editing session loses no
  completed operation in any trial.
- **SC-002**: No kill, at any point including mid-write, produces a recovered session in which a
  partial edit appears complete.
- **SC-003**: A recovered session saves output byte-identical to the same edits performed without
  interruption.
- **SC-004**: Journaling adds no perceptible latency to an edit and no frame-loop stall.
- **SC-005**: A session survives a clean exit and resumes days later with staged work intact.

## Out of Scope

- Validating the *content* being edited ([173](../173-asset-integrity-gate/spec.md)). A faithful
  journal of a path to corrupt output is only half the problem — which is why 173 lands first.
- Multi-user or networked collaboration on a journal.

## Assumptions

- Zarr chunk writes are individual files, so a crash damages at most the chunk in flight and a reader
  can open a partially-written store and see what completed. This is why Zarr is the substrate.
