# Feature Specification: Modern Chunk Completeness Survey & Legacy Build-In Feasibility

**Feature Branch**: `v0.5.4-dev` (v0.6 release line; no per-spec branch)

**Release**: v0.6 (research/doc lane)

**Created**: 2026-09-18

**Status**: Draft (operator-directed 2026-09-18)

**Depends on**: [Spec 239](../239-modern-client-assets/spec.md) (modern WDT/ADT reader),
[Spec 240](../240-format-conformance/spec.md) (wowdev.wiki × reader audit — this continues it),
[Spec 237](../237-adt-v26-terrain/spec.md) (DAT v26 corpus),
[Spec 243](../243-modern-to-legacy-map-conversion/spec.md) and
[Spec 244](../244-modern-liquid-flow/spec.md) (consumers of the findings)

**Input**: operator direction, 2026-09-18 — "we should probably look at all the modern chunks to
understand the data that we are just ignoring since it does not pertain to the versions of clients we
support writing to, but maybe we can find ways to build into via alpha mask and texture id
manipulation."

## Context

The modern-data lane reads just enough of a `wow_classic_beta` WDT/ADT family to render it. Everything
else in those files is currently discarded, and the discard is invisible: nothing lists what was
skipped, so no one can tell an intentional omission from an undiscovered feature. A concrete example
already in the release notes — the WDT `MAI2` chunk (≥ `12.0.5.66330`) is a 4096-entry table whose
first field is `liquidFlowTexture`, with seven further fields documented only as `unknown1..unknown7`
(see [Spec 244](../244-modern-liquid-flow/spec.md)).

The operator's hypothesis is worth testing rather than assuming: some of that ignored data may be
representable in the legacy targets this project *does* write (LK v18 ADT, Alpha 0.5.3 WDT) by
re-expressing it — for example through alpha-mask composition or texture-id selection — rather than
being genuinely lost. This spec produces the evidence that settles, per chunk, which of the three it
is: representable, not representable, or not yet understood.

## User Scenarios & Testing

### User Story 1 — One authoritative inventory of every modern chunk (Priority: P1)

Anyone asking "do we read chunk X?" gets an answer from one table, with the file families it appears
in, how often, and what the code currently does with it.

**Why this priority**: without it, every later decision (reading, converting, or ignoring) is guesswork.

**Independent Test**: pick any chunk name from a real corpus walk and find its row, its current state,
and the evidence for that state.

**Acceptance Scenarios**:

1. **Given** a real modern corpus, **When** the survey runs, **Then** every distinct chunk encountered
   appears in the inventory with the count and file family it was found in.
2. **Given** a chunk the viewer ignores, **When** its row is read, **Then** it states the current
   handling and the code (or absence of code) that proves it.
3. **Given** a chunk whose meaning is not documented, **When** its row is read, **Then** it is marked
   unknown rather than given an invented name.

### User Story 2 — Legacy build-in feasibility, per chunk (Priority: P1)

For every chunk we could start reading, the survey states whether the data can be carried into LK v18
and/or Alpha 0.5.3, and by what mechanism — including alpha-mask and texture-id manipulation — or why
it cannot.

**Independent Test**: pick any "expressible" verdict and follow it to the named target field; pick any
"not expressible" verdict and read the stated reason.

**Acceptance Scenarios**:

1. **Given** a candidate chunk, **When** its row is read, **Then** it names the legacy mechanism or
   states "not representable" with the reason.
2. **Given** an item that needs a lossy re-expression (for example one value mapped onto several
   texture ids), **When** its row is read, **Then** the loss is stated.

### User Story 3 — Reproducible, non-invasive evidence (Priority: P2)

The survey changes no runtime behaviour and can be re-run to the same numbers.

**Independent Test**: run the documented command twice over the same corpus; counts match.

**Acceptance Scenarios**:

1. **Given** the same corpus, **When** the survey is run twice, **Then** counts are identical.
2. **Given** a build after the survey, **When** the viewer runs, **Then** behaviour and output are
   unchanged.

## Requirements

- **FR-001**: The survey MUST cover every chunk in the modern file families actually in use: WDT (and
  its `_occ` and `_lgt` companions), root ADT, `_tex0`, `_obj0`, and `_lod`.
- **FR-002**: For every chunk the survey MUST record: name, file family, occurrence count, current
  handling (parsed / partially parsed / ignored / unknown), and the code reference proving that state.
- **FR-003**: For every ignored chunk the survey MUST record its documented meaning with a source and
  a confidence label (measured / documented / unknown). Unknown fields MUST stay unknown.
- **FR-004**: For every ignored chunk the survey MUST assign a disposition: read now, read later,
  deliberately ignore, or blocked-unknown.
- **FR-005**: For every "read" candidate the survey MUST state legacy feasibility for **both** LK v18
  ADT and Alpha 0.5.3 WDT: the concrete target field/mechanism, or "not representable" with a reason.
- **FR-006**: Mechanisms considered MUST include re-expression through alpha-mask composition and
  texture-id selection, and the survey MUST state the loss incurred by each.
- **FR-007**: The survey MUST be reproducible by one documented command over a named corpus, and MUST
  NOT alter runtime behaviour or existing readers.
- **FR-008**: Findings MUST be handed to [Spec 243](../243-modern-to-legacy-map-conversion/spec.md) /
  [Spec 244](../244-modern-liquid-flow/spec.md) as dated pointers or amendments, never as duplicated
  implementation.
- **FR-009**: The survey MUST ship a receipt per AGENTS.md §9.2 with the corpus identity, the command,
  and counts measured on real data.

## Key Entities

- **Chunk inventory row**: name, families, count, current handling, code reference.
- **Chunk disposition**: read now / read later / deliberately ignore / blocked-unknown.
- **Legacy feasibility verdict**: per target (LK v18, Alpha 0.5.3), mechanism or reason, and loss.

## Success Criteria

- **SC-001**: Every distinct chunk found in the corpus walk appears in the inventory — no chunk seen
  by the tooling is missing from the table.
- **SC-002**: No ignored chunk is left without a disposition and a documented meaning (or an explicit
  unknown).
- **SC-003**: Every "expressible" verdict names a concrete target mechanism; every negative verdict
  states why.
- **SC-004**: Two runs over the same corpus produce identical counts.
- **SC-005**: The viewer build and existing tests are unchanged by the survey.

## Assumptions & open questions

- Corpus: the operator's `wow_classic_beta` 1.60.1 install plus the 700-file DAT v26 set already on
  hand; other builds are not in scope and would be a follow-on.
- Documentation source is the wowdev.wiki; where it is silent the survey records "unknown" and does
  not invent a meaning (AGENTS.md §9.1).
- Whether unread modern chunks can be re-expressed is an *outcome* of this survey, not an assumption:
  the spec requires the verdict, not a positive one.
- The survey may surface new read work; each resulting implementation is its own phase/spec amendment
  and needs operator approval before code.