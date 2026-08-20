# Feature Specification: Asset Integrity Gate

**Feature Branch**: `173-asset-integrity-gate`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Editor Platform](../epic-editor-platform/epic.md) — **read the epic first**, especially the
validation corpus.
**Depends on**: [166](../166-editor-plugin-host/spec.md). Should land **before**
[172](../172-editor-edit-journal/spec.md).

## Scope

Nothing is written from input that did not verify. Assets are validated on read; a write is refused if
any contributing input is quarantined or unverified; every written file is re-read and verified before
being reported as saved. Detection and refusal only — repair is
[174](../174-asset-repair-patterns/spec.md).

**This closes the second Noggit failure mode**: ADTs corrupted "for no reason" because a reader
accepted a malformed input, produced plausible-looking garbage, and a writer serialized it.

## User Story - Nothing is written from input that did not verify (Priority: P1)

Before the Editor writes a file, everything that file's contents derive from has been checked. If any
input failed, the write does not happen and the user is told exactly what failed.

**Independent Test**: Sweep `H:\CLIENTS\WoW335\modernwow\` — a 3.3.5 client full of fuckported assets
that crash the viewer today — and confirm every previously-crashing asset instead produces a **named
verdict**, and that no output is produced from any of them.

**Acceptance Scenarios**:

1. **Given** an asset is read, **When** it fails structural validation, **Then** it is quarantined with
   a diagnostic **naming the constraint violated** — not a generic parse error.
2. **Given** a write is requested, **When** any contributing input is quarantined or unverified,
   **Then** the write is refused, the blocking inputs are named, and no partial file is left behind.
3. **Given** a write proceeds, **When** it completes, **Then** the file is **re-read and verified**
   before being reported as saved; a file that does not read back correctly is not reported as
   success.
4. **Given** validation cannot reach a verdict, **When** the input is used, **Then** it is treated as
   **unverified and blocks the write** — never assumed good.
5. **Given** a lossy downport path runs, **When** it completes, **Then** what it lost is recorded. A
   downport that cannot state its losses is not treated as verified.
6. **Given** any written file, **When** its provenance is inspected, **Then** source build,
   contributing operations, validation results, and losses are all present.

### Edge Cases

- An asset that validates in isolation but is invalid in combination with another.
- A fuckported asset whose defect is *within* budget limits and therefore structurally invisible.
- Killing the process during post-write verification.
- Provenance cannot be recorded — the write is refused; an untraceable write is not permitted.

## Requirements

### Functional Requirements

- **FR-001**: Validate assets structurally on read; failures name the violated constraint specifically.
- **FR-002**: Refuse a write if any contributing input is quarantined or unverified, naming the
  blocking inputs. "Unverified" is treated as failing, **not** as passing.
- **FR-003**: Re-read and verify every written file before reporting it as saved.
- **FR-004**: A refused or failed write leaves no partial file.
- **FR-005**: Lossy downport paths record what was lost, per asset.
- **FR-006**: Every written file carries provenance: source build, contributing operations, validation
  results, repairs applied, losses recorded. A write that cannot record provenance is refused.
- **FR-007**: Sweep the modernwow corpus end to end and publish a **census** — how many assets crash
  today, how many are classified afterwards, by defect class. A sampled check does not satisfy this.
- **FR-008**: Validate the existing 384-group WMO overflow merge against a **real** fuckported WMO and
  confirm whether the target client loads **and renders** the result. Report the finding either way;
  fixing it if broken is separate work.

## Success Criteria

- **SC-001**: Zero assets in `H:\CLIENTS\WoW335\modernwow\` crash the viewer; every previously-crashing
  asset produces a named verdict instead. **A crash converted into a silent skip is a regression, not
  a fix.**
- **SC-002**: A published census states, by defect class, how many assets crash today, how many are
  classified afterwards, and how many have no repair pattern yet.
- **SC-003**: Every known-malformed test asset is refused with a diagnostic naming the specific
  violated constraint — **zero** generic parse-error refusals.
- **SC-004**: No output file is produced from a quarantined or unverified input, across the full
  validation suite.
- **SC-005**: Every file reported as saved reads back and validates — zero
  reported-success-but-unreadable outputs.
- **SC-006**: The 384-group merge's behavior is stated as measured fact, replacing "not sure it even
  works right" with a result either way.

## Out of Scope

- Repairing anything ([174](../174-asset-repair-patterns/spec.md)).
- Fixing the 384-group merge if it proves broken — this spec **measures** it.
- Restructuring the existing converters.

## Assumptions

- The 384-group merge is **unproven, not broken**. It has synthetic unit coverage
  (`Convert_WhenSourceExceedsLegacyGroupLimit_MergesOverflowIntoFinalLegacyGroup`) but no real-data
  validation and no evidence a client accepts its output.
- A "fuckported" asset is any asset re-fitted from a later client to an earlier one, whether by this
  repo's tooling or by a third party before it arrived.
- The census drives which repair patterns 174 writes — coverage follows measurement, not speculation.
