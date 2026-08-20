# Feature Specification: Incremental Processing (Derivation Dedupe)

**Feature Branch**: `181-incremental-processing`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Client Datastore](../epic-client-datastore/epic.md) — **read the epic first**, especially
"the hazard that shapes 181".
**Depends on**: [180](../180-multi-build-datastore/spec.md).

## Scope

**This is the datastore's actual payoff.** Storage dedupe saves disk, which is cheap. Derivation dedupe
saves harvest and processing, which is not. A byte-identical input already processed must cost **zero**
work.

The mechanism is one step from existing: `tools/harvest` already computes `InputSha256` and writes it
into a manifest (`Program.cs:642,689`) — and never compares it to anything.

## User Story - A new build costs only its new content (Priority: P1)

When the next client of the same era is added, everything byte-identical to what was already processed
is recognised and not processed again — only genuinely new content is harvested, derived, and stored.

**Independent Test**: Process build A fully. Process build B, which shares most content with A, and
measure the work actually performed — it must be proportional to B's **unique** content, not its total
content, and the resulting dataset must be identical to processing B from scratch.

**Acceptance Scenarios**:

1. **Given** a build is processed, **When** a later build shares byte-identical inputs, **Then** the
   derived artifacts for those inputs are reused rather than recomputed.
2. **Given** a build added incrementally, **When** compared to processing that build from scratch,
   **Then** the two are **identical**. Reuse never changes output.
3. **Given** a derived artifact, **When** reuse is considered, **Then** the decision accounts for the
   **complete** input set and the processing version — not one input file.
4. **Given** a derivation's complete input set cannot be enumerated with certainty, **When** reuse is
   considered, **Then** the work is **redone**. Uncertain reuse is prohibited.
5. **Given** the processing logic changes, **When** builds are reprocessed, **Then** artifacts derived
   by the old logic are not reused, and the processing version is recorded with each artifact.
6. **Given** an incremental run, **When** it completes, **Then** it reports what was reused and what
   was recomputed, so the saving is a **measured number** rather than a claim.

### Edge Cases

- A derived signal depending on a file **and its neighbours** and a DBC lookup — the key must cover all
  of them.
- Two processing versions in flight during a partial upgrade.
- An input whose bytes match but whose semantic context differs (different build era).
- A reuse index that disagrees with the store it describes.

## Requirements

### Functional Requirements

- **FR-001**: Reuse derived artifacts whose complete input set **and** processing version are unchanged,
  rather than recomputing them.
- **FR-002**: The reuse key covers the **complete** input set of a derivation plus the identity and
  version of the processing that produced it. A single input file's hash is **not** a sufficient key.
- **FR-003**: If a derivation's complete input set cannot be enumerated with certainty, the work is
  **redone**. Uncertain reuse is prohibited.
- **FR-004**: Incremental processing produces output **identical** to processing that build from
  scratch.
- **FR-005**: Changing processing logic invalidates artifacts derived by the previous version; each
  artifact records the processing version that produced it.
- **FR-006**: An incremental run reports what was reused and what was recomputed.
- **FR-007**: Derivation records and the reuse index live **inside** the datastore, not as sidecar
  files (**Constitution V**: the Zarr store is the only on-disk artifact). A cache separable from the
  store it describes can go stale against it.
- **FR-008**: Nearest-build comparison **may** be used to speed up change detection but must **not** be
  a correctness dependency. A wrong "nearest" choice costs only time — never a missed dedupe or a
  different result.

## Success Criteria

- **SC-001**: Processing a second same-era build performs work proportional to its **unique** content,
  reported as a measured ratio of reused to recomputed artifacts.
- **SC-002**: A build processed incrementally produces a dataset identical to that build processed from
  scratch, verified by hash across **every** derived artifact.
- **SC-003**: Changing one processing step causes affected artifacts to be recomputed — verified by
  altering a step and confirming **no stale artifact survives**.
- **SC-004**: A derivation whose input set cannot be fully enumerated is always recomputed, verified by
  constructing such a case.

## Out of Scope

- Storage-level deduplication (180).
- Encoding selection (182).

## Assumptions

- **A stale derivation cache in an ML corpus does not fail loudly — it silently trains on wrong data
  and surfaces months later as an unexplained result.** Recomputing costs time; a wrong reuse costs
  trust in the corpus. Every requirement above follows from that asymmetry.
- Derivation dedupe is expected to dominate storage dedupe in value, but both are **measured**, not
  assumed. If recompute avoidance turns out small, report it plainly.
