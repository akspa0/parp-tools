# Feature Specification: Canonical MPQ Patch-Chain Resolver

**Feature Branch**: `179-patch-chain-resolver`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Client Datastore](../epic-client-datastore/epic.md) — **read the epic first**; it carries
the measured evidence.
**Depends on**: nothing. **This must land first** — everything downstream inherits whatever it
resolves, correctly or not.

## Scope

One patch-priority implementation, shared by viewer and tools, with full letter/numeric/locale
precedence and recorded patch/delete provenance. Today there are **two**, and the dataset builder
inherited the weaker one — a Constitution II violation actively producing wrong data.

## User Story - The patch chain resolves correctly, once (Priority: P1)

Every tool resolves a client's archives the same way, deterministically, with patched and deleted
files recorded rather than inferred.

**Independent Test**: Resolve `H:\CLIENTS\WoW335\modernwow\` and confirm all **9** currently-tied
archives have deterministic, correct precedence, that viewer and builder resolve identically, and that
repeated runs match.

**Acceptance Scenarios**:

1. **Given** a client with letter, numeric, and locale patches, **When** archives are resolved,
   **Then** every archive has a distinct deterministic precedence — **no ties**.
2. **Given** the same client, **When** resolved by viewer and builder, **Then** both produce identical
   resolution for every file.
3. **Given** repeated runs on different machines, **When** compared, **Then** results are identical —
   no dependence on directory enumeration order.
4. **Given** a file replaced by a patch, **When** resolved, **Then** the winning archive and the
   archives it overrode are both recorded.
5. **Given** a file deleted by a patch, **When** resolved, **Then** it is absent **and** the deletion
   is recorded with the archive that performed it — distinguishable from never having existed.
6. **Given** an archive like `patch-enUS-3-orig.MPQ`, **When** archives are admitted, **Then** the
   admission decision is recorded, inspectable, and overridable.
7. **Given** this spec is complete, **When** the codebase is searched, **Then** exactly **one**
   patch-priority implementation remains.

### Edge Cases

- Two archives at genuinely equal precedence by any documented rule — reported, not silently resolved.
- A patch deleting a file a later patch re-adds.
- Zero-length files, which must be distinguishable from deletion markers.
- A client with no patches at all.
- Locale archives for a locale absent from another build.

## Requirements

### Functional Requirements

- **FR-001**: Exactly **one** patch-priority implementation exists, shared by viewer and tools.
- **FR-002**: Handle `patch.mpq`, `patch-N`, `patch-[a-z]`, and locale-qualified patches, each with a
  distinct deterministic precedence — no ties.
- **FR-003**: Resolution must not depend on filesystem enumeration order and must be reproducible
  across machines.
- **FR-004**: Record, per resolved file, the winning archive and the archives it overrode.
- **FR-005**: Record deletions with the archive that performed them, distinguishable from files that
  never existed.
- **FR-006**: Archive admission is a recorded, inspectable, overridable decision — never an incidental
  result of a filename pattern.
- **FR-007**: A file at genuinely equal precedence is **reported**, not silently resolved.
- **FR-008**: MPQ remains a **read-only input**. This spec writes no archive (**Constitution VII**).

## Success Criteria

- **SC-001**: Patch-priority implementations in the repo: **2 → 1**.
- **SC-002**: Archives tied at one rank on `H:\CLIENTS\WoW335\modernwow\`: **9 → 0**.
- **SC-003**: Viewer and builder resolve every file in a client identically, verified file-by-file.
- **SC-004**: Repeated resolution runs produce identical results across machines.
- **SC-005**: Deleted files are queryable as deleted and distinguishable from never-present, verified
  on a client with known patch deletions.

## Out of Scope

- The datastore itself ([180](../180-multi-build-datastore/spec.md)).
- CASC. A CASC reader follows the `IDataSource` seam separately, and is a **reader**.

## Assumptions

- The canonical implementation is expected to be the viewer's `MpqArchiveCatalog`, which is the more
  complete of the two — but this must be **verified against documented client behavior**, not adopted
  merely for being better than its sibling.
- Deletion semantics follow MPQ's zero-length-entry convention, already relied on by both existing
  implementations.
- The `*.mpq`/`*.MPQ` double-enumeration in `NativeMpqService` is **already de-duplicated** by
  `.Distinct(OrdinalIgnoreCase)`. It is not a bug — do not "fix" it.
