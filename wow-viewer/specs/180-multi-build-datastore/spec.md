# Feature Specification: Multi-Build Content-Addressed Datastore

**Feature Branch**: `180-multi-build-datastore`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Client Datastore](../epic-client-datastore/epic.md) — **read the epic first**, especially
the content-addressing decision.
**Depends on**: [179](../179-patch-chain-resolver/spec.md).

## Scope

One Zarr datastore holding several client builds, content-addressed so shared bytes are stored once,
with builds added and removed cheaply — plus verification against source clients.

**Content-addressed storage, not a stored diff format.** A diff format needs a reconstruction path and
a bug there corrupts every version downstream. Content-addressing gets the diffs for free and is
*simpler*. See the epic; this decision is settled.

## User Story - One datastore, many builds, cheap to load and unload (Priority: P1)

Several client versions live in one store. Content shared between them is stored once. Each build is
fully reconstructible, and adding or removing one does not churn the rest.

**Independent Test**: Build a datastore from three 0.x clients; verify every file of every build
reconstructs byte-identically to the source, then remove one build and confirm the others still
verify with their stored bytes unchanged.

**Acceptance Scenarios**:

1. **Given** several clients, **When** a datastore is built, **Then** it contains each build's complete
   resolved file set, individually addressable by build.
2. **Given** two builds sharing a file, **When** storage is inspected, **Then** those bytes are stored
   **once**.
3. **Given** any build, **When** any file is read, **Then** it is byte-identical to what the source
   client's resolved patch chain yields.
4. **Given** a build with deletions, **When** read, **Then** deleted files are absent and the deletion
   is queryable.
5. **Given** two builds, **When** their difference is requested, **Then** added, removed, and changed
   files are reported **without reconstructing either build in full**.
6. **Given** a new build is added, **When** existing builds are checked, **Then** they are unaffected
   and not rewritten.
7. **Given** a build is removed, **When** the store is checked, **Then** only content **unique** to it
   is reclaimed, every remaining build still verifies, and no remaining build's bytes are rewritten.
8. **Given** a datastore and its sources, **When** verification runs, **Then** every file is confirmed
   byte-identical or reported as a named discrepancy.

### Edge Cases

- Two clients claiming the same build identity with different contents.
- A file whose content is identical across builds but whose path differs.
- A datastore larger than available memory during verification.
- Interrupted build — partial datastore must be resumable or cleanly discardable.
- Post-build corruption localized to affected content rather than failing the whole store.

## Requirements

### Functional Requirements

- **FR-001**: One datastore holds multiple builds, each individually addressable.
- **FR-002**: Storage is content-addressed so identical content across builds is stored once.
- **FR-003**: **No delta-chain or diff-reconstruction format.** Every file reconstructs by direct
  lookup.
- **FR-004**: Every file reads back byte-identical to the source client's resolved chain.
- **FR-005**: The datastore represents deletions.
- **FR-006**: Differences between any two builds are computable without full reconstruction.
- **FR-007**: Adding a build does not rewrite or affect existing builds.
- **FR-008**: Removing a build reclaims **only** content unique to it; content referenced by any
  remaining build survives, and remaining builds are not rewritten.
- **FR-009**: Content collection is safe against removing still-referenced content. **If reference
  state cannot be established with certainty, collection is refused** — an over-eager collection
  silently destroys a different build.
- **FR-010**: Reading a build must not extract or copy it to disk; content is read in place.
- **FR-011**: The datastore uses the repo's existing Zarr conventions and is **not**, and cannot be
  exported to, any Blizzard container (**Constitution VII**).
- **FR-012**: An interrupted build leaves a datastore that is resumable or cleanly discardable, never
  silently incomplete.
- **FR-013**: Verification confirms every file byte-identical or names the discrepancy, recording build
  identities, file counts, and hashes.
- **FR-014**: Arrays are self-describing and readable by **standard Zarr tooling with no code from this
  repository** — a preservation requirement, since the store must outlive the code that wrote it.

## Success Criteria

- **SC-001**: Every file in every build reads back byte-identical to its source, verified by hash
  across ≥3 builds.
- **SC-002**: Adding a build leaves prior builds' content bytes unchanged, verified by hash.
- **SC-003**: Removing a build leaves every remaining build verifying byte-identical, bytes unchanged.
- **SC-004**: Space reclaimed by removing a build equals that build's **unique** content, reported as a
  measured number alongside its total size.
- **SC-005**: Loading or switching builds writes no copy of build content to disk.
- **SC-006**: The difference between two builds is computable without reconstructing either.
- **SC-007**: Verification accounts for every file in every build — no unexplained entries.
- **SC-008**: Every array opens and decodes with standard Zarr tooling in a process that has none of
  this repository's code available.
- **SC-009**: No file inside any source client is created, modified, or deleted during a build.

## Out of Scope

- Avoiding recomputation of derived artifacts ([181](../181-incremental-processing/spec.md)). This spec
  dedupes **storage**; 181 dedupes **work**.
- Per-type encoding selection ([182](../182-adaptive-encoding/spec.md)).
- Viewer loading ([183](../183-datastore-viewer-load/spec.md)).

## Assumptions

- "Diffs" are a **view** over manifests, not a stored format.
- Builds are added and removed routinely, not written once — which is why reference-counted
  reclamation and read-in-place are requirements rather than optimizations.
- Builds are identified by the repo's existing build-identity conventions.
