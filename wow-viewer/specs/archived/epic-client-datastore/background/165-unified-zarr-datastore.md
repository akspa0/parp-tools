# Feature Specification: Unified Multi-Version Zarr Datastore

**Feature Branch**: `165-unified-zarr-datastore`

**Created**: 2026-08-19

**Status**: Draft

**Input**: User description: "The viewer should gain a 'Load Zarr Datastore', which should be able to support multiple versions of client data in a single datastore. We need to improve our tooling that builds the single datastore dataset, as it does not currently just look at client data changed in patch/patch-[0-9]/[A-Z].mpq files, nor what files are patched or deleted… We need more tooling around building a proper diffs-store for all versions of 0.x, so we can just refer to the datastore version instead of the original MPQ version, and thus store all the 0.x client data in a single package for all viewing and inference needs. just an idea, it makes everything easier and faster to deal with, maybe?"

**Related**: [164 — Editor Data Safety](../164-editor-data-safety/spec.md) shares the Zarr
conventions but is independent; neither blocks the other.

## Context

### The complaint is correct, and the cause is duplication

There are **two** MPQ patch-priority implementations in this repo, and the dataset builder inherited
the weaker one.

| Case | [`MpqArchiveCatalog`](../../src/core/WowViewer.Core.IO/Files/MpqArchiveCatalog.cs) (viewer runtime) | [`NativeMpqService`](../../src/core/WowViewer.Core.IO/Files/NativeMpqService.cs) (**harvest / builder**) |
|---|---|---|
| `patch.mpq` | ✅ | ✅ |
| `patch-N.mpq` | ✅ | ✅ |
| `patch-[a-z].mpq` | ✅ ranked `500 + letter` | ❌ **all → 1099** |
| `patch-<locale>-N.mpq` | ✅ locale-aware | ❌ **all → 1099** |
| Deletion markers | ✅ skipped, counted (`MpqPatchedDeleteHitCount`) | ⚠️ skipped silently, no provenance |

`tools/harvest` constructs `NativeMpqService` in **8 places**. Since `.OrderBy()` is a stable sort,
every archive that lands on rank 1099 keeps its *filesystem enumeration order* — which is not patch
precedence, and is not even guaranteed stable across machines.

This is a Constitution II violation (one canonical owner per format surface) that is actively
producing wrong data, not merely duplicated code.

### Measured on a real client

`H:\CLIENTS\WoW335\modernwow\` — a 3.3.5 client that is also the fuckported corpus for spec 164 —
puts **9 archives on rank 1099**:

| Class | Archives |
|---|---|
| Letter patches | `Patch-D.mpq`, `Patch-N.MPQ`, `patch-P.MPQ`, `Patch-T.mpq`, `Patch-Y.mpq` |
| Locale patches | `patch-enUS.MPQ`, `patch-enUS-2.MPQ`, `patch-enUS-3.MPQ`, `patch-enUS-3-orig.MPQ` |

Every dataset built from this client has undefined precedence among nine archives. Any file those
archives disagree about was resolved by directory order.

`patch-enUS-3-orig.MPQ` deserves separate mention: it is a **backup that parses as a patch**. Under
both implementations it is loaded as a live archive. Real client directories accumulate this kind of
debris, so archive admission has to be a decision the builder records and the user can inspect —
never an incidental consequence of a filename glob.

*(Checked and cleared: the `*.mpq` + `*.MPQ` double-enumeration in `NativeMpqService` is de-duplicated
by `.Distinct(OrdinalIgnoreCase)`. Not a bug.)*

### What "patched or deleted" actually requires

Resolution order alone is not enough. A patch chain expresses three distinct facts, and only the
first survives today:

1. **This file replaced that one** — needs order (broken above for 9 archives).
2. **This file was deleted** — MPQ marks deletion with a zero-length entry. Both implementations skip
   these, so the *effect* is right; but the builder records no trace, so a dataset cannot distinguish
   "never existed" from "deliberately removed in patch-3."
3. **This file was never touched** — the base-archive majority, which is what makes cross-version
   deduplication worthwhile.

A datastore that cannot express deletion cannot faithfully represent a client, and cannot answer the
question the user actually wants: *what changed between these two builds?*

### The infrastructure exists

Zarr v3 is already the repo's dataset substrate: the harvester writes stores
([`zarr_io.py`](../../data-harvester/src/harvester/zarr_io.py), Blosc+Zstd+bitshuffle, per-key chunk
presets), the viewer reads them ([`ZarrTileDatasetLoader`](../../src/viewer/WoWViewer/Terrain/ZarrTileDatasetLoader.cs)),
and [`DatasetVersionCatalog`](../../src/core/WowViewer.Core/Maps/DatasetVersionCatalog.cs) already
models `DatasetSourceKind.ZarrStore`. Stores exist on disk today (`output/object-library/objlib_0_5_3_3368.zarr`).

The gap is that a store is **one build**. The user wants one store, many builds — confirmed in
session, with the reason: **builds should load and unload cheaply, without lots of disk churn.**

That reason is a requirement, not a nicety, and it constrains the design in two directions that a
"just put them all in one store" reading would miss:

- **Unloading must reclaim only what is unique to the removed build.** Content shared with builds
  that remain must survive untouched. In a content-addressed store this is reference-counted
  collection — natural, but also the one operation that can destroy a *different* build's data if it
  gets the count wrong. It is specified here with that hazard in mind (FR-014a/FR-014b).
- **Loading must not extract.** Reading a build means reading it in place, not unpacking it to a
  temporary directory first. Extraction would reintroduce exactly the disk churn the single store is
  meant to eliminate, and would make switching builds cost a full copy.

### The primary surface is ML/AI, and the dedupe that matters is compute

Confirmed in session: the datastore exists so that **any build can be processed into a dataset once,
and subsequent clients dedupe into it** — the point being to save *time and energy later*, with ML/AI
as the intended consumer more than the viewer.

That reorders the value. Storage dedupe saves disk, which is cheap. **Derivation dedupe saves harvest
and processing, which is not.** If build B's ADT is byte-identical to build A's already-processed
ADT, then every tensor derived from it is already computed — and the correct amount of work to do is
zero. Across many same-era 0.x builds that share the overwhelming majority of their content, that is
the difference between processing every build and processing every build's delta.

The mechanism is one step from existing. The harvest pipeline **already computes an input SHA-256**
and records it in its manifest (`InputSha256`, `tools/harvest/.../Program.cs:642,689`) — and never
compares it to anything. Hashing is a provenance habit here; it has never been used to avoid work.

**The hazard this creates, stated plainly.** A stale derivation cache in an ML corpus is worse than a
slow rebuild, because it does not fail — it silently trains on wrong data, and the error surfaces
much later as an unexplained result. Two rules follow, and they are the reason FR-029..FR-033 are
written the way they are:

1. **The reuse key is the complete input set plus the processing version**, never a single file.
   A signal derived from terrain *and* its neighbours *and* a DBC lookup is only reusable if all of
   those are unchanged and the code that combined them is unchanged.
2. **If the input set cannot be enumerated with certainty, redo the work.** Uncertain reuse is
   prohibited outright — the same discipline spec 164 applies to unverified assets, for the same
   reason. Recomputing costs time; a wrong reuse costs trust in the corpus.

### The approach is already validated at scale

This is not a proposal from zero. A prototype already exists and was measured: **~160 GB of builds
stored as ~40 GB of Zarr — roughly 4:1 — and that was the *unoptimized* case**, carrying multiple
copies of the same data across build-versions. Two independent gains are therefore still unclaimed:

1. **Cross-build deduplication** (US2/US3) removes the duplicate copies that 4:1 was measured *with*.
2. **Per-type encoding** (US7) — the measurement was taken at Blosc **lz4 clevel-1, no shuffle
   specified** (`data-harvester/src/harvester/v25/dataset.py:62`), which is close to the weakest
   setting available.

That reframes the deduplication question from "will this help?" to "**how far past 4:1 does this
go?**" — and it makes 4:1 the floor a multi-build store must beat, not a hope.

*(Noted in passing: there are two codec defaults in the harvester —
[`zarr_io.py:33`](../../data-harvester/src/harvester/zarr_io.py) uses zstd clevel-5 with bitshuffle,
while [`v25/dataset.py:62`](../../data-harvester/src/harvester/v25/dataset.py) uses lz4 clevel-1.
Another duplicated decision worth converging under US7.)*

### Why the decode goes all the way down

The corpus was deliberately decoded **completely** — every field of the ADT and of the assets an ADT
references — so the whole MPQ corpus survives as portable Zarr rather than as opaque blobs. Two
consequences follow, and they pull in the same direction:

- **ML pipelines can operate on the data structurally**, not by re-parsing files at training time.
- **Preservation**: a fully decoded, self-describing store outlives the code that wrote it. A store
  of opaque blobs is only as durable as its parser; a store of named, typed arrays is readable by
  anyone with standard Zarr tooling, years from now, without this repo. That is a requirement here
  (FR-036), not a side effect.

This datastore is also intended as **a node in a larger system** for building Warcraft experiences —
which raises the bar on portability and self-description specifically, since the consumer may not be
this project.

### On nearest-build diffing — a clarification worth making

The request is to "diff a previous build processed against the nearest build that's incoming." Worth
separating two things that look like one:

- **Deduplication does not need it.** With content-addressed storage, a match is found by hash
  regardless of which build first contributed it. Content shared with a build five versions away
  dedupes exactly as well as content shared with the immediate predecessor. Correctness is unaffected
  by comparison order.
- **Scanning does benefit from it.** Choosing the nearest already-processed build as the comparison
  baseline is the fastest way to *find* what changed, because most paths will match immediately.

So nearest-build selection is a **scan optimization, not a dedupe requirement** (FR-037). This matters
because it must never become a correctness dependency: a wrong "nearest" guess should cost time, never
a missed dedupe or a wrong result.

### On the diffs-store idea — a direct answer

The user offered this tentatively (*"just an idea… maybe?"*). It is a good idea, and worth stating
why, because the strongest version is not the one described.

**Do not build a diff format.** Diff formats need a reconstruction path, and a bug in that path
corrupts every version downstream of it — precisely the failure class specs 164 and 165 exist to
eliminate.

**Build content-addressed storage instead, and the diffs are implicit.** Store every file by the hash
of its contents; each build is a manifest mapping paths to hashes. Then:

- Bytes shared between 0.5.3 and 0.5.5 are stored **once**, automatically, with no diff logic.
- The diff between any two builds is a set difference over manifests — computable in any direction,
  between any pair, including pairs never anticipated.
- Every build reconstructs by direct lookup. There is no delta chain, so there is no reconstruction
  bug class.
- Deletion is representable: a path present in one manifest and absent from the next.
- Verification is intrinsic — content hashes are their own checksums.

This gets the user's stated benefits (one package, refer to the datastore version rather than the MPQ,
faster and easier) while being *simpler* than a diff store rather than more complex. The compression
comes from deduplication, which for same-era 0.x builds should be substantial — that ratio is a
measurement this spec must report, not assume.

### Out of scope

- **Replacing the viewer's live MPQ path.** The datastore is an additional source, not a replacement.
- **Writing MPQ or any Blizzard container.** Containers are read-only inputs (Constitution VII); the
  datastore is Zarr. There is no "export a build back to MPQ" path, now or later.
- **CASC.** Follows spec 161's `IDataSource` seam separately.
- **Editing datastore contents.** The datastore is built and read; the Editor writes loose output.
- **Re-deriving existing harvested datasets.** Existing tensor stores keep working.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The patch chain resolves correctly, once (Priority: P1)

Every tool in the repo resolves a client's archives the same way — full letter, numeric, and locale
patch precedence, with patched and deleted files recorded rather than inferred.

**Why this priority**: Everything downstream is wrong if this is wrong, and it is measurably wrong
today for nine archives on a client already on disk. It also removes a duplicate format owner.

**Independent Test**: Resolve `H:\CLIENTS\WoW335\modernwow\` and confirm all 9 previously-tied
archives have deterministic, correct precedence, that resolution is identical across the viewer and
the builder, and that repeated runs produce identical results.

**Acceptance Scenarios**:

1. **Given** a client with letter, numeric, and locale patches, **When** archives are resolved,
   **Then** every archive has a distinct, deterministic precedence with no ties.
2. **Given** the same client, **When** resolved by the viewer and by the builder, **Then** both
   produce identical resolution for every file.
3. **Given** repeated runs on different machines, **When** results are compared, **Then** they are
   identical — no dependence on directory enumeration order.
4. **Given** a file replaced by a patch, **When** it is resolved, **Then** the winning archive and the
   archives it overrode are both recorded.
5. **Given** a file deleted by a patch, **When** it is resolved, **Then** it is absent **and** the
   deletion is recorded with the archive that performed it — distinguishable from never having
   existed.
6. **Given** an archive like `patch-enUS-3-orig.MPQ`, **When** archives are admitted, **Then** the
   admission decision is recorded and inspectable, and the user can override it.
7. **Given** this story is complete, **When** the codebase is searched, **Then** exactly **one**
   patch-priority implementation remains.

---

### User Story 2 - One datastore, many builds (Priority: P1)

A single Zarr datastore holds several client versions. Content shared between them is stored once.
Each build is fully reconstructible.

**Why this priority**: It is the core ask, and it is what makes "refer to the datastore version
instead of the MPQ" possible.

**Independent Test**: Build a datastore from three 0.x clients; verify every file of every build
reconstructs byte-identically to the source client, and report the deduplication ratio achieved.

**Acceptance Scenarios**:

1. **Given** several clients, **When** a datastore is built, **Then** it contains each build's
   complete resolved file set, individually addressable by build.
2. **Given** two builds sharing a file, **When** storage is inspected, **Then** those bytes are stored
   once.
3. **Given** any build in the datastore, **When** any file is read, **Then** it is byte-identical to
   what the source client's resolved patch chain yields.
4. **Given** a build with deletions, **When** it is read, **Then** deleted files are absent and the
   deletion is queryable.
5. **Given** two builds, **When** their difference is requested, **Then** added, removed, and changed
   files are reported without reconstructing either build in full.
6. **Given** a datastore, **When** a new build is added, **Then** existing builds are unaffected and
   are not rewritten.
7. **Given** a datastore with several builds, **When** one is removed, **Then** only content unique
   to it is reclaimed, every remaining build still verifies byte-identical, and no remaining build's
   stored bytes are rewritten.
8. **Given** a build is removed, **When** the reclaimed space is measured, **Then** it corresponds to
   that build's unique content — not to its full size, and not to nothing.

---

### User Story 3 - A new build costs only its new content (Priority: P1)

A client is processed into the datastore. When the next client of the same era is added, everything
byte-identical to what was already processed is **recognised and not processed again** — only genuinely
new content is harvested, derived, and stored.

**Why this priority**: This is the datastore's actual payoff. Storage dedupe saves disk, which is
cheap; **derivation dedupe saves harvest and processing time, which is not.** For a corpus of many
same-era 0.x builds that overwhelmingly share content, this is the difference between processing each
build and processing each build's delta.

**Independent Test**: Process build A fully. Process build B, which shares most content with A, and
measure how much work was actually performed — it must be proportional to B's unique content, not to
B's total content, and the resulting dataset must be identical to processing B from scratch.

**Acceptance Scenarios**:

1. **Given** a build is processed, **When** a later build shares byte-identical inputs, **Then** the
   derived artifacts for those inputs are reused rather than recomputed.
2. **Given** a build is added incrementally, **When** the result is compared to processing that build
   from scratch, **Then** the two are identical. Reuse is never allowed to change the output.
3. **Given** a derived artifact, **When** reuse is considered, **Then** the decision accounts for the
   **complete** input set and the processing version — not one input file.
4. **Given** the complete input set for a derivation cannot be enumerated with certainty, **When**
   reuse is considered, **Then** the work is **redone**. Uncertain reuse is prohibited.
5. **Given** the processing logic changes, **When** builds are reprocessed, **Then** artifacts derived
   by the old logic are not reused, and the processing version is recorded with each artifact.
6. **Given** an incremental run, **When** it completes, **Then** it reports what was reused and what
   was recomputed, so the saving is a measured number rather than a claim.

---

### User Story 4 - The datastore is verifiable against its sources (Priority: P2)

The user can prove a datastore faithfully represents the clients it was built from.

**Why this priority**: A datastore that becomes the reference for viewing and inference must be
provably faithful, or every downstream result inherits an unverified assumption.

**Independent Test**: Verify a datastore against its source clients and confirm the report accounts
for every file — matched, deleted, or explained.

**Acceptance Scenarios**:

1. **Given** a datastore and its sources, **When** verification runs, **Then** every file is confirmed
   byte-identical or reported as a discrepancy.
2. **Given** a discrepancy, **When** it is reported, **Then** the path, build, and nature are named.
3. **Given** verification passes, **When** results are recorded, **Then** they include build
   identities, file counts, and hashes.
4. **Given** a datastore is corrupted after building, **When** verification runs, **Then** the
   affected content is identified rather than the store being declared bad wholesale.

---

### User Story 5 - Load a datastore in the viewer (Priority: P3)

The user picks **Load Zarr Datastore**, chooses a build from those it contains, and the viewer works
as if that client were installed.

**Why this priority**: Useful — one artifact instead of a directory of clients — but the datastore's
primary surface is ML/AI, not viewing. **Demoted to P3 in session**: viewer loading is a convenience
that falls out of a correct datastore, not the reason to build one.

**Independent Test**: Load the same map from a datastore build and from the original client; renders
and loaded data must match.

**Acceptance Scenarios**:

1. **Given** a datastore, **When** the user chooses Load Zarr Datastore, **Then** its builds are listed
   with identity and contents summarized.
2. **Given** a build is selected, **When** the viewer loads, **Then** behavior matches loading the
   original client — same maps, models, tables.
3. **Given** a datastore build is loaded, **When** the user switches builds, **Then** it works without
   restarting, and without extracting either build to disk.
3a. **Given** a build is loaded, **When** disk activity is observed, **Then** no copy of the build's
   content is written anywhere — files are read from the datastore in place.
4. **Given** a datastore is missing data a request needs, **When** the request is made, **Then** the
   gap is reported specifically, not silently rendered as absent content.
5. **Given** a datastore build is loaded, **When** the user checks provenance, **Then** the datastore,
   build, and original client identity are all visible.

---

### User Story 6 - One package for all 0.x, as a training corpus (Priority: P3)

Every available 0.x build lives in one datastore that is the working reference for viewing and
inference.

**Why this priority**: The destination the earlier stories build toward — the corpus ML/AI work
references instead of a shelf of clients. P3 because it is mostly *applying* US1-US4 at scale, and
because its value depends on measured deduplication and measured recompute avoidance.

**Independent Test**: Build the full 0.x datastore, report its size against the sum of source clients,
and confirm every build verifies.

**Acceptance Scenarios**:

1. **Given** all available 0.x clients, **When** one datastore is built, **Then** every build verifies
   against its source.
2. **Given** the datastore, **When** its size is compared to the sum of sources, **Then** the
   deduplication ratio is reported as measured fact.
3. **Given** the datastore, **When** an inference or viewing workflow references a build, **Then** it
   can use the datastore version in place of the original client.
4. **Given** a new 0.x build appears later, **When** it is added, **Then** only its unique content is
   stored.

### User Story 7 - Storage encoding is tuned per data type, by measurement (Priority: P2)

Each kind of array is stored with the codec and level that suit *it*, chosen from measured
compression ratio and decompression throughput — not one global default applied to everything.

**Why this priority**: The current measurement was taken at nearly the weakest setting, so the
headroom is real and unclaimed. P2 because a correct multi-build store (US1-US3) is worth more than a
smaller one, and because tuning is meaningless until deduplication has removed the duplicate copies.

**Independent Test**: Encode the same corpus under the current uniform default and under per-type
selection; report size and decompression throughput for both, per array type.

**Acceptance Scenarios**:

1. **Given** the array types in the corpus, **When** encoding is chosen, **Then** each type's codec
   and level come from a measurement of that type, not from a global default.
2. **Given** a candidate encoding, **When** it is evaluated, **Then** both compression ratio **and**
   decompression throughput are measured, because the primary consumer reads this data repeatedly
   during training.
3. **Given** a type where higher compression costs more time than it saves in size, **When** encoding
   is selected, **Then** the faster setting wins and the reason is recorded.
4. **Given** parser or decoder knowledge suggests an array is highly redundant, **When** that
   suggestion is used, **Then** it is treated as a **hypothesis that selects candidates to measure**,
   never as the decision itself.
5. **Given** encodings are chosen, **When** any array is read, **Then** it decodes correctly through
   standard Zarr tooling with no custom codec.
6. **Given** the repo currently carries two different codec defaults, **When** this story completes,
   **Then** one selection mechanism governs both paths.

---

### Edge Cases

- Two archives at genuinely equal precedence by any documented rule.
- A patch deleting a file another patch later re-adds.
- An archive that fails to open mid-build.
- Two clients claiming the same build identity with different contents.
- A file whose content is identical across builds but whose path differs.
- Zero-length files, which must be distinguishable from deletion markers.
- A datastore larger than available memory during verification.
- Locale archives for a locale not present in another build.
- A client with no patches at all.
- Interrupted build — partial datastore must be resumable or cleanly discardable.

## Requirements *(mandatory)*

### Functional Requirements

**Patch chain resolution**

- **FR-001**: Exactly **one** patch-priority implementation MUST exist, shared by viewer and tools.
- **FR-002**: Resolution MUST handle `patch.mpq`, `patch-N`, `patch-[a-z]`, and locale-qualified
  patches, with a distinct deterministic precedence for each — no ties.
- **FR-003**: Resolution MUST NOT depend on filesystem enumeration order and MUST be reproducible
  across machines.
- **FR-004**: For each resolved file, the winning archive and overridden archives MUST be recorded.
- **FR-005**: Deletions MUST be recorded with the archive that performed them, and MUST be
  distinguishable from files that never existed.
- **FR-006**: Archive admission MUST be a recorded, inspectable, overridable decision — never an
  incidental result of a filename pattern.
- **FR-007**: A file at genuinely equal precedence MUST be reported rather than silently resolved.

**Datastore**

- **FR-008**: One datastore MUST hold multiple builds, each individually addressable.
- **FR-009**: Storage MUST be content-addressed so identical content across builds is stored once.
- **FR-010**: No delta-chain or diff-reconstruction format. Every file MUST be reconstructible by
  direct lookup.
- **FR-011**: Every file MUST read back byte-identical to the source client's resolved chain.
- **FR-012**: The datastore MUST represent deletions.
- **FR-013**: Differences between any two builds MUST be computable without full reconstruction.
- **FR-014**: Adding a build MUST NOT rewrite or affect existing builds.
- **FR-014a**: Removing a build MUST reclaim only content unique to it. Content referenced by any
  remaining build MUST survive, and remaining builds MUST NOT be rewritten.
- **FR-014b**: Content collection MUST be safe against removing still-referenced content. If
  reference state cannot be established with certainty, collection MUST be refused rather than
  performed optimistically — an over-eager collection silently destroys a different build.
- **FR-014c**: Reading a build MUST NOT extract or copy it to disk. Content is read from the
  datastore in place, so switching builds costs no copy.
- **FR-015**: The datastore MUST use the repo's existing Zarr conventions.
- **FR-015a**: The datastore MUST NOT be, contain, or be exportable to any Blizzard container. MPQ is
  an input format only (Constitution VII).
- **FR-016**: An interrupted build MUST leave a datastore that is resumable or cleanly discardable,
  never silently incomplete.

**Incremental processing (derivation dedupe)**

- **FR-029**: Processing a build MUST reuse derived artifacts whose complete input set and processing
  version are unchanged, rather than recomputing them.
- **FR-030**: The reuse key MUST cover the **complete** input set of a derivation plus the identity
  and version of the processing that produced it. A single input file's hash is not a sufficient key.
- **FR-031**: If a derivation's complete input set cannot be enumerated with certainty, the work MUST
  be redone. Uncertain reuse is prohibited.
- **FR-032**: Incremental processing MUST produce output identical to processing that build from
  scratch. Reuse MUST never change results.
- **FR-033**: Changing processing logic MUST invalidate artifacts derived by the previous version;
  each artifact MUST record the processing version that produced it.
- **FR-034**: An incremental run MUST report what was reused and what was recomputed.
- **FR-035**: Derivation records and the reuse index MUST live **inside** the datastore, not as
  sidecar files beside it (Constitution V: the Zarr store is the only on-disk artifact). A cache that
  can be separated from the store it describes can go stale against it.

**Viewer integration**

- **FR-017**: The viewer MUST offer **Load Zarr Datastore**, listing builds with identity and contents.
- **FR-018**: A loaded datastore build MUST behave equivalently to the original client.
- **FR-019**: Switching builds MUST NOT require a restart.
- **FR-020**: Missing data MUST be reported specifically, never silently rendered as absent content.
- **FR-021**: Provenance — datastore, build, original client identity — MUST be visible while loaded.
- **FR-022**: Datastore access MUST go through the existing data-source abstraction.

**Encoding, preservation, and portability**

- **FR-036**: Every array MUST be readable by **standard Zarr tooling without this repository's
  code** — no custom codecs, no undocumented conventions, no parser required to interpret stored
  values. This is a preservation requirement: the store must outlive the code that wrote it.
- **FR-037**: Nearest-build comparison MAY be used to speed up change detection, but MUST NOT be a
  correctness dependency. A wrong "nearest" choice MUST cost only time — never a missed deduplication
  or a different result.
- **FR-038**: Encoding MUST be selected per array type from measured compression ratio **and**
  measured decompression throughput. A single global codec/level applied to all arrays does not
  satisfy this.
- **FR-039**: Parser- or decoder-derived complexity estimates MAY narrow the candidates to measure.
  They MUST NOT select an encoding on their own.
- **FR-040**: One encoding-selection mechanism MUST govern all writers. The repo's two current codec
  defaults MUST converge.
- **FR-041**: Arrays MUST be self-describing — name, dtype, shape, units/semantics, and provenance
  discoverable from the store itself.

**Verification**

- **FR-023**: A datastore MUST be verifiable against its sources, confirming every file byte-identical
  or reporting a named discrepancy.
- **FR-024**: Verification MUST record build identities, file counts, and hashes.
- **FR-025**: Post-build corruption MUST be localized to affected content, not fail the whole store.
- **FR-026**: The deduplication ratio MUST be reported as measured fact.

**Validation**

- **FR-027**: Patch resolution MUST be validated against `H:\CLIENTS\WoW335\modernwow\`, confirming
  all 9 currently-tied archives resolve deterministically and correctly.
- **FR-028**: The datastore MUST be validated against real clients from `H:\CLIENTS` spanning at least
  three builds, with commands, build identity, and hashes recorded.

### Key Entities

- **Archive Chain**: The ordered, deterministic set of archives for one client, with each archive's
  admission decision and precedence recorded.
- **Resolution Record**: For one file — winning archive, overridden archives, or the deletion and the
  archive that performed it.
- **Build Manifest**: One client build's complete resolved state — path to content identity, plus
  deletions and build identity.
- **Content Object**: One stored file's bytes, addressed by content, shared by every build referencing
  it.
- **Datastore**: The container of content objects and build manifests, with its own integrity record.
- **Build Difference**: Added, removed, and changed paths between two manifests.
- **Encoding Profile**: For one array type — the codec and level chosen, the measurements that
  justified them, and the reason.
- **Derivation Record**: One derived artifact's complete input set, processing identity and version,
  and resulting content — the basis for deciding whether it can be reused.
- **Content Reference State**: Which builds reference which content objects — the basis for safely
  reclaiming space when a build is removed.
- **Verification Report**: The outcome of checking a datastore against sources.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The repo contains exactly **one** patch-priority implementation — down from two.
- **SC-002**: All 9 currently-tied archives in `H:\CLIENTS\WoW335\modernwow\` resolve with distinct
  deterministic precedence; zero ties remain.
- **SC-003**: Viewer and builder resolve every file in a client identically, verified file-by-file.
- **SC-004**: Repeated resolution runs produce identical results across machines.
- **SC-005**: Every file in every datastore build reads back byte-identical to its source client,
  verified by hash across at least 3 builds.
- **SC-006**: Deleted files are queryable as deleted and distinguishable from never-present, verified
  on a client with known patch deletions.
- **SC-007**: The deduplication ratio for a multi-build 0.x datastore is reported as a measured
  number, with total size compared to the sum of sources, **and it beats the prototype's measured
  ~4:1 (≈160 GB → ≈40 GB) baseline.** That baseline was achieved with duplicate per-build copies and
  near-minimum compression, so failing to beat it indicates a defect, not a limit.
- **SC-007a**: Per-type encoding selection is reported as a table of array type → chosen codec/level
  → measured ratio → measured decompression throughput, with the reason each was chosen.
- **SC-007b**: Every array in the datastore opens and decodes with standard Zarr tooling in a process
  that has none of this repository's code available.
- **SC-007c**: Exactly one encoding-selection mechanism exists — down from two codec defaults.
- **SC-008**: A user can load a build from a datastore and reach the same rendered result as loading
  the original client.
- **SC-009**: Adding a build to an existing datastore leaves prior builds' content bytes unchanged,
  verified by hash before and after.
- **SC-009a**: Removing a build from a multi-build datastore leaves every remaining build verifying
  byte-identical, with their stored bytes unchanged — verified by hash before and after.
- **SC-009b**: Space reclaimed by removing a build equals that build's unique content, reported as a
  measured number alongside its total size.
- **SC-009c**: Loading and switching builds writes no copy of build content to disk — verified by
  observing writes during a switch.
- **SC-010a**: Processing a second same-era build performs work proportional to its **unique**
  content, not its total content — reported as a measured ratio of reused to recomputed artifacts.
- **SC-010b**: A build processed incrementally produces a dataset identical to that build processed
  from scratch, verified by hash across every derived artifact.
- **SC-010c**: Changing processing logic causes affected artifacts to be recomputed, verified by
  altering one processing step and confirming no stale artifact survives.
- **SC-010**: The difference between two builds is computable without reconstructing either in full.
- **SC-011**: Verification accounts for every file in every build — no unexplained entries.
- **SC-012**: No file inside any source client is created, modified, or deleted during a build,
  verified by hashing before and after.

## Assumptions

- The ~4:1 prototype result (≈160 GB → ≈40 GB) is treated as a **measured floor**, not a target. It
  was obtained with duplicate copies per build-version and lz4 clevel-1, so both deduplication and
  encoding have unclaimed headroom.
- Decompression throughput is a first-class metric, not an afterthought: an ML corpus is read
  repeatedly during training, so an encoding that shrinks the store but slows every epoch can be a
  net loss. Both numbers are measured and reported (SC-007a).
- Long-term preservation is an explicit goal. The store must be interpretable years from now by
  someone with Zarr tooling and no access to this codebase, which is why full decoding and
  self-description are requirements rather than conveniences.
- This datastore is intended to serve consumers beyond this project. Portability and
  self-description are specified accordingly.
- **The datastore's primary consumer is ML/AI, not the viewer.** Where the two pull in different
  directions, the training/inference surface wins; viewer loading is a convenience that falls out of a
  correct store (which is why it is P3).
- Derivation dedupe is expected to dominate storage dedupe in value. Both are measured (SC-007,
  SC-010a) rather than assumed, and if recompute avoidance turns out to be small, that should be
  reported plainly.
- Content-addressed storage is the mechanism; "diffs" are a *view* over manifests, not a stored
  format. This was chosen over a delta format deliberately — it is both simpler and safer, and the
  reasoning is in Context so it can be revisited if measurement contradicts it.
- Deduplication is expected to be substantial for same-era 0.x builds, but that is a hypothesis this
  spec measures (SC-007), not an assumption it relies on. If the ratio is poor, the packaging benefit
  still stands and the spec should say so plainly.
- The datastore is an additional data source; the live MPQ path remains supported.
- Builds are expected to be added and removed routinely, not written once. Cheap load/unload is a
  design goal confirmed in session, which is why reference-counted reclamation and read-in-place are
  requirements rather than optimizations.
- The canonical patch-priority implementation is expected to be the viewer's, which is the more
  complete of the two — but it must be verified against documented client behavior, not adopted on the
  grounds of being better than its sibling.
- Deletion semantics are taken as MPQ's zero-length-entry convention, already relied on by both
  existing implementations.
- Builds are identified by the repo's existing build-identity conventions.
- Existing harvested tensor stores are unaffected; this is a client-content datastore, a different
  thing that shares a substrate.
- Not every historical build will be available. The datastore holds what exists and reports what it
  does not.
