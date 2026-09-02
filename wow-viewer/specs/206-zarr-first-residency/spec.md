# Feature Specification: Zarr-First Asset Residency

**Feature Branch**: `206-zarr-first-residency`

**Created**: 2026-09-01

**Status**: Draft

**Epic**: [Client Datastore](../epic-client-datastore/epic.md) — **read the epic first.** This spec is a
member of that epic, not a new one. It does not build a second store.

**Input**: User description: "latest results of flying around and looking at new and old areas of the
map, it's mainly the storage bottleneck, or how we read data out of the mpq's. If we stuck all the
data we render into a zarr dataset, instead of in a cache folder, we'd be better off every time. Can
we work that in, as the main method the data ends up in a zarr dataset from the beginning, when we
work with the data, in the first place? wouldn't that make the whole program a whole lot more
streamlined, and permit us the idea of providing a universal interchange format for this sort of
archival data?"

---

## Read this first: what the measurement says, and what it does not

The operator's symptom is real and is on screen: **every recent hitch is attributed to
`DeferredAssetLoads`** — 59.6 / 131.8 / 136.8 / 160.3 ms, at 3 FPS, in The Jade Forest (MoP Beta
5.0.1.15464). That is the same population spec 204 measured (12 of 13 hitches, 26.4–68.1 ms, median
frame 75.26 ms, 2047/2048 frames over 33.3 ms).

**But the cost is not the read.** Spec 204's research established that `MpqDataSource` already runs 2
prefetch workers and that root bytes are usually warm; what runs inside the frame is **parse,
adaptation, BLP decode and GL upload**, because `WorldAssetManager` contains no threading at all.

So the storage change that helps is **not** "read the same bytes from a faster container". It is
**"stop needing to decode them at all"** — store what the renderer actually consumes, already
decoded, and the expensive stage is deleted rather than relocated. That is the premise this spec is
built on, and it is why the store's contents (US3) matter more than the store's format (US2).

**Boundary with spec 204 — they are complementary, not alternatives.** 204 moves work off the render
thread. 206 removes work from the pipeline. If both land, the only per-frame asset cost left is the
GPU upload itself, which is 204's to schedule. **Neither substitutes for the other**: a render-ready
store still has to upload off-thread, and off-thread decode still decodes.

## The storage situation today, measured

There are **three** separate places render data lives, and none of them holds it in the form the
renderer consumes.

| # | Where | What it holds | Measured |
|---|---|---|---|
| 1 | MPQ archives, via `MpqDataSource` | original client bytes | prefetches on 2 workers; usually warm |
| 2 | `WorldAssetManager` in-memory LRUs (`_fileDataCache`, `_mdxModels`, `_wmoModels`, …) | raw bytes + built renderers | lost on exit; every miss pays full decode in-frame |
| 3 | `output/cache/` on disk | **byte-for-byte copies of client files** | **~1.75 GB**, incl. `Kalimdor.wdt` **1.03 GB** and `Azeroth.wdt` **752 MB** |

Layer 3 is the "cache folder" in the request, and it is worth naming precisely, because it is not a
performance cache at all. `ViewerApp.cs:12398` writes it under the comment:

```csharp
// Write to cache folder for parsers that expect file paths
Directory.CreateDirectory(CacheDir);
var cachePath = Path.Combine(CacheDir, _loadedFileName!);
File.WriteAllBytes(cachePath, data);
```

It is a **path shim**: a full extraction of client bytes to disk performed only so that parsers
taking a `string path` instead of a `byte[]` can run. It buys no frame time, it is written on every
load, and it is exactly the extraction that the epic's spec 183 FR-004 already forbids for datastore
loads. Retiring it is a self-contained win that does not depend on anything else in this spec.

## ARCHITECTURE CONSTRAINT (operator directive, 2026-09-02) — read before touching this spec

**Python owns the datastore. C# must not implement Zarr or TensorStore reading or writing.**

The storage engine is **Python using TensorStore's Zarr driver** (today the Python side uses bare
`zarr` + `numcodecs`; moving it to TensorStore is part of this spec's work). The operator has been
down the C#-implements-Zarr road before and it **cost two months** to undo.

**This spec's original framing of US2 was wrong.** It called "there is no C# Zarr array reader" a
blocking gap. It is not a gap — it is the intended architecture. A C# reader was written on
2026-09-02 and deleted the same day. The absence is deliberate.

**The correct handoff already exists and is proven**: C# `harvest-stream` emits raw tile blobs in the
`ARRY/ENDS` wire format via `RawArraySerializer`; Python's `harvester.raw_reader.read_tile_blob`
ingests them and `zarr_io.py` / `zarr_store.py` build the store. **C# composes and emits. Python
stores. Conversion to any output format happens from the store.**

So US2 is re-scoped: it is not "give C# a Zarr reader", it is **"make the Python store the single
storage engine, on TensorStore, and route everything the viewer needs through the existing ARRY
handoff"**.

**Pre-existing violations, flagged not extended**: `RosettaDatastoreWriter` writes Zarr v3 from C#,
and `ZarrTileDatasetLoader` validates a store from C#. Both predate this constraint. Do not build on
them; do not silently delete them either.

## The capability that is actually missing

Zarr is already the project's chosen store, and there is more of it in place than a cold reader
expects — but the **C# array read path does not exist**:

- `RosettaDatastoreWriter` (spec 190, implemented, 69/69 tests green) writes a **real Zarr v3 store**
  — `zarr.json` group/array metadata, regular chunk grids, Parquet side tables — but its arrays
  declare `codecs: [bytes]` only: **uncompressed**, so no codec path is exercised.
- `ZarrTileDatasetLoader` (viewer) **validates and discovers** a store, and its `LoadTile` throws
  `NotImplementedException`: *"The remaining work is the Blosc+Zstd+bitshuffle chunk decoder."*
- `StoreIndexReader` states it outright: *"this codebase has no C# Zarr array reader"* — it reads the
  Parquet index and re-derives every tensor from the client rather than reading the store's arrays.
- The Python harvester (`zarr_io.py`, `dataset.py`) reads and writes the compressed arrays fine.

So today the store is **write-mostly from C# and read-only from Python**. Every claim in this spec
about the viewer reading its data from Zarr is blocked on one missing capability: a C# Zarr array
reader that handles the same codecs the Python side writes. That is US2, and it gates US3 and US4.

## What this spec changes about the epic

The epic states: *"The primary consumer is ML/AI, not the viewer. This is a training corpus that a
viewer can also read — not the reverse. Where the two pull in different directions, the training
surface wins. That is why viewer loading (183) is last."*

The operator's request **inverts that ordering** for one specific reason that did not exist when the
epic was written: the viewer now has a measured frame-time problem whose fix is the same store. This
spec therefore adds the **render-ready layer** that 183 does not describe, and asks for it early.

**It does not overturn the epic's design decisions.** Content-addressing, no-MPQ-output, the store as
the only on-disk artifact, incremental output equalling from-scratch output, and full decode as a
preservation requirement all stand and are inherited here.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The viewer stops extracting client files to disk (Priority: P1)

Opening any file through the data source stops writing a byte-for-byte copy into `output/cache/`.
Parsers that need a path are given one from the store or are converted to take bytes.

**Why this priority**: It is the literal "instead of a cache folder" in the request, it is the
largest single on-disk artifact the program creates (~1.75 GB), it is already prohibited for
datastore loads by 183 FR-004, and it is **independent of every other story here** — it needs no
Zarr reader and no store. It can ship on its own.

**Independent Test**: Load a map and several models, then observe `output/cache/`: no client file
copies appear, and rendering is unchanged.

**Acceptance Scenarios**:

1. **Given** a client is loaded, **When** the user opens a WDT, model, or WMO, **Then** no copy of
   its bytes is written to disk and it renders identically to before.
2. **Given** a parser that requires a filesystem path, **When** it is invoked, **Then** it receives
   its input without a client-bytes copy being persisted outside the store.
3. **Given** an existing populated `output/cache/`, **When** the viewer starts, **Then** it does not
   depend on that directory's contents and reports the reclaimable size once.

---

### User Story 2 - The viewer can read Zarr arrays at all (Priority: P1)

A store written by the harvester opens in the viewer and its arrays decode to the same values Python
reads from them, including the compressed codecs the harvester actually writes.

**Why this priority**: It is the missing capability. `LoadTile` throws today, so every statement
about "the viewer reads its data from Zarr" is currently false. Nothing downstream in this spec can
be tested until this exists.

**Independent Test**: Read a known array from a harvester-written store in both Python and the
viewer; values, shape and dtype must match exactly.

**Acceptance Scenarios**:

1. **Given** a store written by the Python harvester, **When** the viewer reads an array, **Then**
   the values, shape and data type match what Python reads from the same array, bit for bit.
2. **Given** an array using a codec chain the reader does not support, **When** it is read, **Then**
   the unsupported codec is named in the failure — never returned as zeros or silently skipped.
3. **Given** a store written by the C# writer, **When** Python opens it with standard Zarr tooling,
   **Then** it opens with no code from this repository.
4. **Given** a partially written or truncated chunk, **When** it is read, **Then** the store reports
   which chunk is bad rather than yielding partial data.

---

### User Story 3 - Render data lives in the store already decoded (Priority: P2)

**Every** content type the renderer consumes — terrain tiles, model geometry, WMO geometry, liquids
and textures — resides in the store in the form it is consumed in, so streaming anything is a read
plus an upload, with no parse, no adaptation and no image decode on the way.

Coverage is complete by decision (2026-09-01): partial residency was rejected because a store that
covers only some content types leaves the frame cost dominated by whatever it does not cover, and the
before/after measurement then reads as "no improvement" for reasons that have nothing to do with the
store. Ship it per content type, but the story is not done until all of them are resident.

**Why this priority**: This is the story that addresses the measured hitch. It is P2 only because it
is untestable until US2 exists, not because it matters less — it is the reason the operator asked.

**Independent Test**: Fly the same recorded path over a region resident in the store and over one
that is not, and compare hitch attribution and frame-time distribution between the two.

**Acceptance Scenarios**:

1. **Given** a region whose render data is resident in the store, **When** the user flies into it,
   **Then** no container parsing or image decoding occurs on the render thread for that content.
2. **Given** the same region loaded from the client instead, **When** both are rendered, **Then** the
   rendered result is the same.
3. **Given** content is resident, **When** an asset is streamed, **Then** the work remaining on the
   render thread is the GPU upload only, and that remainder is attributed as such in the frame panel.
4. **Given** content the store does not have, **When** it is requested, **Then** the viewer falls
   back to the client path and reports the miss specifically — never renders it as absent.

---

### User Story 4 - Data lands in the store on first contact (Priority: P2)

When the program works with client data for the first time, that work produces store residency as
its normal output — not as a separate export step the user has to know to run.

**Why this priority**: This is the "from the beginning, in the first place" in the request. Without
it the store is another thing to remember to build, and the viewer's default path stays the client.
It is P2 because US3 defines what is worth storing; ingesting the wrong form early is wasted work.

**Independent Test**: Point the program at a client it has never seen, use it normally, and confirm
the store gains the corresponding content with correct provenance and no separate command.

**Acceptance Scenarios**:

1. **Given** a client not previously seen, **When** the user loads a map from it, **Then** the
   content that was processed becomes resident in the store with its build identity recorded.
2. **Given** content already resident from an earlier session or another build, **When** it is
   encountered again, **Then** the work is not redone (epic: derivation dedupe).
3. **Given** ingestion is in progress, **When** the user interacts with the viewer, **Then**
   ingestion does not introduce frame hitches — a store that costs frames to fill has moved the
   problem rather than solved it.
4. **Given** ingestion is interrupted, **When** the program restarts, **Then** the store is in a
   consistent state and the incomplete work is redone rather than trusted.

---

### User Story 5 - The store is a universal interchange format (Priority: P3)

A store hands to someone else — with no code from this project — a complete, self-describing,
readable archive of what was preserved, including the render-ready arrays.

**Why this priority**: It is the durability argument and the reason to prefer a standard array format
over a bespoke one, but it is a property to *hold* while building US1–US4 rather than a separate
build. It is verified last because it is only meaningful once there is something in the store.

**Independent Test**: Open a store in third-party tooling on a machine with none of this project's
code and read both a preservation array and a render-ready array without documentation from us.

**Acceptance Scenarios**:

1. **Given** a store, **When** it is opened with standard Zarr tooling, **Then** every array opens and
   its meaning is discoverable from the store's own metadata.
2. **Given** a render-ready array, **When** a third party reads it, **Then** its layout, units and
   coordinate conventions are recorded in the store, not only in this repository.
3. **Given** a store written by a newer version of the tooling, **When** an older reader opens it,
   **Then** it either reads it correctly or names the version difference — it never misreads it.

---

### Edge Cases

- A store on slow or removable storage — residency must not make the viewer *slower* than the client
  path; if it does, that is a reportable finding, not a silent regression.
- A store and a live client both configured, disagreeing about the same file.
- Render-ready arrays that were built by a different version of the decoders they came from — stale
  derived data must be detectable, per the epic's stale-derivation hazard.
- Content the store holds in preservation form but not render-ready form, and vice versa.
- Disk exhaustion during ingestion.
- Two viewer instances writing to the same store.
- A client whose data cannot be fully decoded (spec 199 MCAL, spec 205 MH2O) — what residency means
  for content that is known to decode incorrectly today.

## Requirements *(mandatory)*

### Functional Requirements

**Retiring the extraction cache (US1)**

- **FR-001**: The viewer MUST NOT write byte-for-byte copies of client files to disk in order to
  satisfy parsers that require a filesystem path.
- **FR-002**: Any consumer that requires a path MUST be served without persisting client bytes
  outside the store.
- **FR-003**: The viewer MUST report, once, the size of any pre-existing extraction cache it no
  longer uses, so the user can reclaim it deliberately.

**Reading the store (US2)**

- **FR-004**: The project MUST be able to read Zarr arrays from its own store in the language the
  viewer is written in, including the compression codecs its own writers produce.
- **FR-005**: Reads MUST agree exactly with the reference implementation — same values, shape and
  data type.
- **FR-006**: An unsupported codec, a corrupt chunk, or a truncated array MUST be reported naming the
  array and chunk. Returning zeros, empty, or partial data for a failed read is prohibited.
- **FR-007**: Writers and readers in this project MUST agree on one codec chain; a store this project
  writes MUST be readable by this project and by standard tooling.

**Render-ready residency (US3)**

- **FR-008**: The store MUST hold render-ready forms for **every** content type the renderer consumes
  — terrain tiles, model geometry, WMO geometry, liquids and textures — such that serving any of them
  requires no container parsing and no image decoding.
- **FR-008a**: Textures MUST be resident in **two** forms: a portable decoded-pixel array, which is
  the interchange product, and a GPU-native block-compressed array derived from it, which is the
  upload product. The pixel array MUST be readable and complete on its own; the block array MUST NOT
  be required to interpret it.
- **FR-008b**: The two texture forms MUST be verifiably consistent — a stored block array that does
  not correspond to its pixel array MUST be detectable, and detected before it is served.
- **FR-009**: Content served from the store MUST render identically to the same content loaded from
  the original client.
- **FR-010**: When requested content is not resident, the viewer MUST fall back to the client path
  and report the miss specifically (183 FR-005: never silently rendered as absent).
- **FR-011**: The frame instrumentation MUST attribute asset cost by stage, so "resident" and
  "not resident" are separable numbers rather than a single aggregate. Per spec 201's rule, this must
  land **before** any before/after claim is made about hitch reduction.
- **FR-012**: Render-ready arrays MUST record the version of the decoder that produced them, so
  stale derived data is detectable rather than silently trusted (epic stale-derivation hazard).

**Ingestion (US4)**

- **FR-013**: Working with client data MUST be able to produce store residency as a normal
  consequence, without requiring a separate user-invoked export.
- **FR-014**: Ingestion MUST NOT introduce frame hitches into an interactive session.
- **FR-015**: Ingestion MUST be resumable and MUST leave the store consistent after interruption;
  incomplete work is redone, never trusted (epic: *"if the input set cannot be enumerated with
  certainty, redo the work"*).
- **FR-016**: Content already resident MUST NOT be reprocessed (epic: derivation dedupe).
- **FR-017**: The original client MUST remain a supported source. The store is an additional source
  (183 FR-008), and MUST NOT become a precondition for opening a client.
- **FR-017a**: The store is **derived, not authoritative**. It MUST be fully reconstructible from the
  original client plus the recorded decoder versions, and deleting it MUST cost only time. The client
  remains the source of truth, so a store built by a decoder later found wrong is discarded and
  rebuilt rather than repaired.
- **FR-017a**: The store is **derived, not authoritative**. It MUST be fully reconstructible from the
  original client plus the recorded decoder versions, and deleting it MUST cost time only — never
  content. The client remains the source of truth.
- **FR-017b**: Because the store is derived, a store built by a decoder later found to be wrong MUST
  be discardable and rebuildable per content type, without rebuilding the whole store. This is the
  concrete mitigation for specs 199 and 205 shipping residency built on known-wrong decoders.

**Interchange (US5)**

- **FR-018**: Every array in the store MUST open with standard tooling and no code from this
  repository (epic constraint, extended here to render-ready arrays).
- **FR-019**: Array meaning — layout, units, coordinate convention, provenance — MUST be recorded in
  the store's own metadata.
- **FR-020**: The store MUST carry a schema version, and a reader encountering a newer one MUST name
  the difference rather than misread the data.
- **FR-021**: No Blizzard container may be produced as output (Constitution VII; epic hard
  constraint 1). This spec adds no export-to-MPQ path.

### Key Entities

- **Store**: the single on-disk artifact. Holds many builds, content-addressed, with its reuse index
  inside it rather than beside it (epic hard constraint 2).
- **Preservation array**: the decoded-but-faithful representation of client data; the archival
  product, and the epic's existing subject.
- **Render-ready array**: the same content in the form the renderer consumes. Derived, regenerable,
  and versioned by the decoder that produced it. **New in this spec.**
- **Residency**: whether a given piece of content is present in the store in a given form. Query­able
  per content item, and the thing US3's measurement partitions on.
- **Provenance**: build identity, source archive and patch chain, and decoder version, carried with
  the content rather than in a sidecar.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After a session that loads at least 3 maps and 20 models, **zero bytes** of client file
  copies exist outside the store. Baseline today: ~1.75 GB in `output/cache/`, including a 1.03 GB
  single file.
- **SC-002**: A store array read by this project and by the reference implementation returns
  identical values, shape and dtype, verified on at least one array of every data type the store
  writes.
- **SC-003**: Every read failure in a deliberately corrupted store names the array and chunk. Zero
  failures return zeros, empty or partial data.
- **SC-003a**: Every content type the renderer consumes has a render-ready form in the store, and a
  resident region requires **zero** container parses and **zero** image decodes to render — counted,
  not estimated.
- **SC-003b**: For every resident texture, the block-compressed array is confirmed to derive from the
  pixel array, and an injected mismatch is detected rather than served. The pixel array alone is
  sufficient to read the texture.
- **SC-004**: On a recorded flight over resident content, **no hitch is attributed to container
  parsing or image decoding**, and the per-stage attribution required by FR-011 is in place before
  the comparison is made.
- **SC-005**: The same flight over resident and non-resident content renders identically —
  pixel-compared, not judged by eye.
- **SC-006**: Using a previously unseen client normally results in its content becoming resident,
  with build identity recorded, and **no user-invoked export step**.
- **SC-007**: A second encounter with already-resident content performs no reprocessing, measured as
  work done, not as elapsed time.
- **SC-008**: Interactive frame-time distribution during ingestion is statistically indistinguishable
  from the same session without ingestion.
- **SC-009**: A store opens in third-party tooling on a machine with none of this project's code, and
  both a preservation array and a render-ready array are read and interpreted from store metadata
  alone.
- **SC-010**: An older reader opening a newer store either reads it correctly or names the version
  difference. Zero silent misreads.
- **SC-011**: A store deleted and rebuilt from the same client and decoder versions produces the same
  content, and nothing is lost that the client did not already hold.
- **SC-011**: Every content type the renderer consumes has a render-ready form in the store, and the
  count of content types served from the client on a resident flight is **zero**.
- **SC-012**: For every resident texture, the block-compressed form and the decoded-pixel form
  correspond, checked on the whole resident set rather than sampled; a deliberately mismatched pair is
  detected before it is served.
- **SC-013**: Deleting the store and rebuilding it from the client reproduces it exactly, verified by
  comparing the rebuilt store to the original for at least one build.

## Out of Scope

- **Replacing the MPQ path.** The client stays a first-class source (FR-017).
- **Exporting to any Blizzard container** (FR-021).
- **Moving asset work off the render thread.** That is spec 204 and stays there; this spec removes
  work rather than rescheduling it.
- **Fixing decoders that are known to be wrong.** Specs 199 (MCAL) and 205 (MH2O) own those. Storing
  their output does not make it correct — and residency built on a wrong decoder is exactly the stale
  corpus hazard, which is why FR-012 exists.
- **Editing store contents.** The Editor writes loose output (183 out-of-scope, inherited).
- **Network or remote stores.**

## Dependencies

- **Epic 179–183** — this spec is a member. 179 (patch-chain resolver) determines what content is
  even correctly identified; anything ingested before it lands inherits its resolution errors.
- **Spec 204** — complementary and independent; neither blocks the other. Both touch the asset load
  path, so whichever lands second must re-measure rather than assume.
- **Spec 201** — its per-path attribution is the pattern FR-011 follows; a before/after claim without
  it is unreadable.

## Assumptions

- **"All the data we render" means all of it** — terrain, models, WMOs, liquids and textures
  (operator decision, 2026-09-01). Per-content-type sequencing is a planning decision; per-content-type
  *coverage* is not, and US3 is not complete while any renderer-consumed type is missing.
- **The store is derived and rebuildable, not an authoritative archive** (operator decision,
  2026-09-01). This is what keeps a decoder bug from becoming data loss and is why FR-012's decoder
  versioning is a requirement rather than a nicety.
- **Textures are stored in both forms** (operator decision, 2026-09-01): portable pixels for
  interchange, block-compressed for upload. This roughly doubles texture storage and buys a
  consistency obligation (FR-008b) in exchange for keeping SC-009 intact while still getting the
  memcpy-shaped upload.
- **The store's format is Zarr.** This is the operator's explicit requirement and the project's
  existing choice (specs 129, 165, 179–183, 190); it is treated as a domain constraint, not an
  implementation detail chosen here.
- **The reference implementation is the Python harvester**, because it is the side that currently
  reads compressed arrays correctly. SC-002 is stated against it.
- **A read-only store is a valid configuration** — FR-013's ingestion must be disableable, since a
  shared or archival store may not be writable.
- **The measured MoP frame data generalizes to other eras** in kind but not in magnitude. Success
  criteria are stated as comparisons within a flight, not as absolute frame times, so an era with a
  different baseline still yields a valid result.
