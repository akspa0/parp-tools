# Feature Specification: Adaptive Per-Type Storage Encoding

**Feature Branch**: `182-adaptive-encoding`
**Created**: 2026-08-19
**Status**: Draft
**Epic**: [Client Datastore](../epic-client-datastore/epic.md) — **read the epic first**, especially
the measured 4:1 baseline.
**Depends on**: [180](../180-multi-build-datastore/spec.md). Best measured after
[181](../181-incremental-processing/spec.md), since dedupe removes the duplicate copies the baseline
was measured with.

## Scope

Each kind of array is stored with the codec and level that suit **it**, chosen from measured
compression ratio **and** measured decompression throughput — not one global default applied to
everything.

The prototype measured ~160 GB → ~40 GB (~4:1) at Blosc **lz4 clevel-1, no shuffle specified** — close
to the weakest setting available, and *with* duplicate copies per build. **4:1 is the floor.**

## User Story - Storage encoding is tuned per data type, by measurement (Priority: P2)

Encoding is selected per array type from evidence, and one mechanism governs every writer.

**Independent Test**: Encode the same corpus under the current uniform default and under per-type
selection; report size and decompression throughput for both, per array type.

**Acceptance Scenarios**:

1. **Given** the array types in the corpus, **When** encoding is chosen, **Then** each type's codec and
   level come from a measurement **of that type**, not from a global default.
2. **Given** a candidate encoding, **When** evaluated, **Then** both compression ratio **and**
   decompression throughput are measured — because the primary consumer reads this data repeatedly
   during training.
3. **Given** a type where higher compression costs more time than it saves in size, **When** encoding
   is selected, **Then** the faster setting wins and the reason is recorded.
4. **Given** parser or decoder knowledge suggests an array is highly redundant, **When** that
   suggestion is used, **Then** it is treated as a **hypothesis that selects candidates to measure** —
   never as the decision itself.
5. **Given** encodings are chosen, **When** any array is read, **Then** it decodes through **standard
   Zarr tooling with no custom codec**.
6. **Given** the repo currently carries two codec defaults, **When** this spec completes, **Then** one
   selection mechanism governs both paths.

### Edge Cases

- An array type whose optimal encoding differs between builds.
- A type too rare to measure reliably — falls back to a documented default, recorded as such.
- A codec available at write time but not in a consumer's Zarr version.
- Chunk shape interacting with codec choice.

## Requirements

### Functional Requirements

- **FR-001**: Encoding is selected per array type from measured compression ratio **and** measured
  decompression throughput. A single global codec/level for all arrays does not satisfy this.
- **FR-002**: Parser- or decoder-derived complexity estimates **may** narrow the candidates to measure.
  They **must not** select an encoding on their own.
- **FR-003**: One encoding-selection mechanism governs all writers; the repo's two current codec
  defaults converge.
- **FR-004**: Every array remains readable by **standard Zarr tooling with no code from this
  repository**, and with no custom codec.
- **FR-005**: Arrays are self-describing — name, dtype, shape, units/semantics, and provenance
  discoverable from the store itself.
- **FR-006**: The chosen encoding and the measurements that justified it are recorded per array type.

## Success Criteria

- **SC-001**: Corpus compression **beats the prototype's measured ~4:1 (≈160 GB → ≈40 GB) baseline.**
  That baseline had duplicate per-build copies and near-minimum compression, so failing to beat it
  indicates a defect, not a limit.
- **SC-002**: A published table of array type → chosen codec/level → measured ratio → measured
  decompression throughput → reason.
- **SC-003**: Encoding-selection mechanisms in the repo: **2 → 1**.
- **SC-004**: Every array opens and decodes with standard Zarr tooling in a process with none of this
  repository's code available.

## Out of Scope

- Changing what is stored, or how it is decoded from client files.
- Lossy compression of any kind.

## Assumptions

- **Decompression throughput is a first-class metric, not an afterthought.** An ML corpus is read
  repeatedly during training, so an encoding that shrinks the store but slows every epoch can be a net
  loss. Both numbers are measured and reported.
- The existing defaults (`zarr_io.py` zstd clevel-5 + bitshuffle; `v25/dataset.py` lz4 clevel-1) are
  starting candidates, not answers.
