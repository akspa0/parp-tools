# Feature Specification: MCAL Alpha Map Decode Correctness

**Feature Branch**: `199-mcal-decode-correctness`

**Created**: 2026-09-01

**Status**: Draft

**Input**: User description: "we should figure it out because if we want to properly harvest this data for a dataset, we need to know how to properly read it, no matter what."

## Context

Terrain in Cataclysm/MoP-era maps renders with large blocky single-texture patches that snap to
chunk boundaries (observed on Mogu'shan Palace, MoP Beta 5.0.1.15464, 2026-09-01).

The immediate cause is located: `SynthesizeCataclysm400ResidualAlpha` in
`StandardTerrainAdapter` **fabricates** a 64x64 alpha map for any layer 1-3 that failed to
decode, filling it with `255 - sum(other layers)`. Where the other layers are near zero —
most of a chunk — the result is a fully opaque 64x64 block. It is gated on `useBigAlpha`, so
it fires on Cata/MoP maps and never on the 0.5.3 lane. `StitchCataclysm400ChunkEdges` then
blends those fabricated blocks across seams.

That fabrication is a symptom. The underlying problem is that **this project does not have a
single established answer for how to read MCAL**, and it shows: there are at least **four
independent decoders**.

| Decoder | Location | Notes |
|---|---|---|
| `Mcal.GetAlphaMapForLayer` / `…Relaxed` | `Core.IO/Lk/Mcal.cs` | picks a branch by *inferring spans* from the next layer's offset |
| `AdtMcalDecoder` | `Core.IO/Maps/AdtMcalDecoder.cs` | its own `ReadCompressedAlpha` |
| `DecodeLayerBySpan` + fallbacks + synthesis | `viewer/Terrain/StandardTerrainAdapter.cs` | three more branches, plus the fabrication |
| `AlphaMapService.ReadBigAlpha` | `viewer/Terrain/Vlm/AlphaMapService.cs` | "ported from Warcraft.NET" |

`VlmDatasetExporter` — the **dataset harvest path** — calls `GetAlphaMapForLayer(layer, false)`
with big-alpha hardcoded to `false`, while the renderer derives it from the WDT `MPHD` flags.
So the harvested corpus and the rendered scene disagree about the same bytes, on any map where
`MPHD` sets big alpha.

This violates Constitution II directly: *"Format readers/writers are never duplicated across
tools. One canonical owner per format surface."*

The dataset consequence is the reason this is a correctness spec rather than a rendering
bug: alpha is a harvested signal. A corpus built through a decoder that guesses, or that
disagrees with the renderer, is wrong in a way no amount of downstream modelling can recover.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - One decoder, and it reports what it did (Priority: P1)

Every consumer — renderer, harvester, inspector, converter — reads MCAL through one canonical
decoder that reports, per layer, which decode rule it applied and whether it succeeded.

**Why this priority**: Until there is one answer, "is our alpha correct?" is unanswerable —
different callers get different bytes from the same file. Consolidation is also what makes
the remaining questions measurable: with one decoder, a failure count is a real number.

**Independent Test**: Point the canonical decoder at a corpus spanning 0.5.3, LK, Cata and MoP
and read the per-layer report. Delivers a truthful picture of current decode behaviour with no
decode rule changed yet.

**Acceptance Scenarios**:

1. **Given** an MCNK with layered alpha, **When** any consumer decodes it, **Then** the bytes
   returned are identical regardless of which consumer asked.
2. **Given** a layer that cannot be decoded, **When** decoding runs, **Then** the layer is
   reported as failed and **no alpha map is produced for it** — never a substitute.
3. **Given** a decode run over a corpus, **When** it completes, **Then** per-era counts of
   each decode rule applied, and of failures, are available as data.

---

### User Story 2 - The decode rule is established by measurement (Priority: P1)

For each supported era, the rule that selects between compressed, 4-bit (2048) and big-alpha
(4096) is determined from the client and from real files, not inferred from span arithmetic
at read time.

**Why this priority**: Equal to US1 — consolidating four guesses into one guess is not
progress. This is the "figure it out, no matter what" the operator asked for.

**Independent Test**: Apply the established rule to a corpus and confirm the decoded layer
sizes account for the MCAL payload exactly, with no unexplained remainder.

**Acceptance Scenarios**:

1. **Given** the established rule for an era, **When** a tile's layers are decoded, **Then**
   the bytes consumed sum to the MCAL chunk size with no leftover and no overrun.
2. **Given** a file where the rule does not account for the payload, **When** decoding runs,
   **Then** it is reported as an unexplained file with its path, rather than absorbed by a
   heuristic.

---

### User Story 3 - The fabrication is gone (Priority: P2)

No code path invents alpha data. A layer with no decodable alpha has no alpha map.

**Why this priority**: Depends on US1/US2 — removing the fabrication before the decode is
trusted would replace wrong terrain with missing terrain and lose the diagnostic signal the
blocks currently (accidentally) provide. Sequenced after, it is a deletion.

**Independent Test**: Load Mogu'shan Palace and confirm the blocky patches are gone and that
the decode report shows zero synthesized layers.

**Acceptance Scenarios**:

1. **Given** a Cata/MoP map, **When** terrain loads, **Then** no alpha map originates from
   synthesis, and the blocky full-chunk patches are absent.
2. **Given** a layer that genuinely has no alpha, **When** terrain renders, **Then** it is
   treated as absent rather than as fully opaque.

---

### User Story 4 - Harvest and render agree (Priority: P2)

The dataset harvest reads alpha through the canonical decoder with the same era inputs the
renderer uses, so harvested alpha matches what is drawn.

**Why this priority**: The operator's stated reason for the spec. Sequenced after US1 because
it is a consequence of consolidation.

**Independent Test**: Harvest a tile and render the same tile; compare the alpha arrays.

**Acceptance Scenarios**:

1. **Given** a tile on a map whose `MPHD` sets big alpha, **When** it is both harvested and
   rendered, **Then** the alpha arrays are identical.
2. **Given** a previously harvested corpus, **When** the canonical decoder is applied,
   **Then** the tiles whose alpha changes are enumerated so the corpus can be rebuilt
   knowingly.

---

### Edge Cases

- A layer's alpha offset points outside the MCAL payload.
- The last layer's extent is unbounded — no following offset to infer from.
- `doNotFixAlphaMap` (MCNK flag `0x8000`) set and unset, for each decode rule.
- A chunk whose declared layer count disagrees with the number of alpha maps present.
- 0.5.3-era chunks, which have no alpha at all — must remain untouched and must not be
  reported as failures.
- An MCAL payload with trailing bytes after the last layer.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Exactly one component MUST own MCAL decoding. All other decode implementations
  are removed or become thin delegations to it.
- **FR-002**: The decoder MUST NOT synthesize, substitute, interpolate, or otherwise
  manufacture alpha data. A layer that cannot be decoded yields no alpha map.
- **FR-003**: The decoder MUST report, per layer, which decode rule was applied and whether
  it succeeded, and MUST report per-file when the decoded layers do not account for the MCAL
  payload exactly.
- **FR-004**: Rule selection MUST be driven by era profile and the flags the format defines
  (`MCLY` compression flag, WDT `MPHD` big-alpha flags, `doNotFixAlphaMap`), not by inferring
  extents from neighbouring offsets at read time.
- **FR-005**: Where a rule cannot be established from evidence for an era, the decoder MUST
  report those files as unexplained rather than apply a fallback that appears to succeed.
- **FR-006**: The renderer and the dataset harvest MUST obtain alpha from the same decoder
  with the same era inputs, producing identical bytes for the same input.
- **FR-007**: 0.5.3-era terrain behaviour MUST NOT change.
- **FR-008**: The decoder MUST be usable without a graphics context so it can run in tests
  and in offline harvest.
- **FR-009**: Decode reports MUST be aggregatable across a corpus, so per-era rule and
  failure counts can be produced for the whole client library.

### Key Entities

- **Alpha decode rule**: One of the format's defined encodings — compressed, 4-bit packed,
  or full-byte — plus whether the edge fixup applies. Selected from era and flags.
- **Layer decode outcome**: Per layer — the rule applied, bytes consumed, success or the
  reason for failure. Never carries a fabricated map.
- **Chunk decode outcome**: The layer outcomes plus whether they account for the MCAL payload
  exactly, and the unexplained remainder if not.
- **Corpus decode report**: Aggregated outcomes across files, grouped by era and build, with
  the offending file list for unexplained cases.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: One decoder implementation exists; a search for alternative MCAL decode routines
  across the repository returns only delegations to it.
- **SC-002**: For a named corpus spanning 0.5.3, LK, Cata and MoP, the proportion of chunks
  whose decoded layers account for the MCAL payload exactly is measured and reported per era.
- **SC-003**: The count of synthesized alpha maps produced anywhere in the codebase is zero.
- **SC-004**: The blocky full-chunk patches on Mogu'shan Palace are absent, confirmed by
  capture.
- **SC-005**: For a sample of tiles on a big-alpha map, harvested and rendered alpha arrays
  are byte-identical.
- **SC-006**: Every file the established rules cannot explain is enumerated by path, so the
  remaining unknown is a list rather than an impression.
- **SC-007**: A 0.5.3 reference scene renders pixel-identically to before the change.

## Assumptions

- The three encodings named in the format (compressed, 4-bit packed, full-byte) are the
  complete set for the 0.5.3-5.1 range. If measurement contradicts this, FR-005 makes the
  contradiction visible rather than absorbing it.
- The existing `AdtMcalDecodeProfile` and `AdtFormatProfile.BigAlphaFlagsMask` are the right
  place to express era rules; this spec establishes their content, not new machinery.
- Real-client visual confirmation (SC-004) and corpus-scale runs (SC-002, SC-006) are
  operator-executed per this project's execution boundary.
- Rebuilding any existing harvested corpus is out of scope; SC-005 and the US4 enumeration
  tell the operator what would need rebuilding, and the decision is theirs.
