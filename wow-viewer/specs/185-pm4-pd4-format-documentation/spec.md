# Feature Specification: PM4/PD4 format documentation and terminology restoration

**Feature Branch**: `185-pm4-pd4-format-documentation`

**Created**: 2026-08-23

**Status**: Draft

**Input**: User description: "We still don't know how the 'ck24' is really fully meant to be read, we
just made accurate assumptions about that data and how it's a packed set of parameters from the
wowdev wiki. For some reason, along the way, chatgpt or gemini really f'd me and corrupted the names
of things so they were no longer based on the wowdev wiki documentation, and ideally, we should write
some updates for that documentation without reinventing names that don't really describe the data
properly. write better documentation for pm4 and pd4 formats"

## Context

This repo's PM4 field names drifted away from their wowdev.wiki anchors. The drift is not cosmetic —
it has produced names that **assert semantics the data does not have**, and those names then shaped
how later work read the format. Two of the seven entries in the existing terminology catalog are
falsified by measurement taken on 2026-08-23, and one of them sent the connective-geometry search at
the wrong chunk for months.

### The naming drift is measurable, not a matter of taste

`Pm4TerminologyCatalog` already tracks raw offset → local alias → confidence. That instrument is
correct and should be the backbone of this work. Its **content** is what is wrong:

| raw field | current local alias | status after 2026-08-23 measurement |
|---|---|---|
| `MSUR._0x02` | `AttributeMask` ("bit meanings still open") | **Falsified.** It is a **count** — the length of a window into MSLK. A mask name invites bitwise reading of a scalar. |
| `MSUR._0x18` | `_0x18` ("indexes into MSCN, NOT MDOS") | **Falsified.** It indexes **MSLK**, not MSCN. Against MSCN the windows overrun 6,240 times and cover 93.52%; against MSLK, zero overruns and 100% coverage. |
| `MSUR._0x00` | `GroupKey` (confidence low) | Unverified assertion of a grouping semantic. |
| `MSLK._0x04` | `GroupObjectId` (confidence low) | Already flagged as "not a confirmed identity field" — the name says otherwise. |
| `MSUR._0x1C` | `PackedParams` → derived `CK24`, `Ck24Type`, `Ck24ObjectId`, `Ck24HighByte`, `Ck24LowByte` | **Falsified.** It is an IEEE-754 **float Z coordinate**, not a packed key: r=0.995 with each object's bbox floor Z against controls of −0.125 and 0.011, high byte confined to float exponent bands `0x3D–0x43` **and their sign-set counterparts** `0xBD–0xC3`. Negative values exist, which no object id can have. Every CK24 slice is a slice of a float, and `Ck24Type` is its exponent band. |

### The same struct is named two different ways in this codebase

`Pm4MsurEntry` and `Pd4MsurEntry` describe the **identical 32-byte record**, and five of nine fields
carry divergent names:

| offset | `Pm4MsurEntry` | `Pd4MsurEntry` |
|---|---|---|
| `_0x00` | `GroupKey` | `Flags` |
| `_0x01` | `IndexCount` | `IndexCount` |
| `_0x02` | `AttributeMask` | `Unknown02` |
| `_0x03` | `Padding` | `Padding` |
| `_0x04..0x0F` | `Normal` | `Normal` |
| `_0x10` | `Height` | `Height` |
| `_0x14` | `MsviFirstIndex` | `FirstIndex` |
| `_0x18` | `_0x18` | `RefIndex` |
| `_0x1C` | `PackedParams` | `Zero` |

Nobody reading one model can carry knowledge to the other. Any cross-format finding has to be
re-derived, which is exactly how the MSCN misreading survived.

### MPRR is not decoded, and it has never been given a proper writeup

Measured 2026-08-23 with `pm4 mprr` over the 616-file corpus (502 files carry MPRR):

- 13,978,231 non-sentinel entries in **3,171,410** sentinel-delimited runs.
- **99.9843% of runs have length ≡ 3 (mod 4)** — measured across all **246** distinct run lengths,
  max 5019. Residues 0 and 2 are **empty**; the only 497 exceptions are residue 1. So a run plus its
  terminating sentinel always occupies a whole number of 4-entry (16-byte) blocks, and 75.5% are the
  minimal single block (3 data entries + 1 sentinel).
- The run count matches **no** chunk's entry count (best: MSUR and MSCN at 4/502 = 0.8%), so MPRR is
  not a simple per-entry list for any known chunk.
- Value bound tests are weak and non-discriminating: best non-self domain is MSVI at 67.6% for
  `Value1` and 79.0% for `Value2`. These are **bound tests only** — a value in range does not prove
  ownership.

The 4n+3 blocking is a hard new structural constraint that any MPRR hypothesis must satisfy, and it
is not recorded anywhere. The user's standing read — "some sort of range record, or an index into
range records" — is consistent with fixed-size records and has never been tested against this
constraint.

### MVER is a format version, not a build and not a size

Measured 2026-08-23. The PM4 `MVER` payload is a constant `10 30 00 00` (12304 / 0x3010) across
corpus files spanning a 20x size range, every one with an identical 32-byte MSHD - so it is neither a
size nor any content-derived quantity. PD4 stores `30 00 00 00` (48 / 0x0030). Under a consistent
byte0 reading that is PM4 v16 and PD4 v48, with PM4's `0x30` high byte **undecoded**.

The viewer had been mapping the raw word to the client build string `4.0.1.12304` and printing
"hints build ... from PM4 files" in the status bar. 12304 resembling that real build is a coincidence
of digits. The mapping survives as an **era heuristic** for choosing a base client; the message no
longer claims a build was read from the file. A proposed "PM4 3.30 / PD4 3.31" reading was tested
against the raw bytes and does not reproduce under any byte-swap or uint16-pair interpretation.

### PD4 has no documentation at all in this repo

PD4 is the **per-model** form of the same format: object-local coordinates, no placement, no tile
structures. The garrison reference file carries only `MVER, MCRC, MSPV, MSPI, MSCN, MSLK, MSVI,
MSVT, MSUR`. Its `_0x1C` is zero in 3,447 of 3,447 surfaces — because the file *is* one object and
has no per-object identity to carry, which is direct evidence about what `_0x1C` is for in PM4.
That comparison is the single cleanest lever on the CK24 question and it is undocumented.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Trust a field name again (Priority: P1)

A researcher reads a field name in this codebase and can tell, without cross-checking, whether it is
a wowdev-anchored name, a measured local name, or a guess — and what evidence backs it.

**Why this priority**: Every other story writes documentation. If the names are still lying, the
documentation propagates the lie at scale. This story is what stops that.

**Independent Test**: Take the two falsified entries and confirm the catalog now states the measured
reading with its evidence; take any remaining speculative alias and confirm it is either renamed to
its raw offset or carries an explicit unverified marker.

**Acceptance Scenarios**:

1. **Given** a field whose semantic is measured, **When** its catalog entry is read, **Then** it
   names the measurement and the figure that supports it.
2. **Given** a field whose semantic is assumed, **When** its catalog entry is read, **Then** it is
   marked unverified and its local alias does not assert a semantic (no "Mask", "Key", "Id" on
   unmeasured data).
3. **Given** the field previously called a mask that is in fact a count, **When** any report renders
   it, **Then** it reads as a count everywhere, with no surviving mask-named accessor.
4. **Given** a reader of either the per-tile or per-model model, **When** they look up the same
   record offset, **Then** both models give it the same name.

---

### User Story 2 - Read one accurate format document per file type (Priority: P1)

A researcher — inside this project or on the wiki — reads a document per format that states, per
chunk and per field, the offset, width, what it is known to hold, and how strongly that is known.

**Why this priority**: This is the deliverable the user asked for, and it is what makes the
measurements of the last months survive the next context reset.

**Independent Test**: Every claim in the document traces to a named command and a figure; a reader
can re-run that command and get the stated number.

**Acceptance Scenarios**:

1. **Given** a documented relationship, **When** a reader follows its cited command, **Then** the
   reported figure reproduces.
2. **Given** a field with no settled meaning, **When** the document covers it, **Then** it appears
   with its raw offset, its measured population, and what would settle it — not omitted, and not
   given an invented name.
3. **Given** the per-model format, **When** its document is read, **Then** it states which chunks are
   absent relative to the per-tile format and what that absence implies.
4. **Given** a wiki editor, **When** they take the document, **Then** it uses wowdev's existing field
   names wherever wowdev has one, and proposes a new name only with its evidence attached.

---

### User Story 3 - Constrain MPRR with structure before guessing at values (Priority: P2)

A researcher gets MPRR's structural grammar written down — block quantisation, run population,
sentinel role — so that candidate readings can be rejected on structure without a value-domain hunt.

**Why this priority**: The value-domain sweep has already run and discriminates nothing (best
non-self bound fit 79.0%, and bound fits do not prove ownership). Structure is where the remaining
signal is, and the 4n+3 constraint is unexploited.

**Independent Test**: State the grammar, then test each candidate reading against it and record which
survive and which the constraint eliminates.

**Acceptance Scenarios**:

1. **Given** the corpus, **When** the run grammar is measured, **Then** the block quantisation is
   reported over all distinct run lengths, not a truncated histogram.
2. **Given** a candidate reading of MPRR, **When** it is tested, **Then** it is recorded as surviving
   or eliminated with the specific structural fact that decided it.
3. **Given** the 497 runs that violate the quantisation, **When** they are examined, **Then** they are
   characterised rather than dismissed as noise.
4. **Given** no candidate survives, **When** results are written up, **Then** the negative result is
   recorded as progress with the search space it closes.

---

### User Story 4 - Settle how the packed surface parameter is meant to be read (Priority: P2)

A researcher gets a defensible account of the packed value at the end of each surface record — what
its parts are, which are identity and which are not — or an explicit statement of what remains
assumed.

**Why this priority**: This value underpins object grouping across the whole PM4 stack, and the
current reading is self-described as a local derived identity rather than a format-native one. The
per-model format is a natural control: it holds the same record with this field zeroed throughout.

**Independent Test**: State the reading, then show it holds on the per-tile corpus and explain the
per-model file's all-zero population under the same reading.

**Acceptance Scenarios**:

1. **Given** the per-model reference file, **When** the field's population is measured, **Then** the
   all-zero result is explained by the proposed reading rather than treated as an anomaly.
2. **Given** the proposed decomposition, **When** it is applied corpus-wide, **Then** each part's
   population is reported separately, and any part with no evidence is named as unexplained.
3. **Given** the existing derived identity, **When** the reading changes, **Then** every consumer of
   that identity is enumerated and its behaviour under the new reading is stated.

---

### Edge Cases

- A chunk present in the per-model format but absent from the per-tile one, or the reverse.
- A field that is genuinely a different thing in the two formats despite sharing an offset — the
  end-of-record value is the live candidate.
- Runs that violate the block quantisation (497 measured) and the possibility they mark a distinct
  record kind.
- Version drift: corpus files report version 12304 while the reference per-model file reports 48; a
  documented field may hold only for one of those.
- A wowdev name that is itself wrong; the project must be able to say so with evidence rather than
  silently substituting a private name.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every field name in the codebase for these formats MUST resolve to a catalog entry
  giving its raw offset, its evidence, and its confidence.
- **FR-002**: A local name MUST NOT assert a semantic that has not been measured; unmeasured fields
  MUST carry their raw offset as the name.
- **FR-003**: The two falsified entries MUST be corrected to their measured readings, and the
  correction MUST cite the figure that falsified the old one.
- **FR-004**: The per-tile and per-model models MUST give identical names to identical record
  offsets, and any deliberate divergence MUST be documented as a real format difference with
  evidence.
- **FR-005**: Documentation MUST exist per format, covering every chunk the readers recognise,
  including chunks with no settled meaning.
- **FR-006**: Every documented claim MUST cite the command and figure that produced it.
- **FR-007**: Documentation MUST carry confidence per claim and MUST NOT flatten measured facts and
  working assumptions into one voice.
- **FR-008**: Documentation MUST be usable as a wiki contribution without this repo's private
  vocabulary, preferring existing wowdev names where they exist.
- **FR-009**: The MPRR structural grammar MUST be measured over all distinct run lengths and
  reported with its exceptions characterised.
- **FR-010**: Each MPRR candidate reading MUST be recorded as surviving or eliminated, with the fact
  that decided it; eliminations MUST be retained so the search is not repeated.
- **FR-011**: The packed surface parameter's reading MUST account for the per-model format's all-zero
  population.
- **FR-012**: Renaming MUST NOT change decode behaviour; the corpus figures reported before and after
  MUST be identical.
- **FR-013**: No second reader or parser for these formats may be introduced by this work.
- **FR-014**: Where this work contradicts wowdev, the contradiction MUST be stated explicitly with
  its evidence rather than resolved silently in either direction.

### Key Entities

- **Raw field**: a chunk record offset and width — the stable anchor a name attaches to.
- **Terminology entry**: raw field, local name, confidence, evidence, and wowdev name if one exists.
- **Format document**: the per-format writeup, chunk by chunk, with claims and their citations.
- **Structural grammar**: the shape constraints a stream obeys independent of value meaning.
- **Candidate reading**: a proposed interpretation, with its status and the fact that decided it.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of field names used in these formats' models and reports resolve to a catalog
  entry with evidence and confidence.
- **SC-002**: Zero fields carry a semantic-asserting name without a measurement backing it.
- **SC-003**: The per-tile and per-model models agree on 9 of 9 shared surface-record field names, up
  from 4 of 9.
- **SC-004**: Both format documents cover 100% of chunks the readers recognise.
- **SC-005**: Every claim in both documents cites a reproducible command, and a spot check of claims
  reproduces the stated figures exactly.
- **SC-006**: MPRR's block quantisation is published over all 246 distinct run lengths with the 497
  exceptions characterised rather than dropped.
- **SC-007**: Every MPRR candidate reading is recorded as surviving or eliminated; the number of
  eliminated candidates is reported, and a result of zero survivors is an acceptable outcome.
- **SC-008**: The packed surface parameter has either a reading that explains both the per-tile
  populations and the per-model all-zero population, or an explicit statement of what is still
  assumed and what would settle it.
- **SC-009**: Corpus figures produced by the existing analyzers are byte-identical before and after
  the rename.

## Assumptions

- wowdev.wiki is the naming authority where it has a name; this project's job is to supply evidence
  and propose names only where the wiki has none.
- The existing terminology catalog is the right mechanism and is extended, not replaced.
- Publishing to the wiki itself is a user action; this spec produces the document, not the edit.
- Version drift between the corpus files and the per-model reference file is real and any field claim
  is scoped to the versions it was measured on.
- The MPRR decode may not close. The spec requires the search space to shrink and be recorded, not
  that an answer be found.
- Long corpus-wide sweeps beyond read-only inspection are user-run.

## Out of Scope

- Generating these files — that is spec 184, which consumes this spec's vocabulary.
- Object-to-placed-asset identity matching.
- Any change to coordinate handling or placement math.
- Editing wowdev.wiki.
- Renaming anything outside these two formats.
