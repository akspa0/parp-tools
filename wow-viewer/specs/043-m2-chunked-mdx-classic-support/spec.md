# Feature Specification: 043 — 1.x Classic MDX (Chunked) Model Support

**Feature Branch**: `043-m2-chunked-mdx-classic-support`
**Created**: 2026-06-02
**Status**: In Progress (chunked lane only; 1.12.1 deferred to spec 048)
**Status (2026-06-05)**: 1.12.1 is **not** a chunked-MDX client. Native-client Ghidra evidence (`docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md`) shows 1.12.1's `.mdx` files use the same `MD20` magic as 3.0.1+ but with a different stride set, version `0x100`/`0x101`, and a different view/bone/light/camera/ribbon/particle/sequence layout. The 1.12.1 MD20 reader ships in spec `048-m2-1121-era-aware-md20-reader`. This spec now covers only the **0.5.3 / 0.7.0 / 0.8.0 chunked-MDX** lane. 2.x pre-2.0.0 chunked MDX, if it exists, is also deferred to a follow-up spec.

**Input**: User direction — "The Viewer is our main focus, we need to ensure that we worry more about MDX support from older clients than the Zarr dataset stuff right now." Plus the user's earlier note that 1.12.1 (`4,775,986`-byte `WoW.exe`) is staged. The 0.5.3 / 0.7.0 / 0.8.0 chunked-MDX era uses a chunked MDX format with `MDLX` magic and FourCC sub-chunks (VERS, MODL, SEQS, GLBS, MTLS, TEXS, GEOS, GEOA, BONE, HELP, PIVT, ATCH, LITE, PREM, PRE2, RIBB, EVTS, CAMS, CLID, HTST, TXAN) — completely distinct from the `MD20` magic and external-`.skin` model used in 3.0.1+ (spec 037 scope) and the 1.12.1 MD20 layout (spec 048 scope). The current wow-viewer `M2ModelReader` and `M2SkinReader` (3.3.5-era) reject these chunked files; the viewer needs a sibling adapter that reads the chunked family and emits the existing `M2ModelDocument` contract so the downstream renderer, runtime, and tensor packer consume the result with zero changes.

**Implementation Status (2026-06-03)**: The foundational `WowViewer.Core.IO.M2Chunked` reader/dispatcher, staged smoke tests, active viewer standalone open-file routing, and `WowViewer.Tool.Inspect m2 inspect` support for `MDLX` inputs are now landed. The standalone/runtime M2 viewer path also now treats invalid alias chains and malformed external `.anim` companions as non-fatal: bad sequence selections are rejected, logged, marked invalid in the selector, and the renderer falls back to another usable sequence instead of crashing. Remaining gaps against this draft are primarily companion-file completeness inside `M2ChunkedModelReader` itself (external `.skin` / `.anim` ingestion beyond the generated conversion skin) and the deferred 2.x pre-`2.0.0` research lane.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load a 1.12.1 .mdx into the Viewer (Priority: P1)

A user mounts the WoWArchive, points the viewer at
`I:\parp\parp-tools\output\tmp\wowarchive-clients\1.X_Retail_Windows_enUS_1.12.1.5875\World of Warcraft\`,
opens `File > Open File...`, and selects a `.mdx` from `Data/` (e.g. `Creature\Bear\Bear.mdx`).
The viewer detects the `MDLX` magic, parses the chunked FourCC structure, and renders
the model using the existing renderer/runtime tensor packer.

**Why this priority**: 1.12.1 is the most-requested pre-WotLK classic client in the
10 TB archive. Without chunked-MDX support, opening any 1.x model returns
"unsupported model format" — the viewer is effectively dead for ~3 of the 10 staged
clients (0.5.3, 1.X_Retail_Windows_enUS_1.12.1.5875, and possibly 2.X pre-2.0.0 if
any are chunked). This is the minimum-viable "the viewer actually works on 1.12"
slice the user is asking for.

**Independent Test**: Stand up a `WowViewer.Core.IO.Tests` test that reads
`Bear.mdx` (or any non-trivial 1.12 MDX) from the staged 1.12.1 client and asserts:
(a) `M2ModelDocument.Vertices.Length > 0`, (b) `M2ModelDocument.Geometry.Bones.Length > 0`
or 0 only if the model genuinely has no bones, (c) `M2ModelDocument.Materials.Length > 0`,
(d) no exceptions. Run with `dotnet test`. This is the SC-001 acceptance.

**Acceptance Scenarios**:
1. **Given** a 1.12.1 `.mdx` file with `MDLX` magic, **When** the user opens it
   in the viewer, **Then** the model renders with vertices, materials, and
   (when present) bones.
2. **Given** a 1.12.1 `.mdx` file with companion `.skin` files (`Skin01.skin`,
   `Skin02.skin`, …), **When** the user opens the model, **Then** the viewer
   resolves the skin files from the same directory and loads the active skin.
3. **Given** a 1.12.1 `.mdx` file with companion `.anim` files, **When** the
   user opens the model, **Then** the viewer reads the animations into the
   `M2ModelDocument.Animations` list. (Animation playback is a later slice;
   this slice is "read and surface in the document.")
4. **Given** a 3.3.5 `.m2` file (already supported), **When** the user opens
   it, **Then** the existing 3.3.5 path is taken — the new chunked adapter
   is NOT invoked. The two paths do not interfere.

---

### User Story 2 - Multi-Version Chunked MDX (0.5.3, 1.x, 2.x-pre-2.0.0) (Priority: P1)

The chunked format evolved across three eras. The reader must handle:

- **0.5.3 era** (Vanilla alpha/beta): smallest, fewest chunks. `MDLX` magic,
  `VERS` typically `0x100` or `0x101`. No `.skin` / `.anim` companion files
  on the traced path (skins are inline in `GEOS` sub-chunks).
- **1.x era** (Vanilla release): full chunk set, separate `.skin` (chunked
  with `SKIN` magic) and `.anim` (chunked with `ANIM` magic) companion files.
  `VERS` typically `0x100`..`0x105`. `SEQS` records have legacy-named-record
  variants (gilli `MdxFile.cs:24-49`).
- **2.x pre-2.0.0 era** (TBC beta?): unclear whether this era uses chunked
  MDX or transitions to MD20. **Defer to follow-up slice** — needs Ghidra
  evidence first.

**Why this priority**: The user's 10 TB archive has all three eras. The 0.5.3
client is the oldest and has the trickiest format (legacy record layouts,
inline skins). 1.x is the most-requested. The 2.x transition is research-only
for now.

**Independent Test**: Three separate `WowViewer.Core.IO.Tests` tests, one per
era. Each reads a representative MDX from the corresponding staged client and
asserts the same `M2ModelDocument` contract as US-1. 0.5.3 tests use a smaller
expected-vertex-count check (the model genuinely has fewer vertices).

**Acceptance Scenarios**:
1. **Given** a 0.5.3 `.mdx` with inline skins (no `.skin` companion), **When**
   parsed, **Then** the reader's `M2ModelDocument.Skins[0]` is populated from
   the inline `GEOS`/`VRTX` chunks.
2. **Given** a 1.x `.mdx` with external `.skin` companions, **When** parsed,
   **Then** the reader joins each `%02d.skin` file by `MODL.Name` and
   `MODL.SkinCount`.
3. **Given** an unknown chunk FourCC (e.g. a future client adds a new chunk),
   **When** parsed, **Then** the reader skips the unknown chunk with a
   non-fatal warning rather than aborting the whole load. (Forward compat.)
4. **Given** a chunk with size 0 or a malformed size that overruns the file,
   **When** parsed, **Then** the reader surfaces a clear error naming the
   chunk FourCC and the offset, rather than throwing a generic
   `EndOfStreamException`.

---

### User Story 3 - 1.12.1 Ghidra Cross-Reference (Priority: P2)

For each non-obvious format decision (e.g. how `LOOKS`/look-up tables are
emitted, how `PREM` particle emitter size is computed, how `ATCH` attachment
parent indices are interpreted), the spec records the cross-reference to
either (a) the gillijimproject_refactor `MdxFile.cs` reader, or (b) a
Ghidra decompilation of the 1.12.1 `WoW.exe`'s `MDLFileRead`-equivalent
function. The reader is NOT "ship until Ghidra confirms" — gilli's code is
trusted as the primary reference (it has been used in production for years).
Ghidra is the secondary check for the parts where gilli's choices are
unclear (e.g. what field of `MODL.Name` is the "internal model name" the
native client uses for cache lookup).

**Why this priority**: Ghidra is expensive (the user has to load the
binary). Doing Ghidra up-front is wasteful when gilli's reader already
works. Defer Ghidra to the slice where we have a concrete question
the gilli code can't answer.

**Independent Test**: The spec's "Ghidra Open Questions" section lists
every format decision where the answer came from gilli and not from
Ghidra. When a real-data test fails for one of those decisions, that
becomes the next Ghidra task.

**Acceptance Scenarios**:
1. **Given** the 043 implementation slice, **When** complete, **Then**
   the spec's "Format Decisions" section lists each non-obvious choice
   (skinning algorithm, animation blend mode, attachment parenting,
   particle emitter V1 vs V2 selection) and the source it came from
   (gilli `MdxFile.cs:line` or wowdev.wiki URL).
2. **Given** a future regression that traces to one of those decisions,
   **When** the user opens the spec, **Then** the spec identifies the
   gilli line(s) to revisit and (optionally) the Ghidra function
   address in 1.12.1 `WoW.exe` to re-verify.

---

### User Story 4 - Don't Break 3.3.5 (Priority: P1, hard non-goal)

The 3.3.5 model load path must remain bit-for-bit unchanged. The new
chunked adapter is a sibling, not a replacement. The existing
`M2ModelReader`, `M2SkinReader`, `M2AnimationReader`, `M2GeometryReader`,
`M2ToMdxConverter`, `MdxToM2Converter` are FROZEN.

**Why this priority**: This is the same hard non-goal as spec 037.
Three prior attempts to "fix" the general M2 path for older builds
broke 3.3.5. The user has explicitly forbidden it.

**Independent Test**: Run the full `WowViewer.Core.IO.Tests` M2 family
test suite (existing tests, no modifications) before AND after each
commit. All must pass with 0 changes.

**Acceptance Scenarios**:
1. **Given** the existing 3.3.5 `M2ModelReader`, **When** the 043
   implementation lands, **Then** zero files in
   `wow-viewer/src/core/WowViewer.Core/M2/`,
   `wow-viewer/src/core/WowViewer.Core.IO/M2/`, or
   `wow-viewer/src/core/WowViewer.Core.Runtime/M2/` are modified
   (verified by `git diff --stat`).
2. **Given** a 3.3.5 `.m2` model, **When** opened in the viewer,
   **Then** the load path is identical to the pre-043 build
   (verified by a byte-for-byte comparison of the loaded
   `M2ModelDocument` against a saved golden file).
3. **Given** a `wow-viewer/docs/architecture/m2/implementation-contract.md`
   anchor, **When** 043 lands, **Then** the contract is unchanged.
   The "sibling adapter in a new namespace" rule from spec 037 is
   re-affirmed.

---

### User Story 5 - 0.5.3 Era Out-Of-Scope Marker (Priority: P3)

The 0.5.3 era chunked format is more divergent than 1.x (legacy
record layouts, inline skins, smaller chunk set). For the 043 MVP
slice, 0.5.3 support is a stretch goal — the spec records it as
"supported by 043 but not validated end-to-end" so future slices
can pick it up.

**Why this priority**: 0.5.3 is the oldest client in the archive and
the most likely to need its own Ghidra pass. The 043 slice will
*read* 0.5.3 files (US-2), but won't promise identical visual output
to 3.3.5.

**Independent Test**: A 0.5.3 `.mdx` test runs as part of US-2 and
asserts vertex/material/bone counts. Visual output parity is a
later slice.

**Acceptance Scenarios**:
1. **Given** a 0.5.3 `.mdx`, **When** parsed, **Then** the reader
   produces a non-null `M2ModelDocument` with `Vertices.Length > 0`.
   No assertion on visual parity with the 3.3.5 model of the same
   asset.
2. **Given** a 0.5.3 `.mdx` that uses inline skins, **When** parsed,
   **Then** the reader populates the `Skins[0]` from the inline
   `GEOS` sub-chunk. The external-`.skin` companion code path is
   not exercised.

---

### Edge Cases

- **MDX with no `VERS` chunk**: Some 0.5.3 alphas are missing `VERS`.
  Reader should default to `0x100` and continue.
- **MDX with a `VERS` value the reader doesn't recognize** (e.g.
  `0x200` from a hypothetical future client): Reader should treat
  it as 1.x but log a warning that the version is unknown.
- **MDX with a chunk that has size 0**: Skip the chunk (some clients
  emit empty `MTLS` for models with no materials).
- **MDX with duplicate chunk FourCCs**: Use the first occurrence;
  warn on subsequent ones.
- **Skin file missing the active skin index**: Fall back to skin 0.
- **Skin file present but the parent MDX has a different name**: Match
  by stem (`Bear.mdx` ↔ `Bear00.skin`).
- **Anim file with a `CiRange.Start > CiRange.End`**: Reader should
  not crash; log the malformed range and skip the animation entry.
- **Viewer sequence selector hits an invalid alias chain or malformed external `.anim`**:
  selection should not crash the standalone/runtime viewer; log the failure,
  mark the sequence invalid, and fall back to another usable sequence when one exists.
- **Companion files on a different drive / not next to the MDX**:
  Reader should look in the same directory only; user must place
  files together.
- **File-too-small to even contain `MDLX` magic (less than 4 bytes)**:
  Reader should return a clear "not an MDX file" rather than
  throwing `EndOfStreamException`.
- **Chunk size overflow** (chunk claims to be larger than remaining
  file): Truncate to file end and warn; do not throw.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Reader (US-1, US-2)

- **FR-001**: The chunked MDX reader MUST be placed in a new namespace
  `WowViewer.Core.IO.M2Chunked` and a new folder
  `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/`. No modifications
  to existing files in `WowViewer.Core.IO/M2/` (per spec 037's hard
  non-goal, reaffirmed in US-4).
- **FR-002**: The reader MUST accept a `Stream` or `byte[]` (in-memory)
  and a `string` directory path (for companion file resolution).
- **FR-003**: The reader MUST validate `MDLX` magic (`0x584C444D` LE)
  as the first 4 bytes. Mismatch returns a clear "not a chunked MDX"
  error rather than throwing.
- **FR-004**: The reader MUST walk the chunk list (FourCC + uint32 size
  + payload) and dispatch each chunk to a typed parser. Unknown
  FourCCs are skipped with a warning (FR-018).
- **FR-005**: The reader MUST parse all 25+ chunks from
  `gillijimproject_refactor/MDX-L_Tool/Formats/Mdx/MdxHeaders.cs:13-55`:
  `VERS`, `MODL`, `SEQS`, `GLBS`, `MTLS`, `TEXS`, `GEOS`, `GEOA`, `BONE`,
  `HELP`, `PIVT`, `ATCH`, `LITE`, `PREM`, `PRE2`, `RIBB`, `EVTS`, `CAMS`,
  `CLID`, `HTST`, `TXAN`. Sub-chunks of `GEOS` (`VRTX`, `NRMS`, `PTYP`,
  `PCNT`, `PVTX`, `GNDX`, `MTGC`, `MATS`, `UVAS`, `UVBS`, `BIDX`,
  `BWGT`) are also required.
- **FR-006**: The reader MUST emit an `M2ModelDocument` (existing
  canonical type in `WowViewer.Core/M2/M2ModelDocument.cs`) with:
  vertices, normals, UVs, materials, bones (when present), animations
  (when present), skins (when external or inline).
- **FR-007**: The reader MUST resolve `.skin` companion files from the
  same directory as the `.mdx`, matching `Bear.mdx` ↔ `Bear00.skin`,
  `Bear01.skin`, etc. The active skin index comes from `MODL` or
  defaults to 0.
- **FR-008**: The reader MUST resolve `.anim` companion files the same
  way. Each `ANIM` file becomes one entry in `M2ModelDocument.Animations`.
- **FR-009**: The reader MUST handle inline skins (0.5.3 era) by
  extracting skin data from the `GEOS` sub-chunks themselves.
- **FR-010**: The reader MUST handle the `SEQS` legacy-named-record
  variants (gilli `MdxFile.cs:24-49`) — the 0.5.3 era uses a 128-byte
  record layout, 1.x uses 132+ bytes. Reader auto-detects by size.
- **FR-011**: The reader MUST handle `PREM` (v1) and `PRE2` (v2)
  particle emitters. v2 is used in 1.x; v1 is used in 0.5.3.

#### Dispatch (US-1, US-4)

- **FR-012**: The viewer MUST dispatch `.mdx` files to the new
  `M2ChunkedModelReader` based on the `MDLX` magic. `.m2` files
  continue to dispatch to the existing 3.3.5 reader. The dispatch
  is a single `if (magic == MDLX) return M2ChunkedModelReader.Read(...);
  else return M2ModelReader.Read(...);` at the file-load seam.
- **FR-013**: The dispatch MUST be implemented as a new
  `M2ModelReaderDispatcher` (or similar) in a new namespace. The
  existing `M2ModelReader` is not modified.
- **FR-014**: The dispatch seam MUST be a single new file in
  `WowViewer.Core.IO/M2Chunked/` (or `WowViewer.Core.Runtime/M2/`
  if the dispatch belongs in the runtime). NO modifications to
  any existing M2 file.

#### Format Decisions (US-3)

- **FR-015**: Every non-obvious format decision in the reader MUST
  be documented in the spec's "Format Decisions" appendix with a
  source line (gilli `MdxFile.cs:line` or wowdev.wiki URL).
- **FR-016**: If a real-data test fails for a decision sourced from
  gilli (not Ghidra), the spec MUST add a "Ghidra Open Question"
  row naming the Ghidra function to decompile in 1.12.1 `WoW.exe`.
- **FR-017**: Ghidra work is deferred to a follow-up slice. The 043
  MVP does not require the 1.12.1 `WoW.exe` to be loaded in Ghidra.

#### Robustness (US-1, US-2, US-4)

- **FR-018**: Unknown chunk FourCCs MUST be skipped with a
  `Console.Error.WriteLine` warning naming the FourCC and the
  offset. The load continues.
- **FR-019**: Chunk size overruns (chunk claims more bytes than
  remain in the file) MUST be truncated to the file end with a
  warning. The load continues with whatever data was readable.
- **FR-020**: A `VERS` value the reader doesn't recognize MUST be
  treated as 1.x with a warning. The load continues.
- **FR-021**: A `VERS` chunk missing entirely MUST default to
  `0x100` with a warning. The load continues.

#### Validation (US-1, US-2, US-5)

- **FR-022**: At least three `WowViewer.Core.IO.Tests` integration
  tests MUST pass:
  - One 1.12.1 `.mdx` test (full material + skin + animation).
  - One 0.5.3 `.mdx` test (inline skin, no animation).
  - One 2.x MDX test (if any 2.x chunked MDX exists in the staged
    clients; otherwise skipped with a `[Skip]` attribute).
- **FR-023**: All existing `WowViewer.Core.IO.Tests` M2 family tests
  MUST continue to pass with zero modifications (US-4 hard non-goal).
- **FR-024**: The 043 implementation MUST extend the existing CLI
  command `WowViewer.Tool.Inspect m2 inspect --input <file.mdx|file.mdl>`
  (and the archive-root variant) so it reads a chunked `.mdx` and prints
  a summary (chunk list, vertex count, material count, skin count,
  animation count, magic check). The command is the spec 041 `map`
  command analog but for classic MDX routed through the shared M2
  runtime contract.

### Key Entities

- **M2ChunkedModelReader**: the new reader class. Lives in
  `WowViewer.Core.IO.M2Chunked` namespace. Pure read-only; emits
  `M2ModelDocument` (existing type).
- **M2ChunkedChunkHeader**: `(string FourCC, uint Size, long Offset)`
  pair. Used internally by the reader for dispatch.
- **M2ChunkedChunkParser**: per-chunk typed parser. 25+ parsers
  (one per FR-005 chunk). All in the `M2Chunked` namespace.
- **M2ModelDocumentDispatcher** (or `M2ModelReaderDispatcher`):
  the dispatch seam. Checks magic, routes to either
  `M2ChunkedModelReader` or existing `M2ModelReader`.
- **M2ModelDocument**: the existing canonical output type. No
  changes. The chunked reader's output matches the 3.3.5
  reader's output shape (same property names, same semantics).

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A 1.12.1 `.mdx` (e.g. `Creature\Bear\Bear.mdx`) loads
  via `M2ChunkedModelReader` and produces a non-null
  `M2ModelDocument` with `Vertices.Length > 0`,
  `Materials.Length > 0`, and either `Bones.Length > 0` or 0 only
  if the model genuinely has no bones. Verified by
  `WowViewer.Core.IO.Tests` integration test.
- **SC-002**: All 25+ chunks (FR-005) parse without error on a
  representative 1.12.1 `.mdx`. Verified by a chunk-by-chunk
  test that loads each chunk's typed parser and asserts the
  parser returns a non-null result.
- **SC-003**: Zero modifications to existing files in
  `WowViewer.Core/M2/`, `WowViewer.Core.IO/M2/`, and
  `WowViewer.Core.Runtime/M2/`. Verified by
  `git diff --stat` showing 0 changes in those folders.
- **SC-004**: All existing `WowViewer.Core.IO.Tests` M2 family
  tests pass with 0 modifications. Verified by `dotnet test`
  showing the pre-043 test count is unchanged.
- **SC-005**: A 0.5.3 `.mdx` loads via `M2ChunkedModelReader`
  and produces a non-null `M2ModelDocument` with
  `Vertices.Length > 0`. The inline-skin path is exercised.
  (US-2 AC-1.)
- **SC-006**: The viewer can render a 1.12.1 `.mdx` end-to-end
  (open file → read → render to screen) and the model is
  visually recognizable as a bear (or whatever the test model
  is). Validated manually by the user on a real 1.12.1 client
  build.
- **SC-007**: `WowViewer.Tool.Inspect m2 inspect --input <path-to-1.12.1-.mdx>`
  exits 0 on a valid 1.12.1 `.mdx` and prints the chunk list,
  vertex count, material count, skin count, and animation count.
- **SC-008**: The "Format Decisions" appendix lists every
  non-obvious format choice with a gilli `MdxFile.cs:line`
  reference. Zero "TODO: Ghidra" rows in the MVP slice.

---

## Assumptions

- **A-001**: The 10 TB WoWArchive is read-only. The 043 slice
  reads from `I:\parp\parp-tools\output\tmp\wowarchive-clients\`
  only.
- **A-002**: gillijimproject_refactor's `MDX-L_Tool/Formats/Mdx/`
  code is a trusted reference. The chunk parsers in gilli
  have been used in production for years and are the primary
  format source. Ghidra is secondary.
- **A-003**: The 1.12.1 `WoW.exe` is NOT loaded in Ghidra for
  the MVP slice. Ghidra is deferred to a follow-up slice
  (FR-017, US-3).
- **A-004**: The 2.x pre-2.0.0 era MDX format is unknown until
  Ghidra is loaded. The 043 slice supports 0.5.3 and 1.x
  explicitly. 2.x pre-2.0.0 is deferred.
- **A-005**: Animation playback (timeline, blending) is OUT OF
  SCOPE for the 043 slice. The reader populates
  `M2ModelDocument.Animations` with the parsed data, but
  the runtime does not play them. This is a separate slice.
- **A-006**: Particle emitter rendering (PREM, PRE2, RIBB) is
  OUT OF SCOPE for the 043 slice. The reader reads the
  emitter data into `M2ModelDocument`, but the runtime does
  not render particles. This is a separate slice.
- **A-007**: The `M2ModelDocument` schema is sufficient for
  chunked MDX output. The reader maps chunked-format types
  (e.g. `MdlModel` from gilli) into the canonical
  `M2ModelDocument` shape. No `M2ModelDocument` schema
  changes are needed for the MVP. (If a schema change is
  needed, it goes in a follow-up slice and is tracked as
  a separate spec.)
- **A-008**: The dispatch seam (FR-012 to FR-014) is a
  non-invasive wrapper. The existing 3.3.5 reader keeps
  its current entry point; the dispatch is a thin
  pre-check.

---

## Open Questions

- **OQ-1**: 2.x pre-2.0.0 era — is it chunked MDX or MD20?
  Default assumption: spec 037 is correct (2.0.0.5610 is
  MD20 v0x100). The transition from chunked to MD20
  happened between 1.12.1 and 2.0.0.5600+ (vanilla TBC).
  Needs a 2.x build to confirm. **Action**: 043 MVP
  ships 0.5.3 + 1.x only. 2.x chunked MDX, if it
  exists, is a follow-up slice.

- **OQ-2**: The `LOOKS` table (render flags) — gilli
  `MdxFile.cs` reads it from `MODL`. The 3.3.5 reader
  reads it from a different offset in the MD20 header.
  Does the chunked `MODL` field map directly to
  `M2ModelDocument.RenderFlags`? **Default**: yes, with
  a translation table (gilli uses 32-bit flags, MD20
  uses 32-bit flags, but the bit meanings differ). The
  translation table is in the spec's "Format Decisions"
  appendix.

- **OQ-3**: The `GEOS` chunk has 7+ sub-chunks with their
  own magic. The reader walks the sub-chunk list
  independently of the parent chunk. Are there any
  sub-chunks that span multiple `GEOS` blocks (e.g.
  `NRMS` after the first `GEOS` applies to all)?
  **Default**: no, sub-chunks are scoped to their
  parent `GEOS` block. Confirmed by gilli
  `MdxFile.cs:130-200`.

- **OQ-4**: The `BONE` chunk size in 0.5.3 is 0x50
  (80 bytes). In 1.x it grows to 0xAC (172 bytes)
  with the addition of `BillboardSizes` and
  `UnknownPadding` fields. The reader auto-detects by
  size. **Default**: yes. The 0xAC stride was confirmed
  in the 3.3.5 Ghidra pass for `FUN_0095b870`'s
  animation blend function (uses 0xAC stride).

- **OQ-5**: The "Open Zarr Dataset..." menu item from
  spec 042 lands in a separate slice. The 043 slice
  does NOT touch the menu. The user has confirmed
  Zarr is a lower priority than MDX; menu changes
  land with the 042 work.

---

## Format Decisions Appendix (placeholder, filled during implementation)

Each row in this appendix is one non-obvious format choice. The "Source"
column is either `gillijimproject_refactor/src/MDX-L_Tool/Formats/Mdx/MdxFile.cs:<line>`
or a wowdev.wiki URL. Ghidra rows are deferred to a follow-up slice.

| Decision | Default | Source | Ghidra needed? |
| --- | --- | --- | --- |
| Magic | `0x584C444D` ("MDLX" LE) | `MdxHeaders.cs:9-10` | No |
| Chunk walk order | Sequential, skip unknown | `MdxFile.cs:120-180` | No |
| `SEQS` record size detection | 128 = 0.5.3, 132+ = 1.x | `MdxFile.cs:24-49, 69-86` | No |
| `BONE` stride | 0x50 (0.5.3), 0xAC (1.x) | `MdxFile.cs` (bone parser) | No |
| `PREM` vs `PRE2` | First one wins, version is a sibling chunk | wowdev.wiki/MDX | No |
| `MODL` render flags → `M2ModelDocument.RenderFlags` | Bit-translation table (added in impl) | TBD | No |
| Skin file matching | By stem (`Bear.mdx` ↔ `Bear00.skin`) | `MdxFile.cs` (skin resolver) | No |
| Active skin selection | `MODL` index, default 0 | wowdev.wiki/MDX | No |
| Inline skin (0.5.3) | Pull from `GEOS` sub-chunks | gilli reader | No |
| Animation record format | `MdlAnimBlock` from gilli `MdxTypes.cs` | `MdxTypes.cs` | No |

(Filled in during implementation. Empty cells mean "decided at impl time,
gilli source is the reference.")

---

## Notes

- This spec is an **implementation slice** (not research). The format
  is well-understood via gillijimproject_refactor's `MDX-L_Tool`.
- All new code lives in `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/`
  (new folder, new namespace). NO existing M2 files are modified.
- The hard non-goal from spec 037 is re-affirmed: 3.3.5 path is
  bit-for-bit unchanged.
- Animation playback and particle rendering are explicitly OUT OF SCOPE
  for this slice. The reader reads the data; the runtime does not
  consume it yet.
- 0.5.3 + 1.x are in scope. 2.x chunked MDX (if it exists) is deferred.
- 1.12.1 Ghidra is deferred to a follow-up slice.
