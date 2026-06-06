# Implementation Plan: 048 — 1.12.1 Era-Aware MD20 Reader

**Branch**: `048-m2-1121-era-aware-md20-reader` | **Date**: 2026-06-05 | **Spec**: `wow-viewer/specs/048-m2-1121-era-aware-md20-reader/spec.md`
**Input**: Feature spec + Ghidra trace in `wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md`

## Summary

Add a sibling era-aware MD20 reader (`M2Era1121ModelReader`) for the 1.12.1 Vanilla client. The 1.12.1 native binary (`WoW.exe` `Build 5875`, Ghidra-traced) uses the `MD20` magic (`0x3032444D`) with `version ∈ {0x100, 0x101}` and a flat (count, offset) pointer table — NOT the chunked `MDLX` format spec 043 covers, and NOT the 3.3.5 stride constants `M2ModelReader.cs` uses. The fix is a new reader in a new folder/namespace that emits the existing `M2ModelDocument` shape, plus a one-line version-field branch in the existing dispatcher. The 3.3.5 reader and the 0.5.3/0.7.0/0.8.0 chunked reader are siblings and are FROZEN.

## Technical Context

**Language/Version**: C# / .NET 10 (matches `wow-viewer` baseline).
**Primary Dependencies**: `WowViewer.Core.IO.M2` (frozen 3.3.5 reader), `WowViewer.Core.IO.M2Chunked` (frozen chunked reader, dispatcher is the only edit seam), `WowViewer.Core.M2.M2ModelDocument` (canonical output, unchanged), `System.Buffers.Binary`, `System.Numerics`.
**Storage**: N/A — reader is in-memory only.
**Testing**: `dotnet test i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`. New tests land in `wow-viewer/tests/WowViewer.Core.IO.Tests/`. The 048 implementation slice is responsible for staging a representative 1.12.1 `.mdx` test fixture (see Open Question F-1 below).
**Target Platform**: Win32, .NET 10. Cross-platform: should work on Linux/macOS (no P/Invoke).
**Project Type**: Library (sibling reader) + 1-line dispatcher edit + CLI tool update.
**Performance Goals**: Dispatcher reads 8 bytes (magic + version) per call. 1.12.1 reader reads the file once. No additional allocations vs the 3.3.5 reader.
**Constraints**:
- 3.3.5 reader is FROZEN (spec 037, reaffirmed in spec 043 US-4).
- 0.5.3/0.7.0/0.8.0 chunked reader is FROZEN except for the dispatcher's version-field branch.
- Real-data validation is required (per AGENTS.md "Real-Data Validation" rule + spec SC-001). The test fixture staging is the implementation slice's responsibility, not the spec's.
- Repo independence (no references outside `wow-viewer/`).
**Scale/Scope**: One new reader, one new constants class, one new version enum, one new era-tag enum, ~6-10 new files, ~4 new tests, one dispatcher edit, one CLI command update, 1 spec doc update (043), 1 research doc cleanup, 1 memory bank update.

## Constitution Check

*Gate: must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Repo Independence** (Principle I): PASS. New reader lives in `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/`. No `.csproj` outside `wow-viewer/`.
- **Library-First** (Principle II): PASS. New reader is a shared library. CLI tool change is a thin wrapper. No format reader duplication (the new reader covers the 1.12.1 lane; the 3.3.5 reader covers 3.0.1+; the chunked reader covers 0.5.3/0.7.0/0.8.0).
- **Real-Data Validation** (Principle III): PARTIAL. The Ghidra trace + the canonical binary are the primary proof for the spec. Real-data validation against a 1.12.1 `.mdx` test fixture is an implementation-phase task (F-1 below). The implementation slice MUST stage a 1.12.1 `.mdx` before declaring the slice done.
- **No Game Client Path Assumptions** (Principle VI): PASS. The 048 reader's test fixture path is `I:\parp\parp-tools\output\tmp\wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/...`. No `H:\CLIENTS`.
- **Format Reader/Writer Ownership** (Safety): PASS. The 3.3.5 reader and the chunked reader are NOT rewritten. The 048 reader is a new sibling that does not duplicate work.
- **Spec Docs Are Source of Truth** (Workflow): PASS. The spec is at `specs/048-.../spec.md`. The research doc is at `docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md`. Any code change that contradicts a constant in the research doc is a research-doc update first, code update second, in the same commit.
- **Bite-Sized Plans** (Workflow): PASS. Eight phases below, each ≤10 steps, each independently validatable.
- **Memory Bank Discipline** (Workflow): PASS. The implementation slice ends with an `activeContext.md` / `progress.md` update in `gillijimproject_refactor/memory-bank/`.

**Re-check after Phase 1 design**: Same as above, with the addition that the test fixture path is concretely known (F-1 closed) and the era tag enum's CLI output is mocked in the test.

## Project Structure

### Documentation (this feature)

```text
specs/048-m2-1121-era-aware-md20-reader/
├── spec.md              # The feature spec (this directory)
├── plan.md              # This file
├── research.md          # Pointer to m2-mdx-1121-native-trace-2026-06-05.md
├── data-model.md        # EraTag enum, Version enum, M2Era1121Constants
├── quickstart.md        # How to stage a 1.12.1 .mdx test fixture
├── contracts/           # (empty for now; 1.12.1 .skin/.anim contracts are spec 050)
└── tasks.md             # Phase-by-phase task breakdown
```

### Source Code (changes)

```text
wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/         # NEW
├── M2Era1121ModelReader.cs                              # NEW: the new reader
├── M2Era1121Constants.cs                                # NEW: 1.12.1 offsets + strides
├── M2Era1121Version.cs                                  # NEW: enum { Unknown, V100, V101 }
└── M2Era1121EraTag.cs                                   # NEW: enum for dispatch + CLI output

wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/         # MODIFIED (1 file)
└── M2ModelReaderDispatcher.cs                           # +1 if-branch (version-field check)

wow-viewer/src/tools/WowViewer.Tool.Inspect/             # MODIFIED (1 file)
└── ...                                                  # M2 inspect command prints "Era: 1.12.1 (MD20 v0x100)" tag

wow-viewer/tests/WowViewer.Core.IO.Tests/                # MODIFIED (1 file, +4 tests)
└── M2Era1121ModelReaderTests.cs                         # NEW: 4 tests per FR-017
```

### Documentation (changes)

```text
wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md  # CLEANUP: add "Status: superseded by spec 048" banner
wow-viewer/specs/043-m2-chunked-mdx-classic-support/spec.md         # EDIT: remove 1.12.1 from the chunked lane, point at spec 048
gillijimproject_refactor/memory-bank/activeContext.md                 # APPEND: 048 status, next steps
gillijimproject_refactor/memory-bank/progress.md                      # APPEND: 048 milestone
```

**Structure Decision**: Sibling-folder layout matches the existing 043 pattern (`M2Chunked/` next to `M2/`). The new folder is `M2Era1121/` to make the era explicit in the path. The dispatcher's edit is one branch in one file, not a new dispatcher file (per FR-011).

## Implementation Phases

Each phase is independently validatable. Phases 1-3 are the MVP. Phases 4-5 flesh out the reader. Phases 6-7 add tests and CLI. Phase 8 is doc and memory-bank sync.

### Phase 0 — Test fixture staging (BLOCKER for Phase 1+)

*Goal*: A representative 1.12.1 `.mdx` is staged to `I:\parp\parp-tools\output\tmp\wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/World of Warcraft/Data/` and the test fixture path is recorded in `quickstart.md`.

*Steps* (≤10):

1. Decide which 1.12.1 `.mdx` to use as the test fixture. Pick a model with all the relevant features: sequences, views, lights, bones, colors, texture weights, particle/ribbon/camera records (or as many as 1.12.1 has). Suggested: a creature model like `Creature\Bear\Bear.mdx` if extractable; otherwise any small model with rich metadata.
2. Stage the `.mdx` from the 1.12.1 MPQ archive into the staged client directory. The implementation slice is responsible for the extraction; this is the only step that touches the MPQ.
3. Record the test fixture path in `wow-viewer/specs/048-m2-1121-era-aware-md20-reader/quickstart.md` and reference it from the test file.
4. (Optional) Stage a companion 1.12.1 `.skin` for negative testing (the 048 reader must mark skins as missing/unreadable without crashing).

*Validation*: `Test-Path` of the fixture path returns true; file size > 0; first 4 bytes are `4D 44 32 30` (ASCII "MD20").

*Out of scope*: Repacking the full 10.5TB archive into Zarr. Out of scope per the user's "data landscape" message.

### Phase 1 — Constants, version enum, era-tag enum (FOUNDATION)

*Goal*: The new `M2Era1121Constants.cs`, `M2Era1121Version.cs`, and `M2Era1121EraTag.cs` exist. No reader logic yet.

*Steps* (≤10):

1. Create the folder `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/`.
2. Create `M2Era1121Version.cs` with the enum `{ Unknown, V100, V101 }` and a `bool Is1121` helper.
3. Create `M2Era1121EraTag.cs` with the enum `{ Mdlx = 0, Md20_1X_V100 = 1, Md20_1X_V101 = 2, Md20_3X_V108 = 3, Unknown = 99 }` and a `string ToDisplayString()` helper that returns "1.12.1 (MD20 v0x100)", "1.12.1 (MD20 v0x101)", "3.3.5 (MD20 v0x108)", "MDLX", "Unknown".
4. Create `M2Era1121Constants.cs` as a static class. Transcribe the 1.12.1 offset/stride constants from `m2-mdx-1121-native-trace-2026-06-05.md` sections "Header layout" and "Stride differences vs 3.3.5". The constants include at minimum: `VersionOffset = 0x04`, `NameCountOffset = 0x08`, `NameOffsetOffset = 0x0C`, `FlagsOffset = 0x10`, `GlobalLoopCountOffset = 0x14`, `GlobalLoopOffsetOffset = 0x18`, `SequenceCountOffset = 0x1C`, `SequenceOffsetOffset = 0x20`, `BoneCountOffset = 0x2C`, `BoneOffsetOffset = 0x30`, `ViewCountOffset = 0x3C`, `ColorCountOffset`, `ColorOffsetOffset`, `TextureWeightCountOffset`, `TextureWeightOffsetOffset`, `TextureTransformCountOffset`, `TextureTransformOffsetOffset`, `BoundsOffset`, `BoundsRadiusOffset`, `LightCountOffset`, `LightOffsetOffset`, `CameraCountOffset`, `CameraOffsetOffset`, `RibbonCountOffset`, `RibbonOffsetOffset`, `ParticleCountOffset`, `ParticleOffsetOffset`, plus per-record strides: `SequenceStride = 0x6C`, `BoneStride = ?`, `LightStride = 0x0C`, `CameraStride = 0x2C + per-frame ?`, `RibbonStride = 0x7C`, `ParticleStride = 0xDC`, plus the `0x101`-only table offsets: `V101ExtraCountOffset = 0xDC`, `V101ExtraOffsetOffset = 0xE0`, `V101ExtraEntrySize = 0x1F8`, `V101ExtraSubTableCount = 29`.
5. Add XML doc comments to every constant pointing at the research doc section that sourced it. Example: `/// <summary>Sequence count offset in 1.12.1 MD20. Source: research doc "Header layout" table.</summary>`.
6. (Implementation detail) Where the research doc is uncertain (e.g. `BoneStride` is not in the research doc explicitly; the bone record was not decompiled in the Ghidra trace), leave the constant as `0` with an `// TODO: Ghidra OQ-N` comment and the reader's bone-walk code throws "not yet mapped" per FR-006.
7. Build: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`. No warnings or errors.

*Validation*: The constants class compiles. The version enum and era-tag enum compile. The research doc's "Stride differences vs 3.3.5" table is cross-referenced one-for-one in the XML docs.

### Phase 2 — Era-aware dispatcher branch

*Goal*: The existing dispatcher reads `version` and routes 1.12.1 `.mdx` to the (not-yet-existing) `M2Era1121ModelReader`. 3.3.5 and chunked lanes are unchanged.

*Steps* (≤10):

1. Read `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs` to see the current dispatch shape. (Implementation slice does this; not a spec action.)
2. Add a `ReadEra` static helper that reads bytes 0..7 from a `Stream`, parses the magic and the `version` field at offset 0x04, and returns a `M2Era1121EraTag` value. The helper throws on EOF (file too small to be any M2).
3. In the dispatcher's main method, insert a new branch: if `era == Md20_1X_V100 || era == Md20_1X_V101`, call `M2Era1121ModelReader.Read(stream, sourcePath)`. The branch comes AFTER the `MDLX` check (so chunked is short-circuited) and BEFORE the existing 3.3.5 MD20 fallback.
4. Insert the 2.x TBC rejection: if `magic == MD20 && version ∈ (0x101, 0x108)`, throw `NotSupportedException("MD20 v0x" + version.ToString("X") + " is the 2.x TBC era, which is not yet supported. Tracked under spec 049.")`. Place this BEFORE the 3.3.5 fallback so 2.x doesn't get silently parsed as 3.3.5.
5. The 3.3.5 MD20 branch (`version >= 0x108`) is unchanged.
6. The `MDLX` branch is unchanged.
7. The "unsupported magic" branch is unchanged.
8. Add a public `Era` property to the dispatcher's return type (or wrap the result) so the CLI tool can print the era tag. If the existing dispatcher returns a tuple, extend it. If it returns a class, add a property. The cleanest path: the dispatcher returns `(M2ModelDocument document, M2Era1121EraTag era)` and the CLI tool formats the era tag.
9. Build: `dotnet build`. No new warnings or errors.
10. Run the existing M2 test suite: `dotnet test`. All existing tests pass with 0 changes (FR-018, US-3).

*Validation*: A synthetic test that constructs an 8-byte `MD20 + version=0x100` byte sequence returns `Md20_1X_V100` from `ReadEra`. A 2.x sentinel (version=0x104) throws `NotSupportedException` with the 049 message. The chunked lane is untouched (a `MDLX` byte sequence returns `Mdlx`).

### Phase 3 — `M2Era1121ModelReader` skeleton (header + version + dispatch)

*Goal*: The new reader exists, accepts a `Stream` and `sourcePath`, validates the magic and version, reads the model name and the global-loop count/offset, and returns a minimal `M2ModelDocument` with empty per-record arrays. No sequence/light/bone reading yet.

*Steps* (≤10):

1. Create `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/M2Era1121ModelReader.cs` with the public static `Read(string path)` and `Read(Stream stream, string sourcePath)` methods.
2. The reader's entry point mirrors `M2ModelReader.Read`: read all bytes, validate minimum size, read magic (4 bytes), read `version` (4 bytes at offset 0x04), call a private `ParseM2` that walks the header.
3. `ParseM2` reads the model name via `NameCountOffset`/`NameOffsetOffset`, the `Flags` at `FlagsOffset`, the `ViewCount` at `ViewCountOffset`, and the `GlobalLoops` table at `GlobalLoopCountOffset`/`GlobalLoopOffsetOffset`. All other per-record arrays are passed as empty `IReadOnlyList<>` to the `M2ModelDocument` constructor.
4. The `M2ModelDocument` constructor signature is the existing one. No new fields are introduced (per A-006, schema changes are out of scope).
5. The reader returns the document with `Version = version` (raw uint from offset 0x04), `Flags = flags` (raw uint from offset 0x10), `ViewCount = viewCount` (raw uint from offset 0x3C), and the model name as a `string?`.
6. Bounds (`BoundsMin`, `BoundsMax`, `BoundsRadius`) are read from the offsets recorded in the research doc. The research doc records `BoundsOffset = 0xA0`, `BoundsRadiusOffset = 0xB8` as the 3.3.5 values; the 1.12.1 values are TBD and the reader uses the 3.3.5 offsets with a `// TODO: confirm 1.12.1 bounds offset` comment.
7. Companion `.skin`/`.anim` resolution is NOT attempted in this spec. The reader returns `embeddedSkinProfileCount = 0`, `embeddedSkinProfileOffset = 0`, and the skins/animations are not populated (A-005, FR-016).
8. Add validation: `count > 0 && offset == 0` throws `InvalidDataException` with the label, count, and offset (FR-013). `count × stride > file length` throws (FR-014). Non-finite floats in vector fields throw (FR-015).
9. Build: `dotnet build`. No warnings or errors.
10. Run the 1.12.1 test fixture (from Phase 0) through the new reader. The reader returns a non-null `M2ModelDocument` with `Version = 0x100` or `0x101`, `Flags` non-zero, and `ViewCount > 0` (or 0 only if the model has no views, which is rare). Capture the output as a golden file for the test.

*Validation*: The 1.12.1 test fixture loads, the header values are read correctly per the research doc, and the document passes through the existing 3.3.5-shape contract.

### Phase 4 — Per-record array readers (sequences, bones, colors, texture weights, texture transforms, lights)

*Goal*: The reader populates the per-record arrays. Each array is read with the 1.12.1 stride constant from `M2Era1121Constants`.

*Steps* (≤10):

1. Add a private `ReadSequences` method that reads `SequenceCount` records at `SequenceStride` (`0x6C` per the research doc). Each sequence is mapped to the existing `M2SequenceDefinition` constructor signature. The 1.12.1 sequence record layout is the 0x6C stride; the field offsets within the record are transcribed from the research doc (FR-005).
2. Add `ReadBones` with the 1.12.1 bone stride. The exact stride is `// TODO: Ghidra OQ-N` per Phase 1 step 6; if the stride is unknown, the method throws "not yet mapped" per FR-006.
3. Add `ReadColors`, `ReadTextureWeights`, `ReadTextureTransforms` with the 1.12.1 strides.
4. Add `ReadLights` with `LightStride = 0x0C`. Per OQ-4, the 1.12.1 light record is `(pos, radius)` only. The full `M2LightDefinition` requires color/intensity/attenuation tracks; for 1.12.1, those are set to runtime defaults (per A-006, the schema is unchanged; missing fields become defaults).
5. Wire all the per-record array reads into `ParseM2` so the document is fully populated (subject to the schema constraints in A-006 and the open-question TODOs).
6. The 0x101-only table at `0xDC/0xE0` (29 sub-tables, 0x1F8/entry) is read with bounds checks. If the read succeeds, the records are passed to a new `M2ModelDocument` field or stored in a side dictionary; if it fails, the reader logs a warning and continues (per FR-006, "not yet mapped" is the failing case if the schema cannot absorb the records).
7. Add a helper `EnsureSpan(count, offset, stride, length, label)` that mirrors the 3.3.5 reader's `ValidateSpan` for consistency.
8. Build: `dotnet build`. No warnings or errors.
9. Re-run the 1.12.1 test fixture. The reader returns a fully populated `M2ModelDocument`. Update the golden file.
10. Add unit tests: synthetic 1.12.1 headers with known sequence/light counts, asserting the parsed counts and stride-correct offsets.

*Validation*: All non-deferred per-record arrays populate correctly. Bounds checks fire on truncation. The 0x101-only table is read with bounds checks (no buffer overruns on the 0x100 test fixture, which doesn't have the table).

### Phase 5 — View-record walker + camera/ribbon/particle

*Goal*: The view record at `0x3C` is walked with 0x2C stride and 9 nested sub-tables (per the research doc). Cameras, ribbons, and particles are read with the 1.12.1 strides.

*Steps* (≤10):

1. Add a private `ReadViews` method that walks `ViewCount` records at `ViewStride = 0x2C`. Each view record has 9 nested sub-tables (per the research doc). The sub-tables are read with bounds checks; if a sub-table's span overruns, the view is logged and skipped with a warning.
2. Add `ReadCameras` with `CameraStride = 0x2C + per-frame ?`. Per the research doc, the 1.12.1 camera record is 0x2C with an additional per-frame table; the exact layout is TBD. The reader reads the base 0x2C record and marks the per-frame table as "not yet mapped" if the stride is unknown.
3. Add `ReadRibbons` with `RibbonStride = 0x7C`.
4. Add `ReadParticles` with `ParticleStride = 0xDC`.
5. Wire view, camera, ribbon, and particle reads into `ParseM2`.
6. The view record's LOD distance at `+0x28` is read and stored in a new `M2ViewDefinition` or, if the schema cannot absorb it, logged and skipped. Per A-006, schema changes are out of scope; if the view record cannot be stored, the reader throws "not yet mapped" naming OQ-7 (a new open question: view record schema).
7. Build: `dotnet build`. No warnings or errors.
8. Re-run the 1.12.1 test fixture. The reader returns a fully populated `M2ModelDocument` with view records and camera/ribbon/particle arrays.
9. Update the golden file.
10. Add unit tests for view-record bounds checks (synthetic view records at the end of the file with truncated sub-tables).

*Validation*: View records parse without overruns. Camera/ribbon/particle records use the 1.12.1 strides. The golden file matches.

### Phase 6 — Tests

*Goal*: The four required tests from FR-017 pass. The 3.3.5 and chunked test suites are unchanged.

*Steps* (≤10):

1. Create `wow-viewer/tests/WowViewer.Core.IO.Tests/M2Era1121ModelReaderTests.cs`.
2. **Test US-1**: `M2Era1121ModelReader_Read_TestFixture_ProducesDocument`. Loads the staged 1.12.1 test fixture, asserts `Version ∈ {0x100, 0x101}`, asserts `SequenceCount == expectedCount`, asserts `ViewCount == expectedCount`, asserts no exceptions.
3. **Test US-1 negative**: `M2ModelReaderDispatcher_3X_Model_GoesTo3X_Reader`. Loads a 3.3.5 `.m2` and asserts the dispatcher's `Era == Md20_3X_V108` (i.e. the 1.12.1 reader is not invoked).
4. **Test US-2**: `M2Era1121ModelReader_FlagsPassThrough`. Constructs a synthetic 1.12.1 MD20 with `flags = 0x80000000`, parses, asserts `M2ModelDocument.Flags == 0x80000000`. Also asserts the 1.12.1 reader's public surface has no `M2BatchParticles` or `M2ForceAdditiveParticleSort` cvar accessor (reflection check).
5. **Test US-3 regression**: `M2ModelReaderDispatcher_MDLX_GoesToChunkedReader`. Constructs an `MDLX` byte sequence, dispatches, asserts `Era == Mdlx`. The 1.12.1 reader is not invoked (verified by a counter or by the dispatcher's return path).
6. **Test US-4**: `M2ModelReaderDispatcher_2X_Version_Throws`. Constructs an `MD20 + version=0x104` byte sequence, dispatches, asserts `NotSupportedException` with the 049 spec slot mentioned.
7. **Golden file test**: Parse the 1.12.1 test fixture twice (once for the golden, once for the assertion). Compare the parsed document to a stored golden JSON. (The golden file is regenerated when the reader changes.)
8. Build: `dotnet build`. No warnings or errors.
9. Run: `dotnet test`. All new tests pass. All existing tests pass (FR-018).
10. Capture test output for the milestone commit.

*Validation*: 4 new tests pass (or 5 with the golden file). 0 existing tests fail.

### Phase 7 — CLI era tag

*Goal*: `WowViewer.Tool.Inspect m2 inspect --input <file>` prints the era tag.

*Steps* (≤10):

1. Read `wow-viewer/src/tools/WowViewer.Tool.Inspect/` to find the `m2 inspect` command implementation. (Implementation slice does this.)
2. Update the command to call the dispatcher (or the new reader's `ReadEra` helper) and print "Era: <era tag>" as the first line of output.
3. For chunked `MDLX` files, the era tag is "Era: MDLX (chunked)".
4. For 1.12.1 files, the era tag is "Era: 1.12.1 (MD20 v0x100)" or "Era: 1.12.1 (MD20 v0x101)".
5. For 3.3.5 files, the era tag is "Era: 3.3.5 (MD20 v0x108)".
6. For 2.x files, the command exits non-zero with the `NotSupportedException` message.
7. Build: `dotnet build`. No warnings or errors.
8. Run the command against the 1.12.1 test fixture. Output includes "Era: 1.12.1 (MD20 v0x100)" as the first line. Capture output for the milestone commit.
9. Run the command against an existing 3.3.5 fixture. Output includes "Era: 3.3.5 (MD20 v0x108)" as the first line. Existing behavior unchanged.
10. Run the command against a 0.5.3 fixture. Output includes "Era: MDLX (chunked)" as the first line. Existing behavior unchanged.

*Validation*: The CLI's era tag is consistent across all three lanes.

### Phase 8 — Doc sync + memory bank

*Goal*: Spec 043 is updated to point at 048. The research doc is cleaned up. The memory bank is updated.

*Steps* (≤10):

1. Read `wow-viewer/specs/043-m2-chunked-mdx-classic-support/spec.md`. Edit the "Input" section to remove 1.12.1 from the chunked lane list (move it to a "see also spec 048" reference). The chunked lane is 0.5.3, 0.7.0, 0.8.0; 1.12.1 is MD20 and lives in 048.
2. Add a "Status (2026-06-05)" note to spec 043 explaining the 048 split: 043 is the chunked lane (0.5.3/0.7.0/0.8.0), 048 is the 1.12.1 MD20 lane. Future 049 is the 2.x TBC lane.
3. Edit `wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md` to add a banner: "Status: This research is the source of truth for the 1.12.1 native M2 contract. The implementation slice is spec 048 (`specs/048-m2-1121-era-aware-md20-reader/spec.md`)." The "Open Questions" section is updated to mark OQ-1..OQ-6 with their resolution status from the 048 implementation.
4. Edit `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` (the earlier M2 research doc) to add a 1.12.1 section that points at the new research doc and spec 048.
5. Update `gillijimproject_refactor/memory-bank/activeContext.md` under "M2 / Runtime Continuity" to add: "Spec 048 (1.12.1 era-aware MD20 reader) — 2026-06-05: spec/plan/tasks written; implementation pending test fixture staging and 043/doc/memory updates." Add a "Next Likely Steps" block: stage 1.12.1 .mdx fixture, implement Phase 1 (constants), then Phases 2-7.
6. Update `gillijimproject_refactor/memory-bank/progress.md` under the same section: append "2026-06-05 — Spec 048 written (1.12.1 MD20 era-aware reader). 043 revised to defer 1.12.1 to 048. Research doc cleanup pending. Implementation blocked on test fixture."
7. (Optional) Update `wow-viewer/README.md` if it has a "current specs in flight" section, add 048.
8. Build: `dotnet build`. No warnings or errors.
9. Run: `dotnet test`. All tests pass.
10. Commit. The commit message follows the project's style: a 1-line "why" summary + a bullet list of changes. Suggested: "Spec 048: 1.12.1 era-aware MD20 reader + 043 lane split".

*Validation*: Spec 043 reads correctly (1.12.1 is gone from the chunked lane; spec 048 is referenced). The research doc has the banner. The memory bank has the 048 entry.

## Complexity Tracking

No constitution violations. The implementation is a sibling reader + a 1-line dispatcher branch + CLI string change + docs. No new project files, no cross-cutting refactors, no shared-state additions.

## Open Questions (F-list, separate from research OQ-list)

- **F-1**: Which 1.12.1 `.mdx` to use as the test fixture, and how to stage it. The implementation slice is responsible. (Phase 0.)
- **F-2**: The exact 1.12.1 bone stride (`BoneStride`) is not in the Ghidra trace. The 048 MVP either (a) reads bones with a `// TODO: confirm` placeholder, or (b) defers bones entirely. The decision is at Phase 4 step 2.
- **F-3**: Whether the 0x101-only table at `0xDC/0xE0` needs a new `M2ModelDocument` field. The 048 MVP either (a) reads and stores the table in a side dictionary on the document, or (b) defers the table. The decision is at Phase 4 step 6. Per A-006, schema changes are out of scope; option (b) is the safer default.
- **F-4**: The view record's 9 nested sub-tables. The 048 MVP reads them with bounds checks and either (a) stores them in a side dictionary, or (b) throws "not yet mapped" if the schema cannot absorb them. The decision is at Phase 5 step 6.

These are isolated, deferrable questions. None of them block the spec or the 043/doc/memory updates.
