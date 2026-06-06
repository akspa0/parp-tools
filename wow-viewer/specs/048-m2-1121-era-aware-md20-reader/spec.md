# Feature Specification: 048 — 1.12.1 Era-Aware MD20 Reader

**Feature Branch**: `048-m2-1121-era-aware-md20-reader`
**Created**: 2026-06-05
**Status**: In Progress (Phases 0-12 done, Phase 13 deferred)
**Input**: Ghidra trace of `WoW.exe` `Build 5875` (1.12.1 Vanilla, image base `0x00400000`, .text `0x00401000-0x007fefff`) and the research document `wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md`. The 1.12.1 native client loads model files via the function at `FUN_0071cdf0` (`M2Model.cpp`), which expects the flat `MD20` magic (`0x3032444D`) with version `0x100` or `0x101` and a (count, offset) pointer table — NOT the chunked `MDLX` format that spec 043 covers. The current `M2ModelReader` in `wow-viewer/src/core/WowViewer.Core.IO/M2/M2ModelReader.cs` is hard-coded to 3.3.5 stride constants (`ViewCountOffset=0x44`, `SequenceStride=0x40`, `LightStride=0x9C`, `ParticleStrideClassic=0x1dc`, etc.) and the dispatcher routes anything that is not `MDLX` magic to it. Result: 1.12.1 `.mdx` files (which carry `MD20` magic with the legacy `.mdx` extension) are silently parsed as 3.3.5 MD20 and produce invalid models. Spec 043 must be revised to acknowledge 1.12.1 is MD20, not MDLX. The fix is a sibling era-aware MD20 reader that dispatches on the `version` field at offset `0x04` of the header.

**Implementation Status (2026-06-06)**: Phases 0-8 of the original plan are complete. The 048 reader was extended in place (Phases 9-12) to parse the 1.12.1 inline geometry (vertex indices, positions, normals, UVs, triangles, batches) and attach it to the `M2ModelDocument` as a new `InlineEra1121Geometry` property. All 9 048 tests pass with real 1.12.1 bear.m2 data (test runtime 3-4 seconds confirms real data, not silent no-op). The viewer's load path (`WorldAssetManager.LoadMdxModel`) is NOT yet wired to use the new geometry — that is Phase 13, deferred to a follow-up session. Until Phase 13 lands, the viewer still produces the 0-vertex fallback error for 1.12.1 .mdx files.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read a 1.12.1 .mdx as a real M2 model (Priority: P1)

A user points the viewer or the `WowViewer.Tool.Inspect` CLI at a 1.12.1 `.mdx` file (e.g. a creature or item model from the staged 1.12.1 client). The reader detects `MD20` magic, reads the `version` field at offset `0x04`, and dispatches to a 1.12.1-aware reader when `version ∈ {0x100, 0x101}`. The reader emits an `M2ModelDocument` whose header-level scalar fields (`GlobalLoops`, `Sequences`, `Colors`, `TextureWeights`, `TextureTransforms`, `Lights`, `BoundsMin/Max/Radius`, `EmbeddedSkinProfileCount/Offset`, `ModelName`) are read with 1.12.1-correct offsets and strides, and whose per-record arrays are walked with 1.12.1-correct stride constants. The downstream renderer/runtime sees the same `M2ModelDocument` shape it already consumes.

**Why this priority**: 1.12.1 is the most-requested pre-WotLK Vanilla client. Without this slice, 1.12.1 `.mdx` files silently produce garbage through the 3.3.5 reader and the viewer cannot show any pre-2.0 model. This is the minimum-viable "the viewer works on 1.12.1" deliverable.

**Independent Test**: A `WowViewer.Core.IO.Tests` integration test that takes the staged 1.12.1 `.mdx` bytes (test fixture path supplied by the implementation slice; staging is out of scope here), parses them with the new `M2Era1121ModelReader`, and asserts: (a) `M2ModelDocument.Version ∈ {0x100, 0x101}`, (b) `M2ModelDocument.SequenceCount == sequence_count_for_test_model`, (c) `M2ModelDocument.ViewCount == view_count_for_test_model`, (d) `M2ModelDocument.LightCount == light_count_for_test_model`, (e) all parsed records have valid finite floats (no NaN, no Infinity), (f) no exception is thrown. The 3.3.5 reader is invoked for `version > 0x101` and produces the same outputs it does today.

**Acceptance Scenarios**:

1. **Given** a 1.12.1 `.mdx` file with `MD20` magic and `version == 0x100`, **When** read via the dispatcher, **Then** the new `M2Era1121ModelReader` is invoked and returns a non-null `M2ModelDocument` whose scalar header values match the 1.12.1 Ghidra-traced offsets (`0x08` name count, `0x0C` name offset, `0x10` flags, `0x14` global loop count, `0x18` global loop offset, `0x1C` sequence count, `0x20` sequence offset, `0x2C` bone count, `0x30` bone offset, `0x3C` view count, etc.) and whose per-record array strides match the 1.12.1 constants from the research doc (sequence stride `0x6C`, light stride `0x0C`, etc.).
2. **Given** a 1.12.1 `.mdx` file with `MD20` magic and `version == 0x101`, **When** read via the dispatcher, **Then** the same `M2Era1121ModelReader` is invoked; the reader recognizes the version and reads the `0x101`-only table at `0xDC/0xE0` (29 nested sub-tables, 0x1F8/entry, per research doc open question OQ-1) without throwing.
3. **Given** a 3.3.5 `.m2` file with `MD20` magic and `version == 0x108` (or higher), **When** read via the dispatcher, **Then** the existing 3.3.5 `M2ModelReader` is invoked, NOT the new 1.12.1 reader. The 3.3.5 output is bit-for-bit identical to the pre-048 build (verified by golden-file comparison).
4. **Given** a file whose magic is not `MD20` and not `MDLX` (e.g. an SKIN file, a corrupt file, a 2.x MD20 with version 0x104), **When** read via the dispatcher, **Then** the dispatcher throws a clear "unsupported M2 magic" error naming the magic it found, exactly as it does today. The 1.12.1 reader is not invoked.

---

### User Story 2 - 1.12.1 cvar / flag / version coverage matches the native client (Priority: P1)

The 1.12.1 native client has a smaller M2 cvar set than 3.3.5 and a different flag word bit layout. The 1.12.1 reader must NOT reuse the 3.3.5 cvar or flag helpers. It exposes only the cvars and flag bits the 1.12.1 binary uses, and its public surface flags itself as "1.12.1 era" so the runtime and tool layers know not to mix the two surfaces.

**Why this priority**: Reusing 3.3.5 cvar/flag logic on 1.12.1 data will produce subtle renderer misconfigurations that fail visually rather than throw. The cost of getting this wrong is a renderer that "works" with wrong materials/blend modes/depth-test state.

**Independent Test**: Two unit tests in `WowViewer.Core.IO.Tests`. (a) A test that constructs a synthetic 1.12.1 MD20 header, writes a known flag word (`0x8000_0000`), parses it with `M2Era1121ModelReader`, and asserts the resulting `M2ModelDocument.Flags` equals `0x8000_0000` exactly (no 3.3.5 bit-mangling). (b) A test that asserts the 1.12.1 reader has no public surface referencing the 3.3.5-only cvars (`M2BatchParticles`, `M2ForceAdditiveParticleSort`) — the test uses `Type.GetProperties()` reflection on the 1.12.1 reader's public type and asserts the cvar property set is exactly the 1.12.1 set from `m2-mdx-1121-native-trace-2026-06-05.md` section "M2 cvar / console variable set" (8 cvars: `M2FasterDebug`, `M2Faster`, `M2BatchDoodads`, `M2UsePixelShaders`, `M2UseShaders`, `M2UseThreads`, `M2UseClipPlanes`, `M2UseZFill`).

**Acceptance Scenarios**:

1. **Given** a 1.12.1 MD20 header, **When** parsed, **Then** the resulting `M2ModelDocument.Flags` is the raw `uint` at offset `0x10`. The 1.12.1 reader does NOT apply the 3.3.5 flag bit-translation table.
2. **Given** the 1.12.1 reader's public surface, **When** inspected via reflection, **Then** the cvar accessor set equals the 1.12.1 set (8 cvars). The 3.3.5-only cvars are absent.
3. **Given** a 1.12.1 model with `flags & 0x08` (M2_USE_SHADERS, per the 1.12.1 binary's local convention), **When** the runtime selects shaders, **Then** the runtime uses 1.12.1 shader selection logic, NOT 3.3.5 shader selection logic. (This slice only validates the flag passes through cleanly; runtime shader selection is a downstream slice.)
4. **Given** a 3.3.5 model with `flags & 0x200` (M2_USE_PARTICLES_BATCH, a 3.3.5-only flag), **When** parsed, **Then** the 3.3.5 reader handles the bit (existing behavior). The 1.12.1 reader is not invoked.

---

### User Story 3 - 0.5.3 / 0.7.0 / 0.8.0 era chunked MDLX remains a sibling lane (Priority: P1, hard non-goal)

The pre-1.x era (0.5.3, 0.7.0, 0.8.0) uses the chunked `MDLX` format that spec 043 already covers. The new era-aware dispatch must NOT regress the chunked lane. Any `MDLX` magic file goes to the existing `M2ChunkedModelReader` exactly as it does today.

**Why this priority**: This is the same hard non-goal as spec 043 US-4 and spec 037. The 0.5.3 lane is the oldest of the three pre-2.0 eras and the most likely to break under a dispatcher refactor.

**Independent Test**: Re-run the existing `M2ChunkedModelReader` test suite. All tests pass with zero changes. Additionally, a new test asserts that an `MDLX` magic byte sequence is dispatched to the chunked reader, not the 1.12.1 reader, regardless of any byte following the magic.

**Acceptance Scenarios**:

1. **Given** any file with `MDLX` magic, **When** dispatched, **Then** the chunked reader is invoked, the 1.12.1 reader is not invoked, and the output is bit-for-bit identical to the pre-048 build.
2. **Given** the dispatcher's new version-field check, **When** a 0.5.3 `.mdx` is loaded, **Then** the dispatcher reads the `MDLX` magic first, short-circuits to the chunked lane, and never reads the `version` field at offset `0x04` (which is meaningless for `MDLX`).

---

### User Story 4 - 2.x TBC era is explicitly deferred (Priority: P3, scope guard)

The 2.x (TBC) era MD20 is not covered by this spec. The dispatcher treats any `MD20` file with `version > 0x101` and `< 0x108` (i.e. 2.0.0..2.4.3-ish) as "out of era" and throws a clear, actionable error. This is a scope guard, not a feature.

**Why this priority**: 2.x TBC has its own stride and view-record changes. We do not have a 2.x `WoW.exe` available to Ghidra-trace, and we do not have a staged 2.x client with extracted test data. Speculating about 2.x strides would be guessing. The dispatcher must reject 2.x explicitly so a future "049" spec can pick it up.

**Independent Test**: A unit test in `WowViewer.Core.IO.Tests` that constructs a synthetic 8-byte `MD20` magic + version 0x104 header, calls the dispatcher, and asserts it throws `NotSupportedException` (or the project's chosen "era-not-supported" exception) with a message naming the version and pointing at the open spec slot (049).

**Acceptance Scenarios**:

1. **Given** an `MD20` magic file with `version == 0x104` (or any 2.x value), **When** dispatched, **Then** the dispatcher throws an exception whose message names the version and says "2.x TBC era is not yet supported; tracked under spec 049."
2. **Given** the 048 implementation slice, **When** complete, **Then** `wow-viewer/docs/architecture/m2-native-client-research-2026-03-31.md` (or a successor research doc) contains a one-paragraph "2.x TBC" section that lists the missing Ghidra binary, the missing staged client with extracted test data, and the 049 spec slot. The section is a placeholder, not a research finding.

---

### Edge Cases

- **File with magic `MD20` but length < minimum header size** (`0x110` is the 3.3.5 minimum; 1.12.1's actual minimum is smaller and is to be confirmed from the Ghidra trace during implementation): The reader throws `InvalidDataException` with a clear "file too small" message and the file length, exactly as the 3.3.5 reader does today.
- **File with `MD20` magic and `version == 0x100` but with garbage in the per-record arrays** (e.g. sequence count `0xFFFFFFFF`, sequence offset `0x00000000`): The reader must NOT silently produce an empty document. It must throw with the offending count/offset named.
- **File with `MD20` magic and `version == 0x101` but the `0x101`-only table at `0xDC/0xE0` (29 sub-tables, 0x1F8/entry) is truncated**: The reader attempts to read it, finds the span out of range, and either (a) falls back to a "0x100 equivalent" parse with a warning, or (b) throws with the offset and span named. Behavior is decided at implementation time and tested in the slice.
- **A 1.12.1 `.mdx` whose companion `.skin` files exist in the same directory but are written for 1.12.1 specifically (e.g. with the 1.12.1 `SKIN` magic size variant, not 3.3.5's)**: The existing `M2SkinReader` is FROZEN per spec 037; this spec does NOT add 1.12.1 skin reading. The 1.12.1 `.skin` parsing is a follow-up slice (e.g. spec 050). The 1.12.1 `.mdx` reader must not crash when the companion `.skin` is unreadable — it must mark skins as missing and continue.
- **A 1.12.1 `.mdx` whose companion `.anim` files exist**: Same as skins — 1.12.1 `.anim` parsing is a follow-up slice. The reader must not crash.
- **A 1.12.1 `.mdx` whose `MODL.Name` (mapped to `ModelName` in 1.12.1's MD20 at offset `0x08`/`0x0C`) is empty or all zeros**: The reader produces a document with `ModelName == null` and continues. No exception.
- **A 1.12.1 `.mdx` whose view record (per the Ghidra trace, 0x2C stride, 9 nested sub-tables) is malformed**: The reader attempts the read, finds the span out of range, and either skips the view record with a warning or throws with the view index and span named. Behavior is decided at implementation time and tested in the slice.
- **A 3.3.5 `.m2` file with `version == 0x108` that the user opens immediately after opening a 1.12.1 `.mdx`**: The dispatcher's per-call read of the magic + version must be stateless. No static state leaks between calls.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Reader (US-1, US-2)

- **FR-001**: A new reader class `M2Era1121ModelReader` MUST be created in a new namespace `WowViewer.Core.IO.M2Era1121` and a new folder `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/`. No existing files in `WowViewer.Core.IO/M2/` (the 3.3.5 reader) are modified.
- **FR-002**: The new reader MUST accept the same input shapes as `M2ModelReader` (`string path` and `Stream stream + string sourcePath`).
- **FR-003**: The new reader MUST validate `MD20` magic (`0x3032444D` LE) as the first 4 bytes. Mismatch throws `InvalidDataException` with the actual magic named, exactly as the 3.3.5 reader does.
- **FR-004**: The new reader MUST validate the `version` field at offset `0x04` is in the set `{0x100, 0x101}`. Any other value throws `NotSupportedException` with the actual version named and a message pointing at the deferred-2.x handling. (This is a defensive check; the dispatcher is the primary guard, but the reader should not silently accept wrong versions.)
- **FR-005**: The new reader MUST use the 1.12.1 offset/stride constants from the research doc `m2-mdx-1121-native-trace-2026-06-05.md` (sections "Header layout" and "Stride differences vs 3.3.5"). The constant set is the source of truth; this spec does not enumerate them all. The implementation slice must transcribe the constants from the research doc into a `M2Era1121Constants` static class.
- **FR-006**: The new reader MUST emit the same `M2ModelDocument` shape as `M2ModelReader`. No `M2ModelDocument` schema changes are part of this spec. If a 1.12.1 record cannot be mapped into the existing schema (e.g. the `0x101`-only 0xF8/entry batch table from OQ-1), the reader throws a clear "not yet mapped to M2ModelDocument" error and lists the open question from the research doc.
- **FR-007**: The new reader MUST use a separate flag/flag-bit handling surface from the 3.3.5 reader. It MUST NOT call into `M2ModelReader`'s flag helpers. The 1.12.1 `Flags` field at offset `0x10` is passed through to `M2ModelDocument.Flags` as a raw `uint`.
- **FR-008**: The new reader MUST NOT introduce a cvar accessor on the 1.12.1 public surface for any of the 3.3.5-only cvars (`M2BatchParticles`, `M2ForceAdditiveParticleSort`, or any other cvar absent from the 1.12.1 native binary). The cvar accessor surface is a downstream concern (runtime) and is out of scope for this spec; if the reader exposes any cvar surface at all, it exposes only the 1.12.1 set.

#### Dispatch (US-1, US-2, US-3, US-4)

- **FR-009**: The existing dispatcher in `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs` MUST be extended to read the `version` field at offset `0x04` when the magic is `MD20`. The dispatch logic is: `MDLX` → chunked reader (unchanged); `MD20` + `version ∈ {0x100, 0x101}` → new `M2Era1121ModelReader`; `MD20` + `version ∈ {0x108, ...}` → existing 3.3.5 `M2ModelReader` (unchanged); `MD20` + `version ∈ (0x101, 0x108)` → `NotSupportedException` naming the version; anything else → existing "unsupported magic" error.
- **FR-010**: The dispatcher's version-field read MUST be limited to the first 8 bytes of the file (`magic` + `version`). No additional header fields are read at dispatch time. This keeps the dispatch fast and avoids "read a thousand bytes to decide the era" overhead.
- **FR-011**: The dispatch MUST be implemented as a new branch in the existing dispatcher file (`M2ModelReaderDispatcher.cs`). No new dispatcher file is created. The 3.3.5 reader's entry point is not modified.
- **FR-012**: The 0.5.3 / 0.7.0 / 0.8.0 chunked MDLX lane is a sibling that does not participate in the version-field read. The dispatcher short-circuits on `MDLX` magic before reading the version field. US-3 acceptance scenarios enforce this.

#### Robustness (US-1, US-2)

- **FR-013**: The 1.12.1 reader MUST throw `InvalidDataException` with a clear message and the offending count/offset when a per-record array reports `count > 0` but `offset == 0`, exactly as `M2ModelReader` does.
- **FR-014**: The 1.12.1 reader MUST throw `InvalidDataException` when any read span (count × stride) overruns the file. The message names the label, the count, the offset, the stride, and the file length.
- **FR-015**: The 1.12.1 reader MUST validate all non-finite floats (`NaN`, `Infinity`) in vector/quaternion fields and throw `InvalidDataException` with the field name and offset. This matches the 3.3.5 reader's behavior.
- **FR-016**: The 1.12.1 reader MUST NOT load companion `.skin` or `.anim` files in this spec. When a companion file is missing or unreadable, the reader logs to `Console.Error` and continues with an empty/missing-skin document. The follow-up spec 050 covers 1.12.1 skin/anim parsing.

#### Validation (US-1, US-2, US-3, US-4)

- **FR-017**: At least four `WowViewer.Core.IO.Tests` integration tests MUST be added in this spec:
  - **US-1 test**: Parse a 1.12.1 `.mdx` (test fixture path from `I:\parp\parp-tools\output\tmp\wowarchive-clients/1.X_Retail_Windows_enUS_1.12.1.5875/`; the implementation slice is responsible for staging a representative `.mdx` and recording the test fixture path) and assert `M2ModelDocument` shape.
  - **US-1 negative test**: Parse a 3.3.5 `.m2` and assert the 1.12.1 reader is NOT invoked (verified by a dispatch-trace counter or by a non-finite-3.3.5-result check).
  - **US-2 test**: Synthetic 1.12.1 MD20 with `flags == 0x80000000` round-trips through `M2Era1121ModelReader` and lands in `M2ModelDocument.Flags` unchanged.
  - **US-3 regression test**: A file with `MDLX` magic dispatches to the chunked reader, not the 1.12.1 reader, regardless of the next 4 bytes.
  - **US-4 test**: An `MD20` file with `version == 0x104` throws `NotSupportedException` with the 049 spec slot mentioned.
- **FR-018**: All existing `WowViewer.Core.IO.Tests` M2 family tests MUST continue to pass with zero modifications. Verified by `dotnet test` showing the pre-048 test count is unchanged.
- **FR-019**: The 048 implementation MUST NOT modify any file under `wow-viewer/src/core/WowViewer.Core.IO/M2/` (3.3.5 reader is frozen per spec 037). Verified by `git diff --stat` showing 0 changes in that folder.
- **FR-020**: The 048 implementation MUST NOT modify any file under `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/` that the existing chunked lane depends on, except `M2ModelReaderDispatcher.cs` (the single dispatch seam per FR-011). All other chunked files (chunk parsers, read result, summary, conversion) are frozen.

#### CLI (US-1)

- **FR-021**: The existing `WowViewer.Tool.Inspect m2 inspect --input <file>` command MUST continue to work for `MD20` files of any era. The 1.12.1 `.mdx` case is the new behavior; the 3.3.5 `.m2` case is unchanged. The command prints the same summary fields for 1.12.1 as for 3.3.5 (version, view count, sequence count, light count, etc.).
- **FR-022**: The `WowViewer.Tool.Inspect m2 inspect` command MUST print the detected era in its output (e.g. "Era: 1.12.1 (MD20 v0x100)" or "Era: 3.3.5 (MD20 v0x108)") so the user can tell at a glance which path was taken. The era tag is computed by the dispatcher.

### Key Entities

- **M2Era1121ModelReader**: the new reader class. Lives in `WowViewer.Core.IO.M2Era1121` namespace. Pure read-only; emits `M2ModelDocument` (existing type) with 1.12.1-correct fields.
- **M2Era1121Constants**: static class of 1.12.1 offset/stride constants. Transcribed from the research doc. One file, one namespace, no logic.
- **M2Era1121Version**: enum of supported 1.12.1 versions (`Unknown`, `V100`, `V101`). Used by the dispatcher and reader to select the right table-walk for `0x101`-only sub-tables.
- **M2Era1121EraTag**: enum used by the dispatcher to identify the era for CLI output and logging (`Mdlx = 0`, `Md20_1X_V100 = 1`, `Md20_1X_V101 = 2`, `Md20_3X_V108 = 3`, `Unknown = 99`).
- **M2ModelReaderDispatcher** (existing, modified): the dispatch seam. Reads magic + version, routes to the right reader. Single new `if (magic == MD20 && version ∈ {0x100, 0x101})` branch.
- **M2ModelDocument** (existing, unchanged): the canonical output type. The 1.12.1 reader's output matches the 3.3.5 reader's output shape (same property names, same semantics) so the downstream runtime/tensor packer is unchanged.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A 1.12.1 `.mdx` (e.g. `Creature\Bear\Bear.mdx` from the staged 1.12.1 client, or a 1.12.1-spell-effect model) loads via `M2Era1121ModelReader` and produces a non-null `M2ModelDocument` with `Version ∈ {0x100, 0x101}`, `SequenceCount > 0` when the model has sequences, `ViewCount > 0` when the model has views, and all parsed records have valid finite floats. Verified by `WowViewer.Core.IO.Tests` integration test.
- **SC-002**: A 3.3.5 `.m2` file loads via the existing 3.3.5 `M2ModelReader` with byte-for-byte identical `M2ModelDocument` output to the pre-048 build. Verified by a golden-file comparison test (the existing 3.3.5 tests are the de-facto golden files; running them with 0 changes passes the criterion).
- **SC-003**: A 0.5.3 / 0.7.0 / 0.8.0 chunked `.mdx` loads via the existing `M2ChunkedModelReader` with byte-for-byte identical `M2ChunkedReadResult` output to the pre-048 build. Verified by re-running the existing chunked test suite with 0 changes.
- **SC-004**: An `MD20` file with `version == 0x104` (2.x TBC sentinel) throws `NotSupportedException` whose message names the version and mentions spec 049. Verified by a unit test in `WowViewer.Core.IO.Tests`.
- **SC-005**: `WowViewer.Tool.Inspect m2 inspect --input <path-to-1.12.1-.mdx>` exits 0 on a valid 1.12.1 `.mdx` and prints the summary with the "Era: 1.12.1 (MD20 v0x100/0x101)" tag.
- **SC-006**: `git diff --stat` of the 048 implementation shows 0 changes in `wow-viewer/src/core/WowViewer.Core.IO/M2/` (3.3.5 reader folder is frozen).
- **SC-007**: The 1.12.1 reader's public surface (via `Type.GetProperties()`) exposes only the 1.12.1 cvar set (8 cvars from the research doc). Verified by a unit test that asserts the cvar set equality.

---

## Assumptions

- **A-001**: The 1.12.1 `WoW.exe` (`Build 5875`) is the canonical 1.12.1 reference binary. It is already Ghidra-loaded and the function anchors are recorded in the research doc. No additional Ghidra work is required for the 048 implementation slice beyond transcribing the constants and stride tables.
- **A-002**: The research doc `m2-mdx-1121-native-trace-2026-06-05.md` is the source of truth for the 1.12.1 header layout, view-record structure, relocator catalog, and stride differences. If the implementation slice finds a constant in the research doc is wrong, the doc is updated in the same commit; the code is not.
- **A-003**: The 6 open questions in the research doc (OQ-1 through OQ-6) are NOT blockers for the 048 MVP slice. Where the research doc is uncertain (e.g. the exact 0x101-only sub-table layout from OQ-1), the 1.12.1 reader either reads the table conservatively (with bounds checks) or throws "not yet mapped" per FR-006. The follow-up spec 049 (or 050) addresses the open questions once a real 1.12.1 `.mdx` test fixture is staged and the open questions can be resolved by inspecting the data.
- **A-004**: The 2.x TBC era is out of scope. Spec 049 (not yet created) covers 2.x when a TBC `WoW.exe` and a staged 2.x client with extracted test data are available. The 048 dispatcher's 2.x rejection (US-4) is a scope guard, not a feature.
- **A-005**: The 1.12.1 `.skin` and `.anim` companion file formats are out of scope. Spec 050 covers 1.12.1 companion files. The 048 slice produces a document with empty/missing skins and anims when companions are present-but-unreadable.
- **A-006**: The `M2ModelDocument` schema is sufficient for 1.12.1 output. If the implementation slice finds the 1.12.1 records do not fit (e.g. the 0x101-only batch table from OQ-1 needs a new field), the implementation slice raises a "schema change needed" flag and stops; the schema change is a follow-up spec. The 048 slice does not silently invent schema.
- **A-007**: The 1.12.1 cvar / console-variable accessor surface is a runtime concern, not a reader concern. The 048 reader does not expose cvars; it only flags the era. The runtime's cvar lookup is a follow-up slice.
- **A-008**: Animation playback and particle rendering for 1.12.1 models are out of scope for 048 (and were out of scope for 043). The reader populates the data; the runtime does not consume 1.12.1-specific data yet.

---

## Open Questions

- **OQ-1**: 0x101-only `0xDC/0xE0` table (29 nested sub-tables, 0x1F8/entry, per the research doc) — what is the semantic content? Best guess: a consolidated per-batch record (mesh index, material index, blend mode, etc.) that the 0x100 era stores inline at a different offset. The 048 slice reads the table with bounds checks and passes the raw records to the document (or throws "not yet mapped" per A-006).
- **OQ-2**: 1.12.1 flag word bit meanings. The Ghidra trace identifies the bit positions; the bit semantics (which bits mean "use shaders", "use particles", "is billboard", etc.) need to be cross-referenced against real-data. The 048 slice passes the flag through as a raw `uint`; bit semantics are a runtime concern (A-007).
- **OQ-3**: 1.12.1 vertex table — confirmed to be separate position + normal tables in the Ghidra trace, but the exact ordering of UV sets, blend indices, and bone weights is unknown. The 048 slice uses the 0xC/entry position stride from the research doc for the position table and reads the normal table at the offset recorded in the research doc. UV/blend/bone are not yet mapped (per A-006 they remain 0/empty until the schema is extended).
- **OQ-4**: 1.12.1 light record 0xC/entry is suspiciously small. Best guess: `(pos, radius)` only; the rest is runtime default. The 048 slice reads the 0xC stride as a minimal light record and stores it in the document's `M2LightDefinition`. The full light record (color, intensity, attenuation) is set to runtime defaults. This may need a follow-up slice.
- **OQ-5**: 1.12.1 cvar bit mapping (0x1/0x2/0x4/0x8/0x10/0x20/0x40/0x80/0x100) — what is the read path? Not decompiled in the Ghidra trace. Per A-007, this is a runtime concern; the 048 reader does not need to know.
- **OQ-6**: 1.12.1 likely has no external `.anim` files (the Ghidra trace shows no `.anim` references in `M2Cache.cpp`'s normalize step). The 048 slice does not attempt to load `.anim`; if a 1.12.1 `.mdx` has an `.anim` in the same directory, the reader ignores it. Spec 050 may revisit this if real-data shows otherwise.

---

## Notes

- This spec is an **implementation slice** (not research). The format is well-understood via the Ghidra trace; the open questions are explicit and isolated.
- All new code lives in `wow-viewer/src/core/WowViewer.Core.IO/M2Era1121/` (new folder, new namespace). The 3.3.5 reader folder is frozen (per spec 037); the chunked reader folder is mostly frozen (only the dispatcher is touched, per FR-020).
- The hard non-goal from spec 037 and spec 043 is re-affirmed: 3.3.5 path is bit-for-bit unchanged; chunked 0.5.3/0.7.0/0.8.0 path is bit-for-bit unchanged.
- 2.x TBC era is explicitly out of scope. The dispatcher throws `NotSupportedException` for 2.x versions; spec 049 is the follow-up slot.
- 1.12.1 `.skin` and `.anim` companion files are out of scope. The 048 reader ignores companions; spec 050 is the follow-up slot.
- Animation playback, particle rendering, cvar-driven shader selection, and 1.12.1 runtime behavior are out of scope. The 048 reader reads the data; the runtime does not consume 1.12.1-specific data yet.
- The research doc `m2-mdx-1121-native-trace-2026-06-05.md` is the canonical source of truth for the 1.12.1 constants. The implementation slice transcribes, does not re-research.
- Spec 043 must be updated to acknowledge 1.12.1 is MD20, not MDLX, and that the 1.12.1 lane is now spec 048. This is a follow-up documentation edit, not a code change.

---

## Cross-References

- `wow-viewer/docs/architecture/m2-mdx-1121-native-trace-2026-06-05.md` — full Ghidra trace, header layout, view-record structure, relocator catalog, stride differences, open questions.
- `wow-viewer/specs/043-m2-chunked-mdx-classic-support/spec.md` — chunked `MDLX` lane (0.5.3, 0.7.0, 0.8.0). Does NOT cover 1.12.1 (uses different magic).
- `wow-viewer/specs/037-m2-301-embedded-views-adapter/spec.md` — 3.3.5 era. Hard non-goal reaffirmed: 3.3.5 reader is frozen.
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2ModelReader.cs` — 3.3.5 MD20 reader (frozen).
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2GeometryReader.cs` — 3.3.5 geometry reader (frozen).
- `wow-viewer/src/core/WowViewer.Core.IO/M2/M2SkinReader.cs` — 3.3.5 skin reader (frozen).
- `wow-viewer/src/core/WowViewer.Core.IO/M2Chunked/M2ModelReaderDispatcher.cs` — existing dispatch seam. Modified only at the version-field branch.
- `wow-viewer/src/core/WowViewer.Core/M2/M2ModelDocument.cs` — canonical output type. Unchanged.
